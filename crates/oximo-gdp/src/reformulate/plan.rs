//! Shared row planning, validation, and atomic application.

use super::big_m::{method_m, validate_m};
use super::logic::LogicCompiler;
use super::{GdpError, GdpReformulationOptions};
use oximo_core::gdp::{
    BooleanId, GdpBooleanMapping, GdpPendingStarts, GdpReformulationReport,
    GdpReformulationReportData,
};
use oximo_core::{
    Constraint, ConstraintId, Expr, ExprId, ExprNode, Interval, Model, ParamId,
    ScalarDynamicFunction, UnaryOp, VarId,
};
use oximo_expr::SignedExpr;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Debug, Default)]
pub(super) struct Linear {
    pub(super) constant: f64,
    pub(super) terms: Vec<(VarId, f64)>,
}

impl Linear {
    pub(super) fn constant(x: f64) -> Self {
        Self { constant: x, terms: Vec::new() }
    }

    pub(super) fn variable(v: VarId) -> Self {
        Self { constant: 0.0, terms: vec![(v, 1.0)] }
    }

    pub(super) fn scaled(mut self, scale: f64) -> Self {
        self.constant *= scale;

        for (_, c) in &mut self.terms {
            *c *= scale;
        }
        self
    }

    pub(super) fn add(mut self, rhs: Self) -> Self {
        self.constant += rhs.constant;
        self.terms.extend(rhs.terms);
        self
    }

    pub(super) fn not(self) -> Self {
        Self::constant(1.0).add(self.scaled(-1.0))
    }

    pub(super) fn normalize(
        &mut self,
        slots: &mut FxHashMap<VarId, usize>,
    ) -> Result<(), GdpError> {
        slots.clear();
        let mut length = 0;

        for i in 0..self.terms.len() {
            let (variable, coefficient) = self.terms[i];

            if !coefficient.is_finite() {
                return Err(GdpError::NumericOverflow);
            }

            if let Some(&slot) = slots.get(&variable) {
                self.terms[slot].1 += coefficient;

                if !self.terms[slot].1.is_finite() {
                    return Err(GdpError::NumericOverflow);
                }
            } else {
                slots.insert(variable, length);
                self.terms[length] = (variable, coefficient);
                length += 1;
            }
        }

        self.terms.truncate(length);
        self.terms.retain(|(_, coefficient)| *coefficient != 0.0);

        if !self.constant.is_finite() {
            return Err(GdpError::NumericOverflow);
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Row {
    name: String,
    residual: Vec<SignedExpr>,
    linear: Linear,
    lower: f64,
    upper: f64,
    canonical: bool,
}

#[derive(Debug)]
pub(super) struct PlanBuilder {
    variables: Vec<(VarId, String)>,
    rows: Vec<Row>,
    pub(super) report: GdpReformulationReportData,
    pub(super) locked_variables: FxHashSet<VarId>,
    pub(super) locked_parameters: FxHashSet<ParamId>,
    dependency_nodes: FxHashSet<ExprId>,
    dependency_stack: Vec<ExprId>,
    expression_nodes: usize,
    next_variable: usize,
    next_constraint: usize,
    pub(super) logic: LogicCompiler,
}

#[derive(Debug)]
pub(super) struct ValidatedPlan(PlanBuilder);

impl PlanBuilder {
    pub(super) fn build(
        model: &Model,
        options: &GdpReformulationOptions,
    ) -> Result<ValidatedPlan, GdpError> {
        let data = model.gdp();

        validate_m(options.fallback)?;
        validate_m(method_m(options.default_method))?;

        for (&handle, &method) in &options.methods {
            if handle.model_id() != model.id() || handle.id().index() >= data.disjunctions.len() {
                return Err(GdpError::ForeignHandle("disjunction"));
            }

            validate_m(method_m(method))?;
        }

        for (&handle, &values) in &options.big_m {
            if handle.model_id() != model.id() || handle.id().index() >= data.rows.len() {
                return Err(GdpError::ForeignHandle("conditional row"));
            }

            validate_m(values.lower)?;
            validate_m(values.upper)?;
        }

        let methods: FxHashMap<_, _> = options.methods.iter().map(|(h, m)| (h.id(), *m)).collect();
        let overrides: FxHashMap<_, _> = options.big_m.iter().map(|(h, m)| (h.id(), *m)).collect();
        let mut plan = Self {
            variables: Vec::new(),
            rows: Vec::new(),
            report: GdpReformulationReportData::default(),
            locked_variables: FxHashSet::default(),
            locked_parameters: FxHashSet::default(),
            dependency_nodes: FxHashSet::default(),
            dependency_stack: Vec::new(),
            expression_nodes: 0,
            next_variable: model.num_variables(),
            next_constraint: model.constraints().algebraic().len(),
            logic: LogicCompiler::new(data.booleans.len() + 2),
        };

        plan.report.booleans = data
            .booleans
            .iter()
            .enumerate()
            .map(|(i, b)| GdpBooleanMapping {
                source: BooleanId(u32::try_from(i).expect("Boolean ID")),
                binary: b.binary,
            })
            .collect();

        if !model.has_unreformulated_gdp() {
            return plan.validate(model.arena().len());
        }

        let arena = model.arena();
        let GdpPendingStarts {
            rows: row_start,
            disjunctions: disjunction_start,
            logical_constraints: logical_start,
        } = model.__gdp_pending_starts();

        plan.plan_big_m(model, options, &data, row_start, &methods, &overrides)?;
        plan.plan_disjunctions(model, &data, disjunction_start)?;
        plan.plan_logic(model, &data, logical_start)?;

        plan.validate(arena.len())
    }

    fn validate(mut self, arena_len: usize) -> Result<ValidatedPlan, GdpError> {
        // Merge and validate during planning so failure cannot leave appended
        // variables or rows behind.
        let mut coefficient_slots = FxHashMap::default();

        for row in &mut self.rows {
            if row.canonical {
                row.linear.terms.retain(|(_, coefficient)| *coefficient != 0.0);
            } else {
                row.linear.normalize(&mut coefficient_slots)?;
            }

            if row.lower.is_nan()
                || row.upper.is_nan()
                || row.lower == f64::INFINITY
                || row.upper == f64::NEG_INFINITY
            {
                return Err(GdpError::NumericOverflow);
            }
        }

        self.expression_nodes = self.rows.iter().try_fold(self.variables.len(), |total, row| {
            total
                .checked_add(1 + usize::from(!row.residual.is_empty()))
                .and_then(|count| {
                    count.checked_add(row.residual.iter().filter(|term| term.neg).count())
                })
                .ok_or(GdpError::Capacity)
        })?;
        u32::try_from(arena_len.checked_add(self.expression_nodes).ok_or(GdpError::Capacity)?)
            .map_err(|_| GdpError::Capacity)?;

        Ok(ValidatedPlan(self))
    }

    pub(super) fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub(super) fn row_ids(&self, start: usize) -> Result<Vec<ConstraintId>, GdpError> {
        (start..self.rows.len())
            .map(|i| {
                u32::try_from(self.next_constraint + i)
                    .map(ConstraintId)
                    .map_err(|_| GdpError::Capacity)
            })
            .collect()
    }

    pub(super) fn row(
        &mut self,
        model: &Model,
        source: Option<&[SignedExpr]>,
        linear: Linear,
        lower: f64,
        upper: f64,
    ) -> Result<ConstraintId, GdpError> {
        let index = self.next_constraint.checked_add(self.rows.len()).ok_or(GdpError::Capacity)?;
        let id = ConstraintId(u32::try_from(index).map_err(|_| GdpError::Capacity)?);
        let mut name = format!("__oximo_gdp_row{index}");

        if model.constraint_id(&name).is_some() {
            let base = name;
            let mut suffix = 1_u64;
            name = format!("{base}_{suffix}");

            while model.constraint_id(&name).is_some() {
                suffix += 1;
                name = format!("{base}_{suffix}");
            }
        }

        // Distinct row indices have disjoint names.
        self.rows.push(Row {
            name,
            residual: source.unwrap_or_default().to_vec(),
            linear,
            lower,
            upper,
            canonical: false,
        });

        Ok(id)
    }

    /// Numeric bodies are normalized before checked shifts append or update
    /// distinct indicator coefficients.
    pub(super) fn numeric_row(
        &mut self,
        model: &Model,
        source: Option<&[SignedExpr]>,
        linear: Linear,
        lower: f64,
        upper: f64,
    ) -> Result<ConstraintId, GdpError> {
        let id = self.row(model, source, linear, lower, upper)?;
        self.rows.last_mut().expect("appended row").canonical = true;

        Ok(id)
    }

    pub(super) fn auxiliary(&mut self, model: &Model) -> Result<Linear, GdpError> {
        let index =
            self.next_variable.checked_add(self.variables.len()).ok_or(GdpError::Capacity)?;
        let id = VarId(u32::try_from(index).map_err(|_| GdpError::Capacity)?);
        let name = model.__gdp_unique_variable_name(&format!("__oximo_gdp_aux{index}"));
        self.variables.push((id, name));
        self.report.variables.push(id);

        Ok(Linear::variable(id))
    }

    pub(super) fn dependencies(&mut self, arena: &oximo_expr::ExprArena, root: ExprId) {
        self.dependency_stack.push(root);

        while let Some(id) = self.dependency_stack.pop() {
            if !self.dependency_nodes.insert(id) {
                continue;
            }

            match arena.get(id) {
                ExprNode::Var(v) => {
                    self.locked_variables.insert(*v);
                }
                ExprNode::Param(p) => {
                    self.locked_parameters.insert(*p);
                }
                ExprNode::Linear { coeffs, .. } => self
                    .locked_variables
                    .extend(coeffs.iter().filter(|(_, c)| *c != 0.0).map(|(v, _)| *v)),
                ExprNode::Add(cs) | ExprNode::Mul(cs) | ExprNode::Min(cs) | ExprNode::Max(cs) => {
                    self.dependency_stack.extend(cs);
                }
                ExprNode::Unary(_, c) => self.dependency_stack.push(*c),
                ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                    self.dependency_stack.push(*a);
                    self.dependency_stack.push(*b);
                }
                ExprNode::Const(_) => {}
            }
        }
    }
}

impl ValidatedPlan {
    pub(super) fn apply(self, model: &Model) -> GdpReformulationReport {
        let Self(plan) = self;
        let PlanBuilder {
            variables,
            rows,
            report,
            locked_variables,
            locked_parameters,
            expression_nodes,
            ..
        } = plan;
        let report = GdpReformulationReport::from(report);

        if report.rows.is_empty()
            && report.disjunctions.is_empty()
            && report.logical_constraints.is_empty()
        {
            return report;
        }
        model.__gdp_reserve(variables.len(), rows.len(), expression_nodes);

        for (id, name) in variables {
            let variable = model.__var(name).binary().build();
            assert_eq!(variable.var_id(), Some(id));
        }

        for row in rows {
            let mut arena = model.__sum_context().borrow_mut();
            let correction = if row.linear.terms.is_empty() {
                arena.constant(row.linear.constant)
            } else {
                arena.linear(row.linear.terms, row.linear.constant)
            };

            let id = if row.residual.is_empty() {
                correction
            } else {
                let children = row
                    .residual
                    .into_iter()
                    .map(|term| {
                        if term.neg {
                            arena.push(ExprNode::Unary(UnaryOp::Neg, term.id))
                        } else {
                            term.id
                        }
                    })
                    .chain(std::iter::once(correction))
                    .collect();
                arena.push(ExprNode::Add(children))
            };
            drop(arena);
            let expr: Expr = Expr::new(id, model.__sum_context());

            // Nonlinear residuals retain symbolic bodies.
            model.add_constraint(
                row.name,
                Constraint::new(
                    ScalarDynamicFunction::from(expr),
                    Interval { lower: row.lower, upper: row.upper },
                ),
            );
        }
        model.__gdp_finish(
            report.clone(),
            locked_variables.into_iter().collect(),
            locked_parameters.into_iter().collect(),
        );
        report
    }
}
