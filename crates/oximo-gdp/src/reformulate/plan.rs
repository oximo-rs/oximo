//! Shared row planning, validation, and atomic application.

use super::big_m::{method_m, validate_m};
use super::logic::LogicCompiler;
use super::{
    GdpError, GdpModelSize, GdpPlanningLimits, GdpPlanningSummary, GdpReformulationOptions,
    GdpSizeLimits,
};
use oximo_core::gdp::{
    BooleanId, GdpBooleanMapping, GdpPendingStarts, GdpReformulationReport,
    GdpReformulationReportData,
};
use oximo_core::{
    Constraint, ConstraintId, Domain, Expr, ExprId, ExprNode, Interval, Model, ParamId,
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
    variables: Vec<(VarId, String, Domain, f64, f64)>,
    rows: Vec<Row>,
    pub(super) report: GdpReformulationReportData,
    pub(super) locked_variables: FxHashSet<VarId>,
    pub(super) locked_parameters: FxHashSet<ParamId>,
    dependency_nodes: FxHashSet<ExprId>,
    dependency_stack: Vec<ExprId>,
    expression_nodes: usize,
    next_variable: usize,
    next_constraint: usize,
    source_constraints: usize,
    size_limits: GdpSizeLimits,
    planning_limits: GdpPlanningLimits,
    pub(super) logic: LogicCompiler,
    pub(super) generated_nodes: Vec<ExprNode>,
    pub(super) source_nodes: usize,
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
        super::hull::validate_method(options.default_method)?;

        for (&handle, &method) in &options.methods {
            if handle.model_id() != model.id() || handle.id().index() >= data.disjunctions.len() {
                return Err(GdpError::ForeignHandle("disjunction"));
            }

            validate_m(method_m(method))?;
            super::hull::validate_method(method)?;
        }

        for (&(model_id, variable), &value) in &options.reference_points {
            if model_id != model.id()
                || variable.is_some_and(|v| v.index() >= model.num_variables())
            {
                return Err(GdpError::ForeignHandle("Hull reference variable"));
            }

            if variable.is_none() {
                return Err(GdpError::ExpectedHullReferenceVariable);
            }

            if !value.is_finite() {
                return Err(GdpError::InvalidHullReference(value));
            }
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
            source_constraints: model.num_constraints(),
            size_limits: options.size_limits,
            planning_limits: options.planning_limits,
            logic: LogicCompiler::new(data.booleans.len() + 2),
            generated_nodes: Vec::new(),
            source_nodes: model.arena().len(),
        };

        plan.check_size(0, 0, 0)?;
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

        let has_hull = matches!(options.default_method, super::GdpMethod::Hull(_))
            || methods.values().any(|m| matches!(m, super::GdpMethod::Hull(_)));

        if has_hull {
            plan.plan_hull(
                model,
                options,
                &data,
                row_start,
                disjunction_start,
                &methods,
                &overrides,
            )?;
        } else {
            plan.plan_big_m(model, options, &data, row_start, &methods, &overrides, None)?;
        }

        plan.selection_rows(model, &data, disjunction_start, options, &methods)?;
        plan.plan_logic(model, &data, logical_start)?;

        plan.validate(arena.len())
    }

    fn selection_rows(
        &mut self,
        model: &Model,
        data: &oximo_core::gdp::GdpSnapshot,
        disjunction_start: usize,
        options: &GdpReformulationOptions,
        methods: &FxHashMap<oximo_core::gdp::DisjunctionId, super::GdpMethod>,
    ) -> Result<(), GdpError> {
        let selection_start = self.report.disjunctions.len();
        self.plan_disjunctions(model, data, disjunction_start)?;

        if selection_start != 0 {
            let slots: FxHashMap<_, _> = self.report.disjunctions[..selection_start]
                .iter()
                .enumerate()
                .map(|(i, d)| (d.source, i))
                .collect();

            for artifacts in self.report.disjunctions.drain(selection_start..).collect::<Vec<_>>() {
                if let Some(&slot) = slots.get(&artifacts.source) {
                    self.report.disjunctions[slot].constraints.extend(artifacts.constraints);
                } else {
                    self.report.disjunctions.push(artifacts);
                }
            }
        }

        for d in &mut self.report.disjunctions {
            d.method = match methods.get(&d.source).copied().unwrap_or(options.default_method) {
                super::GdpMethod::BigM(_) => oximo_core::gdp::GdpMethodKind::BigM,
                super::GdpMethod::Hull(_) => oximo_core::gdp::GdpMethodKind::Hull,
            };
        }

        Ok(())
    }

    fn validate(mut self, arena_len: usize) -> Result<ValidatedPlan, GdpError> {
        self.compact_nodes();

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

        let base_nodes = self
            .variables
            .len()
            .checked_add(self.generated_nodes.len())
            .ok_or(GdpError::Capacity)?;
        self.expression_nodes = self.rows.iter().try_fold(base_nodes, |total, row| {
            total
                .checked_add(1 + usize::from(!row.residual.is_empty()))
                .and_then(|count| {
                    count.checked_add(row.residual.iter().filter(|term| term.neg).count())
                })
                .ok_or(GdpError::Capacity)
        })?;
        u32::try_from(arena_len.checked_add(self.expression_nodes).ok_or(GdpError::Capacity)?)
            .map_err(|_| GdpError::Capacity)?;

        self.check_size(0, 0, self.expression_nodes)?;
        Ok(ValidatedPlan(self))
    }

    /// Check total sizes, using a lower bound on nodes until DAG compaction.
    pub(super) fn check_size(
        &self,
        additional_variables: usize,
        additional_rows: usize,
        expression_nodes: usize,
    ) -> Result<(), GdpError> {
        if self.size_limits == GdpSizeLimits::default() {
            return Ok(());
        }

        let variables =
            self.variables.len().checked_add(additional_variables).ok_or(GdpError::Capacity)?;
        let rows = self.rows.len().checked_add(additional_rows).ok_or(GdpError::Capacity)?;
        let minimum_nodes = variables.checked_add(rows).ok_or(GdpError::Capacity)?;

        for (resource, initial, generated, limit) in [
            ("variables", self.next_variable, variables, self.size_limits.variables),
            ("constraints", self.source_constraints, rows, self.size_limits.constraints),
            (
                "expression nodes",
                self.source_nodes,
                expression_nodes.max(minimum_nodes),
                self.size_limits.expression_nodes,
            ),
        ] {
            let required = initial.checked_add(generated).ok_or(GdpError::Capacity)?;

            if let Some(limit) = limit
                && required > limit
            {
                return Err(GdpError::SizeLimit { resource, required, limit });
            }
        }

        Ok(())
    }

    pub(super) fn record_row_expansions(&mut self, count: usize) -> Result<(), GdpError> {
        let required =
            self.report.planning.row_expansions.checked_add(count).ok_or(GdpError::Capacity)?;

        if let Some(limit) = self.planning_limits.row_expansions
            && required > limit
        {
            return Err(GdpError::PlanningLimit { resource: "row expansions", required, limit });
        }

        self.report.planning.row_expansions = required;

        Ok(())
    }

    /// Each copy must contribute two distinct bound rows, even when ancestors
    /// transform them.
    pub(super) fn check_hull_copies(&self, additional: usize) -> Result<(), GdpError> {
        if self.size_limits == GdpSizeLimits::default() {
            return Ok(());
        }

        let copies =
            self.report.hull_variables.len().checked_add(additional).ok_or(GdpError::Capacity)?;
        let rows = copies.checked_mul(2).ok_or(GdpError::Capacity)?;

        self.check_size(additional, rows.saturating_sub(self.rows.len()), 0)
    }

    /// Affine fast paths need no expression rewrites in the committed arena.
    /// Retain only nonlinear DAGs and report activations, preserving sharing.
    fn compact_nodes(&mut self) {
        if self.generated_nodes.is_empty() {
            return;
        }

        let mut reachable = vec![false; self.generated_nodes.len()];
        let mut stack: Vec<_> = self
            .rows
            .iter()
            .flat_map(|r| r.residual.iter().map(|t| t.id))
            .chain(self.report.hull_variables.iter().map(|m| m.activation))
            .collect();

        while let Some(id) = stack.pop() {
            if id.index() < self.source_nodes {
                continue;
            }

            let slot = id.index() - self.source_nodes;
            if std::mem::replace(&mut reachable[slot], true) {
                continue;
            }

            match &self.generated_nodes[slot] {
                ExprNode::Add(cs) | ExprNode::Mul(cs) | ExprNode::Min(cs) | ExprNode::Max(cs) => {
                    stack.extend(cs);
                }
                ExprNode::Unary(_, c) => stack.push(*c),
                ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                _ => {}
            }
        }

        let mut remapping = vec![ExprId(0); reachable.len()];
        let mut next = self.source_nodes;

        for (i, &keep) in reachable.iter().enumerate() {
            if keep {
                remapping[i] = ExprId(u32::try_from(next).expect("planned expression ID"));
                next += 1;
            }
        }

        let remap = |id: ExprId| {
            if id.index() < self.source_nodes {
                id
            } else {
                remapping[id.index() - self.source_nodes]
            }
        };

        self.generated_nodes = std::mem::take(&mut self.generated_nodes)
            .into_iter()
            .enumerate()
            .filter_map(|(i, mut node)| {
                if !reachable[i] {
                    return None;
                }

                remap_node(&mut node, &remap);
                Some(node)
            })
            .collect();

        for row in &mut self.rows {
            for term in &mut row.residual {
                term.id = remap(term.id);
            }
        }

        for mapping in &mut self.report.hull_variables {
            mapping.activation = remap(mapping.activation);
        }
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
        self.check_size(0, 1, 0)?;
        self.record_row_expansions(1)?;
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
        Ok(Linear::variable(self.variable(model, Domain::Binary, 0.0, 1.0)?))
    }

    pub(super) fn variable(
        &mut self,
        model: &Model,
        domain: Domain,
        lower: f64,
        upper: f64,
    ) -> Result<VarId, GdpError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(GdpError::NumericOverflow);
        }

        self.check_size(1, 0, 0)?;
        let index =
            self.next_variable.checked_add(self.variables.len()).ok_or(GdpError::Capacity)?;
        let id = VarId(u32::try_from(index).map_err(|_| GdpError::Capacity)?);
        let name = model.__gdp_unique_variable_name(&format!("__oximo_gdp_aux{index}"));
        self.variables.push((id, name, domain, lower, upper));
        self.report.variables.push(id);

        Ok(id)
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
    pub(super) fn summary(&self) -> GdpPlanningSummary {
        let plan = &self.0;
        let generated = GdpModelSize {
            variables: plan.variables.len(),
            constraints: plan.rows.len(),
            expression_nodes: plan.expression_nodes,
        };

        GdpPlanningSummary {
            planning: plan.report.planning,
            generated,
            total: GdpModelSize {
                variables: plan.next_variable + generated.variables,
                constraints: plan.source_constraints + generated.constraints,
                expression_nodes: plan.source_nodes + generated.expression_nodes,
            },
            hull_copies: plan.report.hull_variables.len(),
            hull_patterns: plan.report.hull_patterns.len(),
        }
    }

    pub(super) fn apply(self, model: &Model) -> GdpReformulationReport {
        let Self(plan) = self;
        let PlanBuilder {
            variables,
            rows,
            report,
            locked_variables,
            locked_parameters,
            expression_nodes,
            generated_nodes,
            source_nodes,
            ..
        } = plan;

        let mut report = report;
        let shift = variables.len();
        let remap = |id: ExprId| {
            if id.index() < source_nodes {
                id
            } else {
                ExprId(u32::try_from(id.index() + shift).expect("validated expression capacity"))
            }
        };

        for mapping in &mut report.hull_variables {
            mapping.activation = remap(mapping.activation);
        }

        let report = GdpReformulationReport::from(report).__with_model_id(model.id());

        if report.rows.is_empty()
            && report.disjunctions.is_empty()
            && report.logical_constraints.is_empty()
        {
            return report;
        }

        model.__gdp_reserve(variables.len(), rows.len(), expression_nodes);

        for (id, name, domain, lower, upper) in variables {
            let variable = model.__var(name).domain(domain).bounds(lower, upper).build();
            assert_eq!(variable.var_id(), Some(id));
        }

        {
            let mut arena = model.__sum_context().borrow_mut();

            for mut node in generated_nodes {
                remap_node(&mut node, &remap);
                arena.push(node);
            }
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
                            arena.push(ExprNode::Unary(UnaryOp::Neg, remap(term.id)))
                        } else {
                            remap(term.id)
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

fn remap_node(node: &mut ExprNode, remap: &impl Fn(ExprId) -> ExprId) {
    match node {
        ExprNode::Add(cs) | ExprNode::Mul(cs) | ExprNode::Min(cs) | ExprNode::Max(cs) => {
            for id in cs {
                *id = remap(*id);
            }
        }
        ExprNode::Unary(_, id) => *id = remap(*id),
        ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
            *a = remap(*a);
            *b = remap(*b);
        }
        _ => {}
    }
}
