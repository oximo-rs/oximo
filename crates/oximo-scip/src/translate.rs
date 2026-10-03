use oximo_core::{
    ConstraintId, Domain, IndicatorConstraint, Model, ModelKind, ObjectiveSense, SosType, VarId,
};
use oximo_expr::ExprId;
use oximo_solver::{SolverError, prepare::LoweringContext};
use russcip::{
    Constraint, Expr, Model as Native, ObjSense, ProblemCreated, ProblemOrSolving, VarType,
    Variable,
};

use crate::{ScipOptions, expression::Formula};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Column {
    pub id: VarId,
    pub name: String,
    pub lb: f64,
    pub ub: f64,
    pub domain: Domain,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Body {
    Linear(Vec<(VarId, f64)>, f64),
    Quadratic(Vec<(VarId, f64)>, Vec<(VarId, VarId, f64)>, f64),
    Nonlinear(Formula),
}

impl Body {
    fn prepare(p: &LoweringContext<'_>, id: ExprId) -> Result<Self, SolverError> {
        // Check vocabulary and constant domains before polynomial extraction,
        // which may otherwise simplify an unsupported function out of existence.
        let formula = Formula::prepare(p.arena(), id, p.variables().len())?;
        if let Some(l) = p.linear(id) {
            return Ok(Self::Linear(l.coeffs.to_vec(), l.constant));
        }
        if let Some(q) = p.quadratic(id) {
            return Ok(Self::Quadratic(
                q.linear.clone(),
                q.hessian
                    .iter()
                    .map(|&(a, b, c)| (a, b, if a == b { c / 2.0 } else { c }))
                    .collect(),
                q.constant,
            ));
        }
        Ok(Self::Nonlinear(formula))
    }
    pub(crate) fn native(&self, vars: &[Variable]) -> Expr {
        match self {
            Self::Linear(cs, k) => {
                Expr::sum_weighted(cs.iter().map(|(v, c)| (*c, Expr::var(&vars[v.index()])))) + *k
            }
            Self::Quadratic(cs, qs, k) => {
                Expr::sum_weighted(cs.iter().map(|(v, c)| (*c, Expr::var(&vars[v.index()]))))
                    + Expr::sum(qs.iter().map(|(a, b, c)| {
                        *c * Expr::var(&vars[a.index()]) * Expr::var(&vars[b.index()])
                    }))
                    + *k
            }
            Self::Nonlinear(f) => f.native(vars),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Row {
    pub name: String,
    pub body: Body,
    pub lower: f64,
    pub upper: f64,
    pub active: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Cone {
    name: String,
    terms: Vec<Formula>,
    bound: Option<Formula>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sos {
    name: String,
    kind: SosType,
    members: Vec<(VarId, f64)>,
    active: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct Indicator {
    name: String,
    trigger: VarId,
    active_value: bool,
    coefficients: Vec<(VarId, f64)>,
    constant: f64,
    lower: f64,
    upper: f64,
    active: bool,
}
impl Indicator {
    fn prepare(p: &LoweringContext<'_>, c: &IndicatorConstraint) -> Result<Self, SolverError> {
        crate::check_name(&c.name)?;
        check_bounds(c.lower, c.upper)?;
        let (coefficients, constant) = if c.active {
            let Body::Linear(coefficients, constant) = Body::prepare(p, c.lhs)? else {
                return Err(crate::backend("indicator consequent must be affine"));
            };
            crate::finite(constant)?;
            for &(_, coefficient) in &coefficients {
                crate::finite(coefficient)?;
            }
            (coefficients, constant)
        } else {
            (vec![], 0.0)
        };
        Ok(Self {
            name: c.name.to_string(),
            trigger: c.trigger,
            active_value: c.active_value,
            coefficients,
            constant,
            lower: c.lower,
            upper: c.upper,
            active: c.active,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Snapshot {
    pub name: String,
    pub columns: Vec<Column>,
    pub rows: Vec<Row>,
    cones: Vec<Cone>,
    sets: Vec<Sos>,
    indicators: Vec<Indicator>,
    pub objective: Body,
    pub sense: ObjectiveSense,
    pub kind: ModelKind,
}

impl Snapshot {
    pub(crate) fn prepare(model: &Model) -> Result<Self, SolverError> {
        crate::check_name(&model.name)?;
        if model.has_active_psd_constraints() {
            return Err(SolverError::UnsupportedConstraint("PSD"));
        }
        let p = LoweringContext::new(model)?;
        let mut columns = Vec::new();
        for v in p.variables() {
            crate::check_name(&v.name)?;
            if let Some(threshold) = v.domain.semi_threshold()
                && (!threshold.is_finite() || threshold <= 0.0)
            {
                return Err(crate::backend(format!(
                    "semi-variable {} requires a finite positive threshold",
                    v.name
                )));
            }
            check_bounds(v.lb, v.ub)?;
            columns.push(Column {
                id: v.id,
                name: v.name.to_string(),
                lb: v.lb,
                ub: v.ub,
                domain: v.domain,
            });
        }
        let mut rows = Vec::new();
        for c in p.constraints().algebraic() {
            crate::check_name(&c.name)?;
            check_bounds(c.lower, c.upper)?;
            rows.push(Row {
                name: c.name.to_string(),
                body: if c.active { Body::prepare(&p, c.lhs)? } else { Body::Linear(vec![], 0.0) },
                lower: c.lower,
                upper: c.upper,
                active: c.active,
            });
        }
        let cones = p
            .constraints()
            .second_order_cones()
            .iter()
            .map(|c| {
                crate::check_name(&c.name)?;
                Ok(Cone {
                    name: c.name.to_string(),
                    terms: if c.active {
                        c.terms
                            .iter()
                            .map(|&e| Formula::prepare(p.arena(), e, columns.len()))
                            .collect::<Result<_, _>>()?
                    } else {
                        vec![]
                    },
                    bound: if c.active {
                        Some(Formula::prepare(p.arena(), c.bound, columns.len())?)
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<_, SolverError>>()?;
        let sets = p
            .constraints()
            .special_ordered_sets()
            .iter()
            .map(|s| {
                crate::check_name(&s.name)?;
                let members = s
                    .members
                    .iter()
                    .map(|m| {
                        crate::finite(m.weight)?;
                        Ok((m.variable, m.weight))
                    })
                    .collect::<Result<_, SolverError>>()?;
                Ok(Sos { name: s.name.to_string(), kind: s.sos_type, members, active: s.active })
            })
            .collect::<Result<_, SolverError>>()?;
        let indicators = p
            .constraints()
            .indicators()
            .iter()
            .map(|c| Indicator::prepare(&p, c))
            .collect::<Result<_, SolverError>>()?;
        Ok(Self {
            name: model.name.to_string(),
            columns,
            rows,
            cones,
            sets,
            indicators,
            objective: p
                .objective()
                .map_or(Ok(Body::Linear(vec![], 0.0)), |o| Body::prepare(&p, o.expr))?,
            sense: p.sense(),
            kind: p.kind(),
        })
    }

    pub(crate) fn extends(&self, old: &Self) -> bool {
        if self.name != old.name
            || self.sense != old.sense
            || !self.columns.starts_with(&old.columns)
            || !self.rows.starts_with(&old.rows)
            || !self.cones.starts_with(&old.cones)
            || !self.sets.starts_with(&old.sets)
            || !self.indicators.starts_with(&old.indicators)
        {
            return false;
        }
        match (&self.objective, &old.objective) {
            (Body::Linear(a, ac), Body::Linear(b, bc)) => {
                if ac.to_bits() != bc.to_bits() {
                    return false;
                }
                let old_costs = costs(b, old.columns.len());
                let new_costs = costs(a, self.columns.len());
                new_costs.starts_with(&old_costs)
            }
            (a, b) => a == b,
        }
    }

    pub(crate) fn append(
        &self,
        model: &mut Native<ProblemCreated>,
        mapping: &mut crate::plugins::ScipMapping,
        old: Option<&Self>,
    ) -> Result<(), SolverError> {
        let objective_costs = match &self.objective {
            Body::Linear(c, _) => costs(c, self.columns.len()),
            _ => vec![0.0; self.columns.len()],
        };
        for c in self.columns.iter().skip(old.map_or(0, |s| s.columns.len())) {
            let (lb, ub) = if c.domain == Domain::Binary {
                (c.lb.max(0.0), c.ub.min(1.0))
            } else if let Some(threshold) = c.domain.semi_threshold() {
                (c.lb.max(threshold), c.ub)
            } else {
                (c.lb, c.ub)
            };
            // SCIP rejects inverted bounds at construction.
            let cost = objective_costs[c.id.index()];
            let v = match c.domain {
                Domain::SemiContinuous { .. } if lb <= ub => {
                    model.add_semi_continuous_var(lb, ub, cost, &c.name)
                }
                Domain::SemiInteger { .. } if lb <= ub => {
                    model.add_semi_integer_var(lb, ub, cost, &c.name)
                }
                Domain::SemiContinuous { .. } | Domain::SemiInteger { .. } => {
                    // A semi-variable can always take zero, even when its
                    // nonzero interval is empty.
                    let ty =
                        if c.domain.is_integer() { VarType::Integer } else { VarType::Continuous };
                    model.add_var(0.0, 0.0, cost, &c.name, ty)
                }
                domain => {
                    let ty = match domain {
                        Domain::Real => VarType::Continuous,
                        Domain::Integer => VarType::Integer,
                        Domain::Binary => VarType::Binary,
                        _ => unreachable!(),
                    };
                    model.add_var(
                        if lb > ub { 0.0 } else { lb },
                        if lb > ub { 0.0 } else { ub },
                        cost,
                        &c.name,
                        ty,
                    )
                }
            };
            mapping.variables.push(v);
            if lb > ub && c.domain.semi_threshold().is_none() {
                model.add_cons(vec![], &[], 1.0, f64::INFINITY, "oximo_infeasible_bounds");
            }
        }
        for (i, row) in self.rows.iter().enumerate().skip(old.map_or(0, |s| s.rows.len())) {
            let mut named = row.clone();
            named.name = format!("oximo_c{i}_{}", row.name);
            let native =
                if row.active { Some(add_row(model, &mapping.variables, &named)?) } else { None };
            mapping.constraints.push(native);
            mapping.linear.push(row.active && matches!(row.body, Body::Linear(..)));
            debug_assert_eq!(i + 1, mapping.constraints.len());
        }
        for cone in self.cones.iter().skip(old.map_or(0, |s| s.cones.len())) {
            let Some(bound) = &cone.bound else {
                continue;
            };
            // Direct norm expression.
            let norm = Expr::pow(
                Expr::sum(cone.terms.iter().map(|f| Expr::pow(f.native(&mapping.variables), 2.0))),
                0.5,
            );
            let e = model
                .build_expr(&(norm - bound.native(&mapping.variables)))
                .map_err(crate::backend)?;
            model.add_cons_nonlinear(&e, f64::NEG_INFINITY, 0.0, &cone.name);
        }
        for sos in self.sets.iter().skip(old.map_or(0, |s| s.sets.len())) {
            if sos.active {
                let vars = sos.members.iter().map(|(v, _)| &mapping.variables[v.index()]).collect();
                let weights = sos.members.iter().map(|(_, w)| *w).collect::<Vec<_>>();
                match sos.kind {
                    SosType::Sos1 => {
                        model.add_cons_sos1(vars, Some(&weights), &sos.name);
                    }
                    SosType::Sos2 => {
                        model.add_cons_sos2(vars, Some(&weights), &sos.name);
                    }
                }
            }
        }
        for (i, indicator) in
            self.indicators.iter().enumerate().skip(old.map_or(0, |s| s.indicators.len()))
        {
            add_indicator(model, &mapping.variables, indicator, i)?;
        }
        if old.is_none() {
            add_objective(model, mapping, &self.objective, self.sense)?;
        }
        Ok(())
    }

    pub(crate) fn build(
        &self,
        opts: &ScipOptions,
    ) -> Result<(Native<ProblemCreated>, crate::plugins::ScipMapping), SolverError> {
        let native = Native::try_new()
            .map_err(crate::backend)?
            .include_default_plugins()
            .create_prob(&self.name);
        let native = native.set_obj_sense(if self.sense == ObjectiveSense::Minimize {
            ObjSense::Minimize
        } else {
            ObjSense::Maximize
        });
        let mut native = opts.apply(native)?;
        let mut mapping = crate::plugins::ScipMapping::default();
        self.append(&mut native, &mut mapping, None)?;
        Ok((native, mapping))
    }
}

fn add_indicator(
    model: &mut Native<ProblemCreated>,
    vars: &[Variable],
    indicator: &Indicator,
    index: usize,
) -> Result<(), SolverError> {
    if !indicator.active || (!indicator.lower.is_finite() && !indicator.upper.is_finite()) {
        return Ok(());
    }
    let complement = if indicator.active_value {
        None
    } else {
        let complement =
            model.add_var(0.0, 1.0, 0.0, &format!("oximo_i{index}_complement"), VarType::Binary);
        model.add_cons(
            vec![&vars[indicator.trigger.index()], &complement],
            &[1.0, 1.0],
            1.0,
            1.0,
            &format!("oximo_i{index}_complement_definition"),
        );
        Some(complement)
    };
    let trigger = complement.as_ref().unwrap_or(&vars[indicator.trigger.index()]);
    let row_vars =
        indicator.coefficients.iter().map(|(id, _)| &vars[id.index()]).collect::<Vec<_>>();
    if indicator.upper.is_finite() {
        let rhs = indicator.upper - indicator.constant;
        crate::finite(rhs)?;
        let mut coefficients = indicator.coefficients.iter().map(|(_, c)| *c).collect::<Vec<_>>();
        model.add_cons_indicator(
            trigger,
            row_vars.clone(),
            &mut coefficients,
            rhs,
            &format!("oximo_i{index}_upper_{}", indicator.name),
        );
    }
    if indicator.lower.is_finite() {
        let rhs = indicator.constant - indicator.lower;
        crate::finite(rhs)?;
        let mut coefficients = indicator.coefficients.iter().map(|(_, c)| -*c).collect::<Vec<_>>();
        model.add_cons_indicator(
            trigger,
            row_vars,
            &mut coefficients,
            rhs,
            &format!("oximo_i{index}_lower_{}", indicator.name),
        );
    }
    Ok(())
}

fn add_objective(
    model: &mut Native<ProblemCreated>,
    mapping: &mut crate::plugins::ScipMapping,
    body: &Body,
    sense: ObjectiveSense,
) -> Result<(), SolverError> {
    match body {
        Body::Linear(_, constant) => {
            if *constant != 0.0 {
                model.add_var(1.0, 1.0, *constant, "oximo_objective_constant", VarType::Continuous);
            }
        }
        body => {
            let aux = model.add_var(
                f64::NEG_INFINITY,
                f64::INFINITY,
                1.0,
                "oximo_objective",
                VarType::Continuous,
            );
            let e = model
                .build_expr(&(body.native(&mapping.variables) - Expr::var(&aux)))
                .map_err(crate::backend)?;
            let (lb, ub) = if sense == ObjectiveSense::Minimize {
                (f64::NEG_INFINITY, 0.0)
            } else {
                (0.0, f64::INFINITY)
            };
            model.add_cons_nonlinear(&e, lb, ub, "oximo_objective_definition");
            mapping.objective_aux = Some(aux);
        }
    }
    Ok(())
}

fn costs(terms: &[(VarId, f64)], len: usize) -> Vec<f64> {
    let mut c = vec![0.0; len];
    for &(v, a) in terms {
        c[v.index()] += a;
    }
    c
}

pub(crate) fn check_bounds(lb: f64, ub: f64) -> Result<(), SolverError> {
    if lb.is_nan() || ub.is_nan() || lb == f64::INFINITY || ub == f64::NEG_INFINITY {
        Err(crate::backend("invalid bounds"))
    } else {
        Ok(())
    }
}

fn add_row(
    model: &mut Native<ProblemCreated>,
    vars: &[Variable],
    row: &Row,
) -> Result<Constraint, SolverError> {
    if row.lower > row.upper {
        return Ok(model.add_cons(vec![], &[], 1.0, f64::INFINITY, &row.name));
    }
    match &row.body {
        Body::Linear(c, k) => Ok(model.add_cons(
            c.iter().map(|(v, _)| &vars[v.index()]).collect(),
            &c.iter().map(|(_, a)| *a).collect::<Vec<_>>(),
            row.lower - k,
            row.upper - k,
            &row.name,
        )),
        Body::Quadratic(c, q, k) => Ok(model.add_cons_quadratic(
            c.iter().map(|(v, _)| &vars[v.index()]).collect(),
            &mut c.iter().map(|(_, a)| *a).collect::<Vec<_>>(),
            q.iter().map(|(v, _, _)| &vars[v.index()]).collect(),
            q.iter().map(|(_, v, _)| &vars[v.index()]).collect(),
            &mut q.iter().map(|(_, _, a)| *a).collect::<Vec<_>>(),
            row.lower - k,
            row.upper - k,
            &row.name,
        )),
        Body::Nonlinear(f) => {
            let expr = model.build_expr(&f.native(vars)).map_err(crate::backend)?;
            Ok(model.add_cons_nonlinear(&expr, row.lower, row.upper, &row.name))
        }
    }
}

pub(crate) fn id(i: usize) -> ConstraintId {
    ConstraintId(u32::try_from(i).expect("oximo constraint ID"))
}
