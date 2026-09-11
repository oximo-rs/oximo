use std::borrow::Cow;
use std::time::Duration;

use oximo_core::{
    ConstraintId, ConstraintRef, Expr, IndexKey, IndexedVar, Model, SocConstraintId, VarId,
};
use oximo_expr::{EvalContext, ExprArena, ExprId, ParamId, evaluate};
use rustc_hash::FxHashMap;

use crate::status::{PrimalStatus, TerminationStatus};

/// A single primal point returned by a solver.
///
/// Most solves yield one point, but a global solver asked to enumerate solutions
/// may returns several. In a [`SolverResult`] the points live in [`SolverResult::solutions`].
/// Index `0` is always the best/incumbent.
#[derive(Clone, Debug, Default)]
pub struct SolutionPoint {
    pub primal: FxHashMap<VarId, f64>,
    pub objective: Option<f64>,
}

struct PointContext<'a>(&'a FxHashMap<VarId, f64>);

impl EvalContext for PointContext<'_> {
    fn var(&self, id: VarId) -> Option<f64> {
        self.0.get(&id).copied()
    }

    fn param(&self, _id: ParamId) -> Option<f64> {
        None
    }
}

impl SolutionPoint {
    /// Look up a primal value by `VarId`.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.primal.get(&id).copied()
    }

    /// Evaluate an expression at this primal point.
    ///
    /// Returns `None` when any variable needed by the expression is absent.
    /// Parameter values are read from the expression's model arena at query
    /// time.
    pub fn value_of(&self, expr: Expr<'_>) -> Option<f64> {
        let arena = expr.arena.borrow();
        evaluate(&arena, expr.id, &PointContext(&self.primal)).ok()
    }

    /// Look up the primal value for a specific index of an [`IndexedVar`].
    ///
    /// Returns `None` if `key` is not in the variable's set or the solver did
    /// not return a primal value for that scalar.
    pub fn value_of_idx<V, K: Into<IndexKey>>(
        &self,
        var: &IndexedVar<'_, V>,
        key: K,
    ) -> Option<f64> {
        var.get(key).and_then(|e| self.value_of(e))
    }

    /// Iterate over primal values for all entries of an [`IndexedVar`].
    ///
    /// Yields `(&IndexKey, f64)` for every index whose primal value is present
    /// in the solution.
    pub fn values_of<'iv, 'a, V>(
        &'iv self,
        var: &'iv IndexedVar<'a, V>,
    ) -> impl Iterator<Item = (&'iv IndexKey, f64)> + 'iv {
        var.iter().filter_map(|(k, e)| self.value_of(*e).map(|v| (k, v)))
    }
}

/// Availability and quality of the dual solution returned by a solver.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DualStatus {
    /// The backend reports that no dual solution is available.
    #[default]
    NoSolution,
    /// A usable dual point is available.
    FeasiblePoint,
    /// The backend cannot distinguish unavailable from unreported duals.
    Unknown,
}

/// Activity and feasibility information for an algebraic constraint.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ConstraintEvaluation {
    pub activity: f64,
    pub lower_slack: Option<f64>,
    pub upper_slack: Option<f64>,
    pub violation: f64,
}

/// Activity and feasibility information for an explicit second-order cone.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SocEvaluation {
    pub norm: f64,
    pub bound: f64,
    pub slack: f64,
    pub violation: f64,
}

fn evaluate_at(arena: &ExprArena, id: ExprId, point: &SolutionPoint) -> Option<f64> {
    evaluate(arena, id, &PointContext(&point.primal)).ok()
}

fn evaluate_constraint_at(
    point: &SolutionPoint,
    model: &Model,
    id: ConstraintId,
) -> Option<ConstraintEvaluation> {
    let arena = model.arena();
    let constraints = model.constraints();
    let constraint = constraints.algebraic().get(id.index())?;
    let activity = evaluate_at(&arena, constraint.lhs, point)?;
    let lower_slack = constraint.lower.is_finite().then_some(activity - constraint.lower);
    let upper_slack = constraint.upper.is_finite().then_some(constraint.upper - activity);
    let violation = lower_slack
        .into_iter()
        .chain(upper_slack)
        .map(|slack| (-slack).max(0.0))
        .fold(0.0, f64::max);
    Some(ConstraintEvaluation { activity, lower_slack, upper_slack, violation })
}

fn evaluate_soc_at(
    point: &SolutionPoint,
    model: &Model,
    id: SocConstraintId,
) -> Option<SocEvaluation> {
    let arena = model.arena();
    let socs = model.soc_constraints();
    let constraint = socs.get(id.index())?;
    let squared_norm = constraint.terms.iter().try_fold(0.0, |sum, &term| {
        evaluate_at(&arena, term, point).map(|value| sum + value * value)
    })?;
    let norm = squared_norm.sqrt();
    let bound = evaluate_at(&arena, constraint.bound, point)?;
    let slack = bound - norm;
    Some(SocEvaluation { norm, bound, slack, violation: (-slack).max(0.0) })
}

/// A solver's final result on a model.
///
/// `termination` expresses why the solver stopped and `primal_status` says
/// whether the point in `solutions` is usable. Primal points are held in
/// `solutions` (index `0` is the best/incumbent, empty when no solution was
/// found). `dual` and `reduced_costs` apply to the best continuous point and are
/// sparse maps, so a solver that does not return duals (e.g. MILP) can simply
/// leave them empty. `best_bound` is populated when a global bound is available.
/// `gap` is the solver-reported gap, whose convention is backend-specific.
#[derive(Clone, Debug)]
pub struct SolverResult {
    pub termination: TerminationStatus,
    pub primal_status: PrimalStatus,
    pub dual_status: DualStatus,
    pub solutions: Vec<SolutionPoint>,
    pub dual: FxHashMap<ConstraintId, f64>,
    pub soc_dual: FxHashMap<SocConstraintId, f64>,
    pub reduced_costs: FxHashMap<VarId, f64>,
    /// The best objective bound reported by the backend.
    pub best_bound: Option<f64>,
    /// The backend's relative optimality gap.
    pub gap: Option<f64>,
    pub solve_time: Duration,
    /// A backend-defined aggregate iteration count.
    pub iterations: u64,
    pub node_count: Option<u64>,
    /// A compact native status label or code, distinct from [`Self::raw_log`].
    pub raw_status: Option<Cow<'static, str>>,
    pub raw_log: Option<String>,
    pub solver_name: Option<Cow<'static, str>>,
    pub solver_version: Option<Cow<'static, str>>,
}

impl Default for SolverResult {
    fn default() -> Self {
        Self {
            termination: TerminationStatus::NotSolved,
            primal_status: PrimalStatus::NoSolution,
            dual_status: DualStatus::NoSolution,
            solutions: Vec::new(),
            dual: FxHashMap::default(),
            soc_dual: FxHashMap::default(),
            reduced_costs: FxHashMap::default(),
            best_bound: None,
            gap: None,
            solve_time: Duration::ZERO,
            iterations: 0,
            node_count: None,
            raw_status: None,
            raw_log: None,
            solver_name: None,
            solver_version: None,
        }
    }
}

impl SolverResult {
    /// The number of primal points the solver returned (`0` when infeasible or
    /// unsolved).
    pub fn result_count(&self) -> usize {
        self.solutions.len()
    }

    /// The `i`-th primal point, where index `0` is the best/incumbent.
    pub fn solution(&self, i: usize) -> Option<&SolutionPoint> {
        self.solutions.get(i)
    }

    /// The best primal point, or `None` when no solution was found.
    pub fn best(&self) -> Option<&SolutionPoint> {
        self.solutions.first()
    }

    /// Whether a usable primal point is available, regardless of why the solver
    /// stopped. Driven by [`PrimalStatus`], so an incumbent returned at a time
    /// or iteration limit still counts.
    pub fn has_solution(&self) -> bool {
        self.primal_status.has_solution()
    }

    /// The objective value of the best solution, or `None` when none was found.
    pub fn objective(&self) -> Option<f64> {
        self.solutions.first().and_then(|s| s.objective)
    }

    /// The best solution's primal map, or `None` when no solution was found.
    pub fn primal(&self) -> Option<&FxHashMap<VarId, f64>> {
        self.solutions.first().map(|s| &s.primal)
    }

    /// Look up a primal value by `VarId` in the best solution.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.solutions.first().and_then(|s| s.value(id))
    }

    /// Evaluate an expression at the best solution.
    pub fn value_of(&self, expr: Expr<'_>) -> Option<f64> {
        self.solutions.first().and_then(|s| s.value_of(expr))
    }

    /// Evaluate an algebraic constraint at the best solution.
    ///
    /// The result and model must describe the same solve. Parameter values are
    /// read from `model` at query time, so do not combine an old result with a
    /// subsequently modified model.
    pub fn constraint_evaluation(
        &self,
        model: &Model,
        id: ConstraintId,
    ) -> Option<ConstraintEvaluation> {
        self.constraint_evaluation_at(model, id, 0)
    }

    /// Evaluate an algebraic constraint at solution `solution_index`.
    pub fn constraint_evaluation_at(
        &self,
        model: &Model,
        id: ConstraintId,
        solution_index: usize,
    ) -> Option<ConstraintEvaluation> {
        evaluate_constraint_at(self.solution(solution_index)?, model, id)
    }

    /// Evaluate an explicit second-order-cone constraint at the best solution.
    pub fn soc_evaluation(&self, model: &Model, id: SocConstraintId) -> Option<SocEvaluation> {
        self.soc_evaluation_at(model, id, 0)
    }

    /// Evaluate an explicit second-order-cone constraint at solution
    /// `solution_index`.
    pub fn soc_evaluation_at(
        &self,
        model: &Model,
        id: SocConstraintId,
        solution_index: usize,
    ) -> Option<SocEvaluation> {
        evaluate_soc_at(self.solution(solution_index)?, model, id)
    }

    pub fn dual_of(&self, c: ConstraintId) -> Option<f64> {
        self.dual.get(&c).copied()
    }

    /// The norm-form bound multiplier of an explicit SOC constraint,
    /// or `None` when the backend did not compute it.
    pub fn soc_dual_of(&self, c: SocConstraintId) -> Option<f64> {
        self.soc_dual.get(&c).copied()
    }

    /// Look up the best solution's primal value for a specific index of an
    /// [`IndexedVar`].
    pub fn value_of_idx<V, K: Into<IndexKey>>(
        &self,
        var: &IndexedVar<'_, V>,
        key: K,
    ) -> Option<f64> {
        var.get(key).and_then(|e| self.value_of(e))
    }

    /// Iterate over the best solution's primal values for all entries of an
    /// [`IndexedVar`]. Yields nothing when no solution was found.
    pub fn values_of<'iv, 'a, V>(
        &'iv self,
        var: &'iv IndexedVar<'a, V>,
    ) -> impl Iterator<Item = (&'iv IndexKey, f64)> + 'iv {
        var.iter().filter_map(|(k, e)| self.value_of(*e).map(|v| (k, v)))
    }

    /// A human-readable, model-aware summary of this result.
    ///
    /// It lists the solver, model kind and sense, status,
    /// objective and work counters, then every variable's value
    /// (with its reduced cost when the solver returned duals) and every
    /// constraint's dual.
    pub fn report<'a>(&'a self, model: &'a Model) -> ModelReport<'a> {
        ModelReport { result: self, model }
    }
}

/// A printable, model-aware summary of a [`SolverResult`]. Created by
/// [`SolverResult::report`].
#[derive(Debug)]
pub struct ModelReport<'a> {
    result: &'a SolverResult,
    model: &'a Model,
}

/// Format a value with up to six decimals, trimming trailing zeros so whole
/// numbers render as `5` rather than `5.000000`.
fn num(x: f64) -> String {
    let s = format!("{x:.6}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-0" { "0".to_owned() } else { trimmed.to_owned() }
}

impl std::fmt::Display for ModelReport<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.result;
        let m = self.model;

        writeln!(f, "solution summary")?;
        let solver = match (r.solver_name.as_deref(), r.solver_version.as_deref()) {
            (Some(name), Some(version)) => format!("{name} {version}"),
            (Some(name), None) => name.to_owned(),
            (None, _) => "(unknown)".to_owned(),
        };
        writeln!(f, "  solver     : {solver}")?;
        let objective = m.objective();
        if let Some(objective) = objective.as_ref() {
            writeln!(f, "  model      : {}  ({}, {})", m.name, m.kind(), objective.sense)?;
        } else {
            writeln!(f, "  model      : {}  ({}, no objective)", m.name, m.kind())?;
        }
        writeln!(f, "  termination: {:?}", r.termination)?;
        writeln!(f, "  primal     : {:?}", r.primal_status)?;
        writeln!(f, "  dual       : {:?}", r.dual_status)?;
        if let Some(raw) = r.raw_status.as_deref() {
            writeln!(f, "  raw status : {raw}")?;
        }
        writeln!(f, "  solutions  : {}", r.result_count())?;
        match r.objective() {
            Some(v) => writeln!(f, "  objective  : {}", num(v))?,
            None => writeln!(f, "  objective  : (none)")?,
        }
        if let Some(b) = r.best_bound {
            writeln!(f, "  best bound : {}", num(b))?;
        }
        if let Some(g) = r.gap {
            writeln!(f, "  gap        : {}", num(g))?;
        }
        writeln!(f, "  solve time : {:?}", r.solve_time)?;
        writeln!(f, "  iterations : {}", r.iterations)?;
        if let Some(nodes) = r.node_count {
            writeln!(f, "  nodes      : {nodes}")?;
        }

        // Variables
        let vars = m.variables();
        writeln!(f, "\nvariables ({})", vars.len())?;
        if let Some(best) = r.best() {
            let width = vars.iter().map(|v| v.name.len()).max().unwrap_or(0);
            let show_rc = !r.reduced_costs.is_empty();
            for v in vars.iter() {
                let val = best.value(v.id).map_or_else(|| "n/a".to_owned(), num);
                match (show_rc, r.reduced_costs.get(&v.id)) {
                    (true, Some(rc)) => {
                        writeln!(f, "  {:<width$} = {val}   (reduced cost {})", v.name, num(*rc))?;
                    }
                    _ => writeln!(f, "  {:<width$} = {val}", v.name)?,
                }
            }
        } else {
            writeln!(f, "  (no primal solution)")?;
        }

        // Constraint duals, only when the solver returned any
        if !r.dual.is_empty() {
            let model_constraints = m.constraints();
            let cons: Vec<_> = model_constraints
                .iter()
                .filter_map(|constraint| match constraint {
                    ConstraintRef::Algebraic { id, constraint } => Some((id, constraint)),
                    ConstraintRef::SecondOrderCone { .. }
                    | ConstraintRef::SpecialOrderedSet { .. } => None,
                })
                .collect();
            writeln!(f, "\nconstraints ({})", cons.len())?;
            let width = cons.iter().map(|(_, c)| c.name.len()).max().unwrap_or(0);
            for (id, c) in cons {
                let d = r.dual_of(id).map_or_else(|| "n/a".to_owned(), num);
                writeln!(f, "  {:<width$}  dual = {d}", c.name)?;
            }
        }

        // SOC bound multipliers, only when the solver returned any
        if !r.soc_dual.is_empty() {
            let socs = m.soc_constraints();
            writeln!(f, "\nsoc constraints ({})", socs.len())?;
            let width = socs.iter().map(|s| s.name.len()).max().unwrap_or(0);
            for (i, s) in socs.iter().enumerate() {
                let id = SocConstraintId(u32::try_from(i).expect("soc index fits u32"));
                let d = r.soc_dual_of(id).map_or_else(|| "n/a".to_owned(), num);
                writeln!(f, "  {:<width$}  dual = {d}", s.name)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_result_has_no_solution() {
        let r = SolverResult::default();
        assert_eq!(r.result_count(), 0);
        assert!(r.best().is_none());
        assert!(r.objective().is_none());
        assert!(r.primal().is_none());
        assert!(r.value(VarId(0)).is_none());
        assert!(r.solution(0).is_none());
        assert_eq!(r.dual_status, DualStatus::NoSolution);
        assert!(r.raw_status.is_none());
        assert!(r.solver_version.is_none());
        assert!(r.node_count.is_none());
    }

    #[test]
    fn value_of_evaluates_linear_quadratic_nonlinear_and_parameterized_expressions() {
        use oximo_core::{param, variable};

        let m = Model::new("expressions");
        param!(m, p = 2.0);
        variable!(m, x);
        variable!(m, y);
        let mut primal = FxHashMap::default();
        primal.insert(x.var_id().unwrap(), 3.0);
        primal.insert(y.var_id().unwrap(), 4.0);
        let point = SolutionPoint { primal, objective: None };

        assert_eq!(point.value_of(x), Some(3.0));
        assert_eq!(point.value_of(2.0 * x + y - 1.0), Some(9.0));
        assert_eq!(point.value_of(x.powi(2) + x * y), Some(21.0));
        assert_eq!(point.value_of(x.sin()), Some(3.0_f64.sin()));
        assert_eq!(point.value_of(p * x + y), Some(10.0));

        let incomplete = SolutionPoint::default();
        assert!(incomplete.value_of(x + y).is_none());
    }

    #[test]
    fn algebraic_constraint_evaluations_cover_all_bound_shapes_and_solution_indices() {
        use oximo_core::{constraint, variable};

        let m = Model::new("constraint evaluation");
        variable!(m, x);
        let equality = constraint!(m, equality, x == 2.0);
        let lower = constraint!(m, lower, x >= 1.0);
        let upper = constraint!(m, upper, x <= 3.0);
        constraint!(m, ranged, 1.5 <= x <= 2.5);
        let ranged = m.constraint_id("ranged").unwrap();

        let point = |value| {
            let mut primal = FxHashMap::default();
            primal.insert(x.var_id().unwrap(), value);
            SolutionPoint { primal, objective: None }
        };
        let result = SolverResult {
            primal_status: PrimalStatus::FeasiblePoint,
            solutions: vec![point(2.0), point(4.0)],
            ..Default::default()
        };

        assert_eq!(
            result.constraint_evaluation(&m, equality),
            Some(ConstraintEvaluation {
                activity: 2.0,
                lower_slack: Some(0.0),
                upper_slack: Some(0.0),
                violation: 0.0,
            })
        );
        assert_eq!(result.constraint_evaluation(&m, lower).unwrap().lower_slack, Some(1.0));
        assert_eq!(result.constraint_evaluation(&m, lower).unwrap().upper_slack, None);
        assert_eq!(result.constraint_evaluation(&m, upper).unwrap().lower_slack, None);
        assert_eq!(result.constraint_evaluation(&m, upper).unwrap().upper_slack, Some(1.0));
        assert!(result.constraint_evaluation(&m, ranged).unwrap().violation.abs() < f64::EPSILON);
        assert!(
            (result.constraint_evaluation_at(&m, ranged, 1).unwrap().violation - 1.5).abs()
                < f64::EPSILON
        );
        assert!(result.constraint_evaluation_at(&m, ranged, 2).is_none());
        assert!(result.constraint_evaluation(&m, ConstraintId(u32::MAX)).is_none());
    }

    #[test]
    fn soc_evaluation_reports_norm_slack_and_violation() {
        use oximo_core::{soc_constraint, variable};

        let m = Model::new("soc evaluation");
        variable!(m, x);
        variable!(m, y);
        variable!(m, t);
        let cone = soc_constraint!(m, cone, [x, y] <= t);
        let point = |x_value, y_value, t_value| {
            let mut primal = FxHashMap::default();
            primal.insert(x.var_id().unwrap(), x_value);
            primal.insert(y.var_id().unwrap(), y_value);
            primal.insert(t.var_id().unwrap(), t_value);
            SolutionPoint { primal, objective: None }
        };
        let result = SolverResult {
            primal_status: PrimalStatus::FeasiblePoint,
            solutions: vec![point(3.0, 4.0, 6.0), point(3.0, 4.0, 4.0)],
            ..Default::default()
        };

        assert_eq!(
            result.soc_evaluation(&m, cone),
            Some(SocEvaluation { norm: 5.0, bound: 6.0, slack: 1.0, violation: 0.0 })
        );
        assert!(
            (result.soc_evaluation_at(&m, cone, 1).unwrap().violation - 1.0).abs() < f64::EPSILON
        );
        assert!(result.soc_evaluation_at(&m, cone, 2).is_none());
    }

    #[test]
    fn best_is_solution_zero() {
        let mut p0 = FxHashMap::default();
        p0.insert(VarId(0), 1.5);
        let mut p1 = FxHashMap::default();
        p1.insert(VarId(0), 2.5);
        let r = SolverResult {
            termination: TerminationStatus::Optimal,
            primal_status: PrimalStatus::OptimalPoint,
            solutions: vec![
                SolutionPoint { primal: p0, objective: Some(10.0) },
                SolutionPoint { primal: p1, objective: Some(9.0) },
            ],
            ..Default::default()
        };
        assert_eq!(r.result_count(), 2);
        assert_eq!(r.objective(), Some(10.0));
        assert_eq!(r.value(VarId(0)), Some(1.5));
        assert_eq!(r.solution(1).unwrap().value(VarId(0)), Some(2.5));
    }

    #[test]
    fn report_renders_sections() {
        use oximo_core::{constraint, objective, variable};

        let m = Model::new("toy");
        variable!(m, x >= 0.0);
        let c = constraint!(m, c, x <= 5.0);
        objective!(m, Max, x);

        let mut primal = FxHashMap::default();
        primal.insert(x.var_id().unwrap(), 5.0);
        let mut dual = FxHashMap::default();
        dual.insert(c, 1.0);

        let r = SolverResult {
            termination: TerminationStatus::Optimal,
            primal_status: PrimalStatus::OptimalPoint,
            solutions: vec![SolutionPoint { primal, objective: Some(5.0) }],
            dual,
            solver_name: Some("TestSolver".into()),
            solver_version: Some("1.2.3".into()),
            raw_status: Some("native optimal".into()),
            dual_status: DualStatus::FeasiblePoint,
            node_count: Some(7),
            ..Default::default()
        };

        let out = r.report(&m).to_string();
        assert!(out.contains("solver     : TestSolver 1.2.3"), "{out}");
        assert!(out.contains("termination: Optimal"), "{out}");
        assert!(out.contains("primal     : OptimalPoint"), "{out}");
        assert!(out.contains("dual       : FeasiblePoint"), "{out}");
        assert!(out.contains("raw status : native optimal"), "{out}");
        assert!(out.contains("nodes      : 7"), "{out}");
        assert!(out.contains("objective  : 5"), "{out}");
        assert!(out.contains("(LP, maximize)"), "{out}");
        assert!(out.contains("x = 5"), "{out}");
        assert!(out.contains("dual = 1"), "{out}");
    }

    #[test]
    fn report_keeps_algebraic_duals_paired_when_skipping_soc_rows() {
        use oximo_core::{constraint, objective, variable};

        let m = Model::new("mixed");
        variable!(m, x >= 0.0);
        variable!(m, t >= 0.0);
        let first = constraint!(m, first, x <= 1.0);
        let second = constraint!(m, second, x >= 0.5);
        m.add_soc_constraint("cone", [x], t);
        objective!(m, Min, x);

        let mut dual = FxHashMap::default();
        dual.insert(first, 1.0);
        dual.insert(second, 2.0);
        let r = SolverResult { dual, ..Default::default() };

        let out = r.report(&m).to_string();
        assert!(out.contains("first   dual = 1"), "{out}");
        assert!(out.contains("second  dual = 2"), "{out}");
    }
}
