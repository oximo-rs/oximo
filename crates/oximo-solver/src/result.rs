use std::borrow::Cow;
use std::time::Duration;

use oximo_core::{
    ConstraintHandle, ConstraintId, ConstraintRef, Expr, IndexKey, IndexedVar, Model, ModelId,
    ModelMismatchError, SocConstraintHandle, SocConstraintId, VarId,
};
use oximo_expr::{EvalContext, ExprArena, ExprId, ExprNode, ParamId, evaluate};
use rustc_hash::FxHashMap;

use crate::status::{PrimalStatus, TerminationStatus};

/// A single primal point returned by a solver.
///
/// Most solves yield one point, but a global solver asked to enumerate solutions
/// may returns several. In a [`SolverResult`] the points live in [`SolverResult::solutions`].
/// Index `0` is always the best/incumbent.
#[derive(Clone, Debug)]
pub struct SolutionPoint {
    pub model_id: ModelId,
    pub primal: FxHashMap<VarId, f64>,
    pub objective: Option<f64>,
}

impl Default for SolutionPoint {
    fn default() -> Self {
        Self { model_id: ModelId::UNASSIGNED, primal: FxHashMap::default(), objective: None }
    }
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
    /// Model that produced this point.
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Look up a primal value by raw `VarId`. Raw IDs carry no model provenance;
    /// prefer [`Self::value_of`] when an expression handle is available.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.primal.get(&id).copied()
    }

    /// Evaluate an expression at this primal point.
    ///
    /// Returns `Ok(None)` when any variable needed by the expression is absent
    /// and [`ModelMismatchError`] when the expression belongs to another model.
    /// Parameter values are read from the expression's model arena at query
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `expr` belongs to another model.
    #[inline]
    pub fn value_of(&self, expr: Expr<'_>) -> Result<Option<f64>, ModelMismatchError> {
        ensure_model_id(self.model_id, expr.model_id())?;
        let arena = expr.arena.borrow();
        if let ExprNode::Var(id) = arena.get(expr.id) {
            return Ok(self.value(*id));
        }
        Ok(evaluate(&arena, expr.id, &PointContext(&self.primal)).ok())
    }

    /// Look up the primal value for a specific index of an [`IndexedVar`].
    ///
    /// Returns `Ok(None)` if `key` is not in the variable's set or the solver did
    /// not return a primal value for that scalar. Returns [`ModelMismatchError`]
    /// when the indexed variable belongs to another model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `var` belongs to another model.
    pub fn value_of_idx<V, K: Into<IndexKey>>(
        &self,
        var: &IndexedVar<'_, V>,
        key: K,
    ) -> Result<Option<f64>, ModelMismatchError> {
        ensure_model_id(self.model_id, var.model_id())?;
        var.get(key).map_or(Ok(None), |e| self.value_of(e))
    }

    /// Iterate over primal values for all entries of an [`IndexedVar`].
    ///
    /// Yields `(&IndexKey, f64)` for every index whose primal value is present
    /// in the solution.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `var` belongs to another model.
    pub fn values_of<'iv, 'a, V>(
        &'iv self,
        var: &'iv IndexedVar<'a, V>,
    ) -> Result<impl Iterator<Item = (&'iv IndexKey, f64)> + 'iv, ModelMismatchError> {
        ensure_model_id(self.model_id, var.model_id())?;
        Ok(var.iter().filter_map(|(k, e)| e.var_id().and_then(|id| self.value(id)).map(|v| (k, v))))
    }
}

#[inline]
fn ensure_model_id(expected: ModelId, actual: ModelId) -> Result<(), ModelMismatchError> {
    if actual == expected { Ok(()) } else { model_mismatch(expected, actual) }
}

#[cold]
#[inline(never)]
fn model_mismatch(expected: ModelId, actual: ModelId) -> Result<(), ModelMismatchError> {
    Err(ModelMismatchError::new(expected, actual))
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
    pub model_id: ModelId,
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
            model_id: ModelId::UNASSIGNED,
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
    /// Model that produced this result.
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

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

    /// Look up a primal value by raw `VarId` in the best solution. Raw IDs carry
    /// no model provenance; prefer [`Self::value_of`] when possible.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.solutions.first().and_then(|s| s.value(id))
    }

    /// Evaluate an expression at the best solution, rejecting expressions from
    /// another model with [`ModelMismatchError`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `expr` belongs to another model.
    #[inline]
    pub fn value_of(&self, expr: Expr<'_>) -> Result<Option<f64>, ModelMismatchError> {
        ensure_model_id(self.model_id, expr.model_id())?;
        self.solutions.first().map_or(Ok(None), |s| s.value_of(expr))
    }

    /// Evaluate an algebraic constraint at the best solution.
    ///
    /// The result and model must describe the same solve. Parameter values are
    /// read from `model` at query time, so do not combine an old result with a
    /// subsequently modified model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `model` or the selected solution point
    /// does not belong to this result's model.
    pub fn constraint_evaluation(
        &self,
        model: &Model,
        constraint: ConstraintHandle,
    ) -> Result<Option<ConstraintEvaluation>, ModelMismatchError> {
        self.constraint_evaluation_at(model, constraint, 0)
    }

    /// Evaluate an algebraic constraint at solution `solution_index`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `model` or the selected solution point
    /// does not belong to this result's model.
    pub fn constraint_evaluation_at(
        &self,
        model: &Model,
        constraint: ConstraintHandle,
        solution_index: usize,
    ) -> Result<Option<ConstraintEvaluation>, ModelMismatchError> {
        ensure_model_id(self.model_id, model.id())?;
        ensure_model_id(self.model_id, constraint.model_id())?;
        let Some(point) = self.solution(solution_index) else { return Ok(None) };
        ensure_model_id(self.model_id, point.model_id)?;
        Ok(evaluate_constraint_at(point, model, constraint.id()))
    }

    /// Evaluate an explicit second-order-cone constraint at the best solution.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `model` or the best solution point
    /// does not belong to this result's model.
    pub fn soc_evaluation(
        &self,
        model: &Model,
        constraint: SocConstraintHandle,
    ) -> Result<Option<SocEvaluation>, ModelMismatchError> {
        self.soc_evaluation_at(model, constraint, 0)
    }

    /// Evaluate an explicit second-order-cone constraint at solution
    /// `solution_index`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `model` or the selected solution point
    /// does not belong to this result's model.
    pub fn soc_evaluation_at(
        &self,
        model: &Model,
        constraint: SocConstraintHandle,
        solution_index: usize,
    ) -> Result<Option<SocEvaluation>, ModelMismatchError> {
        ensure_model_id(self.model_id, model.id())?;
        ensure_model_id(self.model_id, constraint.model_id())?;
        let Some(point) = self.solution(solution_index) else { return Ok(None) };
        ensure_model_id(self.model_id, point.model_id)?;
        Ok(evaluate_soc_at(point, model, constraint.id()))
    }

    /// Look up an algebraic constraint multiplier, rejecting a handle from a
    /// different model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `constraint` belongs to another model.
    pub fn dual_of(&self, constraint: ConstraintHandle) -> Result<Option<f64>, ModelMismatchError> {
        ensure_model_id(self.model_id, constraint.model_id())?;
        Ok(self.dual.get(&constraint.id()).copied())
    }

    /// The norm-form bound multiplier of an explicit SOC constraint,
    /// or `None` when the backend did not compute it.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `constraint` belongs to another model.
    pub fn soc_dual_of(
        &self,
        constraint: SocConstraintHandle,
    ) -> Result<Option<f64>, ModelMismatchError> {
        ensure_model_id(self.model_id, constraint.model_id())?;
        Ok(self.soc_dual.get(&constraint.id()).copied())
    }

    /// Look up the best solution's primal value for a specific index of an
    /// [`IndexedVar`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `var` belongs to another model.
    pub fn value_of_idx<V, K: Into<IndexKey>>(
        &self,
        var: &IndexedVar<'_, V>,
        key: K,
    ) -> Result<Option<f64>, ModelMismatchError> {
        ensure_model_id(self.model_id, var.model_id())?;
        var.get(key).map_or(Ok(None), |e| self.value_of(e))
    }

    /// Iterate over the best solution's primal values for all entries of an
    /// [`IndexedVar`]. Yields nothing when no solution was found.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `var` or the best solution point does
    /// not belong to this result's model.
    pub fn values_of<'iv, 'a, V>(
        &'iv self,
        var: &'iv IndexedVar<'a, V>,
    ) -> Result<impl Iterator<Item = (&'iv IndexKey, f64)> + 'iv, ModelMismatchError> {
        ensure_model_id(self.model_id, var.model_id())?;
        if let Some(point) = self.best() {
            ensure_model_id(self.model_id, point.model_id)?;
        }
        let point = self.best();
        Ok(var.iter().filter_map(move |(k, e)| {
            e.var_id().and_then(|id| point.and_then(|solution| solution.value(id))).map(|v| (k, v))
        }))
    }

    /// A human-readable, model-aware summary of this result.
    ///
    /// It lists the solver, model kind and sense, status,
    /// objective and work counters, then every variable's value
    /// (with its reduced cost when the solver returned duals) and every
    /// constraint's dual.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `model` or the best solution point
    /// does not belong to this result's model.
    pub fn report<'a>(&'a self, model: &'a Model) -> Result<ModelReport<'a>, ModelMismatchError> {
        ensure_model_id(self.model_id, model.id())?;
        if let Some(point) = self.best() {
            ensure_model_id(self.model_id, point.model_id)?;
        }
        Ok(ModelReport { result: self, model })
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
                    | ConstraintRef::SpecialOrderedSet { .. }
                    | ConstraintRef::Indicator { .. } => None,
                })
                .collect();
            writeln!(f, "\nconstraints ({})", cons.len())?;
            let width = cons.iter().map(|(_, c)| c.name.len()).max().unwrap_or(0);
            for (id, c) in cons {
                let d = r.dual.get(&id).copied().map_or_else(|| "n/a".to_owned(), num);
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
                let d = r.soc_dual.get(&id).copied().map_or_else(|| "n/a".to_owned(), num);
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
        let point = SolutionPoint { model_id: m.id(), primal, objective: None };

        assert_eq!(point.value_of(x), Ok(Some(3.0)));
        assert_eq!(point.value_of(2.0 * x + y - 1.0), Ok(Some(9.0)));
        assert_eq!(point.value_of(x.powi(2) + x * y), Ok(Some(21.0)));
        assert_eq!(point.value_of(x.sin()), Ok(Some(3.0_f64.sin())));
        assert_eq!(point.value_of(p * x + y), Ok(Some(10.0)));

        let incomplete = SolutionPoint { model_id: m.id(), ..Default::default() };
        assert!(incomplete.value_of(x + y).unwrap().is_none());
    }

    #[test]
    fn expression_queries_reject_foreign_models_with_colliding_variable_ids() {
        use oximo_core::variable;

        let source = Model::new("source");
        variable!(source, x);
        let foreign = Model::new("foreign");
        variable!(foreign, y);
        assert_eq!(x.var_id(), y.var_id());

        let point = SolutionPoint {
            model_id: source.id(),
            primal: [(x.var_id().unwrap(), 4.0)].into_iter().collect(),
            objective: None,
        };
        let result = SolverResult {
            model_id: source.id(),
            primal_status: PrimalStatus::FeasiblePoint,
            solutions: vec![point.clone()],
            ..Default::default()
        };
        let mismatch = ModelMismatchError::new(source.id(), foreign.id());

        assert_eq!(point.value_of(y), Err(mismatch));
        assert_eq!(result.value_of(y), Err(mismatch));
        assert!(matches!(result.report(&foreign), Err(error) if error == mismatch));
        assert_eq!(point.value_of(x), Ok(Some(4.0)));
    }

    #[test]
    fn constraint_queries_reject_foreign_handles_with_colliding_ids() {
        use oximo_core::{constraint, soc_constraint, variable};

        let source = Model::new("source");
        variable!(source, x);
        variable!(source, t);
        let source_row = constraint!(source, row, x <= 1.0);
        let source_cone = soc_constraint!(source, cone, [x] <= t);

        let foreign = Model::new("foreign");
        variable!(foreign, y);
        variable!(foreign, u);
        let foreign_row = constraint!(foreign, row, y <= 2.0);
        let foreign_cone = soc_constraint!(foreign, cone, [y] <= u);
        assert_eq!(source_row.id(), foreign_row.id());
        assert_eq!(source_cone.id(), foreign_cone.id());

        let result = SolverResult {
            model_id: source.id(),
            dual: [(source_row.id(), 3.0)].into_iter().collect(),
            soc_dual: [(source_cone.id(), 4.0)].into_iter().collect(),
            ..Default::default()
        };
        let mismatch = ModelMismatchError::new(source.id(), foreign.id());

        assert_eq!(result.dual_of(foreign_row), Err(mismatch));
        assert_eq!(result.soc_dual_of(foreign_cone), Err(mismatch));
        assert_eq!(result.constraint_evaluation(&source, foreign_row), Err(mismatch));
        assert_eq!(result.soc_evaluation(&source, foreign_cone), Err(mismatch));
        assert_eq!(result.dual_of(source_row), Ok(Some(3.0)));
        assert_eq!(result.soc_dual_of(source_cone), Ok(Some(4.0)));
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
        let ranged = m.constraint_handle("ranged").unwrap();

        let point = |value| {
            let mut primal = FxHashMap::default();
            primal.insert(x.var_id().unwrap(), value);
            SolutionPoint { model_id: m.id(), primal, objective: None }
        };
        let result = SolverResult {
            model_id: m.id(),
            primal_status: PrimalStatus::FeasiblePoint,
            solutions: vec![point(2.0), point(4.0)],
            ..Default::default()
        };

        assert_eq!(
            result.constraint_evaluation(&m, equality),
            Ok(Some(ConstraintEvaluation {
                activity: 2.0,
                lower_slack: Some(0.0),
                upper_slack: Some(0.0),
                violation: 0.0,
            }))
        );
        assert_eq!(
            result.constraint_evaluation(&m, lower).unwrap().unwrap().lower_slack,
            Some(1.0)
        );
        assert_eq!(result.constraint_evaluation(&m, lower).unwrap().unwrap().upper_slack, None);
        assert_eq!(result.constraint_evaluation(&m, upper).unwrap().unwrap().lower_slack, None);
        assert_eq!(
            result.constraint_evaluation(&m, upper).unwrap().unwrap().upper_slack,
            Some(1.0)
        );
        assert!(
            result.constraint_evaluation(&m, ranged).unwrap().unwrap().violation.abs()
                < f64::EPSILON
        );
        assert!(
            (result.constraint_evaluation_at(&m, ranged, 1).unwrap().unwrap().violation - 1.5)
                .abs()
                < f64::EPSILON
        );
        assert!(result.constraint_evaluation_at(&m, ranged, 2).unwrap().is_none());
        assert!(m.constraint_handle_from_id(ConstraintId(u32::MAX)).is_none());
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
            SolutionPoint { model_id: m.id(), primal, objective: None }
        };
        let result = SolverResult {
            model_id: m.id(),
            primal_status: PrimalStatus::FeasiblePoint,
            solutions: vec![point(3.0, 4.0, 6.0), point(3.0, 4.0, 4.0)],
            ..Default::default()
        };

        assert_eq!(
            result.soc_evaluation(&m, cone),
            Ok(Some(SocEvaluation { norm: 5.0, bound: 6.0, slack: 1.0, violation: 0.0 }))
        );
        assert!(
            (result.soc_evaluation_at(&m, cone, 1).unwrap().unwrap().violation - 1.0).abs()
                < f64::EPSILON
        );
        assert!(result.soc_evaluation_at(&m, cone, 2).unwrap().is_none());
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
                SolutionPoint { primal: p0, objective: Some(10.0), ..Default::default() },
                SolutionPoint { primal: p1, objective: Some(9.0), ..Default::default() },
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
        dual.insert(c.id(), 1.0);

        let r = SolverResult {
            model_id: m.id(),
            termination: TerminationStatus::Optimal,
            primal_status: PrimalStatus::OptimalPoint,
            solutions: vec![SolutionPoint { model_id: m.id(), primal, objective: Some(5.0) }],
            dual,
            solver_name: Some("TestSolver".into()),
            solver_version: Some("1.2.3".into()),
            raw_status: Some("native optimal".into()),
            dual_status: DualStatus::FeasiblePoint,
            node_count: Some(7),
            ..Default::default()
        };

        let out = r.report(&m).unwrap().to_string();
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
        dual.insert(first.id(), 1.0);
        dual.insert(second.id(), 2.0);
        let r = SolverResult { model_id: m.id(), dual, ..Default::default() };

        let out = r.report(&m).unwrap().to_string();
        assert!(out.contains("first   dual = 1"), "{out}");
        assert!(out.contains("second  dual = 2"), "{out}");
    }
}
