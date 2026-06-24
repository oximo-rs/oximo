use std::borrow::Cow;
use std::time::Duration;

use oximo_core::{
    ConstraintId, Expr, ExprNode, IndexKey, IndexedVar, Model, ObjectiveSense, VarId,
};
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

impl SolutionPoint {
    /// Look up a primal value by `VarId`.
    pub fn value(&self, id: VarId) -> Option<f64> {
        self.primal.get(&id).copied()
    }

    /// Look up the primal value for an `Expr` that points at a `Var` node.
    /// Returns `None` for any expression that is not a single variable, so
    /// callers that need the value of a derived expression should evaluate it
    /// against the primal solution explicitly.
    pub fn value_of(&self, expr: Expr<'_>) -> Option<f64> {
        let arena = expr.arena.borrow();
        match arena.get(expr.id) {
            ExprNode::Var(v) => self.primal.get(v).copied(),
            _ => None,
        }
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

/// A solver's final result on a model.
///
/// `termination` expresses why the solver stopped and `primal_status` says
/// whether the point in `solutions` is usable. Primal points are held in
/// `solutions` (index `0` is the best/incumbent, empty when no solution was
/// found). `dual` and `reduced_costs` apply to the best continuous point and are
/// sparse maps, so a solver that does not return duals (e.g. MILP) can simply
/// leave them empty. `best_bound` and `gap` are populated by branch-and-bound
/// backends when available.
#[derive(Clone, Debug)]
pub struct SolverResult {
    pub termination: TerminationStatus,
    pub primal_status: PrimalStatus,
    pub solutions: Vec<SolutionPoint>,
    pub dual: FxHashMap<ConstraintId, f64>,
    pub reduced_costs: FxHashMap<VarId, f64>,
    pub best_bound: Option<f64>,
    pub gap: Option<f64>,
    pub solve_time: Duration,
    pub iterations: u64,
    pub raw_log: Option<String>,
    pub solver_name: Option<Cow<'static, str>>,
}

impl Default for SolverResult {
    fn default() -> Self {
        Self {
            termination: TerminationStatus::NotSolved,
            primal_status: PrimalStatus::NoSolution,
            solutions: Vec::new(),
            dual: FxHashMap::default(),
            reduced_costs: FxHashMap::default(),
            best_bound: None,
            gap: None,
            solve_time: Duration::ZERO,
            iterations: 0,
            raw_log: None,
            solver_name: None,
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

    /// Look up the best solution's primal value for an `Expr` that points at a
    /// `Var` node. Returns `None` for any expression that is not a single
    /// variable.
    pub fn value_of(&self, expr: Expr<'_>) -> Option<f64> {
        self.solutions.first().and_then(|s| s.value_of(expr))
    }

    pub fn dual_of(&self, c: ConstraintId) -> Option<f64> {
        self.dual.get(&c).copied()
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

        let sense = {
            let obj = m.objective();
            match obj.as_ref().map(|o| o.sense) {
                Some(ObjectiveSense::Minimize) => "minimize",
                Some(ObjectiveSense::Maximize) => "maximize",
                None => "no objective",
            }
        };

        writeln!(f, "solution summary")?;
        writeln!(f, "  solver     : {}", r.solver_name.as_deref().unwrap_or("(unknown)"))?;
        writeln!(f, "  model      : {}  ({:?}, {sense})", m.name, m.kind())?;
        writeln!(f, "  termination: {:?}", r.termination)?;
        writeln!(f, "  primal     : {:?}", r.primal_status)?;
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
            let cons = m.constraints();
            writeln!(f, "\nconstraints ({})", cons.len())?;
            let width = cons.iter().map(|c| c.name.len()).max().unwrap_or(0);
            for (i, c) in cons.iter().enumerate() {
                let id = ConstraintId(u32::try_from(i).expect("constraint index fits u32"));
                let d = r.dual_of(id).map_or_else(|| "n/a".to_owned(), num);
                writeln!(f, "  {:<width$}  dual = {d}", c.name)?;
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
            ..Default::default()
        };

        let out = r.report(&m).to_string();
        assert!(out.contains("solver     : TestSolver"), "{out}");
        assert!(out.contains("termination: Optimal"), "{out}");
        assert!(out.contains("primal     : OptimalPoint"), "{out}");
        assert!(out.contains("objective  : 5"), "{out}");
        assert!(out.contains("(LP, maximize)"), "{out}");
        assert!(out.contains("x = 5"), "{out}");
        assert!(out.contains("dual = 1"), "{out}");
    }
}
