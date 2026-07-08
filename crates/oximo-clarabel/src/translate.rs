//! Translate an oximo [`Model`] into Clarabel's conic form
//! `min 0.5 x'Px + q'x  s.t.  Ax + s = b, s in K` and read the result back.
//!
//! Row layout:
//! 1. `ZeroCone`: equality constraints, then fixed variables (`lb == ub`).
//! 2. `NonnegativeCone`: `<=` rows as `(+a, ub)`, `>=` rows as `(-a, -lb)`,
//!    two-sided ranges as both, then finite variable bounds.
//! 3. One `SecondOrderCone` block per cone constraint, explicit
//!    [`SocConstraint`]s first, then SOC-shaped quadratic constraints detected
//!    by [`detect_soc`] in constraint order. Each block is
//!    `s0 = bound(x), s_i = term_i(x)` via `A` rows holding the negated
//!    affine coefficients.
//!
//! A `Maximize` objective negates `P`/`q` on the way in and the objective
//! value and duals on the way out. Duals follow the LP convention
//! `gradient = A' y` of the problem as posed. SOC blocks and variable-bound
//! rows carry no [`ConstraintId`] and are skipped. Reduced costs are not
//! reported.

// TODO: Add convex-QCP-to-SOC reformulation?

use std::time::{Duration, Instant};

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use oximo_core::{
    ConstraintId, Model, ObjectiveSense, Sense, SocConstraintId, SocForm, Variable, detect_soc,
    explicit_soc_form, var_name,
};
use oximo_expr::{LinearTerms, VarId, describe_nonlinear_term, extract_linear, extract_quadratic};
use oximo_solver::{PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
use rustc_hash::FxHashMap;

use crate::{ClarabelDirectSolve, ClarabelOptions};

/// The linear-or-conic view of one algebraic constraint.
enum Row {
    Lin(LinearTerms),
    Soc(SocForm),
}

/// `(row, col, value)` sparse-matrix entries.
type Triplets = Vec<(usize, usize, f64)>;

/// Accumulates the Clarabel `A` rows (as [`Triplets`]), the `b` vector, and the
/// per-row dual mapping while constraints and bounds are lowered into cone rows.
#[derive(Default)]
struct Rows {
    a_trip: Triplets,
    b: Vec<f64>,
    row_duals: Vec<Option<(ConstraintId, f64)>>,
}

impl Rows {
    /// Append one `A` row `scale * t.coeffs` with right-hand side `rhs`, carrying
    /// no dual. The caller attaches one with `set_last_dual` when the row maps
    /// back to a constraint.
    fn push(&mut self, t: &LinearTerms, scale: f64, rhs: f64) {
        let row = self.b.len();
        for &(var, coef) in &t.coeffs {
            self.a_trip.push((row, var.index(), scale * coef));
        }
        self.b.push(rhs);
        self.row_duals.push(None);
    }

    /// Append a single-entry row `scale * x_col = rhs` (a fixed variable or a
    /// finite bound), carrying no dual.
    fn push_bound(&mut self, col: usize, scale: f64, rhs: f64) {
        let row = self.b.len();
        self.a_trip.push((row, col, scale));
        self.b.push(rhs);
        self.row_duals.push(None);
    }

    /// Attach a dual mapping to the row just pushed.
    fn set_last_dual(&mut self, id: ConstraintId, scale: f64) {
        *self.row_duals.last_mut().unwrap() = Some((id, scale));
    }
}

/// Readback metadata, turn a solved [`DefaultSolver`] back into a
/// generic [`SolverResult`].
pub(crate) struct Meta {
    /// `-1.0` for a maximize objective (Clarabel always minimizes), else `1.0`.
    sign: f64,
    /// Folded objective constant, added back after solving.
    obj_constant: f64,
    /// Per `A`-row dual mapping: `Some((constraint, scale))`, or `None` for
    /// bound/cone rows that carry no [`ConstraintId`].
    row_duals: Vec<Option<(ConstraintId, f64)>>,
    /// Row index of each SOC block's `s0` entry.
    soc_block_starts: Vec<usize>,
    /// Count of leading SOC blocks that are explicit cones (duals reported).
    n_explicit: usize,
}

/// A translated Clarabel problem, conic data plus the [`Meta`] to read its
/// solution back. The persistent handle keeps the last one to test whether an
/// in-place [`DefaultSolver::update_data`] is possible.
pub(crate) struct Problem {
    pub(crate) p_mat: CscMatrix<f64>,
    pub(crate) q: Vec<f64>,
    pub(crate) a_mat: CscMatrix<f64>,
    pub(crate) b: Vec<f64>,
    pub(crate) cones: Vec<SupportedConeT<f64>>,
    pub(crate) meta: Meta,
}

impl Problem {
    /// True when `other` shares this problem's dimensions, cone layout, and
    /// `P`/`A` sparsity patterns.
    pub(crate) fn same_structure(&self, other: &Problem) -> bool {
        self.q.len() == other.q.len()
            && self.b.len() == other.b.len()
            && self.cones == other.cones
            && self.p_mat.colptr == other.p_mat.colptr
            && self.p_mat.rowval == other.p_mat.rowval
            && self.a_mat.colptr == other.a_mat.colptr
            && self.a_mat.rowval == other.a_mat.rowval
    }
}

/// Translate `model` into a Clarabel problem, solve, and return the generic
/// [`SolverResult`].
///
/// # Errors
///
/// Returns [`SolverError::UnsupportedKind`] for anything but continuous
/// LP/QP/SOCP, [`SolverError::Backend`] for semicontinuous/semi-integer
/// domains or a Clarabel setup failure, and [`SolverError::Nonlinear`] if an
/// expression defeats extraction.
pub fn solve(model: &Model, opts: &ClarabelOptions) -> Result<SolverResult, SolverError> {
    let problem = build_problem(model)?;
    let settings = build_settings(opts);
    let mut solver = DefaultSolver::new(
        &problem.p_mat,
        &problem.q,
        &problem.a_mat,
        &problem.b,
        &problem.cones,
        settings,
    )
    .map_err(|e| SolverError::Backend(format!("Clarabel setup: {e:?}")))?;
    let started = Instant::now();
    solver.solve();
    let elapsed = started.elapsed();
    Ok(read_result(&solver, &problem.meta, elapsed))
}

/// Translate `model` into Clarabel's conic form without solving.
///
/// # Errors
///
/// Returns [`SolverError::UnsupportedKind`] for anything but continuous
/// LP/QP/SOCP, [`SolverError::Backend`] for a semicontinuous/semi-integer
/// domain or an out-of-arena SOC member, and [`SolverError::Nonlinear`] if an
/// expression defeats extraction.
///
/// # Panics
///
/// Panics if variable or constraint indices overflow `u32`.
pub(crate) fn build_problem(model: &Model) -> Result<Problem, SolverError> {
    model.ensure_objective_declared().map_err(SolverError::Core)?;
    let kind = model.kind();
    if !crate::supported(kind) {
        return Err(SolverError::UnsupportedKind(kind));
    }
    let vars = model.variables();
    reject_semi_domains(&vars)?;
    let n = vars.len();

    let (sign, p_trip, q, obj_constant) = objective_data(model, n)?;
    let rows = classify_rows(model)?;
    let (mut acc, m_zero, m_nonneg) = linear_rows(model, &rows);
    let (soc_sizes, soc_block_starts, n_explicit) = soc_blocks(model, &rows, &mut acc)?;

    let Rows { a_trip, b, row_duals } = acc;
    let m = b.len();
    let a_mat = csc_from_triplets(m, n, a_trip);
    let p_mat = csc_from_triplets(n, n, p_trip);
    let cones = build_cones(m_zero, m_nonneg, &soc_sizes);

    Ok(Problem {
        p_mat,
        q,
        a_mat,
        b,
        cones,
        meta: Meta { sign, obj_constant, row_duals, soc_block_starts, n_explicit },
    })
}

/// The objective in Clarabel form: `sign` (`-1` maximize, else `+1`. It negates
/// `P`/`q` so a maximize is posed as a minimize.), the upper-triangular `P`
/// triplets, the `q` vector, and the folded constant.
///
/// `extract_quadratic` subsumes the linear case. oximo's Hessian is
/// lower-triangular with doubled diagonal (the `0.5 x'Qx` convention), matching
/// Clarabel's `P` up to the triangle side, so entries transpose to
/// upper-triangular with no scaling.
fn objective_data(model: &Model, n: usize) -> Result<(f64, Triplets, Vec<f64>, f64), SolverError> {
    let arena = model.arena();
    let objective = model.objective();
    let sign = match objective.as_ref().map(|o| o.sense) {
        Some(ObjectiveSense::Maximize) => -1.0,
        _ => 1.0,
    };
    let Some(obj) = objective.as_ref() else {
        return Ok((sign, Triplets::new(), vec![0.0; n], 0.0));
    };
    let vars = model.variables();
    let quad = extract_quadratic(&arena, obj.expr).ok_or_else(|| SolverError::Nonlinear {
        location: "the objective".into(),
        term: describe_nonlinear_term(&arena, obj.expr, &|v| var_name(&vars, v))
            .unwrap_or_else(|| "<nonlinear>".into()),
    })?;
    let mut q = vec![0.0; n];
    for &(var, coef) in &quad.linear {
        q[var.index()] += sign * coef;
    }
    let p_trip =
        quad.hessian.iter().map(|&(row, col, h)| (col.index(), row.index(), sign * h)).collect();
    Ok((sign, p_trip, q, quad.constant))
}

/// Classify every algebraic constraint as linear or SOC. The kind gate
/// guarantees a quadratic constraint detects as SOC, so a miss here is a
/// genuine nonlinear.
fn classify_rows(model: &Model) -> Result<Vec<Row>, SolverError> {
    let arena = model.arena();
    let vars = model.variables();
    model
        .constraints()
        .iter()
        .map(|c| match extract_linear(&arena, c.lhs) {
            Some(t) => Ok(Row::Lin(t)),
            None => {
                detect_soc(&arena, &vars, c).map(Row::Soc).ok_or_else(|| SolverError::Nonlinear {
                    location: format!("constraint {:?}", c.name),
                    term: describe_nonlinear_term(&arena, c.lhs, &|v| var_name(&vars, v))
                        .unwrap_or_else(|| "<nonlinear>".into()),
                })
            }
        })
        .collect()
}

/// Lower the linear constraints and variable bounds into cone rows: the
/// `ZeroCone` block (equalities, then fixed variables) followed by the
/// `NonnegativeCone` block (inequalities/ranges, then finite bounds). Returns
/// the accumulated [`Rows`] and the two block sizes.
fn linear_rows(model: &Model, rows: &[Row]) -> (Rows, usize, usize) {
    let vars = model.variables();
    let constraints = model.constraints();
    let is_fixed = |v: &Variable| v.lb.is_finite() && v.lb.total_cmp(&v.ub).is_eq();
    let mut acc = Rows::default();

    // ZeroCone rows: equalities, then fixed variables.
    for (i, (con, row)) in constraints.iter().zip(rows).enumerate() {
        if let Row::Lin(lt) = row {
            if let Some((Sense::Eq, rhs)) = con.as_single() {
                let id = ConstraintId(u32::try_from(i).expect("constraint count overflow"));
                acc.push(lt, 1.0, rhs - lt.constant);
                acc.set_last_dual(id, -1.0);
            }
        }
    }
    for var in vars.iter().filter(|&v| is_fixed(v)) {
        acc.push_bound(var.id.index(), 1.0, var.lb);
    }
    let m_zero = acc.b.len();

    // NonnegativeCone rows: inequalities and ranges, then variable bounds.
    for (i, (con, row)) in constraints.iter().zip(rows).enumerate() {
        let Row::Lin(lt) = row else { continue };
        let id = ConstraintId(u32::try_from(i).expect("constraint count overflow"));
        match con.as_single() {
            Some((Sense::Le, rhs)) => {
                acc.push(lt, 1.0, rhs - lt.constant);
                acc.set_last_dual(id, -1.0);
            }
            Some((Sense::Ge, rhs)) => {
                acc.push(lt, -1.0, -(rhs - lt.constant));
                acc.set_last_dual(id, 1.0);
            }
            None if con.is_range() => {
                acc.push(lt, -1.0, -(con.lower - lt.constant));
                acc.set_last_dual(id, 1.0);
                acc.push(lt, 1.0, con.upper - lt.constant);
                acc.set_last_dual(id, -1.0);
            }
            Some((Sense::Eq, _)) | None => {}
        }
    }
    for var in vars.iter().filter(|&v| !is_fixed(v)) {
        if var.ub.is_finite() {
            acc.push_bound(var.id.index(), 1.0, var.ub);
        }
        if var.lb.is_finite() {
            acc.push_bound(var.id.index(), -1.0, -var.lb);
        }
    }
    let m_nonneg = acc.b.len() - m_zero;

    (acc, m_zero, m_nonneg)
}

/// Append the second-order-cone blocks (explicit cones first, then detected
/// ones) to `acc`. Returns each block's size, the row index of each block's
/// `s0` entry, and how many leading blocks are explicit cones.
///
/// Only explicit cones' duals are reported: their bound multiplier `z0` maps
/// back to the constraint, whereas `SocForm` normalizes away a detected cone's
/// original quadratic scaling, so its `z0` cannot be mapped back.
fn soc_blocks(
    model: &Model,
    rows: &[Row],
    acc: &mut Rows,
) -> Result<(Vec<usize>, Vec<usize>, usize), SolverError> {
    let arena = model.arena();
    let socs = model.soc_constraints();
    let explicit_forms = socs
        .iter()
        .map(|s| {
            explicit_soc_form(&arena, s).ok_or_else(|| {
                SolverError::Backend(format!(
                    "SOC constraint '{}' has a member outside this model's arena",
                    s.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let detected_forms = rows.iter().filter_map(|r| match r {
        Row::Soc(f) => Some(f.clone()),
        Row::Lin(_) => None,
    });

    let n_explicit = explicit_forms.len();
    let mut soc_sizes: Vec<usize> = Vec::new();
    let mut soc_block_starts: Vec<usize> = Vec::new();
    for form in explicit_forms.into_iter().chain(detected_forms) {
        soc_block_starts.push(acc.b.len());
        acc.push(&form.bound, -1.0, form.bound.constant);
        for term in &form.terms {
            acc.push(term, -1.0, term.constant);
        }
        soc_sizes.push(1 + form.terms.len());
    }
    Ok((soc_sizes, soc_block_starts, n_explicit))
}

/// Assemble the cone list in row order: zero cone, nonnegative cone, then one
/// second-order cone per SOC block.
fn build_cones(m_zero: usize, m_nonneg: usize, soc_sizes: &[usize]) -> Vec<SupportedConeT<f64>> {
    let mut cones = Vec::new();
    if m_zero > 0 {
        cones.push(SupportedConeT::ZeroConeT(m_zero));
    }
    if m_nonneg > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(m_nonneg));
    }
    cones.extend(soc_sizes.iter().map(|&k| SupportedConeT::SecondOrderConeT(k)));
    cones
}

/// Read a solved [`DefaultSolver`] back into a generic [`SolverResult`],
/// applying the sign/constant folding and dual mapping recorded in `meta`.
pub(crate) fn read_result(
    solver: &DefaultSolver<f64>,
    meta: &Meta,
    elapsed: Duration,
) -> SolverResult {
    let termination = map_status(solver.solution.status);
    let has_point = termination.admits_primal();

    let mut solutions = Vec::new();
    let mut dual: FxHashMap<ConstraintId, f64> = FxHashMap::default();
    let mut soc_dual: FxHashMap<SocConstraintId, f64> = FxHashMap::default();
    if has_point {
        let primal: FxHashMap<VarId, f64> = solver
            .solution
            .x
            .iter()
            .enumerate()
            .map(|(i, &val)| (VarId(u32::try_from(i).expect("variable count overflow")), val))
            .collect();
        let objective = Some(meta.sign * solver.solution.obj_val + meta.obj_constant);
        solutions.push(SolutionPoint { primal, objective });

        for (r, &z) in solver.solution.z.iter().enumerate() {
            if let Some(Some((id, s))) = meta.row_duals.get(r) {
                *dual.entry(*id).or_insert(0.0) += meta.sign * s * z;
            }
        }
        for (k, &start) in meta.soc_block_starts.iter().take(meta.n_explicit).enumerate() {
            if let Some(&z0) = solver.solution.z.get(start) {
                soc_dual.insert(SocConstraintId(u32::try_from(k).expect("SOC count overflow")), z0);
            }
        }
    }
    let primal_status = PrimalStatus::infer(&termination, !solutions.is_empty());

    SolverResult {
        termination,
        primal_status,
        solutions,
        dual,
        soc_dual,
        reduced_costs: FxHashMap::default(),
        best_bound: None,
        gap: None,
        solve_time: elapsed,
        iterations: u64::from(solver.info.iterations),
        raw_log: None,
        solver_name: Some(crate::NAME.into()),
    }
}

/// Assemble an `m x n` [`CscMatrix`] from `(row, col, value)` triplets,
/// merging duplicates.
fn csc_from_triplets(m: usize, n: usize, mut trip: Triplets) -> CscMatrix<f64> {
    trip.sort_unstable_by_key(|&(row, col, _)| (col, row));
    let mut colptr = vec![0_usize; n + 1];
    let mut rowval: Vec<usize> = Vec::with_capacity(trip.len());
    let mut nzval: Vec<f64> = Vec::with_capacity(trip.len());
    let mut last: Option<(usize, usize)> = None;
    for (row, col, val) in trip {
        if last == Some((row, col)) {
            *nzval.last_mut().unwrap() += val;
        } else {
            colptr[col + 1] += 1;
            rowval.push(row);
            nzval.push(val);
            last = Some((row, col));
        }
    }
    for col in 0..n {
        colptr[col + 1] += colptr[col];
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

pub(crate) fn build_settings(o: &ClarabelOptions) -> DefaultSettings<f64> {
    let mut s = DefaultSettings {
        verbose: o.universal.verbose.unwrap_or(false),
        ..DefaultSettings::default()
    };
    if let Some(d) = o.universal.time_limit {
        s.time_limit = d.as_secs_f64();
    }
    // Universal `threads` feeds Clarabel's `max_threads` (only affects
    // multithreaded KKT solvers, the default qdldl solver is single-threaded).
    if let Some(n) = o.universal.threads {
        s.max_threads = n;
    }
    if let Some(m) = o.direct_solve_method {
        s.direct_solve_method = kkt_str(m).to_string();
    }

    // Every scalar `ClarabelOptions` field shares its name with the matching
    // `DefaultSettings` field, so apply each verbatim when set.
    macro_rules! apply_opt {
        ($($field:ident),* $(,)?) => {
            $(if let Some(v) = o.$field { s.$field = v; })*
        };
    }
    apply_opt!(
        max_iter,
        max_step_fraction,
        tol_gap_abs,
        tol_gap_rel,
        tol_feas,
        tol_infeas_abs,
        tol_infeas_rel,
        tol_ktratio,
        reduced_tol_gap_abs,
        reduced_tol_gap_rel,
        reduced_tol_feas,
        reduced_tol_infeas_abs,
        reduced_tol_infeas_rel,
        reduced_tol_ktratio,
        equilibrate_enable,
        equilibrate_max_iter,
        equilibrate_min_scaling,
        equilibrate_max_scaling,
        linesearch_backtrack_step,
        min_switch_step_length,
        min_terminate_step_length,
        static_regularization_enable,
        static_regularization_constant,
        static_regularization_proportional,
        dynamic_regularization_enable,
        dynamic_regularization_eps,
        dynamic_regularization_delta,
        iterative_refinement_enable,
        iterative_refinement_reltol,
        iterative_refinement_abstol,
        iterative_refinement_max_iter,
        iterative_refinement_stop_ratio,
        presolve_enable,
        input_sparse_dropzeros,
    );
    s
}

/// Lower a [`ClarabelDirectSolve`] to the string Clarabel's settings expect.
fn kkt_str(m: ClarabelDirectSolve) -> &'static str {
    match m {
        ClarabelDirectSolve::Auto => "auto",
        ClarabelDirectSolve::Qdldl => "qdldl",
        #[cfg(feature = "faer")]
        ClarabelDirectSolve::Faer => "faer",
    }
}

fn reject_semi_domains(vars: &[Variable]) -> Result<(), SolverError> {
    for v in vars {
        if v.domain.semi_threshold().is_some() {
            return Err(SolverError::Backend(format!(
                "variable {} has a semicontinuous/semi-integer domain, \
                which Clarabel does not support",
                v.name
            )));
        }
    }
    Ok(())
}

fn map_status(s: SolverStatus) -> TerminationStatus {
    match s {
        SolverStatus::Solved => TerminationStatus::Optimal,
        SolverStatus::AlmostSolved => TerminationStatus::Interrupted,
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
            TerminationStatus::Infeasible
        }
        SolverStatus::DualInfeasible | SolverStatus::AlmostDualInfeasible => {
            TerminationStatus::Unbounded
        }
        SolverStatus::MaxIterations => TerminationStatus::IterationLimit,
        SolverStatus::MaxTime => TerminationStatus::TimeLimit,
        SolverStatus::NumericalError | SolverStatus::InsufficientProgress => {
            TerminationStatus::NumericError
        }
        SolverStatus::Unsolved => TerminationStatus::NotSolved,
        other @ SolverStatus::CallbackTerminated => TerminationStatus::Other(format!("{other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;
    use oximo_solver::UniversalOptionsExt;

    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn lp_known_optimum_maximize() {
        // max 3x + 2y  s.t.  x + y <= 4, 0 <= x, y <= 3.
        // Optimum x = 3, y = 1, objective 11.
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 3.0);
        variable!(m, 0.0 <= y <= 3.0);
        constraint!(m, cap, x + y <= 4.0);
        objective!(m, Max, 3.0 * x + 2.0 * y);
        assert_eq!(m.kind(), ModelKind::LP);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 11.0, 1e-6));
        assert!(close(res.value_of(x).unwrap(), 3.0, 1e-6));
        assert!(close(res.value_of(y).unwrap(), 1.0, 1e-6));
    }

    #[test]
    fn lp_range_constraint() {
        // min x + y  s.t.  1 <= x + y <= 3, x, y >= 0. Optimum objective 1.
        let m = Model::new("range");
        variable!(m, x >= 0.0);
        variable!(m, y >= 0.0);
        constraint!(m, band, 1.0 <= x + y <= 3.0);
        objective!(m, Min, x + y);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 1.0, 1e-6));
    }

    #[test]
    fn qp_cvxopt_quickstart() {
        // min 2 x0^2 + x0 x1 + x1^2 + x0 + x1  s.t.  x0 + x1 = 1, x >= 0.
        // x = [0.25, 0.75], objective = 1.875.
        let m = Model::new("cvxopt");
        variable!(m, x0 >= 0.0);
        variable!(m, x1 >= 0.0);
        constraint!(m, eq, x0 + x1 == 1.0);
        objective!(m, Min, 2.0 * x0.powi(2) + x0 * x1 + x1.powi(2) + x0 + x1);
        assert_eq!(m.kind(), ModelKind::QP);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.value_of(x0).unwrap(), 0.25, 1e-5));
        assert!(close(res.value_of(x1).unwrap(), 0.75, 1e-5));
        assert!(close(res.objective().unwrap(), 1.875, 1e-5));
    }

    #[test]
    fn qp_objective_constant_is_added_back() {
        // min (x - 1)^2  ->  x = 1, objective 0 (folded constant 1).
        let m = Model::new("shift");
        variable!(m, -5.0 <= x <= 5.0);
        objective!(m, Min, (x - 1.0).powi(2));
        assert_eq!(m.kind(), ModelKind::QP);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.value_of(x).unwrap(), 1.0, 1e-5));
        assert!(res.objective().unwrap().abs() < 1e-5);
    }

    #[test]
    fn explicit_socp_min_linear_over_disk() {
        // min x + y  s.t.  ||(x, y)||_2 <= 1. Optimum objective -sqrt(2).
        // KKT: (1, 1) + z0 * (x, y)/||(x, y)|| = 0  =>  z0 = sqrt(2).
        let m = Model::new("socp");
        variable!(m, x);
        variable!(m, y);
        variable!(m, t >= 0.0);
        m.fix(t, 1.0);
        let disk = m.add_soc_constraint("disk", [x, y], t);
        objective!(m, Min, x + y);
        assert_eq!(m.kind(), ModelKind::SOCP);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), -std::f64::consts::SQRT_2, 1e-6));

        let z0 = res.soc_dual_of(disk).expect("SOC dual missing");
        assert!(close(z0, std::f64::consts::SQRT_2, 1e-6), "z0 = {z0}");
    }

    #[test]
    fn detected_socp_hypotenuse() {
        // min t  s.t.  x^2 + y^2 <= t^2, t >= 0, x = 3, y = 4. Optimum t = 5.
        let m = Model::new("socp_detected");
        variable!(m, x);
        variable!(m, y);
        variable!(m, t >= 0.0);
        m.fix(x, 3.0);
        m.fix(y, 4.0);
        constraint!(m, cone, x.powi(2) + y.powi(2) <= t.powi(2));
        objective!(m, Min, t);
        assert_eq!(m.kind(), ModelKind::SOCP);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 5.0, 1e-5));
    }

    #[test]
    fn socp_with_quadratic_objective() {
        // min x^2 + y  s.t.  ||(x, y)|| <= 2, y >= -2: unconstrained in x at
        // x = 0, then y = -2 on the cone boundary. Objective -2.
        let m = Model::new("socp_qobj");
        variable!(m, x);
        variable!(m, y);
        variable!(m, t >= 0.0);
        m.fix(t, 2.0);
        m.add_soc_constraint("disk", [x, y], t);
        objective!(m, Min, x.powi(2) + y);
        assert_eq!(m.kind(), ModelKind::SOCP);

        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), -2.0, 1e-5));
    }

    #[test]
    fn milp_is_unsupported() {
        let m = Model::new("milp");
        variable!(m, 0.0 <= x <= 5.0, Int);
        objective!(m, Min, x);
        let err = solve(&m, &ClarabelOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::MILP)));
    }

    #[test]
    fn qcp_is_unsupported() {
        // Bilinear constraint is not SOC-shaped, so this stays QCP and is
        // rejected.
        let m = Model::new("qcp");
        variable!(m, x >= 0.0);
        variable!(m, y >= 0.0);
        constraint!(m, c, x * y <= 4.0);
        objective!(m, Min, x + y);
        let err = solve(&m, &ClarabelOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::QCP)));
    }

    #[test]
    fn nlp_is_unsupported() {
        let m = Model::new("nlp");
        variable!(m, x >= 0.1);
        objective!(m, Min, x.sin());
        let err = solve(&m, &ClarabelOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::UnsupportedKind(ModelKind::NLP)));
    }

    #[test]
    fn semi_domain_is_rejected() {
        let m = Model::new("semi");
        variable!(m, s <= 10.0, SemiCont(2.0));
        objective!(m, Min, s);
        let err = solve(&m, &ClarabelOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::Backend(_)));
    }

    #[test]
    fn infeasible_lp_is_reported() {
        let m = Model::new("infeas");
        variable!(m, 0.0 <= x <= 1.0);
        constraint!(m, c, x >= 2.0);
        objective!(m, Min, x);
        let res = solve(&m, &ClarabelOptions::default()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Infeasible);
        assert!(!res.has_solution());
    }

    #[test]
    fn lp_dual_signs_match_highs() {
        // min 2x + 3y  s.t.  x + y >= 2, x - y <= 1, x, y >= 0.
        fn build() -> Model {
            let m = Model::new("duals");
            variable!(m, x >= 0.0);
            variable!(m, y >= 0.0);
            constraint!(m, demand, x + y >= 2.0);
            constraint!(m, link, x - y <= 1.0);
            objective!(m, Min, 2.0 * x + 3.0 * y);
            m
        }

        let m = build();
        let ours = solve(&m, &ClarabelOptions::default()).unwrap();
        let reference = oximo_highs::solve(&m, &oximo_highs::HighsOptions::default()).unwrap();
        assert_eq!(ours.termination, TerminationStatus::Optimal);
        for (id, want) in &reference.dual {
            let got = ours.dual.get(id).copied().unwrap_or(0.0);
            assert!(close(got, *want, 1e-5), "dual {id:?}: clarabel {got} vs highs {want}");
        }
    }

    /// Every option set to a valid, near-default value.
    fn all_options_set() -> ClarabelOptions {
        ClarabelOptions::default()
            .threads(1)
            .direct_solve_method(ClarabelDirectSolve::Auto)
            .max_iter(500)
            .max_step_fraction(0.99)
            .tol_gap_abs(1e-8)
            .tol_gap_rel(1e-8)
            .tol_feas(1e-8)
            .tol_infeas_abs(1e-8)
            .tol_infeas_rel(1e-8)
            .tol_ktratio(1e-6)
            .reduced_tol_gap_abs(5e-5)
            .reduced_tol_gap_rel(5e-5)
            .reduced_tol_feas(1e-4)
            .reduced_tol_infeas_abs(5e-12)
            .reduced_tol_infeas_rel(5e-5)
            .reduced_tol_ktratio(1e-4)
            .equilibrate_enable(true)
            .equilibrate_max_iter(10)
            .equilibrate_min_scaling(1e-4)
            .equilibrate_max_scaling(1e4)
            .linesearch_backtrack_step(0.8)
            .min_switch_step_length(1e-1)
            .min_terminate_step_length(1e-4)
            .static_regularization_enable(true)
            .static_regularization_constant(1e-8)
            .static_regularization_proportional(1e-30)
            .dynamic_regularization_enable(true)
            .dynamic_regularization_eps(1e-13)
            .dynamic_regularization_delta(2e-7)
            .iterative_refinement_enable(true)
            .iterative_refinement_reltol(1e-13)
            .iterative_refinement_abstol(1e-12)
            .iterative_refinement_max_iter(10)
            .iterative_refinement_stop_ratio(5.0)
            .presolve_enable(true)
            .input_sparse_dropzeros(false)
    }

    #[test]
    fn builder_sets_all_fields() {
        let o = all_options_set();
        // universal + enum + a representative scalar of each type.
        assert_eq!(o.universal.threads, Some(1));
        assert_eq!(o.direct_solve_method, Some(ClarabelDirectSolve::Auto));
        assert_eq!(o.max_iter, Some(500)); // u32
        assert_eq!(o.tol_gap_abs, Some(1e-8)); // f64
        assert_eq!(o.reduced_tol_infeas_abs, Some(5e-12));
        assert_eq!(o.equilibrate_max_iter, Some(10));
        assert_eq!(o.presolve_enable, Some(true)); // bool
        assert_eq!(o.input_sparse_dropzeros, Some(false));
        assert_eq!(o.iterative_refinement_stop_ratio, Some(5.0));
    }

    #[test]
    fn apply_all_options_solves() {
        // Setting every option must still produce settings `DefaultSolver::new`
        // accepts and an LP that solves to the known optimum.
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 3.0);
        variable!(m, 0.0 <= y <= 3.0);
        constraint!(m, cap, x + y <= 4.0);
        objective!(m, Max, 3.0 * x + 2.0 * y);

        let res = solve(&m, &all_options_set()).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 11.0, 1e-6));
    }

    #[test]
    fn direct_solve_method_qdldl_solves() {
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 3.0);
        variable!(m, 0.0 <= y <= 3.0);
        constraint!(m, cap, x + y <= 4.0);
        objective!(m, Max, 3.0 * x + 2.0 * y);

        let opts = ClarabelOptions::default().direct_solve_method(ClarabelDirectSolve::Qdldl);
        let res = solve(&m, &opts).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 11.0, 1e-6));
    }

    #[cfg(feature = "faer")]
    #[test]
    fn direct_solve_method_faer_solves() {
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 3.0);
        variable!(m, 0.0 <= y <= 3.0);
        constraint!(m, cap, x + y <= 4.0);
        objective!(m, Max, 3.0 * x + 2.0 * y);

        let opts = ClarabelOptions::default().direct_solve_method(ClarabelDirectSolve::Faer);
        let res = solve(&m, &opts).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 11.0, 1e-6));
    }

    #[test]
    fn low_max_iter_caps_iterations() {
        let m = Model::new("cvxopt");
        variable!(m, x0 >= 0.0);
        variable!(m, x1 >= 0.0);
        constraint!(m, eq, x0 + x1 == 1.0);
        objective!(m, Min, 2.0 * x0.powi(2) + x0 * x1 + x1.powi(2) + x0 + x1);

        let res = solve(&m, &ClarabelOptions::default().max_iter(1)).unwrap();
        assert!(res.iterations <= 1, "iterations = {}", res.iterations);
    }

    #[test]
    fn threads_maps_to_max_threads() {
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 3.0);
        variable!(m, 0.0 <= y <= 3.0);
        constraint!(m, cap, x + y <= 4.0);
        objective!(m, Max, 3.0 * x + 2.0 * y);

        let res = solve(&m, &ClarabelOptions::default().threads(2)).unwrap();
        assert_eq!(res.termination, TerminationStatus::Optimal);
        assert!(close(res.objective().unwrap(), 11.0, 1e-6));
    }
}
