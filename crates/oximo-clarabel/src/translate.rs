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
use oximo_solver::{
    DualStatus, PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{ClarabelDirectSolve, ClarabelOptions};

const PAR_ROW_THRESHOLD: usize = 256;

/// The linear-or-conic view of one algebraic constraint.
#[derive(Debug)]
enum Row {
    Lin(LinearTerms<'static>),
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
    fn with_capacity(rows: usize, nonzeros: usize) -> Self {
        Self {
            a_trip: Vec::with_capacity(nonzeros),
            b: Vec::with_capacity(rows),
            row_duals: Vec::with_capacity(rows),
        }
    }

    /// Append one `A` row `scale * t.coeffs` with right-hand side `rhs`, carrying
    /// no dual. The caller attaches one with `set_last_dual` when the row maps
    /// back to a constraint.
    fn push(&mut self, t: &LinearTerms<'_>, scale: f64, rhs: f64) {
        let row = self.b.len();
        for &(var, coef) in t.coeffs.iter() {
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
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
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
    if model.has_active_sos_constraints() {
        return Err(SolverError::UnsupportedSos);
    }
    build_problem_with(model, None)
}

fn build_problem_with(model: &Model, parallel: Option<bool>) -> Result<Problem, SolverError> {
    model.ensure_objective_declared().map_err(SolverError::Core)?;
    let kind = model.kind();
    if !crate::supported(kind) {
        return Err(SolverError::UnsupportedKind { kind, supported: crate::SUPPORTED_KINDS });
    }
    let vars = model.variables();
    reject_semi_domains(&vars)?;
    let n = vars.len();

    let (sign, p_trip, q, obj_constant) = objective_data(model, n)?;
    let rows = classify_rows_with(model, parallel)?;
    let arena = model.arena();
    let socs = model.soc_constraints();
    let explicit_forms = explicit_soc_forms(&arena, &socs, parallel)?;
    let (row_capacity, nonzero_capacity, soc_capacity) =
        translation_capacities(model, &rows, &explicit_forms);
    let acc = Rows::with_capacity(row_capacity, nonzero_capacity);
    let (mut acc, m_zero, m_nonneg) = linear_rows(model, &rows, acc);
    let (soc_sizes, soc_block_starts, n_explicit) =
        soc_blocks(&rows, &explicit_forms, &mut acc, soc_capacity);

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
fn classify_rows_with(model: &Model, parallel: Option<bool>) -> Result<Vec<Row>, SolverError> {
    let arena = model.arena();
    let vars = model.variables();
    let model_constraints = model.constraints();
    let constraints = model_constraints.algebraic();
    let arena_ref = &*arena;
    let vars_ref = &*vars;
    let classify_non_linear = |c: &oximo_core::Constraint| {
        detect_soc(arena_ref, vars_ref, c).map(Row::Soc).ok_or_else(|| SolverError::Nonlinear {
            location: format!("constraint {:?}", c.name),
            term: describe_nonlinear_term(arena_ref, c.lhs, &|v| var_name(vars_ref, v))
                .unwrap_or_else(|| "<nonlinear>".into()),
        })
    };

    // Preserve the very cheap LP fast path by extracting affine rows serially
    // and then fan out only a large set of quadratic/SOC candidates.
    // This means LP rows never take the Rayon path.
    let mut rows: Vec<Option<Row>> = (0..constraints.len()).map(|_| None).collect();
    let mut pending = Vec::new();
    for (index, constraint) in constraints.iter().enumerate() {
        match extract_linear(arena_ref, constraint.lhs).map(LinearTerms::into_owned) {
            Some(terms) => rows[index] = Some(Row::Lin(terms)),
            None => pending.push((index, constraint)),
        }
    }
    let use_parallel =
        parallel.unwrap_or(pending.len() >= PAR_ROW_THRESHOLD && rayon::current_num_threads() > 1);
    if use_parallel {
        let detected: Vec<Result<Row, SolverError>> =
            pending.par_iter().map(|(_, c)| classify_non_linear(c)).collect();
        for ((index, _), result) in pending.into_iter().zip(detected) {
            rows[index] = Some(result?);
        }
    } else {
        for (index, constraint) in pending {
            rows[index] = Some(classify_non_linear(constraint)?);
        }
    }
    Ok(rows.into_iter().map(Option::unwrap).collect())
}

/// Exact capacities for the final Clarabel row vectors and SOC metadata.
fn translation_capacities(
    model: &Model,
    rows: &[Row],
    explicit_forms: &[SocForm],
) -> (usize, usize, usize) {
    let vars = model.variables();
    let model_constraints = model.constraints();
    let constraints = model_constraints.algebraic();
    let is_fixed = |v: &Variable| v.lb.is_finite() && v.lb.total_cmp(&v.ub).is_eq();
    let mut row_count = 0;
    let mut nonzero_count = 0;

    for (constraint, row) in constraints.iter().zip(rows) {
        let Row::Lin(terms) = row else { continue };
        let multiplicity =
            usize::from(constraint.as_single().is_some()) + usize::from(constraint.is_range());
        row_count += multiplicity;
        nonzero_count += multiplicity * terms.coeffs.len();
    }
    for var in vars.iter() {
        let multiplicity = if is_fixed(var) {
            1
        } else {
            usize::from(var.ub.is_finite()) + usize::from(var.lb.is_finite())
        };
        row_count += multiplicity;
        nonzero_count += multiplicity;
    }

    let detected_forms = rows.iter().filter_map(|row| match row {
        Row::Soc(form) => Some(form),
        Row::Lin(_) => None,
    });
    let mut soc_count = 0;
    for form in explicit_forms.iter().chain(detected_forms) {
        soc_count += 1;
        row_count += 1 + form.terms.len();
        nonzero_count += form.bound.coeffs.len()
            + form.terms.iter().map(|term| term.coeffs.len()).sum::<usize>();
    }
    (row_count, nonzero_count, soc_count)
}

/// Lower the linear constraints and variable bounds into cone rows: the
/// `ZeroCone` block (equalities, then fixed variables) followed by the
/// `NonnegativeCone` block (inequalities/ranges, then finite bounds). Returns
/// the accumulated [`Rows`] and the two block sizes.
fn linear_rows(model: &Model, rows: &[Row], mut acc: Rows) -> (Rows, usize, usize) {
    let vars = model.variables();
    let model_constraints = model.constraints();
    let constraints = model_constraints.algebraic();
    let is_fixed = |v: &Variable| v.lb.is_finite() && v.lb.total_cmp(&v.ub).is_eq();
    // ZeroCone rows: equalities, then fixed variables.
    for (i, (con, row)) in constraints.iter().zip(rows).enumerate() {
        if let Row::Lin(lt) = row
            && let Some((Sense::Eq, rhs)) = con.as_single()
        {
            let id = ConstraintId(u32::try_from(i).expect("constraint count overflow"));
            acc.push(lt, 1.0, rhs - lt.constant);
            acc.set_last_dual(id, -1.0);
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
    rows: &[Row],
    explicit_forms: &[SocForm],
    acc: &mut Rows,
    soc_capacity: usize,
) -> (Vec<usize>, Vec<usize>, usize) {
    let detected_forms = rows.iter().filter_map(|r| match r {
        Row::Soc(f) => Some(f),
        Row::Lin(_) => None,
    });

    let n_explicit = explicit_forms.len();
    let mut soc_sizes: Vec<usize> = Vec::with_capacity(soc_capacity);
    let mut soc_block_starts: Vec<usize> = Vec::with_capacity(soc_capacity);
    for form in explicit_forms.iter().chain(detected_forms) {
        soc_block_starts.push(acc.b.len());
        acc.push(&form.bound, -1.0, form.bound.constant);
        for term in &form.terms {
            acc.push(term, -1.0, term.constant);
        }
        soc_sizes.push(1 + form.terms.len());
    }
    (soc_sizes, soc_block_starts, n_explicit)
}

fn explicit_soc_forms(
    arena_ref: &oximo_expr::ExprArena,
    socs: &[oximo_core::SocConstraint],
    parallel: Option<bool>,
) -> Result<Vec<SocForm>, SolverError> {
    let extract = |s: &oximo_core::SocConstraint| {
        explicit_soc_form(arena_ref, s).ok_or_else(|| {
            SolverError::Backend(format!(
                "SOC constraint '{}' has a member outside this model's arena",
                s.name
            ))
        })
    };
    let use_parallel = parallel.unwrap_or(false);
    let explicit_results: Vec<Result<SocForm, SolverError>> = if use_parallel {
        socs.par_iter().map(extract).collect()
    } else {
        socs.iter().map(extract).collect()
    };
    explicit_results.into_iter().collect()
}

/// Assemble the cone list in row order: zero cone, nonnegative cone, then one
/// second-order cone per SOC block.
fn build_cones(m_zero: usize, m_nonneg: usize, soc_sizes: &[usize]) -> Vec<SupportedConeT<f64>> {
    let mut cones = Vec::with_capacity(2 + soc_sizes.len());
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
    let native_status = solver.solution.status;
    let termination = map_status(native_status);
    let has_point = status_has_point(native_status);

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
    let (best_bound, gap) =
        mapped_objective_bound(meta, solver.solution.obj_val, solver.solution.obj_val_dual);

    SolverResult {
        termination,
        primal_status,
        dual_status: if has_point { DualStatus::FeasiblePoint } else { DualStatus::NoSolution },
        solutions,
        dual,
        soc_dual,
        reduced_costs: FxHashMap::default(),
        best_bound,
        gap,
        solve_time: elapsed,
        iterations: u64::from(solver.info.iterations),
        node_count: None,
        raw_status: Some(format!("{native_status:?}").into()),
        raw_log: None,
        solver_name: Some(crate::NAME.into()),
        solver_version: None,
    }
}

/// Map Clarabel's primal and dual objectives back to the model's objective
/// convention and compute their relative difference when both are finite.
fn mapped_objective_bound(meta: &Meta, primal: f64, dual: f64) -> (Option<f64>, Option<f64>) {
    let mapped_primal = meta.sign * primal + meta.obj_constant;
    let mapped_dual = meta.sign * dual + meta.obj_constant;
    if !mapped_primal.is_finite() || !mapped_dual.is_finite() {
        return (None, None);
    }

    let relative_gap =
        (mapped_primal - mapped_dual).abs() / (mapped_primal.abs().max(mapped_dual.abs()) + 1e-10);
    let gap = relative_gap.is_finite().then_some(relative_gap);
    (Some(mapped_dual), gap)
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
        SolverStatus::AlmostSolved => TerminationStatus::Feasible,
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

fn status_has_point(status: SolverStatus) -> bool {
    matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved)
}

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use oximo_core::constraint::Relate;

    use super::*;

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const ROW_THRESHOLD: usize = PAR_ROW_THRESHOLD;
    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const SOC_THRESHOLD: usize = 1_024;

    pub fn row_model(rows: usize, soc: bool) -> Model {
        let model = Model::new("clarabel_row_bench");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        let t = model.__var("t").lb(0.0).build();
        model.__minimize(t);
        for i in 0..rows {
            let lhs = if soc { x.powi(2) + y.powi(2) - t.powi(2) } else { x + 2.0 * y - t };
            model.__add_constraint_auto(lhs.le(if soc { 0.0 } else { i as f64 + 10.0 }));
        }
        model
    }

    pub fn qp_model(rows: usize) -> Model {
        let model = Model::new("clarabel_qp_bench");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        model.__minimize(x.powi(2) + y.powi(2));
        for i in 0..rows {
            model.__add_constraint_auto((x + 2.0 * y).le(i as f64 + 10.0));
        }
        model
    }

    pub fn explicit_soc_model(count: usize) -> Model {
        let model = Model::new("clarabel_explicit_soc_bench");
        let x = model.__var("x").build();
        let y = model.__var("y").build();
        let t = model.__var("t").lb(0.0).build();
        for i in 0..count {
            model.add_soc_constraint(format!("soc{i}"), [x, y], t);
        }
        model.__minimize(t);
        model
    }

    pub fn classify(model: &Model, parallel: bool) -> Result<usize, SolverError> {
        classify_rows_with(model, Some(parallel)).map(|rows| rows.len())
    }

    pub fn explicit_socs(model: &Model, parallel: bool) -> Result<usize, SolverError> {
        let arena = model.arena();
        let socs = model.soc_constraints();
        explicit_soc_forms(&arena, &socs, Some(parallel)).map(|forms| forms.len())
    }

    pub fn translate(model: &Model) -> Result<(usize, usize, usize), SolverError> {
        build_problem(model)
            .map(|problem| (problem.a_mat.nzval.len(), problem.b.len(), problem.cones.len()))
    }
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss)]
mod tests {
    use oximo_core::prelude::*;
    use oximo_solver::UniversalOptionsExt;

    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn only_solution_statuses_admit_primal_and_dual_points() {
        assert!(status_has_point(SolverStatus::Solved));
        assert!(status_has_point(SolverStatus::AlmostSolved));
        assert!(!status_has_point(SolverStatus::MaxIterations));
        assert!(!status_has_point(SolverStatus::MaxTime));
    }

    #[test]
    fn finite_dual_objective_maps_bound_and_gap() {
        let meta = Meta {
            sign: -1.0,
            obj_constant: 5.0,
            row_duals: Vec::new(),
            soc_block_starts: Vec::new(),
            n_explicit: 0,
        };

        let (bound, gap) = mapped_objective_bound(&meta, -10.0, -8.0);
        assert_eq!(bound, Some(13.0));
        assert!(close(gap.unwrap(), 2.0 / 15.0, 1e-12));
    }

    #[test]
    fn nonfinite_objective_leaves_bound_and_gap_unset() {
        let meta = Meta {
            sign: 1.0,
            obj_constant: 0.0,
            row_duals: Vec::new(),
            soc_block_starts: Vec::new(),
            n_explicit: 0,
        };

        for (primal, dual) in
            [(f64::NAN, 1.0), (1.0, f64::NAN), (f64::INFINITY, 1.0), (1.0, f64::NEG_INFINITY)]
        {
            assert_eq!(mapped_objective_bound(&meta, primal, dual), (None, None));
        }
    }

    #[test]
    fn forced_serial_and_parallel_translation_are_identical() {
        let m = Model::new("ordered");
        variable!(m, x);
        variable!(m, y);
        variable!(m, t >= 0.0);
        for i in 0..PAR_ROW_THRESHOLD + 3 {
            if i % 2 == 0 {
                m.__add_constraint_auto((x + (i as f64 + 1.0) * y).le(i as f64 + 10.0));
            } else {
                m.__add_constraint_auto((x.powi(2) + y.powi(2) - t.powi(2)).le(0.0));
            }
        }
        for i in 0..PAR_ROW_THRESHOLD + 3 {
            m.add_soc_constraint(format!("soc{i}"), [x, y], t);
        }
        objective!(m, Min, t);

        let serial = build_problem_with(&m, Some(false)).unwrap();
        let parallel = build_problem_with(&m, Some(true)).unwrap();
        let automatic = build_problem(&m).unwrap();
        assert_eq!(serial.p_mat.colptr, parallel.p_mat.colptr);
        assert_eq!(serial.p_mat.rowval, parallel.p_mat.rowval);
        assert_eq!(serial.p_mat.nzval, parallel.p_mat.nzval);
        assert_eq!(serial.q, parallel.q);
        assert_eq!(serial.a_mat.colptr, parallel.a_mat.colptr);
        assert_eq!(serial.a_mat.rowval, parallel.a_mat.rowval);
        assert_eq!(serial.a_mat.nzval, parallel.a_mat.nzval);
        assert_eq!(serial.b, parallel.b);
        assert_eq!(serial.cones, parallel.cones);
        assert_eq!(serial.meta.row_duals, parallel.meta.row_duals);
        assert_eq!(serial.meta.soc_block_starts, parallel.meta.soc_block_starts);
        assert_eq!(serial.meta.n_explicit, parallel.meta.n_explicit);
        assert_eq!(serial.a_mat.colptr, automatic.a_mat.colptr);
        assert_eq!(serial.a_mat.rowval, automatic.a_mat.rowval);
        assert_eq!(serial.a_mat.nzval, automatic.a_mat.nzval);
        assert_eq!(serial.b, automatic.b);
        assert_eq!(serial.cones, automatic.cones);
        assert_eq!(serial.meta.row_duals, automatic.meta.row_duals);
    }

    #[test]
    fn automatic_large_detected_soc_translation_matches_forced_paths() {
        let m = Model::new("automatic_detected_soc");
        variable!(m, x);
        variable!(m, y);
        variable!(m, t >= 0.0);
        objective!(m, Min, t);
        for _ in 0..PAR_ROW_THRESHOLD + 3 {
            m.__add_constraint_auto((x.powi(2) + y.powi(2) - t.powi(2)).le(0.0));
        }

        let serial = build_problem_with(&m, Some(false)).unwrap();
        let automatic = build_problem(&m).unwrap();
        let parallel = build_problem_with(&m, Some(true)).unwrap();
        assert_eq!(serial.a_mat.colptr, automatic.a_mat.colptr);
        assert_eq!(serial.a_mat.rowval, automatic.a_mat.rowval);
        assert_eq!(serial.a_mat.nzval, automatic.a_mat.nzval);
        assert_eq!(serial.b, automatic.b);
        assert_eq!(serial.cones, automatic.cones);
        assert_eq!(automatic.a_mat.colptr, parallel.a_mat.colptr);
        assert_eq!(automatic.a_mat.rowval, parallel.a_mat.rowval);
        assert_eq!(automatic.a_mat.nzval, parallel.a_mat.nzval);
        assert_eq!(automatic.b, parallel.b);
        assert_eq!(automatic.cones, parallel.cones);
    }

    #[test]
    fn parallel_row_classification_keeps_first_error_order() {
        let m = Model::new("errors");
        variable!(m, x);
        variable!(m, y);
        variable!(m, z);
        m.__add_constraint("first", (x * y * z).le(1.0));
        m.__add_constraint("second", x.exp().le(2.0));
        objective!(m, Min, x);
        let serial = classify_rows_with(&m, Some(false)).unwrap_err();
        let automatic = classify_rows_with(&m, None).unwrap_err();
        let parallel = classify_rows_with(&m, Some(true)).unwrap_err();
        assert_eq!(serial.to_string(), parallel.to_string());
        assert_eq!(serial.to_string(), automatic.to_string());
        assert!(serial.to_string().contains("first"));
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
        assert!(matches!(
            err,
            SolverError::UnsupportedKind {
                kind: ModelKind::MISOCP,
                supported: crate::SUPPORTED_KINDS
            }
        ));
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
        assert!(matches!(
            err,
            SolverError::UnsupportedKind {
                kind: ModelKind::QCP,
                supported: crate::SUPPORTED_KINDS
            }
        ));
    }

    #[test]
    fn nlp_is_unsupported() {
        let m = Model::new("nlp");
        variable!(m, x >= 0.1);
        objective!(m, Min, x.sin());
        let err = solve(&m, &ClarabelOptions::default()).unwrap_err();
        assert!(matches!(
            err,
            SolverError::UnsupportedKind {
                kind: ModelKind::NLP,
                supported: crate::SUPPORTED_KINDS
            }
        ));
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
        if res.termination == TerminationStatus::IterationLimit {
            assert!(!res.has_solution());
            assert_eq!(res.dual_status, DualStatus::NoSolution);
        }
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
