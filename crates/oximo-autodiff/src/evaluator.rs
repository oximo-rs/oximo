//! [`NlpEvaluator`]: derivative oracle for a continuous model, built from a [`Model`] and
//! classifying every function as linear, quadratic, or nonlinear. Linear and
//! quadratic functions use closed forms for values and derivatives, only
//! nonlinear functions go through the Enzyme-differentiated tape interpreter.
//!
//! Built once from a [`Model`], owns compiled tapes, sparsity patterns, and
//! scratch buffers, so evaluation never touches the model's `RefCell`s again.

use std::cell::RefCell;

use oximo_core::Model;
use oximo_expr::ExprId;
use rayon::prelude::*;

use crate::enzyme::{tape_gradient, tape_hvp};
use crate::error::AutodiffError;
use crate::slot::{
    FunctionSlot, SlotKind, linear_gradient_add, linear_value, quadratic_gradient_add,
    quadratic_value,
};
use crate::sparsity::{hessian_lagrangian_structure, jacobian_structure, star_hessian_coloring};
use crate::tape::Tape;

// Above these counts a derivative call fans its independent units of work out
// across rayon's thread pool; below, it stays on the single reusable scratch
// buffer (the zero-allocation fast path). Counts are a coarse proxy for work:
// a Hessian seed is a whole forward-over-reverse tape pass, and a Jacobian row
// / nonlinear constraint value is a reverse pass, so linear/quadratic slots
// make the constraint thresholds conservative.
// TODO: benchmark and tune these thresholds (ideally weighting by tape size and
// the nonlinear-slot count rather than a raw element count).
const PAR_SEED_THRESHOLD: usize = 16;
const PAR_CONSTRAINT_THRESHOLD: usize = 64;

/// Where each Lagrangian-tape multiplier slot takes its weight from.
#[derive(Copy, Clone, Debug)]
enum NlSource {
    Objective,
    Constraint(usize),
}

/// One compressed Hessian-vector product. The direction seeds every column in
/// a structurally orthogonal group, and `fills` scatters the result.
#[derive(Debug)]
struct Seed {
    /// Columns receiving 1.0 in the HVP direction.
    cols: Vec<usize>,
    /// `(position-in-vals, dense row)` of the nonlinear-pattern entries this
    /// HVP fills. Built from the nonlinear-only pattern.
    fills: Vec<(usize, usize)>,
}

/// Per-worker-thread scratch for the parallel derivative paths, allocated once
/// per rayon worker by `map_init`. The serial paths keep using the shared
/// [`Scratch`] behind the `RefCell`. Only the above-threshold parallel branches
/// allocate these, so we keep the zero-allocation fast path for small problems.
struct ParScratch {
    regs: Vec<f64>,
    dregs: Vec<f64>,
    regs_t: Vec<f64>,
    dregs_t: Vec<f64>,
    dir: Vec<f64>,
    grad: Vec<f64>,
    hv: Vec<f64>,
}

impl ParScratch {
    fn new(max_regs: usize, n_vars: usize) -> Self {
        Self {
            regs: vec![0.0; max_regs],
            dregs: vec![0.0; max_regs],
            regs_t: vec![0.0; max_regs],
            dregs_t: vec![0.0; max_regs],
            dir: vec![0.0; n_vars],
            grad: vec![0.0; n_vars],
            hv: vec![0.0; n_vars],
        }
    }
}

#[derive(Debug, Default)]
struct Scratch {
    /// Dense gradient buffer, length `n_vars`.
    grad: Vec<f64>,
    /// Dense HVP seed direction, length `n_vars`.
    dir: Vec<f64>,
    /// Dense `H·dir` output, length `n_vars`.
    hv: Vec<f64>,
    /// Lagrangian-tape multipliers.
    mults: Vec<f64>,
    /// Tape register files and their reverse/tangent shadows, sized for the
    /// largest tape.
    regs: Vec<f64>,
    dregs: Vec<f64>,
    regs_t: Vec<f64>,
    dregs_t: Vec<f64>,
}

/// Derivative oracle for a continuous model.
/// Objective/constraint values, gradients, sparse Jacobian,
/// and the sparse lower-triangle Hessian of the Lagrangian
/// `sigma * laplacian(f) + sum_i(lambda_i * laplacian(g_i))`.
///
/// All nonlinear Hessian contributions come from a single weighted "Lagrangian
/// tape", one Hessian-vector product per structurally orthogonal column group
/// of the exact nonlinear pattern.
///
/// Variable domains are ignored here, we keep integrality in the solver.
#[derive(Debug)]
pub struct NlpEvaluator {
    n_vars: usize,
    params: Vec<f64>,
    /// `None` for a feasibility problem, whose objective is the constant zero.
    objective_expr: Option<ExprId>,
    constraint_exprs: Vec<ExprId>,
    objective: FunctionSlot,
    constraints: Vec<FunctionSlot>,
    /// Weighted tape over the nonlinear functions (empty if there are none).
    lagrangian: Tape,
    /// Weight source for each Lagrangian multiplier slot.
    nl_sources: Vec<NlSource>,
    jac_structure: Vec<(usize, usize)>,
    hess_structure: Vec<(usize, usize)>,
    /// Position in `hess_structure` of each entry of the objective's quadratic
    /// Hessian (in `QuadraticTerms::hessian` order). Empty unless the objective
    /// is quadratic.
    obj_hess_pos: Vec<usize>,
    /// Same as `obj_hess_pos`, per constraint.
    con_hess_pos: Vec<Vec<usize>>,
    /// Compressed HVP seeds covering the nonlinear Hessian pattern.
    seeds: Vec<Seed>,
    /// Largest tape register count, sizing both [`Scratch`] and [`ParScratch`].
    max_regs: usize,
    scratch: RefCell<Scratch>,
}

impl NlpEvaluator {
    /// Build the oracle from `model`, classifying every function and
    /// compiling tapes for the nonlinear ones.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::NoObjective`] if the model declares neither an
    /// objective nor a feasibility problem.
    pub fn new(model: &Model) -> Result<Self, AutodiffError> {
        let arena = model.arena();
        let n_vars = model.variables().len();
        model.ensure_objective_declared().map_err(|_| AutodiffError::NoObjective)?;
        let objective_expr: Option<ExprId> = model.objective().as_ref().map(|o| o.expr);
        let constraint_exprs: Vec<ExprId> = model.constraints().iter().map(|c| c.lhs).collect();

        let objective = match objective_expr {
            Some(e) => FunctionSlot::classify(&arena, e),
            None => FunctionSlot::zero(),
        };
        let constraints: Vec<FunctionSlot> =
            constraint_exprs.iter().map(|&e| FunctionSlot::classify(&arena, e)).collect();

        // Lagrangian tape over the nonlinear functions only.
        let mut nl_sources = Vec::new();
        let mut nl_exprs = Vec::new();
        if let Some(e) = objective_expr {
            if objective.is_nonlinear() {
                nl_sources.push(NlSource::Objective);
                nl_exprs.push(e);
            }
        }
        for (i, slot) in constraints.iter().enumerate() {
            if slot.is_nonlinear() {
                nl_sources.push(NlSource::Constraint(i));
                nl_exprs.push(constraint_exprs[i]);
            }
        }
        let lagrangian = Tape::compile_weighted(&arena, &nl_exprs);

        let params = crate::tape::params_snapshot(&arena);
        drop(arena);

        let jac_structure = jacobian_structure(&constraints);
        let hess_structure =
            hessian_lagrangian_structure(std::iter::once(&objective).chain(&constraints));

        let obj_hess_pos = quad_scatter_positions(&objective, &hess_structure);
        let con_hess_pos =
            constraints.iter().map(|s| quad_scatter_positions(s, &hess_structure)).collect();

        let seeds = build_seeds(&objective, &constraints, &hess_structure);

        let max_regs = std::iter::once(&objective)
            .chain(&constraints)
            .filter_map(|s| match &s.kind {
                SlotKind::Nonlinear(t) => Some(t.n_regs()),
                _ => None,
            })
            .chain(std::iter::once(lagrangian.n_regs()))
            .max()
            .unwrap_or(0);

        let scratch = RefCell::new(Scratch {
            grad: vec![0.0; n_vars],
            dir: vec![0.0; n_vars],
            hv: vec![0.0; n_vars],
            mults: vec![0.0; nl_sources.len()],
            regs: vec![0.0; max_regs],
            dregs: vec![0.0; max_regs],
            regs_t: vec![0.0; max_regs],
            dregs_t: vec![0.0; max_regs],
        });

        Ok(Self {
            n_vars,
            params,
            objective_expr,
            constraint_exprs,
            objective,
            constraints,
            lagrangian,
            nl_sources,
            jac_structure,
            hess_structure,
            obj_hess_pos,
            con_hess_pos,
            seeds,
            max_regs,
            scratch,
        })
    }

    /// Re-snapshot parameter values (and the parameter-dependent linear /
    /// quadratic coefficients) after `set_param` on the model. Tapes are not
    /// recompiled and sparsity structures are kept. A coefficient that was
    /// exactly zero at construction and became nonzero may fall outside the
    /// original pattern, so build the evaluator with representative parameter
    /// values.
    ///
    /// Only the cached quadratic scatter positions are rebuilt, `jac_structure`,
    /// `hess_structure`, and `seeds` are reused as-is. Nonlinear tapes (and
    /// therefore the nonlinear Hessian pattern the seeds cover) are
    /// parameter-independent, and the representative-parameter assumption
    /// keeps the linear/quadratic patterns fixed too. If that assumption is
    /// violated so a new quadratic entry appears outside the pattern, the
    /// `expect` in `quad_scatter_positions` panics. Use [`Self::try_refresh`]
    /// when the caller cannot guarantee representative parameters.
    pub fn refresh_params(&mut self, model: &Model) {
        let arena = model.arena();
        self.params = crate::tape::params_snapshot(&arena);
        if let Some(e) = self.objective_expr {
            self.objective = self.objective.reclassify(&arena, e);
        }
        for (slot, &expr) in self.constraints.iter_mut().zip(&self.constraint_exprs) {
            *slot = slot.reclassify(&arena, expr);
        }
        // Reclassification can drop a zeroed quadratic entry or flip a slot's
        // kind, so realign the cached scatter positions with the new
        // `QuadraticTerms`. The Hessian pattern is assumed unchanged (see the
        // representative-parameter note above), so `hess_structure` still
        // contains every entry.
        self.rebuild_quad_scatter();
    }

    /// Try to reuse this evaluator for `model` after a `set_param`/bound
    /// change, validating that the sparsity structure is unchanged.
    ///
    /// Returns `true` when the model has the same variables, the same objective
    /// and constraint expressions and, after re-extracting linear/quadratic
    /// coefficients at the new parameter values, the same Jacobian and
    /// Lagrangian-Hessian patterns. The evaluator is refreshed in place
    /// and is ready to solve.
    /// Returns `false` when anything structural changed, leaving the evaluator
    /// unmodified so the caller can rebuild with [`Self::new`].
    /// Enables a resident/persistent solver to skip retaping across
    /// a parametric sweep while staying correct.
    pub fn try_refresh(&mut self, model: &Model) -> bool {
        let arena = model.arena();
        if model.variables().len() != self.n_vars
            || model.objective().as_ref().map(|o| o.expr) != self.objective_expr
        {
            return false;
        }
        let con_exprs: Vec<ExprId> = model.constraints().iter().map(|c| c.lhs).collect();
        if con_exprs != self.constraint_exprs {
            return false;
        }

        let objective = match self.objective_expr {
            Some(e) => self.objective.reclassify(&arena, e),
            None => FunctionSlot::zero(),
        };
        let constraints: Vec<FunctionSlot> = self
            .constraints
            .iter()
            .zip(&con_exprs)
            .map(|(s, &e)| s.reclassify(&arena, e))
            .collect();

        // A parameter can move a coefficient across zero and change the pattern.
        // Rebuild rather than evaluate against a stale structure.
        let jac = jacobian_structure(&constraints);
        let hess = hessian_lagrangian_structure(std::iter::once(&objective).chain(&constraints));
        if jac != self.jac_structure || hess != self.hess_structure {
            return false;
        }

        self.params = crate::tape::params_snapshot(&arena);
        self.objective = objective;
        self.constraints = constraints;
        self.rebuild_quad_scatter();
        true
    }

    /// Recompute the cached quadratic-Hessian scatter positions from the current
    /// slots against `hess_structure`. Called after any reclassification.
    fn rebuild_quad_scatter(&mut self) {
        self.obj_hess_pos = quad_scatter_positions(&self.objective, &self.hess_structure);
        self.con_hess_pos = self
            .constraints
            .iter()
            .map(|s| quad_scatter_positions(s, &self.hess_structure))
            .collect();
    }

    pub fn num_variables(&self) -> usize {
        self.n_vars
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Number of Hessian-vector products performed per
    /// [`Self::eval_hessian_lagrangian`] call (compressed seed count).
    pub fn num_hessian_seeds(&self) -> usize {
        self.seeds.len()
    }

    /// Objective value at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` differs from [`Self::num_variables`].
    pub fn eval_objective(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.n_vars, "point dimension");
        let scratch = &mut *self.scratch.borrow_mut();
        slot_value(&self.objective, x, &self.params, &mut scratch.regs)
    }

    /// Dense objective gradient at `x` into `grad` (overwritten).
    ///
    /// # Panics
    ///
    /// Panics if `x` or `grad` have the wrong length.
    pub fn eval_objective_gradient(&self, x: &[f64], grad: &mut [f64]) {
        assert_eq!(x.len(), self.n_vars, "point dimension");
        assert_eq!(grad.len(), self.n_vars, "gradient dimension");
        let scratch = &mut *self.scratch.borrow_mut();
        grad.fill(0.0);
        slot_gradient_into(
            &self.objective,
            x,
            &self.params,
            &mut scratch.regs,
            &mut scratch.dregs,
            grad,
        );
    }

    /// Constraint LHS values at `x` into `g` (overwritten), in declaration
    /// order.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `g` have the wrong length.
    pub fn eval_constraint(&self, x: &[f64], g: &mut [f64]) {
        assert_eq!(x.len(), self.n_vars, "point dimension");
        assert_eq!(g.len(), self.constraints.len(), "constraint dimension");
        if self.constraints.len() < PAR_CONSTRAINT_THRESHOLD {
            self.eval_constraint_serial(x, g);
        } else {
            self.eval_constraint_parallel(x, g);
        }
    }

    /// Serial constraint values.
    fn eval_constraint_serial(&self, x: &[f64], g: &mut [f64]) {
        let scratch = &mut *self.scratch.borrow_mut();
        for (out, slot) in g.iter_mut().zip(&self.constraints) {
            *out = slot_value(slot, x, &self.params, &mut scratch.regs);
        }
    }

    /// Parallel constraint values.
    fn eval_constraint_parallel(&self, x: &[f64], g: &mut [f64]) {
        let (params, max_regs) = (self.params.as_slice(), self.max_regs);
        let values: Vec<f64> = self
            .constraints
            .par_iter()
            .map_init(|| vec![0.0; max_regs], |regs, slot| slot_value(slot, x, params, regs))
            .collect();
        g.copy_from_slice(&values);
    }

    /// `(constraint, variable)` Jacobian pattern, row-major.
    /// Rows are sorted by variable.
    pub fn jacobian_structure(&self) -> &[(usize, usize)] {
        &self.jac_structure
    }

    /// Jacobian values at `x` into `vals`, aligned with
    /// [`Self::jacobian_structure`].
    ///
    /// # Panics
    ///
    /// Panics if `x` or `vals` have the wrong length.
    pub fn eval_constraint_jacobian(&self, x: &[f64], vals: &mut [f64]) {
        assert_eq!(x.len(), self.n_vars, "point dimension");
        assert_eq!(vals.len(), self.jac_structure.len(), "jacobian nnz");
        if self.constraints.len() < PAR_CONSTRAINT_THRESHOLD {
            self.eval_constraint_jacobian_serial(x, vals);
        } else {
            self.eval_constraint_jacobian_parallel(x, vals);
        }
    }

    /// Serial Jacobian.
    fn eval_constraint_jacobian_serial(&self, x: &[f64], vals: &mut [f64]) {
        let scratch = &mut *self.scratch.borrow_mut();
        let mut out = 0;
        for slot in &self.constraints {
            // Linear/Quadratic add into grad, so pre-zero just their support.
            // Nonlinear's tape_gradient overwrites the whole buffer.
            for &v in &slot.support {
                scratch.grad[v as usize] = 0.0;
            }
            slot_gradient_into(
                slot,
                x,
                &self.params,
                &mut scratch.regs,
                &mut scratch.dregs,
                &mut scratch.grad,
            );
            for &v in &slot.support {
                vals[out] = scratch.grad[v as usize];
                out += 1;
            }
        }
        debug_assert_eq!(out, vals.len());
    }

    // TODO: Can we write disjoint slices of `vals` from worker threads if we
    // add unsafe?

    /// Parallel Jacobian
    /// Each row's gradient is independent. Compute each into a er-thread
    /// scratch, collect the support values, then concatenate in row order.
    fn eval_constraint_jacobian_parallel(&self, x: &[f64], vals: &mut [f64]) {
        let (params, max_regs, n_vars) = (self.params.as_slice(), self.max_regs, self.n_vars);
        let rows: Vec<Vec<f64>> = self
            .constraints
            .par_iter()
            .map_init(
                || ParScratch::new(max_regs, n_vars),
                |sc, slot| {
                    for &v in &slot.support {
                        sc.grad[v as usize] = 0.0;
                    }
                    slot_gradient_into(slot, x, params, &mut sc.regs, &mut sc.dregs, &mut sc.grad);
                    slot.support.iter().map(|&v| sc.grad[v as usize]).collect()
                },
            )
            .collect();
        let mut out = 0;
        for row in &rows {
            vals[out..out + row.len()].copy_from_slice(row);
            out += row.len();
        }
        debug_assert_eq!(out, vals.len());
    }

    /// Lower-triangle (`row >= col`) Hessian-of-the-Lagrangian pattern,
    /// sorted and deduplicated.
    pub fn hessian_lagrangian_structure(&self) -> &[(usize, usize)] {
        &self.hess_structure
    }

    /// Hessian of `obj_factor * f + sum(lambda[i] * g_i)` at `x` into `vals`,
    /// aligned with [`Self::hessian_lagrangian_structure`].
    ///
    /// # Panics
    ///
    /// Panics if `x`, `lambda`, or `vals` have the wrong length.
    pub fn eval_hessian_lagrangian(
        &self,
        x: &[f64],
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    ) {
        assert_eq!(x.len(), self.n_vars, "point dimension");
        assert_eq!(lambda.len(), self.constraints.len(), "multiplier dimension");
        assert_eq!(vals.len(), self.hess_structure.len(), "hessian nnz");
        vals.fill(0.0);

        // Constant quadratic contributions, scaled by sigma/lambda. The
        // scatter positions are cached (see `obj_hess_pos`/`con_hess_pos`), so
        // this reads `h` live but does no per-call binary search.
        if let SlotKind::Quadratic(q) = &self.objective.kind {
            for (&pos, &(_, _, h)) in self.obj_hess_pos.iter().zip(&q.hessian) {
                vals[pos] += obj_factor * h;
            }
        }
        for ((slot, positions), &mult) in
            self.constraints.iter().zip(&self.con_hess_pos).zip(lambda)
        {
            if let SlotKind::Quadratic(q) = &slot.kind {
                for (&pos, &(_, _, h)) in positions.iter().zip(&q.hessian) {
                    vals[pos] += mult * h;
                }
            }
        }

        if self.seeds.is_empty() {
            return;
        }

        if self.seeds.len() < PAR_SEED_THRESHOLD {
            self.hessian_seeds_serial(x, obj_factor, lambda, vals);
        } else {
            self.hessian_seeds_parallel(x, obj_factor, lambda, vals);
        }
    }

    /// Serial nonlinear Hessian contributions.
    /// One HVP per seed on the shared scratch, each result
    /// scattered (added) into `vals`, which must already
    /// hold the closed-form quadratic contributions.
    fn hessian_seeds_serial(&self, x: &[f64], obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let scratch = &mut *self.scratch.borrow_mut();
        for (k, source) in self.nl_sources.iter().enumerate() {
            scratch.mults[k] = mult_of(source, obj_factor, lambda);
        }
        for seed in &self.seeds {
            scratch.dir.fill(0.0);
            for &col in &seed.cols {
                scratch.dir[col] = 1.0;
            }
            tape_hvp(
                &self.lagrangian,
                x,
                &scratch.dir,
                &self.params,
                &scratch.mults,
                &mut scratch.regs,
                &mut scratch.regs_t,
                &mut scratch.dregs,
                &mut scratch.dregs_t,
                &mut scratch.grad,
                &mut scratch.hv,
            );
            for &(pos, row) in &seed.fills {
                vals[pos] += scratch.hv[row];
            }
        }
    }

    /// Parallel nonlinear Hessian contributions.
    /// Each seed is an independent forward-over-reverse HVP over the whole
    /// Lagrangian tape, and seeds fill disjoint `vals` positions.
    /// Run them on the pool with per-thread scratch, returning each seed's
    /// `(pos, value)` contributions, then apply serially.
    /// `vals` must already hold the closed-form quadratic contributions.
    fn hessian_seeds_parallel(&self, x: &[f64], obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let mults: Vec<f64> =
            self.nl_sources.iter().map(|s| mult_of(s, obj_factor, lambda)).collect();
        let (lagrangian, params) = (&self.lagrangian, self.params.as_slice());
        let (max_regs, n_vars) = (self.max_regs, self.n_vars);
        let contributions: Vec<Vec<(usize, f64)>> = self
            .seeds
            .par_iter()
            .map_init(
                || ParScratch::new(max_regs, n_vars),
                |sc, seed| {
                    sc.dir.fill(0.0);
                    for &col in &seed.cols {
                        sc.dir[col] = 1.0;
                    }
                    tape_hvp(
                        lagrangian,
                        x,
                        &sc.dir,
                        params,
                        &mults,
                        &mut sc.regs,
                        &mut sc.regs_t,
                        &mut sc.dregs,
                        &mut sc.dregs_t,
                        &mut sc.grad,
                        &mut sc.hv,
                    );
                    seed.fills.iter().map(|&(pos, row)| (pos, sc.hv[row])).collect()
                },
            )
            .collect();
        for contribution in &contributions {
            for &(pos, v) in contribution {
                vals[pos] += v;
            }
        }
    }
}

/// Value of `slot` at `x`. Nonlinear slots use `regs` as tape scratch, the
/// closed-form kinds ignore it.
fn slot_value(slot: &FunctionSlot, x: &[f64], params: &[f64], regs: &mut [f64]) -> f64 {
    match &slot.kind {
        SlotKind::Linear(t) => linear_value(t, x),
        SlotKind::Quadratic(q) => quadratic_value(q, x),
        SlotKind::Nonlinear(tape) => tape.value(x, params, &[], regs),
    }
}

/// Write `slot`'s dense gradient at `x` into `grad`. Linear/Quadratic add
/// into `grad`.
/// Nonlinear overwrites the full dense buffer via `tape_gradient`.
/// `regs`/`dregs` are the tape scratch, passed as separate slices to
/// keep the caller's disjoint borrows of a `Scratch`/`ParScratch` valid.
fn slot_gradient_into(
    slot: &FunctionSlot,
    x: &[f64],
    params: &[f64],
    regs: &mut [f64],
    dregs: &mut [f64],
    grad: &mut [f64],
) {
    match &slot.kind {
        SlotKind::Linear(t) => linear_gradient_add(t, 1.0, grad),
        SlotKind::Quadratic(q) => quadratic_gradient_add(q, x, 1.0, grad),
        SlotKind::Nonlinear(tape) => {
            tape_gradient(tape, x, params, &[], regs, dregs, grad);
        }
    }
}

/// Weight of a Lagrangian multiplier slot.
fn mult_of(source: &NlSource, obj_factor: f64, lambda: &[f64]) -> f64 {
    match source {
        NlSource::Objective => obj_factor,
        NlSource::Constraint(i) => lambda[*i],
    }
}

/// Position in `hess` of each entry of `slot`'s quadratic Hessian, in
/// `QuadraticTerms::hessian` order, so `eval_hessian_lagrangian` can scatter the
/// constant quadratic terms without a per-call binary search. Empty for
/// non-quadratic slots.
fn quad_scatter_positions(slot: &FunctionSlot, hess: &[(usize, usize)]) -> Vec<usize> {
    match &slot.kind {
        SlotKind::Quadratic(q) => q
            .hessian
            .iter()
            .map(|&(r, c, _)| {
                hess.binary_search(&(r.index(), c.index()))
                    .expect("quadratic entry is in the pattern by construction")
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Compressed HVP seeds over the exact non-linear pattern only.
/// The quadratic entries in `hess_structure` are filled in
/// closed form and must neither constrain the grouping nor
/// receive `hv` values.
fn build_seeds(
    objective: &FunctionSlot,
    constraints: &[FunctionSlot],
    hess_structure: &[(usize, usize)],
) -> Vec<Seed> {
    let mut nl_pattern: Vec<(usize, usize)> = std::iter::once(objective)
        .chain(constraints)
        .filter(|s| s.is_nonlinear())
        .flat_map(|s| s.hess_pairs.iter().map(|&(r, c)| (r as usize, c as usize)))
        .collect();
    nl_pattern.sort_unstable();
    nl_pattern.dedup();

    // Star-color the nonlinear pattern.
    let coloring = star_hessian_coloring(&nl_pattern);
    let mut seeds: Vec<Seed> =
        coloring.groups.into_iter().map(|cols| Seed { cols, fills: Vec::new() }).collect();
    for (&(row, col), &(group, hv_row)) in nl_pattern.iter().zip(&coloring.recover) {
        let pos = hess_structure
            .binary_search(&(row, col))
            .expect("nonlinear entry is in the merged pattern by construction");
        seeds[group].fills.push((pos, hv_row));
    }
    seeds
}

#[cfg(test)]
mod tests {
    //! Serial vs parallel equivalence of the threshold-gated derivative paths.
    //!
    //! `eval_constraint`, `eval_constraint_jacobian`, and `eval_hessian_lagrangian`
    //! each pick a serial or parallel implementation by problem size. These tests
    //! call both implementations directly and assert they agree bit-for-bit.
    use super::*;
    use oximo_core::Model;
    use oximo_core::prelude::*;

    fn mixed_model() -> Model {
        let m = Model::new("equiv");
        variable!(m, -3.0 <= x <= 3.0);
        variable!(m, -3.0 <= y <= 3.0);
        variable!(m, -3.0 <= z <= 3.0);
        objective!(m, Min, x.sin() * y + z.powi(3) + x * z);
        constraint!(m, cq, x.powi(2) + y.powi(2) <= 10.0);
        constraint!(m, cn, x * y.exp() + z.sin() <= 5.0);
        constraint!(m, cl, 2.0 * x + 3.0 * z <= 7.0);
        m
    }

    const POINT: [f64; 3] = [0.7, -1.1, 0.4];

    #[test]
    fn constraint_values_serial_and_parallel_agree() {
        let ev = NlpEvaluator::new(&mixed_model()).unwrap();
        let mut serial = vec![0.0; ev.num_constraints()];
        let mut parallel = vec![0.0; ev.num_constraints()];
        ev.eval_constraint_serial(&POINT, &mut serial);
        ev.eval_constraint_parallel(&POINT, &mut parallel);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn constraint_jacobian_serial_and_parallel_agree() {
        let ev = NlpEvaluator::new(&mixed_model()).unwrap();
        let nnz = ev.jacobian_structure().len();
        let mut serial = vec![0.0; nnz];
        let mut parallel = vec![0.0; nnz];
        ev.eval_constraint_jacobian_serial(&POINT, &mut serial);
        ev.eval_constraint_jacobian_parallel(&POINT, &mut parallel);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn hessian_seeds_serial_and_parallel_agree() {
        let ev = NlpEvaluator::new(&mixed_model()).unwrap();
        let sigma = 0.9;
        let lambda = [1.2, -0.7, 0.3];
        let nnz = ev.hessian_lagrangian_structure().len();
        let mut serial = vec![0.0; nnz];
        let mut parallel = vec![0.0; nnz];
        ev.hessian_seeds_serial(&POINT, sigma, &lambda, &mut serial);
        ev.hessian_seeds_parallel(&POINT, sigma, &lambda, &mut parallel);
        assert_eq!(serial, parallel);
    }
}
