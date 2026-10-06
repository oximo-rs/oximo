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

use crate::enzyme::{tape_gradient, tape_hvp};
use crate::error::AutodiffError;
use crate::slot::{
    FunctionSlot, SlotKind, linear_gradient_add, linear_value, quadratic_gradient_add,
    quadratic_value,
};
use crate::sparsity::{
    SparsityWorkspace, hessian_lagrangian_patterns, hessian_lagrangian_structure,
    jacobian_structure, star_hessian_coloring,
};
use crate::tape::Tape;

// Minimum estimated work per task, calibrated with repeated native callback
// measurements (Enzyme, fat LTO).
const PAR_VALUE_WORK: usize = 2_048;
const PAR_JACOBIAN_WORK: usize = 1_024;
const PAR_HESSIAN_WORK: usize = 1_024;

/// Affine subexpressions are packed into one instruction, whose coefficient
/// loop still contributes work proportional to its number of terms.
fn tape_work(tape: &Tape) -> usize {
    tape.n_regs() + tape.parts().4.len()
}

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

/// Scratch owned by a parallel task, retained across calls.
/// Task buffers are borrowed exclusively before entering Rayon,
/// so work stealing needs neither thread-local state nor locks.
#[derive(Debug, Default)]
struct ParScratch {
    regs: Vec<f64>,
    dregs: Vec<f64>,
    regs_t: Vec<f64>,
    dregs_t: Vec<f64>,
    basis: Vec<f64>,
    dir: Vec<f64>,
    grad: Vec<f64>,
    hv: Vec<f64>,
}

impl ParScratch {
    fn prepare(&mut self, n_regs: usize, n_vars: usize, gradient: bool, hessian: bool) {
        grow(&mut self.regs, n_regs);
        if gradient || hessian {
            grow(&mut self.dregs, n_regs);
            grow(&mut self.grad, n_vars);
            #[cfg(target_arch = "wasm32")]
            grow(&mut self.basis, n_vars);
        }
        if hessian {
            grow(&mut self.regs_t, n_regs);
            grow(&mut self.dregs_t, n_regs);
            grow(&mut self.dir, n_vars);
            grow(&mut self.hv, n_vars);
        }
    }
}

fn grow(buffer: &mut Vec<f64>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
}

#[derive(Debug, Default)]
struct ParallelWorkspace {
    workers: Vec<ParScratch>,
    hessian_values: Vec<f64>,
}

impl ParallelWorkspace {
    fn prepare(
        &mut self,
        count: usize,
        n_regs: usize,
        n_vars: usize,
        gradient: bool,
        hessian: bool,
    ) {
        self.workers.resize_with(self.workers.len().max(count), ParScratch::default);
        for worker in &mut self.workers[..count] {
            worker.prepare(n_regs, n_vars, gradient, hessian);
        }
    }
}

/// Prefix sums of estimated work and output widths.
#[derive(Debug)]
struct WorkPlan {
    work: Vec<usize>,
    offsets: Vec<usize>,
}

impl WorkPlan {
    fn new(units: impl Iterator<Item = (usize, usize)>) -> Self {
        let mut work = vec![0_usize];
        let mut offsets = vec![0];
        for (cost, width) in units {
            work.push(work.last().unwrap().saturating_add(cost.max(1)));
            offsets.push(offsets.last().unwrap() + width);
        }
        Self { work, offsets }
    }

    fn workers(&self, min_work: usize) -> usize {
        rayon::current_num_threads()
            .min(self.work.len() - 1)
            .min(self.work.last().unwrap() / min_work)
    }

    fn parallel_workers(&self, min_work: usize) -> usize {
        self.workers(min_work).max(2).min(rayon::current_num_threads()).min(self.work.len() - 1)
    }

    /// Choose between the valid boundaries bracketing the proportional target,
    /// minimizing the larger estimated load per worker on either side.
    fn split(&self, start: usize, end: usize, left_count: usize, right_count: usize) -> usize {
        let total = (self.work[end] - self.work[start]) as u128;
        let count = (left_count + right_count) as u128;
        let target = total * left_count as u128;
        let upper = self.work[start..end]
            .partition_point(|&work| (work - self.work[start]) as u128 * count < target)
            + start;
        // Every leaf needs a unit, even when one tape dominates the work.
        let first = start + left_count;
        let last = end - right_count;
        let upper = upper.clamp(first, last);
        let lower = upper.saturating_sub(1).max(first);
        // Multiplying by the common denominator to avoid floating-point rounding
        // and usize overflow when comparing the two groups' work per worker.
        let load = |mid| {
            let left = (self.work[mid] - self.work[start]) as u128 * right_count as u128;
            let right = (self.work[end] - self.work[mid]) as u128 * left_count as u128;
            left.max(right)
        };
        if load(lower) <= load(upper) { lower } else { upper }
    }
}

#[derive(Debug)]
struct ConstraintWork {
    values: WorkPlan,
    jacobian: WorkPlan,
    max_regs: usize,
}

impl ConstraintWork {
    fn new(slots: &[FunctionSlot], n_vars: usize) -> Self {
        let cost = |slot: &FunctionSlot| match &slot.kind {
            SlotKind::Linear(t) => (t.coeffs.len() + 1, t.coeffs.len() + slot.support.len()),
            SlotKind::Quadratic(q) => {
                let terms = q.linear.len() + 2 * q.hessian.len();
                (terms + 1, terms + 2 * slot.support.len())
            }
            SlotKind::Nonlinear(t) => {
                let derivative_passes = if cfg!(target_arch = "wasm32") { n_vars } else { 2 };
                (tape_work(t), derivative_passes * tape_work(t) + n_vars + slot.support.len())
            }
        };
        Self {
            values: WorkPlan::new(slots.iter().map(|s| (cost(s).0, 1))),
            jacobian: WorkPlan::new(slots.iter().map(|s| (cost(s).1, s.support.len()))),
            max_regs: slots
                .iter()
                .filter_map(|s| match &s.kind {
                    SlotKind::Nonlinear(t) => Some(t.n_regs()),
                    _ => None,
                })
                .max()
                .unwrap_or(0),
        }
    }
}

/// Divide both scratch and caller-owned output before spawning work.
fn parallel_units(
    plan: &WorkPlan,
    start: usize,
    end: usize,
    workers: &mut [ParScratch],
    output: &mut [f64],
    evaluate: &(impl Fn(usize, &mut ParScratch, &mut [f64]) + Sync),
) {
    if workers.len() == 1 {
        let mut remaining = output;
        for i in start..end {
            let width = plan.offsets[i + 1] - plan.offsets[i];
            let (out, tail) = remaining.split_at_mut(width);
            evaluate(i, &mut workers[0], out);
            remaining = tail;
        }
        return;
    }
    let left_count = workers.len() / 2;
    let mid = plan.split(start, end, left_count, workers.len() - left_count);
    let (left_workers, right_workers) = workers.split_at_mut(left_count);
    let (left_output, right_output) = output.split_at_mut(plan.offsets[mid] - plan.offsets[start]);
    rayon::join(
        || parallel_units(plan, start, mid, left_workers, left_output, evaluate),
        || parallel_units(plan, mid, end, right_workers, right_output, evaluate),
    );
}

#[derive(Debug, Default)]
struct Scratch {
    /// One-hot forward-mode basis seed, length `n_vars`.
    basis: Vec<f64>,
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
    constraint_work: ConstraintWork,
    hessian_work: WorkPlan,
    scratch: RefCell<Scratch>,
    parallel: RefCell<ParallelWorkspace>,
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
        let constraint_exprs: Vec<ExprId> =
            model.constraints().algebraic().iter().map(|c| c.lhs).collect();

        let mut sparsity_workspace = SparsityWorkspace::default();
        let objective = match objective_expr {
            Some(e) => FunctionSlot::classify_with_workspace(&arena, e, &mut sparsity_workspace),
            None => FunctionSlot::zero(),
        };
        let constraints: Vec<FunctionSlot> = constraint_exprs
            .iter()
            .map(|&e| FunctionSlot::classify_with_workspace(&arena, e, &mut sparsity_workspace))
            .collect();

        // Lagrangian tape over the nonlinear functions only.
        let mut nl_sources = Vec::new();
        let mut nl_exprs = Vec::new();
        if let Some(e) = objective_expr
            && objective.is_nonlinear()
        {
            nl_sources.push(NlSource::Objective);
            nl_exprs.push(e);
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
        let (hess_structure, nl_pattern) =
            hessian_lagrangian_patterns(std::iter::once(&objective).chain(&constraints));

        let obj_hess_pos = quad_scatter_positions(&objective, &hess_structure);
        let con_hess_pos =
            constraints.iter().map(|s| quad_scatter_positions(s, &hess_structure)).collect();

        let seeds = build_seeds(&nl_pattern, &hess_structure);

        let max_regs = std::iter::once(&objective)
            .chain(&constraints)
            .filter_map(|s| match &s.kind {
                SlotKind::Nonlinear(t) => Some(t.n_regs()),
                _ => None,
            })
            .chain(std::iter::once(lagrangian.n_regs()))
            .max()
            .unwrap_or(0);

        let constraint_work = ConstraintWork::new(&constraints, n_vars);
        let hvp_passes = if cfg!(target_arch = "wasm32") { n_vars } else { 4 };
        let hessian_work = WorkPlan::new(seeds.iter().map(|seed| {
            (hvp_passes * tape_work(&lagrangian) + 4 * n_vars + seed.fills.len(), seed.fills.len())
        }));
        let scratch = RefCell::new(Scratch {
            basis: vec![0.0; n_vars],
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
            constraint_work,
            hessian_work,
            scratch,
            parallel: RefCell::default(),
        })
    }

    /// Re-snapshot parameter values (and the parameter-dependent linear /
    /// quadratic coefficients) after `set_param` on the model. Tapes are not
    /// recompiled and sparsity structures are kept. A coefficient that was
    /// exactly zero at construction and became nonzero may fall outside the
    /// original pattern, so build the evaluator with representative parameter
    /// values.
    ///
    /// The cached quadratic scatter positions and constraint work plans are rebuilt,
    /// `jac_structure`, `hess_structure`, and `seeds` are reused as-is. Nonlinear
    /// tapes (and therefore the nonlinear Hessian pattern the seeds cover) are
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
        self.constraint_work = ConstraintWork::new(&self.constraints, self.n_vars);
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
        let con_exprs: Vec<ExprId> =
            model.constraints().algebraic().iter().map(|c| c.lhs).collect();
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
        self.constraint_work = ConstraintWork::new(&self.constraints, self.n_vars);
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
            &mut scratch.basis,
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
        if self.constraint_work.values.workers(PAR_VALUE_WORK) < 2 {
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

    /// Write constraint values directly into disjoint caller-owned slices.
    fn eval_constraint_parallel(&self, x: &[f64], g: &mut [f64]) {
        let plan = &self.constraint_work.values;
        let count = plan.parallel_workers(PAR_VALUE_WORK);
        if count == 0 {
            return;
        }
        let workspace = &mut *self.parallel.borrow_mut();
        workspace.prepare(count, self.constraint_work.max_regs, self.n_vars, false, false);
        let (constraints, params) = (&self.constraints, self.params.as_slice());
        parallel_units(
            plan,
            0,
            constraints.len(),
            &mut workspace.workers[..count],
            g,
            &|i, sc, out| {
                out[0] = slot_value(&constraints[i], x, params, &mut sc.regs);
            },
        );
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
        if self.constraint_work.jacobian.workers(PAR_JACOBIAN_WORK) < 2 {
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
                &mut scratch.basis,
                &mut scratch.grad,
            );
            for &v in &slot.support {
                vals[out] = scratch.grad[v as usize];
                out += 1;
            }
        }
        debug_assert_eq!(out, vals.len());
    }

    /// Compute each row into retained scratch and gather directly into its
    /// disjoint slice of the sparse Jacobian, in declaration/support order.
    fn eval_constraint_jacobian_parallel(&self, x: &[f64], vals: &mut [f64]) {
        let plan = &self.constraint_work.jacobian;
        let count = plan.parallel_workers(PAR_JACOBIAN_WORK);
        if count == 0 {
            return;
        }
        let workspace = &mut *self.parallel.borrow_mut();
        workspace.prepare(count, self.constraint_work.max_regs, self.n_vars, true, false);
        let (constraints, params) = (&self.constraints, self.params.as_slice());
        parallel_units(
            plan,
            0,
            constraints.len(),
            &mut workspace.workers[..count],
            vals,
            &|i, sc, out| {
                let slot = &constraints[i];
                for &v in &slot.support {
                    sc.grad[v as usize] = 0.0;
                }
                slot_gradient_into(
                    slot,
                    x,
                    params,
                    &mut sc.regs,
                    &mut sc.dregs,
                    &mut sc.basis,
                    &mut sc.grad,
                );
                for (value, &v) in out.iter_mut().zip(&slot.support) {
                    *value = sc.grad[v as usize];
                }
            },
        );
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

        if self.hessian_work.workers(PAR_HESSIAN_WORK) < 2 {
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
        let n_regs = self.lagrangian.n_regs();
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
                &mut scratch.regs[..n_regs],
                &mut scratch.regs_t[..n_regs],
                &mut scratch.dregs[..n_regs],
                &mut scratch.dregs_t[..n_regs],
                &mut scratch.basis,
                &mut scratch.grad,
                &mut scratch.hv,
            );
            for &(pos, row) in &seed.fills {
                vals[pos] += scratch.hv[row];
            }
        }
    }

    /// Seeds recover interleaved sparse positions, so retain their values in
    /// seed/fill order and scatter serially after parallel HVPs.
    fn hessian_seeds_parallel(&self, x: &[f64], obj_factor: f64, lambda: &[f64], vals: &mut [f64]) {
        let plan = &self.hessian_work;
        let count = plan.parallel_workers(PAR_HESSIAN_WORK);
        if count == 0 {
            return;
        }
        let scratch = &mut *self.scratch.borrow_mut();
        for (weight, source) in scratch.mults.iter_mut().zip(&self.nl_sources) {
            *weight = mult_of(source, obj_factor, lambda);
        }
        let n_regs = self.lagrangian.n_regs();
        let workspace = &mut *self.parallel.borrow_mut();
        workspace.prepare(count, n_regs, self.n_vars, true, true);
        grow(&mut workspace.hessian_values, *plan.offsets.last().unwrap());
        let (lagrangian, params, seeds, mults) =
            (&self.lagrangian, self.params.as_slice(), &self.seeds, scratch.mults.as_slice());
        parallel_units(
            plan,
            0,
            seeds.len(),
            &mut workspace.workers[..count],
            &mut workspace.hessian_values,
            &|i, sc, out| {
                let seed = &seeds[i];
                sc.dir.fill(0.0);
                for &col in &seed.cols {
                    sc.dir[col] = 1.0;
                }
                tape_hvp(
                    lagrangian,
                    x,
                    &sc.dir,
                    params,
                    mults,
                    &mut sc.regs[..n_regs],
                    &mut sc.regs_t[..n_regs],
                    &mut sc.dregs[..n_regs],
                    &mut sc.dregs_t[..n_regs],
                    &mut sc.basis,
                    &mut sc.grad,
                    &mut sc.hv,
                );
                for (value, &(_, row)) in out.iter_mut().zip(&seed.fills) {
                    *value = sc.hv[row];
                }
            },
        );
        for (&(pos, _), &value) in
            seeds.iter().flat_map(|s| &s.fills).zip(&workspace.hessian_values)
        {
            vals[pos] += value;
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
#[expect(clippy::inline_always, reason = "Sparse rows need inlined closed-form dispatch")]
#[inline(always)]
fn slot_gradient_into(
    slot: &FunctionSlot,
    x: &[f64],
    params: &[f64],
    regs: &mut [f64],
    dregs: &mut [f64],
    basis: &mut [f64],
    grad: &mut [f64],
) {
    match &slot.kind {
        SlotKind::Linear(t) => linear_gradient_add(t, 1.0, grad),
        SlotKind::Quadratic(q) => quadratic_gradient_add(q, x, 1.0, grad),
        SlotKind::Nonlinear(tape) => {
            let n_regs = tape.n_regs();
            tape_gradient(
                tape,
                x,
                params,
                &[],
                &mut regs[..n_regs],
                &mut dregs[..n_regs],
                basis,
                grad,
            );
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
fn build_seeds(nl_pattern: &[(usize, usize)], hess_structure: &[(usize, usize)]) -> Vec<Seed> {
    // Star-color the nonlinear pattern.
    let coloring = star_hessian_coloring(nl_pattern);
    let mut seeds: Vec<Seed> =
        coloring.groups.into_iter().map(|cols| Seed { cols, fills: Vec::new() }).collect();
    let mut hess_pos = 0;
    for (&(row, col), &(group, hv_row)) in nl_pattern.iter().zip(&coloring.recover) {
        while hess_structure[hess_pos] < (row, col) {
            hess_pos += 1;
        }
        assert_eq!(
            hess_structure[hess_pos],
            (row, col),
            "nonlinear entry is in the merged pattern by construction"
        );
        seeds[group].fills.push((hess_pos, hv_row));
    }
    seeds
}

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
pub mod benchmark_support {
    use oximo_core::constraint::Relate;
    use rayon::prelude::*;

    use super::{ExprId, FunctionSlot, Model, NlpEvaluator, PAR_JACOBIAN_WORK, PAR_VALUE_WORK};

    /// Worker counts selected by the public constraint callbacks in the current
    /// Rayon pool. Counts below two select serial execution.
    pub fn constraint_workers(evaluator: &NlpEvaluator) -> (usize, usize) {
        (
            evaluator.constraint_work.values.workers(PAR_VALUE_WORK),
            evaluator.constraint_work.jacobian.workers(PAR_JACOBIAN_WORK),
        )
    }

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const THRESHOLD: usize = 1_024;

    pub fn model(rows: usize, degree: usize) -> Model {
        let model = Model::new("enzyme_classification_bench");
        let x = model.__var("x").lb(-3.0).ub(3.0).build();
        let y = model.__var("y").lb(-3.0).ub(3.0).build();
        let z = model.__var("z").lb(-3.0).ub(3.0).build();
        model.__minimize(x + y + z);
        for i in 0..rows {
            let lhs = match degree {
                1 => (x + 2.0 * y - z).erase(),
                2 => x.powi(2) + y * z,
                _ => (x * y * z + x.sin()).erase(),
            };
            model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        model
    }

    pub fn classify(model: &Model, parallel: bool) -> usize {
        let arena = model.arena().clone();
        let exprs: Vec<_> = model.constraints().algebraic().iter().map(|c| c.lhs).collect();
        classify_slots(&arena, &exprs, parallel).iter().map(|slot| slot.support.len()).sum()
    }

    #[derive(Debug)]
    pub struct Refresh {
        slots: Vec<FunctionSlot>,
        exprs: Vec<ExprId>,
    }

    impl Refresh {
        pub fn new(model: &Model) -> Self {
            let arena = model.arena().clone();
            let exprs: Vec<_> = model.constraints().algebraic().iter().map(|c| c.lhs).collect();
            let slots = classify_slots(&arena, &exprs, false);
            Self { slots, exprs }
        }

        pub fn run(&mut self, model: &Model, parallel: bool) -> usize {
            let arena = model.arena().clone();
            self.slots = reclassify(&arena, &self.slots, &self.exprs, parallel);
            self.slots.iter().map(|slot| slot.support.len()).sum()
        }
    }

    fn classify_slots(
        arena: &oximo_expr::ExprArena,
        exprs: &[ExprId],
        parallel: bool,
    ) -> Vec<FunctionSlot> {
        if parallel {
            exprs.par_iter().map(|&expr| FunctionSlot::classify(arena, expr)).collect()
        } else {
            exprs.iter().map(|&expr| FunctionSlot::classify(arena, expr)).collect()
        }
    }

    fn reclassify(
        arena: &oximo_expr::ExprArena,
        slots: &[FunctionSlot],
        exprs: &[ExprId],
        parallel: bool,
    ) -> Vec<FunctionSlot> {
        if parallel {
            slots
                .par_iter()
                .zip(exprs.par_iter())
                .map(|(slot, &expr)| slot.reclassify(arena, expr))
                .collect()
        } else {
            slots.iter().zip(exprs).map(|(slot, &expr)| slot.reclassify(arena, expr)).collect()
        }
    }
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

    fn with_threads(threads: usize, run: impl FnOnce() + Send) {
        #[cfg(not(target_arch = "wasm32"))]
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(run);
        #[cfg(target_arch = "wasm32")]
        {
            let _ = threads;
            run();
        }
    }

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

    fn check_parallel_callbacks(ev: &NlpEvaluator, point: &[f64], sigma: f64, lambda: &[f64]) {
        let mut serial = vec![0.0; ev.num_constraints()];
        let mut parallel = serial.clone();
        ev.eval_constraint_serial(point, &mut serial);
        ev.eval_constraint_parallel(point, &mut parallel);
        assert_eq!(serial, parallel);

        serial.resize(ev.jacobian_structure().len(), 0.0);
        parallel.resize(serial.len(), 0.0);
        ev.eval_constraint_jacobian_serial(point, &mut serial);
        ev.eval_constraint_jacobian_parallel(point, &mut parallel);
        assert_eq!(serial, parallel);

        // Simulate quadratic contributions.
        serial = vec![2.0; ev.hessian_lagrangian_structure().len()];
        parallel.clone_from(&serial);
        ev.hessian_seeds_serial(point, sigma, lambda, &mut serial);
        ev.hessian_seeds_parallel(point, sigma, lambda, &mut parallel);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn retained_workers_handle_empty_rows_refresh_and_pool_changes() {
        let m = Model::new("retained_workers");
        let n = 24;
        variable!(m, -2.0 <= x[i in 0..n] <= 2.0);
        param!(m, weight = 0.7);
        objective!(m, Min, sum!(x[i] for i in 0..n).sin());
        for i in 0..96 {
            match i % 4 {
                0 => {
                    m.__add_constraint_auto((0.0 * x[0]).le(1.0));
                }
                1 => {
                    m.__add_constraint_auto((weight * x[i % n] + x[(i + 1) % n]).le(2.0));
                }
                2 => {
                    m.__add_constraint_auto((weight * x[i % n].powi(2) + x[0] * x[1]).le(3.0));
                }
                _ => {
                    let mut expr = (x[i % n] + x[(i + 1) % n]).sin();
                    for _ in 0..(if i % 8 == 3 { 1 } else { 32 }) {
                        expr = (0.2 * expr + x[(i + 2) % n]).sin();
                    }
                    m.__add_constraint_auto((weight * expr).le(3.0));
                }
            }
        }
        let mut ev = NlpEvaluator::new(&m).unwrap();
        assert!(ev.constraints.iter().any(|s| s.support.is_empty()));
        assert_eq!(ev.num_hessian_seeds(), n);
        for threads in [1, 4, 2, 3, 4] {
            with_threads(threads, || {
                // Capture the evaluator mutably.
                let ev = &mut ev;
                for step in [0.1, -0.3, 0.0] {
                    let point = vec![step; n];
                    let lambda = vec![step; ev.num_constraints()];
                    check_parallel_callbacks(ev, &point, 1.0 + step, &lambda);
                    let workspace = ev.parallel.borrow();
                    let capacities: Vec<_> = workspace
                        .workers
                        .iter()
                        .map(|w| {
                            [
                                w.regs.capacity(),
                                w.dregs.capacity(),
                                w.regs_t.capacity(),
                                w.dregs_t.capacity(),
                                w.basis.capacity(),
                                w.dir.capacity(),
                                w.grad.capacity(),
                                w.hv.capacity(),
                            ]
                        })
                        .collect();
                    let values_capacity = workspace.hessian_values.capacity();
                    drop(workspace);
                    check_parallel_callbacks(ev, &point, 0.0, &lambda);
                    let workspace = ev.parallel.borrow();
                    for (w, expected) in workspace.workers.iter().zip(capacities) {
                        assert_eq!(
                            [
                                w.regs.capacity(),
                                w.dregs.capacity(),
                                w.regs_t.capacity(),
                                w.dregs_t.capacity(),
                                w.basis.capacity(),
                                w.dir.capacity(),
                                w.grad.capacity(),
                                w.hv.capacity()
                            ],
                            expected
                        );
                    }
                    assert_eq!(workspace.hessian_values.capacity(), values_capacity);
                }
            });
            weight.set_param_value(1.3);
            assert!(ev.try_refresh(&m));
            let fresh = NlpEvaluator::new(&m).unwrap();
            assert_eq!(ev.constraint_work.jacobian.offsets, fresh.constraint_work.jacobian.offsets);
            assert_eq!(ev.constraint_work.jacobian.work, fresh.constraint_work.jacobian.work);
            let point = vec![0.2; n];
            let lambda = vec![0.4; ev.num_constraints()];
            let mut refreshed = vec![0.0; ev.hess_structure.len()];
            let mut rebuilt = refreshed.clone();
            ev.eval_hessian_lagrangian(&point, 0.8, &lambda, &mut refreshed);
            fresh.eval_hessian_lagrangian(&point, 0.8, &lambda, &mut rebuilt);
            assert_eq!(refreshed, rebuilt);
            weight.set_param_value(0.7);
            ev.refresh_params(&m);
        }
    }

    #[test]
    fn empty_parallel_callbacks_and_single_seed_are_supported() {
        let m = Model::new("empty_callbacks");
        variable!(m, -2.0 <= x <= 2.0);
        objective!(m, Min, x.sin());
        let ev = NlpEvaluator::new(&m).unwrap();
        check_parallel_callbacks(&ev, &[0.3], 1.0, &[]);
        m.__minimize(x);
        let linear = NlpEvaluator::new(&m).unwrap();
        check_parallel_callbacks(&linear, &[0.3], 0.0, &[]);
    }

    #[test]
    fn work_estimates_distinguish_tape_cost_from_row_count() {
        with_threads(4, || {
            let cheap = WorkPlan::new((0..128).map(|_| (3, 1)));
            let expensive = WorkPlan::new((0..4).map(|_| (PAR_JACOBIAN_WORK, 1)));
            assert!(cheap.workers(PAR_JACOBIAN_WORK) < 2);
            assert_eq!(expensive.workers(PAR_JACOBIAN_WORK), rayon::current_num_threads());
        });
    }

    #[test]
    fn skewed_work_chooses_the_better_adjacent_split() {
        with_threads(2, || {
            let costs = [4_096_u16, 5_120, 1_024];
            let plan = WorkPlan::new(costs.iter().map(|&cost| (usize::from(cost), 1)));
            assert_eq!(plan.split(0, 3, 1, 1), 1);
            let mut workers: Vec<_> =
                (0..2).map(|_| ParScratch { grad: vec![0.0], ..ParScratch::default() }).collect();
            let mut output = vec![0.0; costs.len()];
            parallel_units(&plan, 0, costs.len(), &mut workers, &mut output, &|i, sc, out| {
                out[0] = f64::from(costs[i]);
                sc.grad[0] += out[0];
            });
            assert_eq!(output, costs.map(f64::from).to_vec());
            assert_eq!(
                workers.iter().map(|w| w.grad[0]).collect::<Vec<_>>(),
                vec![4_096.0, 6_144.0]
            );
        });
    }

    #[test]
    fn work_split_accounts_for_worker_counts_and_valid_boundaries() {
        // Unequal groups need the smallest maximum work per worker.
        let unequal = WorkPlan::new([3, 3, 3, 3, 2].into_iter().map(|cost| (cost, 1)));
        assert_eq!(unequal.split(0, 5, 1, 2), 1);
        let uniform = WorkPlan::new((0..7).map(|_| (1, 1)));
        assert_eq!(uniform.split(0, 7, 2, 3), 3);

        let offset = WorkPlan::new([1, 4_096, 5_120, 1_024, 1].into_iter().map(|cost| (cost, 1)));
        assert_eq!(offset.split(1, 4, 1, 1), 2);
        let left_heavy = WorkPlan::new([10_000, 1, 1, 1, 1].into_iter().map(|cost| (cost, 1)));
        assert_eq!(left_heavy.split(0, 5, 2, 2), 2);
        let right_heavy = WorkPlan::new([1, 1, 1, 1, 10_000].into_iter().map(|cost| (cost, 1)));
        assert_eq!(right_heavy.split(0, 5, 2, 2), 3);
        let tied = WorkPlan::new([2, 1, 2].into_iter().map(|cost| (cost, 1)));
        assert_eq!(tied.split(0, 3, 1, 1), 1);
    }

    #[test]
    fn work_estimates_include_packed_affine_terms() {
        let mut arena = oximo_expr::ExprArena::new();
        let linear = arena.linear((0..128).map(|i| (oximo_expr::VarId(i), 1.0)).collect(), 0.0);
        let root = arena.push(oximo_expr::ExprNode::Unary(oximo_expr::UnaryOp::Sin, linear));
        let tape = Tape::compile(&arena, root);
        assert!(tape_work(&tape) >= 128);
        assert!(tape.n_regs() < 128);
    }

    #[test]
    fn wide_sparse_objective_builds_the_expected_pattern() {
        let model = Model::new("wide_sparse");
        let vars: Vec<_> = (0..1_025).map(|i| model.__var(format!("x{i}")).build()).collect();
        let mut linear = vars[0];
        for &var in &vars[1..] {
            linear = linear + var;
        }
        model.__minimize((vars[0] + vars[1_024]).sin() + linear);

        let evaluator = NlpEvaluator::new(&model).unwrap();
        assert_eq!(evaluator.num_variables(), 1_025);
        assert_eq!(evaluator.hessian_lagrangian_structure(), &[(0, 0), (1_024, 0), (1_024, 1_024)]);
    }
}
