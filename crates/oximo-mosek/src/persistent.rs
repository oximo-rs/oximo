use mosek::{Streamtype, TaskCB};
use oximo_core::{Model, ModelKind};
use oximo_solver::{Snapshot, Solver, SolverError, SolverResult, snapshot};

use crate::translate::{Meta, build_task, solve_task};
use crate::{MosekOptions, NAME};

/// The persistent MOSEK task and the information needed to update it safely.
struct State {
    task: TaskCB,
    meta: Meta,
    snapshot: Option<Snapshot>,
    initial: Vec<Option<f64>>,
}

/// A stateful MOSEK handle that keeps a translated model resident across solves.
///
/// Created by [`Mosek::persistent`](crate::Mosek). For LP and MILP models,
/// changes limited to objective coefficients, the objective constant, and
/// variable bounds are pushed directly into the retained MOSEK task. MOSEK can
/// then reuse its available basis or incumbent information. Changes to rows,
/// integrality, objective sense, initial values, or model dimensions rebuild the
/// task transparently. Quadratic and conic models currently rebuild on every
/// call, retaining the same correctness guarantee while leaving their native
/// incremental update support for future work.
///
/// Options are applied before every optimization and persist on a resident task.
/// If an option is omitted later it keeps its previous native value. Call
/// [`reset`](Self::reset), or use a fresh handle, to restore cold-solve option
/// behavior as well as discard warm-start information.
#[derive(Default)]
pub struct MosekPersistent {
    state: Option<State>,
}

impl std::fmt::Debug for MosekPersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MosekPersistent").field("resident", &self.state.is_some()).finish()
    }
}

impl MosekPersistent {
    /// Create an empty resident handle. The first solve builds its task.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Discard the persistent MOSEK task. The next solve builds a fresh task.
    pub fn reset(&mut self) {
        self.state = None;
    }

    fn rebuild(&mut self, model: &Model, opts: &MosekOptions) -> Result<(), SolverError> {
        let (task, meta) = build_task(model, opts)?;
        let snapshot = linear_snapshot(model)?;
        let initial = initial_values(model);
        self.state = Some(State { task, meta, snapshot, initial });
        Ok(())
    }

    fn solve_resident(
        &mut self,
        model: &Model,
        opts: &MosekOptions,
    ) -> Result<SolverResult, SolverError> {
        let fresh = linear_snapshot(model)?;
        let initial = initial_values(model);
        let mut updated = false;

        if let (Some(fresh), Some(state)) = (fresh, self.state.as_mut())
            && state.snapshot.as_ref().is_some_and(|base| base.fingerprint == fresh.fingerprint)
            && state.initial == initial
        {
            apply_linear_delta(
                &mut state.task,
                state.snapshot.as_ref().expect("snapshot"),
                &fresh,
            )?;
            opts.apply_cb(&mut state.task)?;
            attach_verbose_stream(&mut state.task, opts)?;
            state.snapshot = Some(fresh);
            updated = true;
        }

        if !updated {
            self.rebuild(model, opts)?;
        }

        let state = self.state.as_mut().expect("resident state after rebuild");
        solve_task(model, &mut state.task, &state.meta)
    }
}

impl Solver for MosekPersistent {
    type Options = MosekOptions;

    fn name(&self) -> &str {
        NAME
    }

    fn supports(&self, kind: ModelKind) -> bool {
        crate::supported(kind)
    }

    fn solve(&mut self, model: &Model, opts: &MosekOptions) -> Result<SolverResult, SolverError> {
        match self.solve_resident(model, opts) {
            Ok(result) => Ok(result),
            Err(error) => {
                self.state = None;
                Err(error)
            }
        }
    }
}

fn linear_snapshot(model: &Model) -> Result<Option<Snapshot>, SolverError> {
    if matches!(model.kind(), ModelKind::LP | ModelKind::MILP) {
        snapshot(model).map(Some)
    } else {
        Ok(None)
    }
}

fn initial_values(model: &Model) -> Vec<Option<f64>> {
    model.variables().iter().map(|variable| variable.initial).collect()
}

fn apply_linear_delta(
    task: &mut TaskCB,
    base: &Snapshot,
    fresh: &Snapshot,
) -> Result<(), SolverError> {
    for index in 0..fresh.obj_costs.len() {
        let column = i32::try_from(index).map_err(|_| {
            SolverError::Backend("MOSEK: variable count exceeds the native index range".into())
        })?;
        if fresh.obj_costs[index].to_bits() != base.obj_costs[index].to_bits() {
            task.put_c_j(column, fresh.obj_costs[index]).map_err(backend)?;
        }
        if fresh.lb[index].to_bits() != base.lb[index].to_bits()
            || fresh.ub[index].to_bits() != base.ub[index].to_bits()
        {
            let (key, lower, upper) = bounds(fresh.lb[index], fresh.ub[index]);
            task.put_var_bound(column, key, lower, upper).map_err(backend)?;
        }
    }
    if fresh.obj_constant.to_bits() != base.obj_constant.to_bits() {
        task.put_cfix(fresh.obj_constant).map_err(backend)?;
    }
    Ok(())
}

fn attach_verbose_stream(task: &mut TaskCB, opts: &MosekOptions) -> Result<(), SolverError> {
    if opts.universal.verbose.unwrap_or(false) {
        task.put_stream_callback(Streamtype::LOG, |message| print!("{message}"))
            .map_err(backend)?;
    }
    Ok(())
}

fn bounds(lower: f64, upper: f64) -> (i32, f64, f64) {
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => (mosek::Boundkey::FR, 0.0, 0.0),
        (true, false) => (mosek::Boundkey::LO, lower, 0.0),
        (false, true) => (mosek::Boundkey::UP, 0.0, upper),
        (true, true) if lower.total_cmp(&upper).is_eq() => (mosek::Boundkey::FX, lower, upper),
        (true, true) => (mosek::Boundkey::RA, lower, upper),
    }
}

fn backend(message: String) -> SolverError {
    SolverError::Backend(format!("MOSEK: {message}"))
}
