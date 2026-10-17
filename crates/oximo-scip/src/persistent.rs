use crate::{
    Scip, ScipMapping, ScipOptions, ScipPluginRegistry, ScipSolveOutput,
    plugins::{Shared, SharedState},
    translate::Snapshot,
};
use oximo_core::{Model, ModelKind, SosType, VarId};
use oximo_solver::{Solver, SolverError, SolverResult};
use russcip::{Model as Native, ProblemCreated, ProblemOrSolving, Solved};
use rustc_hash::FxHashMap;
use std::{cell::RefCell, rc::Rc};

struct State {
    native: Native<Solved>,
    mapping: ScipMapping,
    snapshot: Snapshot,
    options: ScipOptions,
    shared: SharedState,
    incumbent: FxHashMap<VarId, f64>,
}

/// Reuses original SCIP problems on unchanged and append-only solves. Existing
/// entry edits, option changes, and plugin-enabled solves rebuild. SCIP's
/// transformed search tree and LP basis are not retained by `free_transform`.
#[derive(Default)]
pub struct ScipPersistent {
    solver: Scip,
    state: Option<State>,
    builds: u64,
    reuses: u64,
}
impl std::fmt::Debug for ScipPersistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipPersistent")
            .field("resident", &self.state.is_some())
            .field("builds", &self.builds)
            .field("reuses", &self.reuses)
            .finish_non_exhaustive()
    }
}
impl ScipPersistent {
    pub fn new() -> Self {
        Self::default()
    }
    pub(crate) fn with_solver(solver: Scip) -> Self {
        Self { solver, ..Self::default() }
    }
    pub fn reset(&mut self) {
        self.state = None;
    }
    pub fn build_count(&self) -> u64 {
        self.builds
    }
    pub fn reuse_count(&self) -> u64 {
        self.reuses
    }
    /// Solve with the resident SCIP problem and return detailed results.
    ///
    /// # Errors
    ///
    /// Returns translation, registration, options, or solve errors. Any error
    /// clears the resident state so the next solve starts from a fresh problem.
    pub fn solve_detailed(
        &mut self,
        model: &Model,
        opts: &ScipOptions,
    ) -> Result<ScipSolveOutput, SolverError> {
        let result = crate::boundary(|| self.run(model, opts));
        if result.is_err() {
            self.state = None;
        }
        result
    }
    fn run(&mut self, model: &Model, opts: &ScipOptions) -> Result<ScipSolveOutput, SolverError> {
        if opts.concurrent && self.solver.factory.is_some() {
            return Err(crate::backend("concurrent solving cannot copy custom Rust plugins"));
        }
        let snapshot = Snapshot::prepare(model)?;
        let old = self.state.take();
        let reusable = old.as_ref().is_some_and(|s| {
            self.solver.factory.is_none() && s.options == *opts && snapshot.extends(&s.snapshot)
        });
        let previous = old
            .as_ref()
            .filter(|s| snapshot.columns.starts_with(&s.snapshot.columns))
            .map(|s| s.incumbent.clone())
            .unwrap_or_default();
        let (mut native, mapping, shared) = if reusable {
            let s = old.expect("resident state");
            let mut native = s.native.free_transform();
            let mut mapping = s.mapping;
            *s.shared.borrow_mut() = Shared::default();
            snapshot.append(&mut native, &mut mapping, Some(&s.snapshot))?;
            self.reuses += 1;
            (native, mapping, s.shared)
        } else {
            drop(old);
            let (native, mapping) = snapshot.build(opts)?;
            self.builds += 1;
            (native, mapping, Rc::new(RefCell::new(Shared::default())))
        };
        if let Some(factory) = &self.solver.factory {
            factory(&mut ScipPluginRegistry::new(&mut native, &mapping, shared.clone()))?;
        }
        crate::plugins::refresh_prices(&mapping, &shared);
        if !reusable
            && snapshot.kind == ModelKind::LP
            && snapshot.columns.iter().all(|column| column.domain.semi_threshold().is_none())
            && !opts.concurrent
            && self.solver.factory.is_none()
        {
            crate::plugins::install_prices(&mut native, &mapping, shared.clone());
        }
        add_start(&native, &mapping, model, &previous)?;
        let solved =
            if opts.concurrent { native.try_solve_concurrent() } else { native.try_solve() }
                .map_err(crate::backend)?;
        if let Some(error) = shared.borrow().error.as_ref() {
            return Err(crate::backend(error));
        }
        let output = crate::result::collect(&solved, &mapping, &snapshot, model, &shared)?;
        let incumbent = output.result.primal().cloned().unwrap_or_default();
        self.state = Some(State {
            native: solved,
            mapping,
            snapshot,
            options: opts.clone(),
            shared,
            incumbent,
        });
        Ok(output)
    }
}
impl Solver for ScipPersistent {
    type Options = ScipOptions;
    fn name(&self) -> &str {
        "SCIP"
    }

    fn supports(&self, kind: ModelKind) -> bool {
        self.solver.supports(kind)
    }

    fn domain_support(&self, domain: oximo_core::Domain) -> oximo_solver::Capability {
        self.solver.domain_support(domain)
    }

    fn supports_sos(&self, kind: SosType) -> bool {
        self.solver.supports_sos(kind)
    }

    fn supports_indicators(&self) -> bool {
        self.solver.supports_indicators()
    }

    fn supports_model(&self, model: &Model) -> bool {
        self.solver.supports_model(model)
    }

    fn validate_model_extra(
        &self,
        model: &Model,
        opts: &Self::Options,
    ) -> Vec<oximo_solver::CapabilityDiagnostic> {
        self.solver.validate_model_extra(model, opts)
    }

    fn solve(&mut self, model: &Model, opts: &ScipOptions) -> Result<SolverResult, SolverError> {
        Ok(self.solve_detailed(model, opts)?.result)
    }
}

fn add_start(
    native: &Native<ProblemCreated>,
    mapping: &ScipMapping,
    model: &Model,
    previous: &FxHashMap<VarId, f64>,
) -> Result<(), SolverError> {
    let values = model
        .variables()
        .iter()
        .filter_map(|v| v.initial.or_else(|| previous.get(&v.id).copied()).map(|x| (v.id, x)))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(());
    }
    for &(_, value) in &values {
        crate::finite(value)?;
    }
    // Partial starts permit new variables and nonlinear objective auxiliaries
    // to be completed by SCIP rather than incorrectly fixing unknowns to zero.
    let sol = native.create_partial_sol();
    for (id, value) in values {
        sol.set_val(&mapping.variables[id.index()], value);
    }
    let _accepted = native.add_sol(sol);
    Ok(())
}
