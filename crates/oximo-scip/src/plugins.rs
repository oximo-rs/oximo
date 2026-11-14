//! Oximo-ID-aware SCIP callbacks. Plugin factories are invoked for each build.
//! Callbacks run on SCIP's solving thread and must not retain native handles.
use oximo_core::{ConstraintId, VarId};
use oximo_solver::SolverError;
use russcip::{
    Model, ModelWithProblem, ProblemCreated, ProblemOrSolving, Solving, WithSolutions,
    WithSolvingStats,
};
use rustc_hash::FxHashMap;
use std::{
    cell::OnceCell,
    cell::RefCell,
    cmp::Ordering,
    collections::HashSet,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
};

pub use russcip::{
    ConshdlrResult, EventMask, HeurResult, HeurTiming, Node, PricerResult, PricerResultState,
    SeparationResult,
};

/// Mapping available during native configuration. Handles refer to original
/// SCIP entities and must not be captured by a plugin.
#[derive(Debug, Default)]
pub struct ScipMapping {
    pub(crate) variables: Vec<russcip::Variable>,
    pub(crate) constraints: Vec<Option<russcip::Constraint>>,
    pub(crate) linear: Vec<bool>,
    pub(crate) objective_aux: Option<russcip::Variable>,
}
impl ScipMapping {
    pub fn variable(&self, id: VarId) -> Option<&russcip::Variable> {
        self.variables.get(id.index())
    }
    pub fn constraint(&self, id: ConstraintId) -> Option<&russcip::Constraint> {
        self.constraints.get(id.index())?.as_ref()
    }
}

/// Separate identity space for dynamically priced columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScipGeneratedVarId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScipVarId {
    Original(VarId),
    Generated(ScipGeneratedVarId),
}
impl From<VarId> for ScipVarId {
    fn from(v: VarId) -> Self {
        Self::Original(v)
    }
}
impl From<ScipGeneratedVarId> for ScipVarId {
    fn from(v: ScipGeneratedVarId) -> Self {
        Self::Generated(v)
    }
}

#[derive(Debug, Clone)]
pub struct ScipLinearRow {
    pub name: String,
    pub coefficients: Vec<(ScipVarId, f64)>,
    pub lower: f64,
    pub upper: f64,
    pub local: bool,
}

#[derive(Debug, Clone)]
pub struct ScipColumn {
    pub name: String,
    pub lower: f64,
    pub upper: f64,
    pub objective: f64,
    pub integer: bool,
    pub coefficients: Vec<(ConstraintId, f64)>,
}

/// Registration priorities and scheduling common to SCIP plugins.
/// Specialized native scheduling remains available through the native configuration hook.
#[derive(Debug, Clone)]
pub struct ScipPluginOptions {
    pub name: String,
    pub description: String,
    pub priority: i32,
    pub frequency: i32,
    pub max_depth: i32,
    pub timing: HeurTiming,
}
impl ScipPluginOptions {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            priority: 0,
            frequency: 1,
            max_depth: -1,
            timing: HeurTiming::AFTER_LP_NODE,
        }
    }
}

#[derive(Debug, Clone)]
struct Map {
    vars: Vec<usize>,
    original_by_index: FxHashMap<usize, VarId>,
    constraints: Vec<Option<String>>,
    linear: Vec<bool>,
}
impl From<&ScipMapping> for Map {
    fn from(m: &ScipMapping) -> Self {
        Self {
            vars: m.variables.iter().map(russcip::Variable::index).collect(),
            original_by_index: m
                .variables
                .iter()
                .enumerate()
                .map(|(i, v)| (v.index(), VarId(u32::try_from(i).expect("variable id"))))
                .collect(),
            constraints: m
                .constraints
                .iter()
                .map(|c| c.as_ref().map(russcip::Constraint::name))
                .collect(),
            linear: m.linear.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Generated {
    pub index: usize,
    pub objective: f64,
    pub name: String,
}
#[derive(Debug, Default)]
pub(crate) struct Shared {
    price_map: Option<Map>,
    pub error: Option<String>,
    pub generated: Vec<Generated>,
    generated_by_index: FxHashMap<usize, ScipGeneratedVarId>,
    pub dual: FxHashMap<ConstraintId, f64>,
    pub reduced: FxHashMap<VarId, f64>,
    pub lp_values: FxHashMap<VarId, f64>,
    pub lp_objective: Option<f64>,
}
pub(crate) type SharedState = Rc<RefCell<Shared>>;

fn guarded<T>(
    shared: &SharedState,
    fallback: impl FnOnce() -> T,
    f: impl FnOnce() -> Result<T, SolverError>,
) -> T {
    if shared.borrow().error.is_some() {
        return fallback();
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => {
            shared.borrow_mut().error = Some(e.to_string());
            fallback()
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("callback panic");
            shared.borrow_mut().error = Some(format!("SCIP plugin panicked: {msg}"));
            fallback()
        }
    }
}

/// Solving-stage view. Values use original IDs even after presolve.
/// Native access is borrowed for this callback only.
pub struct ScipContext<'a> {
    model: Model<Solving>,
    map: &'a Map,
    shared: SharedState,
    original_by_native_index: OnceCell<FxHashMap<usize, russcip::Variable>>,
    transformed_by_index: OnceCell<FxHashMap<usize, VarId>>,
}
impl std::fmt::Debug for ScipContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipContext").finish_non_exhaustive()
    }
}
impl ScipContext<'_> {
    fn new<'a>(model: Model<Solving>, map: &'a Map, shared: SharedState) -> ScipContext<'a> {
        ScipContext {
            model,
            map,
            shared,
            original_by_native_index: OnceCell::new(),
            transformed_by_index: OnceCell::new(),
        }
    }
    fn transformed_by_index(&self) -> &FxHashMap<usize, VarId> {
        self.transformed_by_index.get_or_init(|| {
            self.model
                .orig_vars()
                .into_iter()
                .filter_map(|original| {
                    let id = *self.map.original_by_index.get(&original.index())?;
                    Some((original.transformed()?.index(), id))
                })
                .collect()
        })
    }
    pub fn native(&self) -> &Model<Solving> {
        &self.model
    }
    pub fn native_mut(&mut self) -> &mut Model<Solving> {
        &mut self.model
    }
    pub fn variable(&self, id: impl Into<ScipVarId>) -> Option<russcip::Variable> {
        match id.into() {
            ScipVarId::Original(v) => {
                let index = *self.map.vars.get(v.index())?;
                self.original_by_native_index
                    .get_or_init(|| {
                        self.model.orig_vars().into_iter().map(|v| (v.index(), v)).collect()
                    })
                    .get(&index)
                    .cloned()
            }
            ScipVarId::Generated(v) => {
                let index = self.shared.borrow().generated.get(v.0)?.index;
                self.model.vars().into_iter().find(|v| v.index() == index)
            }
        }
    }
    pub fn value(&self, id: impl Into<ScipVarId>) -> Option<f64> {
        self.variable(id).map(|v| self.model.current_val(&v))
    }
    pub fn incumbent_value(&self, id: impl Into<ScipVarId>) -> Option<f64> {
        Some(self.model.best_sol()?.val(&self.variable(id)?))
    }
    pub fn best_bound(&self) -> f64 {
        self.model.best_bound()
    }
    pub fn node_count(&self) -> usize {
        self.model.n_nodes()
    }
    pub fn identify(&self, var: &russcip::Variable) -> Option<ScipVarId> {
        if var.is_original() {
            return self.map.original_by_index.get(&var.index()).copied().map(ScipVarId::Original);
        }
        if let Some(&id) = self.transformed_by_index().get(&var.index()) {
            return Some(ScipVarId::Original(id));
        }
        self.shared.borrow().generated_by_index.get(&var.index()).copied().map(ScipVarId::Generated)
    }
    pub fn constraint(&self, id: ConstraintId) -> Option<russcip::Constraint> {
        let c = self.model.find_cons(self.map.constraints.get(id.index())?.as_deref()?)?;
        c.transformed().or(Some(c))
    }
    /// Submit a complete candidate. Unspecified original/generated variables
    /// retain SCIP's zero default. Infeasible candidates return `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn submit_solution(&self, values: &[(ScipVarId, f64)]) -> Result<bool, SolverError> {
        let sol = self.model.create_sol();
        for &(id, value) in values {
            crate::finite(value)?;
            let v =
                self.variable(id).ok_or_else(|| crate::backend("unknown candidate variable"))?;
            sol.set_val(&v, value);
        }
        Ok(self.model.add_sol(sol).is_ok())
    }
    fn row_vars(&self, row: &ScipLinearRow) -> Result<Vec<(russcip::Variable, f64)>, SolverError> {
        crate::check_name(&row.name)?;
        crate::translate::check_bounds(row.lower, row.upper)?;
        if row.lower > row.upper {
            return Err(crate::backend("inverted plugin row bounds"));
        }
        row.coefficients
            .iter()
            .map(|&(id, c)| {
                crate::finite(c)?;
                Ok((self.variable(id).ok_or_else(|| crate::backend("unknown cut variable"))?, c))
            })
            .collect()
    }
}

pub struct ScipSeparationContext<'a> {
    pub context: ScipContext<'a>,
    separator: russcip::SCIPSeparator,
}
impl ScipSeparationContext<'_> {
    /// Returns true if the cut makes the current node infeasible.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn add_cut(&mut self, row: &ScipLinearRow) -> Result<bool, SolverError> {
        let vars = self.context.row_vars(row)?;
        let mut native = self
            .separator
            .create_empty_row(
                self.context.native(),
                &row.name,
                row.lower,
                row.upper,
                row.local,
                false,
                true,
            )
            .map_err(crate::backend)?;
        for (v, c) in vars {
            native.set_coeff(&v, c);
        }
        Ok(self.context.model.add_cut(native, false))
    }
}
impl std::fmt::Debug for ScipSeparationContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipSeparationContext").finish_non_exhaustive()
    }
}

pub struct ScipPricingContext<'a> {
    pub context: ScipContext<'a>,
    pub farkas: bool,
}
impl std::fmt::Debug for ScipPricingContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipPricingContext").field("farkas", &self.farkas).finish_non_exhaustive()
    }
}
impl ScipPricingContext<'_> {
    pub fn price(&self, id: ConstraintId) -> Option<f64> {
        let c = self.context.constraint(id)?;
        if self.farkas { c.farkas_dual_sol() } else { c.dual_sol() }
    }
    /// Add a priced column and return its generated variable ID.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn add_column(&mut self, column: &ScipColumn) -> Result<ScipGeneratedVarId, SolverError> {
        crate::check_name(&column.name)?;
        crate::translate::check_bounds(column.lower, column.upper)?;
        crate::finite(column.objective)?;
        if column.lower > column.upper {
            return Err(crate::backend("inverted priced column bounds"));
        }
        let constraints = column
            .coefficients
            .iter()
            .map(|&(id, c)| {
                crate::finite(c)?;
                let cons = self
                    .context
                    .constraint(id)
                    .ok_or_else(|| crate::backend("unknown pricing row"))?;
                if !self.context.map.linear.get(id.index()).copied().unwrap_or(false)
                    || !cons.is_modifiable()
                {
                    return Err(crate::backend(
                        "pricing requires a declared modifiable linear row",
                    ));
                }
                Ok((cons, c))
            })
            .collect::<Result<Vec<_>, SolverError>>()?;
        let var = self.context.model.add_priced_var(
            column.lower,
            column.upper,
            column.objective,
            &column.name,
            if column.integer { russcip::VarType::Integer } else { russcip::VarType::Continuous },
        );
        for (cons, c) in constraints {
            self.context.model.add_cons_coef(&cons, &var, c);
        }
        let mut shared = self.context.shared.borrow_mut();
        let id = ScipGeneratedVarId(shared.generated.len());
        shared.generated_by_index.insert(var.index(), id);
        shared.generated.push(Generated {
            index: var.index(),
            objective: column.objective,
            name: column.name.clone(),
        });
        Ok(id)
    }
}

pub struct ScipConstraintContext<'a> {
    pub context: ScipContext<'a>,
}
impl std::fmt::Debug for ScipConstraintContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipConstraintContext").finish_non_exhaustive()
    }
}
impl ScipConstraintContext<'_> {
    /// Add a linear constraint during a constraint-handler callback.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn add_constraint(&mut self, row: &ScipLinearRow) -> Result<(), SolverError> {
        let vars = self.context.row_vars(row)?;
        let mut builder = russcip::prelude::cons().name(&row.name).bounds(row.lower, row.upper);
        for (v, c) in &vars {
            builder = builder.coef(v, *c);
        }
        if row.local {
            self.context.model.add_cons_local(&builder);
        } else {
            self.context.model.add(builder);
        }
        Ok(())
    }
}

/// A branch candidate is scoped to the current callback. `variable` is None
/// for SCIP-generated auxiliaries not belonging to oximo or a pricer.
#[derive(Clone, Debug)]
pub struct ScipBranchCandidate {
    pub variable: Option<ScipVarId>,
    pub value: f64,
    pub fraction: f64,
}
#[derive(Clone, Debug)]
pub enum ScipBranchingResult {
    DidNotRun,
    BranchOn(usize),
    CutOff,
    CustomBranching,
}

pub trait ScipEventHandler {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve.
    fn execute(
        &mut self,
        context: &mut ScipContext<'_>,
        event: EventMask,
    ) -> Result<(), SolverError>;
}
pub trait ScipSeparator {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve.
    fn execute(
        &mut self,
        context: &mut ScipSeparationContext<'_>,
    ) -> Result<SeparationResult, SolverError>;
}
pub trait ScipPricer {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve.
    fn generate_columns(
        &mut self,
        context: &mut ScipPricingContext<'_>,
    ) -> Result<PricerResult, SolverError>;
}
pub trait ScipBranchRule {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve.
    fn execute(
        &mut self,
        context: &mut ScipContext<'_>,
        candidates: &[ScipBranchCandidate],
    ) -> Result<ScipBranchingResult, SolverError>;
}
pub trait ScipHeuristic {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve.
    fn execute(
        &mut self,
        context: &mut ScipContext<'_>,
        timing: HeurTiming,
        node_infeasible: bool,
    ) -> Result<HeurResult, SolverError>;
}
pub trait ScipConstraintHandler {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve.
    fn check(
        &mut self,
        context: &mut ScipContext<'_>,
        values: &FxHashMap<VarId, f64>,
    ) -> Result<bool, SolverError>;
    /// Enforce the handler's constraints at the current SCIP node.
    ///
    /// # Errors
    ///
    /// Errors invalidate the enclosing solve result.
    fn enforce(
        &mut self,
        context: &mut ScipConstraintContext<'_>,
    ) -> Result<ConshdlrResult, SolverError>;
}
pub trait ScipNodeSelector {
    /// Execute this plugin operation at the current SCIP stage.
    ///
    /// # Errors
    ///
    /// Errors invalidate the solve; adapters contain unwinding panics.
    fn select(&mut self, context: &mut ScipContext<'_>) -> Result<Option<Node>, SolverError>;
    /// Compare two nodes for selection priority.
    ///
    /// # Errors
    ///
    /// Errors invalidate the enclosing solve result.
    fn compare(&mut self, a: &Node, b: &Node) -> Result<Ordering, SolverError>;
}

struct Adapter<T: ?Sized> {
    plugin: Box<T>,
    map: Map,
    shared: SharedState,
    events: EventMask,
}

impl russcip::Eventhdlr for Adapter<dyn ScipEventHandler> {
    fn get_type(&self) -> EventMask {
        self.events
    }
    fn execute(&mut self, model: Model<Solving>, _: russcip::SCIPEventhdlr, event: russcip::Event) {
        let mut context = ScipContext::new(model, &self.map, self.shared.clone());
        guarded(&self.shared, || (), || self.plugin.execute(&mut context, event.event_type()));
    }
}
impl russcip::Separator for Adapter<dyn ScipSeparator> {
    fn execute_lp(
        &mut self,
        model: Model<Solving>,
        separator: russcip::SCIPSeparator,
    ) -> SeparationResult {
        let mut context = ScipSeparationContext {
            context: ScipContext::new(model, &self.map, self.shared.clone()),
            separator,
        };
        guarded(&self.shared, || SeparationResult::DidNotRun, || self.plugin.execute(&mut context))
    }
}
impl russcip::Pricer for Adapter<dyn ScipPricer> {
    fn generate_columns(
        &mut self,
        model: Model<Solving>,
        _: russcip::SCIPPricer,
        farkas: bool,
    ) -> PricerResult {
        let mut context = ScipPricingContext {
            context: ScipContext::new(model, &self.map, self.shared.clone()),
            farkas,
        };
        guarded(
            &self.shared,
            || PricerResult { state: PricerResultState::DidNotRun, lower_bound: None },
            || self.plugin.generate_columns(&mut context),
        )
    }
}
impl russcip::BranchRule for Adapter<dyn ScipBranchRule> {
    fn execute(
        &mut self,
        model: Model<Solving>,
        _: russcip::SCIPBranchRule,
        candidates: Vec<russcip::BranchingCandidate>,
    ) -> russcip::BranchingResult {
        let mut context = ScipContext::new(model, &self.map, self.shared.clone());
        guarded(
            &self.shared,
            || russcip::BranchingResult::DidNotRun,
            || {
                let cs = candidates
                    .iter()
                    .map(|c| ScipBranchCandidate {
                        variable: context
                            .model
                            .var_in_prob(c.var_prob_id)
                            .and_then(|v| context.identify(&v)),
                        value: c.lp_sol_val,
                        fraction: c.frac,
                    })
                    .collect::<Vec<_>>();
                Ok(match self.plugin.execute(&mut context, &cs)? {
                    ScipBranchingResult::DidNotRun => russcip::BranchingResult::DidNotRun,
                    ScipBranchingResult::CutOff => russcip::BranchingResult::CutOff,
                    ScipBranchingResult::CustomBranching => {
                        russcip::BranchingResult::CustomBranching
                    }
                    ScipBranchingResult::BranchOn(i) => russcip::BranchingResult::BranchOn(
                        candidates
                            .get(i)
                            .ok_or_else(|| crate::backend("invalid branch candidate index"))?
                            .clone(),
                    ),
                })
            },
        )
    }
}
impl russcip::Heuristic for Adapter<dyn ScipHeuristic> {
    fn execute(&mut self, model: Model<Solving>, timing: HeurTiming, node_inf: bool) -> HeurResult {
        let mut context = ScipContext::new(model, &self.map, self.shared.clone());
        guarded(
            &self.shared,
            || HeurResult::DidNotRun,
            || self.plugin.execute(&mut context, timing, node_inf),
        )
    }
}
impl russcip::Conshdlr for Adapter<dyn ScipConstraintHandler> {
    fn check(
        &mut self,
        model: Model<Solving>,
        _: russcip::SCIPConshdlr,
        solution: &russcip::Solution<'_>,
    ) -> bool {
        let map = &self.map;
        let mut context = ScipContext::new(model, map, self.shared.clone());
        guarded(
            &self.shared,
            || false,
            || {
                let values = map
                    .vars
                    .iter()
                    .enumerate()
                    .filter_map(|(i, _)| {
                        let id = VarId(u32::try_from(i).ok()?);
                        Some((id, solution.val(&context.variable(id)?)))
                    })
                    .collect();
                self.plugin.check(&mut context, &values)
            },
        )
    }
    /// Forward the enforcement callback and record plugin errors.
    ///
    /// # Errors
    ///
    /// Errors invalidate the enclosing solve result.
    fn enforce(&mut self, model: Model<Solving>, _: russcip::SCIPConshdlr) -> ConshdlrResult {
        let mut context = ScipConstraintContext {
            context: ScipContext::new(model, &self.map, self.shared.clone()),
        };
        guarded(&self.shared, || ConshdlrResult::CutOff, || self.plugin.enforce(&mut context))
    }
}
impl russcip::NodeSel for Adapter<dyn ScipNodeSelector> {
    fn select(&mut self, model: Model<Solving>) -> Option<Node> {
        let mut context = ScipContext::new(model, &self.map, self.shared.clone());
        guarded(&self.shared, || None, || self.plugin.select(&mut context))
    }
    fn comp(&mut self, a: Node, b: Node) -> Ordering {
        guarded(&self.shared, || Ordering::Equal, || self.plugin.compare(&a, &b))
    }
}

/// Registers plugins on a newly built problem. Factories should construct new
/// plugin state here.
pub struct ScipPluginRegistry<'a> {
    model: &'a mut Model<ProblemCreated>,
    mapping: &'a ScipMapping,
    map: Map,
    shared: SharedState,
    names: HashSet<String>,
}
impl std::fmt::Debug for ScipPluginRegistry<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScipPluginRegistry").field("names", &self.names).finish_non_exhaustive()
    }
}
impl<'a> ScipPluginRegistry<'a> {
    pub(crate) fn new(
        model: &'a mut Model<ProblemCreated>,
        mapping: &'a ScipMapping,
        shared: SharedState,
    ) -> Self {
        Self { model, mapping, map: Map::from(mapping), shared, names: HashSet::new() }
    }
    /// Advanced safe russcip configuration. Do not retain owning model/entity
    /// handles in callbacks, and do not remove or reinterpret mapped entities.
    pub fn native(&mut self) -> (&mut Model<ProblemCreated>, &ScipMapping) {
        (self.model, self.mapping)
    }
    fn validate(&mut self, o: &ScipPluginOptions) -> Result<(), SolverError> {
        crate::check_name(&o.name)?;
        crate::check_name(&o.description)?;
        if o.name.is_empty()
            || !self.names.insert(o.name.clone())
            || o.frequency < -1
            || o.max_depth < -1
        {
            return Err(crate::backend("invalid or duplicate plugin registration"));
        }
        Ok(())
    }
    fn adapter<T: ?Sized>(&self, plugin: Box<T>, events: EventMask) -> Adapter<T> {
        Adapter { plugin, map: self.map.clone(), shared: self.shared.clone(), events }
    }
    /// Register an event handler for the selected SCIP events.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn event_handler(
        &mut self,
        o: ScipPluginOptions,
        events: EventMask,
        p: impl ScipEventHandler + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        let a = self.adapter(Box::new(p) as Box<dyn ScipEventHandler>, events);
        self.model.include_eventhdlr(&o.name, &o.description, Box::new(a));
        Ok(())
    }
    /// Register a separator for LP solutions.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn separator(
        &mut self,
        o: ScipPluginOptions,
        p: impl ScipSeparator + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        let a = self.adapter(Box::new(p) as Box<dyn ScipSeparator>, EventMask::DISABLED);
        self.model.include_separator(
            &o.name,
            &o.description,
            o.priority,
            o.frequency,
            1.0,
            false,
            false,
            Box::new(a),
        );
        Ok(())
    }
    /// Register a pricer and mark its declared linear rows as modifiable.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn pricer(
        &mut self,
        o: ScipPluginOptions,
        rows: &[ConstraintId],
        p: impl ScipPricer + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        for &id in rows {
            if !self.map.linear.get(id.index()).copied().unwrap_or(false) {
                return Err(crate::backend("pricing requires active linear rows"));
            }
            let c =
                self.mapping.constraint(id).ok_or_else(|| crate::backend("unknown pricing row"))?;
            self.model.set_cons_modifiable(c, true);
        }
        let a = self.adapter(Box::new(p) as Box<dyn ScipPricer>, EventMask::DISABLED);
        self.model.include_pricer(&o.name, &o.description, o.priority, false, Box::new(a));
        Ok(())
    }
    /// Register a custom branching rule.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn branch_rule(
        &mut self,
        o: ScipPluginOptions,
        p: impl ScipBranchRule + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        let a = self.adapter(Box::new(p) as Box<dyn ScipBranchRule>, EventMask::DISABLED);
        self.model.include_branch_rule(
            &o.name,
            &o.description,
            o.priority,
            o.max_depth,
            1.0,
            Box::new(a),
        );
        Ok(())
    }
    /// Register a primal heuristic.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn heuristic(
        &mut self,
        o: ScipPluginOptions,
        p: impl ScipHeuristic + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        let a = self.adapter(Box::new(p) as Box<dyn ScipHeuristic>, EventMask::DISABLED);
        self.model.include_heur(
            &o.name,
            &o.description,
            o.priority,
            'O',
            o.frequency,
            0,
            o.max_depth,
            o.timing,
            false,
            Box::new(a),
        );
        Ok(())
    }
    /// Register a custom constraint handler.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn constraint_handler(
        &mut self,
        o: ScipPluginOptions,
        p: impl ScipConstraintHandler + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        let a = self.adapter(Box::new(p) as Box<dyn ScipConstraintHandler>, EventMask::DISABLED);
        self.model.include_conshdlr(&o.name, &o.description, o.priority, o.priority, Box::new(a));
        Ok(())
    }
    /// Register a custom node selector.
    ///
    /// # Errors
    ///
    /// Rejects invalid data, unknown entities, or incompatible plugin configuration.
    /// Native failures are reported by the enclosing solve.
    pub fn node_selector(
        &mut self,
        o: ScipPluginOptions,
        p: impl ScipNodeSelector + 'static,
    ) -> Result<(), SolverError> {
        self.validate(&o)?;
        let a = self.adapter(Box::new(p) as Box<dyn ScipNodeSelector>, EventMask::DISABLED);
        self.model.include_nodesel(&o.name, &o.description, o.priority, o.priority, Box::new(a));
        Ok(())
    }
}

// Capture prices only while the LP is known to be optimal, not after SCIP has
// destroyed its LP state. No plugin state owns any native reference.
struct Prices {
    map: Map,
    shared: SharedState,
}
impl russcip::Eventhdlr for Prices {
    fn get_type(&self) -> EventMask {
        EventMask::FIRST_LP_SOLVED | EventMask::LP_SOLVED
    }
    fn execute(&mut self, model: Model<Solving>, _: russcip::SCIPEventhdlr, _: russcip::Event) {
        guarded(
            &self.shared,
            || (),
            || {
                if model.lp_status() != russcip::LPStatus::Optimal {
                    return Ok(());
                }
                let (dual, reduced, values, lp_objective) = {
                    let shared = self.shared.borrow();
                    let map = shared.price_map.as_ref().unwrap_or(&self.map);
                    let ctx = ScipContext::new(model, map, self.shared.clone());
                    let mut dual = FxHashMap::default();
                    let mut reduced = FxHashMap::default();
                    let mut values = FxHashMap::default();
                    for i in 0..map.vars.len() {
                        let id = VarId(u32::try_from(i).map_err(crate::backend)?);
                        if let Some(v) = ctx.variable(id) {
                            values.insert(id, ctx.model.current_val(&v));
                            if let Some(t) = v.transformed()
                                && let Some(rc) = t.redcost().filter(|v| v.is_finite())
                            {
                                reduced.insert(id, rc);
                            }
                        }
                    }
                    for (i, _) in map.constraints.iter().enumerate() {
                        let id = crate::translate::id(i);
                        if map.linear[i]
                            && let Some(v) = ctx
                                .constraint(id)
                                .and_then(|c| c.dual_sol())
                                .filter(|v| v.is_finite())
                        {
                            dual.insert(id, v);
                        }
                    }
                    (dual, reduced, values, ctx.model.lp_obj_val())
                };
                let mut shared = self.shared.borrow_mut();
                shared.dual = dual;
                shared.reduced = reduced;
                shared.lp_values = values;
                shared.lp_objective = Some(lp_objective);
                Ok(())
            },
        );
    }
}
pub(crate) fn install_prices(
    model: &mut Model<ProblemCreated>,
    mapping: &ScipMapping,
    shared: SharedState,
) {
    model.include_eventhdlr(
        "oximo_lp_prices",
        "original LP dual capture",
        Box::new(Prices { map: Map::from(mapping), shared }),
    );
}
pub(crate) fn refresh_prices(mapping: &ScipMapping, shared: &SharedState) {
    shared.borrow_mut().price_map = Some(Map::from(mapping));
}
