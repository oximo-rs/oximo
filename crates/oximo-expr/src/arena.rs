use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, MutexGuard};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use thiserror::Error;

use crate::classify::{Degree, ExprClassCache};

static NEXT_MODEL_ID: AtomicU32 = AtomicU32::new(1);

/// Immutable parameter values captured from an expression arena.
#[derive(Clone, Debug, Default)]
pub struct ParameterSnapshot(Arc<Vec<f64>>);

impl PartialEq for ParameterSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.same_bits(other)
    }
}

impl Eq for ParameterSnapshot {}

impl ParameterSnapshot {
    /// Saved value, or `None` when the parameter was absent at capture time.
    pub fn value(&self, id: ParamId) -> Option<f64> {
        self.0.get(id.index()).copied()
    }

    /// Saved values in parameter-ID order.
    pub fn values(&self) -> &[f64] {
        &self.0
    }

    /// Keep only parameter IDs below `count`, preserving exact value bits.
    /// Shares the existing storage when no values need to be removed.
    pub fn prefix(&self, count: usize) -> Self {
        if count >= self.0.len() { self.clone() } else { Self(Arc::new(self.0[..count].to_vec())) }
    }

    /// Compare exact parameter representations, including signed zero and NaN payloads.
    pub fn same_bits(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.len() == other.0.len()
                && self.0.iter().zip(other.0.iter()).all(|(a, b)| a.to_bits() == b.to_bits()))
    }

    /// Parameter IDs whose exact values differ between two snapshots.
    ///
    /// A parameter present in only one snapshot is reported as changed. The
    /// result is ordered by parameter ID. This compares values only.
    ///
    /// # Panics
    /// Panics if a snapshot contains more parameters than a [`ParamId`] can
    /// address.
    pub fn changed_parameters(&self, newer: &Self) -> Vec<ParamId> {
        if Arc::ptr_eq(&self.0, &newer.0) {
            return Vec::new();
        }

        let len = self.0.len().max(newer.0.len());
        let mut changed = Vec::new();

        for index in 0..len {
            let before = self.0.get(index).map(|value| value.to_bits());
            let after = newer.0.get(index).map(|value| value.to_bits());
            if before != after {
                changed.push(ParamId(u32::try_from(index).expect("parameter count fits u32")));
            }
        }

        changed
    }
}

/// Stable identity shared by a model, its expression handles, and its results.
///
/// IDs are unique within a process. A transformed or cloned model receives a new
/// identity even when it preserves the source model's numeric variable IDs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelId(u32);

impl ModelId {
    /// Sentinel used by detached/default result values that have not been
    /// associated with a model.
    pub const UNASSIGNED: Self = Self(0);

    fn fresh() -> Self {
        let mut id = NEXT_MODEL_ID.load(Ordering::Relaxed);
        loop {
            let next = id.checked_add(1).expect("model ID space exhausted");
            match NEXT_MODEL_ID.compare_exchange_weak(
                id,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Self(id),
                Err(actual) => id = actual,
            }
        }
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// An expression, indexed family, model, or result came from another model.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
#[error("model mismatch: expected model {expected}, got model {actual}")]
pub struct ModelMismatchError {
    pub expected: ModelId,
    pub actual: ModelId,
}

impl ModelMismatchError {
    #[must_use]
    pub const fn new(expected: ModelId, actual: ModelId) -> Self {
        Self { expected, actual }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

impl ExprId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Result of building a node while holding the arena's construction lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConstructionResult {
    pub(crate) id: ExprId,
    pub(crate) reusable: bool,
}

impl ConstructionResult {
    pub(crate) fn from_arena(arena: &mut (impl ArenaAccess + ?Sized), id: ExprId) -> Self {
        Self { id, reusable: arena.reusable_root(id) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

impl VarId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(pub u32);

impl ParamId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

pub type Children = SmallVec<[ExprId; 4]>;

/// Unary scalar functions represented by [`ExprNode::Unary`].
///
/// This enum is public so expression consumers can exhaustively describe the
/// nonlinear vocabulary they support. Adding a variant is therefore a
/// deliberate source-breaking extension for exhaustive matches.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Cbrt,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
}

impl UnaryOp {
    /// Canonical modeling-language spelling used in rendering and diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Cbrt => "cbrt",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Expm1 => "expm1",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
        }
    }

    /// Apply this operation with Rust's IEEE-754 `f64` semantics.
    #[must_use]
    #[inline]
    pub fn apply(self, value: f64) -> f64 {
        match self {
            Self::Neg => -value,
            Self::Abs => value.abs(),
            Self::Sqrt => value.sqrt(),
            Self::Cbrt => value.cbrt(),
            Self::Exp => value.exp(),
            Self::Exp2 => value.exp2(),
            Self::Expm1 => value.exp_m1(),
            Self::Log => value.ln(),
            Self::Log2 => value.log2(),
            Self::Log10 => value.log10(),
            Self::Log1p => value.ln_1p(),
            Self::Sin => value.sin(),
            Self::Cos => value.cos(),
            Self::Tan => value.tan(),
            Self::Asin => value.asin(),
            Self::Acos => value.acos(),
            Self::Atan => value.atan(),
            Self::Sinh => value.sinh(),
            Self::Cosh => value.cosh(),
            Self::Tanh => value.tanh(),
            Self::Asinh => value.asinh(),
            Self::Acosh => value.acosh(),
            Self::Atanh => value.atanh(),
        }
    }

    /// Whether this operation is nonsmooth on its ordinary real domain.
    #[must_use]
    pub const fn is_nonsmooth(self) -> bool {
        matches!(self, Self::Abs)
    }
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Small numeric affine operations use a `Linear` fast path.
/// Large operations retain immutable `Add`/`Mul` nodes to avoid copying
/// coefficient prefixes.
/// `AffineBuilder` can emit a single `Linear` node for a large numeric sum.

#[derive(Clone, Debug)]
pub enum ExprNode {
    Const(f64),
    Var(VarId),
    Param(ParamId),
    Add(Children),
    Mul(Children),
    /// Unary scalar operation.
    Unary(UnaryOp, ExprId),
    Pow(ExprId, ExprId),
    Div(ExprId, ExprId),
    /// Two-argument arctangent in `y.atan2(x)` order.
    Atan2(ExprId, ExprId),
    /// N-ary minimum.
    Min(Children),
    /// N-ary maximum.
    Max(Children),
    Linear {
        coeffs: Vec<(VarId, f64)>,
        constant: f64,
    },
    /// Immutable numeric polynomial coefficients.
    /// A diagonal coefficient `c` represents `c * x * x`.
    Quadratic(Arc<crate::QuadraticTerms>),
}

#[derive(Debug)]
enum AffineOwnedIds {
    Inline(SmallVec<[ExprId; 4]>),
    Hashed(FxHashSet<ExprId>),
}

impl Default for AffineOwnedIds {
    fn default() -> Self {
        Self::Inline(SmallVec::new())
    }
}

impl AffineOwnedIds {
    fn contains(&self, id: ExprId) -> bool {
        match self {
            Self::Inline(ids) => ids.contains(&id),
            Self::Hashed(ids) => ids.contains(&id),
        }
    }

    fn insert(&mut self, id: ExprId) -> bool {
        match self {
            Self::Inline(ids) => {
                if ids.contains(&id) {
                    return false;
                }
                if ids.len() < ids.inline_size() {
                    ids.push(id);
                    return true;
                }

                let mut hashed =
                    FxHashSet::with_capacity_and_hasher(ids.len() + 1, rustc_hash::FxBuildHasher);
                hashed.extend(ids.iter().copied());
                hashed.insert(id);
                *self = Self::Hashed(hashed);
                true
            }

            Self::Hashed(ids) => ids.insert(id),
        }
    }

    fn remove(&mut self, id: ExprId) -> bool {
        match self {
            Self::Inline(ids) => {
                let Some(index) = ids.iter().position(|owned| *owned == id) else {
                    return false;
                };
                ids.swap_remove(index);
                true
            }
            Self::Hashed(ids) => ids.remove(&id),
        }
    }

    fn clear(&mut self) {
        // Publication resets the common case and releases any hash-table allocation.
        *self = Self::default();
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        match self {
            Self::Inline(ids) => ids.len(),
            Self::Hashed(ids) => ids.len(),
        }
    }

    #[cfg(test)]
    fn spilled(&self) -> bool {
        match self {
            Self::Inline(ids) => ids.spilled(),
            Self::Hashed(_) => true,
        }
    }
}

#[derive(Debug, Default)]
struct RootOwnership {
    affine_owned: AffineOwnedIds,
    quadratic_pending: bool,
    oversized_affine: FxHashSet<ExprId>,
    #[cfg(test)]
    child_inspections: usize,
    prefix: usize,
    sealed: FxHashSet<ExprId>,
    slots: FxHashMap<ExprId, crate::quadratic_builder::QuadraticSlots>,
}

const MAX_OVERSIZED_AFFINE_ROOTS: usize = 64;

impl RootOwnership {
    fn needs_child_sealing(&self) -> bool {
        self.quadratic_pending || !self.affine_owned.is_empty()
    }

    fn seal(&mut self, id: ExprId) {
        if !self.contains(id) {
            self.sealed.insert(id);
            self.slots.remove(&id);
        }
    }

    fn contains(&self, id: ExprId) -> bool {
        id.index() < self.prefix || self.sealed.contains(&id)
    }

    fn slots(&mut self, id: ExprId) -> &mut crate::quadratic_builder::QuadraticSlots {
        self.slots.entry(id).or_default()
    }

    fn mark_oversized_affine(&mut self, id: ExprId) {
        if self.oversized_affine.contains(&id) {
            return;
        }
        if self.oversized_affine.len() == MAX_OVERSIZED_AFFINE_ROOTS {
            self.oversized_affine.clear();
        }
        self.oversized_affine.insert(id);
    }

    fn finish_affine_merge(&mut self, lhs: ExprId, rhs: ExprId, destination: ExprId, linear: bool) {
        // Non-destination inputs have been consumed, or were already sealed by
        // a borrowed operand.
        if lhs != destination {
            self.affine_owned.remove(lhs);
        }

        if rhs != destination {
            self.affine_owned.remove(rhs);
        }

        if !linear {
            self.affine_owned.remove(destination);
        }
    }

    fn seal_all(&mut self, prefix: usize) {
        self.affine_owned.clear();
        self.quadratic_pending = false;
        self.oversized_affine.clear();
        self.prefix = prefix;
        self.sealed.clear();
        self.slots.clear();
    }
}

#[derive(Debug, Default)]
pub struct ExprArena {
    nodes: Arc<Vec<ExprNode>>,
    ownership: OnceLock<Box<Mutex<RootOwnership>>>,
    param_values: Arc<Vec<f64>>,
    parameter_revision: u64,
    locked_parameters: Arc<FxHashSet<ParamId>>,
    bound_checks: Arc<FxHashMap<VarId, crate::model_error::BoundCheck>>,
    bound_dependencies: Arc<FxHashMap<ParamId, FxHashSet<VarId>>>,
    lock_sources: Arc<FxHashMap<ParamId, String>>,
    classifications: Mutex<Option<Box<ExprClassCache>>>,
    analysis: OnceLock<Box<Mutex<crate::ExpressionAnalysis>>>,
    #[cfg(test)]
    classification_counters: crate::classify::ClassificationCounters,
}

fn checked_id_range(start: usize, count: usize, kind: &str) -> std::ops::Range<usize> {
    let end = start.checked_add(count).unwrap_or_else(|| panic!("{kind} overflow"));

    if count != 0 {
        u32::try_from(end - 1).unwrap_or_else(|_| panic!("{kind} overflow"));
    }
    start..end
}

// The cache holds derived values only, never calls user code while locked, and
// cannot leave the expression nodes partially mutated when unwinding.
impl std::panic::RefUnwindSafe for ExprArena {}

impl Clone for ExprArena {
    fn clone(&self) -> Self {
        self.seal_all();
        Self {
            ownership: OnceLock::new(),
            nodes: Arc::clone(&self.nodes),
            param_values: Arc::clone(&self.param_values),
            parameter_revision: self.parameter_revision,
            locked_parameters: Arc::clone(&self.locked_parameters),
            bound_checks: Arc::clone(&self.bound_checks),
            bound_dependencies: Arc::clone(&self.bound_dependencies),
            lock_sources: Arc::clone(&self.lock_sources),
            classifications: Mutex::default(),
            analysis: OnceLock::new(),
            #[cfg(test)]
            classification_counters: self.cloned_classification_counters(),
        }
    }
}

impl ExprArena {
    /// Retain exact symbolic degrees for at most 131,072 roots in this arena.
    #[doc(hidden)]
    pub fn __enable_preparation_classifications(&self) {
        *self.classifications.lock() = Some(Box::new(ExprClassCache::for_preparation()));
        self.analysis.get_or_init(|| Box::new(Mutex::default()));
    }

    /// Snapshot-local structural facts.
    #[doc(hidden)]
    pub fn __preparation_expression_summary(
        &self,
        root: ExprId,
    ) -> Option<crate::ExpressionSummary> {
        self.analysis.get().map(|analysis| {
            crate::ExpressionAnalysis::leaf(self, root)
                .unwrap_or_else(|| analysis.lock().summarize(self, root))
        })
    }

    #[doc(hidden)]
    pub fn __cached_expression_summary(&self, root: ExprId) -> Option<crate::ExpressionSummary> {
        self.analysis.get().and_then(|analysis| {
            crate::ExpressionAnalysis::leaf(self, root)
                .or_else(|| analysis.lock().known(self, root))
        })
    }

    #[cfg(test)]
    fn cloned_classification_counters(&self) -> crate::classify::ClassificationCounters {
        let counters = crate::classify::ClassificationCounters::default();
        counters
            .enabled
            .store(self.classification_counters.enabled.load(Ordering::Relaxed), Ordering::Relaxed);
        counters
    }

    /// Enable counters for this arena and later snapshots.
    #[cfg(test)]
    pub fn enable_classification_counters(&self) {
        self.classification_counters.enabled.store(true, Ordering::Relaxed);
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn __use_small_classification_cache(&self) {
        *self.classifications.lock() = None;
    }

    #[cfg(test)]
    pub fn classification_stats(&self) -> crate::classify::ClassificationStats {
        let counters = &self.classification_counters;
        crate::classify::ClassificationStats {
            calls: counters.calls.load(Ordering::Relaxed),
            cache_misses: counters.misses.load(Ordering::Relaxed),
            visited_nodes: counters.visits.load(Ordering::Relaxed),
            retained_roots: self.classifications.lock().as_deref().map_or(0, ExprClassCache::len),
        }
    }

    /// Permanently publish every node currently in the arena.
    /// A raw [`Self::push`] publishes each quadratic root immediately, while a
    /// raw linear root is simply left untracked and therefore not reusable.
    /// Internal builder paths may retain ownership of fresh quadratic roots via
    /// [`ArenaAccess::push`] and affine roots via `push_owned_linear`.
    pub fn seal_all(&self) {
        if let Some(ownership) = self.ownership.get() {
            ownership.lock().seal_all(self.len());
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Arc::new(Vec::with_capacity(cap)), ..Self::default() }
    }

    /// Clone this arena while reserving room for nodes that will immediately
    /// be appended to the copy.
    #[doc(hidden)]
    #[must_use]
    pub fn __clone_with_additional_capacity(&self, additional_nodes: usize) -> Self {
        self.seal_all();
        let mut nodes = Vec::with_capacity(self.nodes.len().saturating_add(additional_nodes));
        nodes.extend_from_slice(&self.nodes);
        Self {
            nodes: Arc::new(nodes),
            ownership: OnceLock::new(),
            param_values: Arc::new(self.param_values.as_ref().clone()),
            parameter_revision: self.parameter_revision,
            locked_parameters: Arc::clone(&self.locked_parameters),
            bound_checks: Arc::clone(&self.bound_checks),
            bound_dependencies: Arc::clone(&self.bound_dependencies),
            lock_sources: Arc::clone(&self.lock_sources),
            classifications: Mutex::default(),
            analysis: OnceLock::new(),
            #[cfg(test)]
            classification_counters: self.cloned_classification_counters(),
        }
    }

    /// Consume a private arena without copying its expression nodes.
    #[doc(hidden)]
    pub fn __into_nodes(self) -> Vec<ExprNode> {
        Arc::unwrap_or_clone(self.nodes)
    }

    /// Reserve room for additional expression nodes without changing IDs.
    #[doc(hidden)]
    pub fn __reserve_nodes(&mut self, additional: usize) {
        Arc::make_mut(&mut self.nodes).reserve(additional);
    }

    /// Append a contiguous variable family, detaching node storage once.
    /// Returns the expression indices in variable order.
    ///
    /// # Panics
    /// Panics if the variable or expression indices exceed `u32::MAX`.
    #[doc(hidden)]
    pub fn __extend_vars(&mut self, variables: std::ops::Range<usize>) -> std::ops::Range<usize> {
        let expressions = checked_id_range(self.nodes.len(), variables.len(), "expression arena");
        checked_id_range(variables.start, variables.len(), "variable count");
        let nodes = Arc::make_mut(&mut self.nodes);
        nodes.extend(
            variables.map(|id| ExprNode::Var(VarId(u32::try_from(id).expect("checked ID")))),
        );
        expressions
    }

    /// Append parameter values and their expression nodes, detaching each buffer once.
    /// Returns parameter indices and expression indices in input order.
    ///
    /// # Panics
    /// Panics if the parameter or expression indices exceed `u32::MAX`.
    #[doc(hidden)]
    pub fn __extend_params(
        &mut self,
        values: impl ExactSizeIterator<Item = f64>,
    ) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        let parameters = checked_id_range(self.param_values.len(), values.len(), "parameter arena");
        let expressions = checked_id_range(self.nodes.len(), values.len(), "expression arena");
        let values: Vec<_> = values.collect();
        assert!(values.iter().all(|v| v.is_finite()), "parameter initial values must be finite");
        Arc::make_mut(&mut self.param_values).extend(values);
        Arc::make_mut(&mut self.nodes).extend(
            parameters
                .clone()
                .map(|id| ExprNode::Param(ParamId(u32::try_from(id).expect("checked ID")))),
        );
        (parameters, expressions)
    }

    #[doc(hidden)]
    pub(crate) fn __append_nodes(&mut self, nodes: Vec<ExprNode>) {
        Arc::make_mut(&mut self.nodes).extend(nodes);
        self.seal_all();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append a raw node. A raw quadratic root is immediately published and
    /// cannot be reused. Internal arithmetic uses `ArenaAccess::push` for fresh
    /// quadratic roots whose builders retain ownership, and
    /// `ArenaAccess::push_owned_linear` for fresh affine roots.
    ///
    /// # Panics
    ///
    /// Panics if the number of expressions exceeds `u32::MAX` (expression arena overflow).
    pub fn push(&mut self, node: ExprNode) -> ExprId {
        let id = self.push_internal(node, false);
        if let Some(ownership) = self.ownership.get_mut()
            && matches!(self.nodes[id.index()], ExprNode::Quadratic(_))
        {
            ownership.get_mut().seal(id);
        }
        id
    }

    fn push_internal(&mut self, node: ExprNode, owned_linear: bool) -> ExprId {
        if self.ownership.get_mut().is_none()
            && !owned_linear
            && !matches!(node, ExprNode::Quadratic(_))
        {
            let id = ExprId(u32::try_from(self.nodes.len()).expect("expression arena overflow"));
            Arc::make_mut(&mut self.nodes).push(node);
            return id;
        }
        let needs_child_sealing = self
            .ownership
            .get_mut()
            .is_some_and(|ownership| ownership.get_mut().needs_child_sealing());
        if needs_child_sealing {
            seal_children(&node, |id| {
                #[cfg(test)]
                {
                    self.ownership
                        .get_mut()
                        .expect("root ownership")
                        .get_mut()
                        .child_inspections += 1;
                }
                match self.nodes[id.index()] {
                    ExprNode::Linear { .. } => {
                        self.ownership
                            .get_mut()
                            .expect("root ownership")
                            .get_mut()
                            .affine_owned
                            .remove(id);
                    }
                    ExprNode::Quadratic(_) => {
                        self.ownership.get_mut().expect("root ownership").get_mut().seal(id);
                    }
                    _ => {}
                }
            });
        }
        let owned_linear = owned_linear && matches!(node, ExprNode::Linear { .. });
        let id = ExprId(u32::try_from(self.nodes.len()).expect("expression arena overflow"));
        if matches!(node, ExprNode::Quadratic(_)) || owned_linear {
            let prefix = ArenaAccess::len(self);
            self.ownership.get_or_init(|| {
                Box::new(Mutex::new(RootOwnership { prefix, ..RootOwnership::default() }))
            });
            let ownership = self.ownership.get_mut().expect("root ownership").get_mut();
            if owned_linear {
                ownership.affine_owned.insert(id);
            }
            if matches!(node, ExprNode::Quadratic(_)) {
                ownership.quadratic_pending = true;
            }
        }
        Arc::make_mut(&mut self.nodes).push(node);
        id
    }

    #[inline]
    /// Inspect a node without publishing it or changing ownership.
    /// Use [`Self::seal`] before retaining its identity across in-place updates.
    pub fn get(&self, id: ExprId) -> &ExprNode {
        self.node(id)
    }

    /// Permanently protect a node identity from in-place affine/quadratic updates.
    ///
    /// # Panics
    /// Panics if `id` is not a valid node in this arena.
    pub fn seal(&self, id: ExprId) {
        let _ = self.node(id);
        ArenaAccess::seal(self, id);
    }

    #[inline]
    pub(crate) fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, id: ExprId) -> &mut ExprNode {
        self.seal_all();
        // A changed child can invalidate any cached root. Appending nodes and
        // rebinding parameters preserve structural degree and need no reset.
        *self.classifications.get_mut() = None;
        self.analysis.take();
        &mut Arc::make_mut(&mut self.nodes)[id.index()]
    }

    /// Inspect all nodes without changing ownership.
    /// Use [`Self::seal_all`] before retaining all current node identities across consuming updates.
    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }

    pub fn constant(&mut self, v: f64) -> ExprId {
        self.push(ExprNode::Const(v))
    }

    pub fn var(&mut self, v: VarId) -> ExprId {
        self.push(ExprNode::Var(v))
    }

    pub fn param(&mut self, p: ParamId) -> ExprId {
        self.push(ExprNode::Param(p))
    }

    /// Allocate a fresh parameter initialized to `value`, returning its
    /// [`ParamId`]. Push a [`ExprNode::Param`] with [`Self::param`] to reference
    /// it inside an expression.
    ///
    /// # Panics
    ///
    /// Panics if the number of parameters exceeds `u32::MAX`.
    pub fn new_param(&mut self, value: f64) -> ParamId {
        assert!(value.is_finite(), "parameter initial value must be finite: {value}");
        let id = ParamId(u32::try_from(self.param_values.len()).expect("parameter arena overflow"));
        Arc::make_mut(&mut self.param_values).push(value);
        id
    }

    #[inline]
    pub fn num_params(&self) -> usize {
        self.param_values.len()
    }

    /// Revision for caches whose numerical assumptions depend on parameter values.
    pub fn parameter_revision(&self) -> u64 {
        self.parameter_revision
    }

    /// Capture parameter values without retaining the expression arena.
    pub fn parameter_snapshot(&self) -> ParameterSnapshot {
        ParameterSnapshot(Arc::clone(&self.param_values))
    }

    /// Current value bound to parameter `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` was not allocated by [`Self::new_param`] on this arena.
    #[inline]
    pub fn param_value(&self, p: ParamId) -> f64 {
        self.param_values[p.index()]
    }

    /// Look up the value of `p`, returning `None` if `p` is out of range.
    #[inline]
    pub fn try_param_value(&self, p: ParamId) -> Option<f64> {
        self.param_values.get(p.index()).copied()
    }

    /// Re-bind parameter `p` to `value`. Takes effect on the next extraction or
    /// evaluation.
    ///
    /// # Errors
    /// Rejects unknown, locked, non-finite, or bound-infeasible assignments.
    #[inline]
    pub fn set_param_value(&mut self, p: ParamId, value: f64) -> Result<(), crate::ModelError> {
        self.validate_single_param(p, value)?;

        // Fast path for the common single write.
        // a lone update cannot be duplicated, so it commits without
        // the batch maps that `validate_param_values` builds.
        if self.bound_dependencies.get(&p).is_some_and(|variables| !variables.is_empty()) {
            return self.set_param_values(&[(p, value)]);
        }
        self.__commit_param_values(&[(p, value)]);
        Ok(())
    }

    /// Validate one update's parameter identity, value, and lock state.
    #[inline]
    fn validate_single_param(&self, p: ParamId, value: f64) -> Result<(), crate::ModelError> {
        use crate::ModelError;
        if self.try_param_value(p).is_none() {
            return Err(ModelError::UnknownParam(p));
        }
        if !value.is_finite() {
            return Err(ModelError::ParamDomain { param: p, value });
        }
        if self.is_param_locked(p) {
            return Err(ModelError::FrozenByReformulation {
                param: p,
                source_name: self
                    .lock_sources
                    .get(&p)
                    .cloned()
                    .unwrap_or_else(|| "generated rows".into()),
            });
        }
        Ok(())
    }

    /// Validate a complete parameter batch, including dependent bound ordering,
    /// without applying it.
    ///
    /// # Errors
    /// Rejects the entire batch if a parameter is unknown, duplicated,
    /// non-finite, or frozen by a reformulation, or if a dependent variable
    /// bound is infeasible.
    pub fn validate_param_values(
        &self,
        updates: &[(ParamId, f64)],
    ) -> Result<(), crate::ModelError> {
        use crate::ModelError;
        let mut values = FxHashMap::default();
        for &(p, value) in updates {
            self.validate_single_param(p, value)?;
            if values.insert(p, value).is_some() {
                return Err(ModelError::DuplicateParam(p));
            }
        }
        let variables: FxHashSet<_> = updates
            .iter()
            .flat_map(|(p, _)| self.bound_dependencies.get(p).into_iter().flatten().copied())
            .collect();
        for variable in variables {
            let check = &self.bound_checks[&variable];
            let resolve = |b| match b {
                crate::Bound::Constant(v) => v,
                crate::Bound::Parameter(p) => {
                    values.get(&p).copied().unwrap_or_else(|| self.param_value(p))
                }
            };
            let (lower, upper) = (resolve(check.lower), resolve(check.upper));
            if lower.is_nan()
                || upper.is_nan()
                || lower > upper
                || lower == f64::INFINITY
                || upper == f64::NEG_INFINITY
            {
                return Err(ModelError::BoundsInfeasible {
                    variable: check.variable,
                    lower,
                    upper,
                });
            }
        }
        Ok(())
    }

    /// Validate and atomically accept a complete assignment.
    ///
    /// # Errors
    /// Rejects the entire batch if any parameter or dependent bound is invalid.
    pub fn set_param_values(
        &mut self,
        updates: &[(ParamId, f64)],
    ) -> Result<(), crate::ModelError> {
        self.validate_param_values(updates)?;
        self.__commit_param_values(updates);
        Ok(())
    }

    /// Commit a previously validated batch without a second validation pass.
    /// Callers must have successfully called `validate_param_values` on this arena
    /// with this exact batch, with no intervening parameter or bound changes.
    ///
    /// This is intended for transactional consumers with independently owned arenas.
    #[doc(hidden)]
    pub fn __commit_param_values(&mut self, updates: &[(ParamId, f64)]) {
        if updates.iter().any(|&(p, v)| self.param_value(p).to_bits() != v.to_bits()) {
            self.parameter_revision = self.parameter_revision.wrapping_add(1);
            let accepted = Arc::make_mut(&mut self.param_values);
            for &(p, v) in updates {
                debug_assert!(v.is_finite());
                accepted[p.index()] = v;
            }
        }
    }

    /// Register or replace a parameter-bound validation record.
    #[doc(hidden)]
    pub fn __set_bound_check(&mut self, variable: VarId, lower: crate::Bound, upper: crate::Bound) {
        let checks = Arc::make_mut(&mut self.bound_checks);
        let dependencies = Arc::make_mut(&mut self.bound_dependencies);
        if let Some(previous) = checks.remove(&variable) {
            for p in [previous.lower, previous.upper].iter().filter_map(|b| b.parameter()) {
                if let Some(vars) = dependencies.get_mut(&p) {
                    vars.remove(&variable);
                }
            }
        }
        if lower.parameter().is_some() || upper.parameter().is_some() {
            checks.insert(variable, crate::model_error::BoundCheck { variable, lower, upper });
            for p in [lower, upper].iter().filter_map(|b| b.parameter()) {
                dependencies.entry(p).or_default().insert(variable);
            }
        }
    }

    /// Lock a bound parameter together with the source declaration for diagnostics.
    #[doc(hidden)]
    pub fn __lock_bound_parameter(&mut self, parameter: ParamId, source: String) {
        Arc::make_mut(&mut self.locked_parameters).insert(parameter);
        Arc::make_mut(&mut self.lock_sources).insert(parameter, source);
    }

    /// Whether a reformulation has fixed this parameter's value.
    pub fn is_param_locked(&self, parameter: ParamId) -> bool {
        self.locked_parameters.contains(&parameter)
    }

    /// Preserve parameter values embedded in generated reformulation rows.
    #[doc(hidden)]
    pub fn __lock_parameters(&mut self, parameters: impl IntoIterator<Item = ParamId>) {
        Arc::make_mut(&mut self.locked_parameters).extend(parameters);
    }

    pub fn linear(&mut self, coeffs: Vec<(VarId, f64)>, constant: f64) -> ExprId {
        self.push(ExprNode::Linear { coeffs, constant })
    }
}

/// Read/push surface shared by the canonical and worker-local arenas.
pub(crate) trait ArenaAccess {
    #[cfg(test)]
    fn record_classification(&self, _miss: bool) {}
    #[cfg(test)]
    fn record_classification_visit(&self) {}
    fn len(&self) -> usize;
    fn get(&self, id: ExprId) -> &ExprNode;
    fn quadratic_root(&mut self, id: ExprId) -> bool;
    fn linear_root(&mut self, _id: ExprId) -> bool {
        false
    }
    fn reusable_root(&mut self, id: ExprId) -> bool {
        self.linear_root(id) || self.quadratic_root(id)
    }
    fn accumulate_linear(
        &mut self,
        _lhs: ExprId,
        _rhs: ExprId,
        _destination: ExprId,
    ) -> Option<ExprId> {
        None
    }
    /// Opt a normalized, small numeric Linear node into consuming addition reuse.
    fn push_owned_linear(&mut self, node: ExprNode) -> ExprId {
        self.push(node)
    }
    /// Whether this Add root already failed the bounded affine probe.
    fn oversized_affine(&self, _id: ExprId) -> bool {
        false
    }
    /// Record an immutable Add root that can never become a small affine root.
    fn mark_oversized_affine(&mut self, _id: ExprId) {}
    /// Retire a consumed affine root copied into a fresh result.
    fn retire_affine_root(&mut self, _id: ExprId) {}
    fn seal(&self, id: ExprId);
    fn accumulate_quadratic(&mut self, lhs: ExprId, rhs: ExprId, scale: f64) -> Option<ExprId>;
    fn scale_quadratic(&mut self, id: ExprId, scale: f64) -> Option<ExprId>;
    fn param_value(&self, id: ParamId) -> f64;
    fn push(&mut self, node: ExprNode) -> ExprId;
    fn push_immutable(&mut self, node: ExprNode) -> ExprId {
        self.push(node)
    }
    fn cached_degree(&self, id: ExprId) -> Option<Degree>;
    fn cache_degree(&self, id: ExprId, class: Degree);
    fn preparation_degree(&self, _id: ExprId) -> Option<Degree> {
        None
    }

    fn constant(&mut self, value: f64) -> ExprId {
        self.push_immutable(ExprNode::Const(value))
    }

    fn var(&mut self, id: VarId) -> ExprId {
        self.push_immutable(ExprNode::Var(id))
    }
}

impl ArenaAccess for ExprArena {
    fn reusable_root(&mut self, id: ExprId) -> bool {
        let Some(ownership) = self.ownership.get_mut() else {
            return false;
        };
        let ownership = ownership.get_mut();
        match self.nodes[id.index()] {
            ExprNode::Linear { .. } => ownership.affine_owned.contains(id),
            ExprNode::Quadratic(_) => !ownership.contains(id),
            _ => false,
        }
    }

    fn linear_root(&mut self, id: ExprId) -> bool {
        // Membership is authoritative.
        self.ownership.get_mut().is_some_and(|o| o.get_mut().affine_owned.contains(id))
    }

    fn accumulate_linear(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        destination: ExprId,
    ) -> Option<ExprId> {
        if (destination != lhs && destination != rhs) || !self.linear_root(destination) {
            return None;
        }

        let nodes = Arc::get_mut(&mut self.nodes)?;
        let result = crate::linear::accumulate_linear_nodes(
            &nodes[lhs.index()],
            &nodes[rhs.index()],
            destination == lhs,
        )?;

        result.write_into(&mut nodes[destination.index()]);
        let linear = matches!(nodes[destination.index()], ExprNode::Linear { .. });
        self.ownership.get_mut()?.get_mut().finish_affine_merge(lhs, rhs, destination, linear);
        *self.classifications.get_mut() = None;
        self.analysis.take();

        Some(destination)
    }

    fn push_owned_linear(&mut self, node: ExprNode) -> ExprId {
        self.push_internal(node, true)
    }

    fn oversized_affine(&self, id: ExprId) -> bool {
        let Some(ownership) = self.ownership.get() else { return false };
        let ownership = ownership.lock();
        !ownership.oversized_affine.is_empty() && ownership.oversized_affine.contains(&id)
    }

    fn mark_oversized_affine(&mut self, id: ExprId) {
        let prefix = ArenaAccess::len(self);
        self.ownership.get_or_init(|| {
            Box::new(Mutex::new(RootOwnership { prefix, ..RootOwnership::default() }))
        });
        self.ownership.get_mut().expect("root ownership").get_mut().mark_oversized_affine(id);
    }

    fn retire_affine_root(&mut self, id: ExprId) {
        if let Some(ownership) = self.ownership.get_mut() {
            ownership.get_mut().affine_owned.remove(id);
        }
    }

    fn quadratic_root(&mut self, id: ExprId) -> bool {
        self.ownership.get_mut().is_some_and(|o| !o.get_mut().contains(id))
            && matches!(self.node(id), ExprNode::Quadratic(_))
    }
    fn preparation_degree(&self, id: ExprId) -> Option<Degree> {
        self.analysis.get()?;
        let mut budget = 64;
        crate::classify::degree_hint(self, id, &mut budget, 0)
    }

    #[cfg(test)]
    fn record_classification(&self, miss: bool) {
        let counters = &self.classification_counters;
        if counters.enabled.load(Ordering::Relaxed) {
            counters.calls.fetch_add(1, Ordering::Relaxed);
            if miss {
                counters.misses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[cfg(test)]
    fn record_classification_visit(&self) {
        let counters = &self.classification_counters;
        if counters.enabled.load(Ordering::Relaxed) {
            counters.visits.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn cached_degree(&self, id: ExprId) -> Option<Degree> {
        self.classifications.lock().as_deref_mut().and_then(|cache| cache.get(id))
    }

    fn cache_degree(&self, id: ExprId, class: Degree) {
        self.classifications.lock().get_or_insert_with(Box::default).insert(id, class);
    }

    fn seal(&self, id: ExprId) {
        if let Some(ownership) = self.ownership.get() {
            match self.nodes[id.index()] {
                ExprNode::Linear { .. } => {
                    ownership.lock().affine_owned.remove(id);
                }
                ExprNode::Quadratic(_) => ownership.lock().seal(id),
                _ => {}
            }
        }
    }
    fn accumulate_quadratic(&mut self, lhs: ExprId, rhs: ExprId, scale: f64) -> Option<ExprId> {
        if self.ownership.get_mut()?.get_mut().contains(lhs)
            || !matches!(self.nodes[lhs.index()], ExprNode::Quadratic(_))
        {
            return None;
        }
        crate::quadratic_builder::accumulate_nodes(
            Arc::get_mut(&mut self.nodes)?,
            lhs.index(),
            rhs.index(),
            self.ownership.get_mut()?.get_mut().slots(lhs),
            scale,
        )?;
        *self.classifications.get_mut() = None;
        self.analysis.take();
        Some(lhs)
    }
    fn scale_quadratic(&mut self, id: ExprId, scale: f64) -> Option<ExprId> {
        if self.ownership.get_mut()?.get_mut().contains(id) {
            return None;
        }
        let ExprNode::Quadratic(q) = &mut Arc::get_mut(&mut self.nodes)?[id.index()] else {
            return None;
        };
        crate::quadratic_builder::scale_unique(
            Arc::get_mut(q)?,
            self.ownership.get_mut()?.get_mut().slots(id),
            scale,
        )?;
        *self.classifications.get_mut() = None;
        self.analysis.take();
        Some(id)
    }
    fn get(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.index()]
    }

    fn param_value(&self, id: ParamId) -> f64 {
        self.param_value(id)
    }

    fn push(&mut self, node: ExprNode) -> ExprId {
        self.push_internal(node, false)
    }

    fn push_immutable(&mut self, node: ExprNode) -> ExprId {
        debug_assert!(!matches!(node, ExprNode::Quadratic(_)));
        let needs_child_sealing = self
            .ownership
            .get_mut()
            .is_some_and(|ownership| ownership.get_mut().needs_child_sealing());
        if !needs_child_sealing {
            let id = ExprId(u32::try_from(self.nodes.len()).expect("expression arena overflow"));
            Arc::make_mut(&mut self.nodes).push(node);
            return id;
        }
        if !matches!(
            node,
            ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. }
        ) {
            return self.push_internal(node, false);
        }
        let id = ExprId(u32::try_from(self.nodes.len()).expect("expression arena overflow"));
        Arc::make_mut(&mut self.nodes).push(node);
        id
    }
}

#[derive(Clone, Debug, Default)]
#[doc(hidden)]
pub struct FrozenExprArena {
    nodes: Arc<Vec<ExprNode>>,
    param_values: Arc<Vec<f64>>,
    classifications: ExprClassCache,
}

impl FrozenExprArena {
    fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug)]
struct ForkedExprArena {
    ownership: OnceLock<Box<Mutex<RootOwnership>>>,
    base: FrozenExprArena,
    nodes: Vec<ExprNode>,
    classifications: RefCell<ExprClassCache>,
}

impl ForkedExprArena {
    fn new(base: FrozenExprArena) -> Self {
        let classifications = RefCell::new(base.classifications.clone());
        Self { ownership: OnceLock::new(), base, nodes: Vec::new(), classifications }
    }

    fn push_internal(&mut self, node: ExprNode, owned_linear: bool) -> ExprId {
        let needs_child_sealing = self
            .ownership
            .get_mut()
            .is_some_and(|ownership| ownership.get_mut().needs_child_sealing());
        if needs_child_sealing {
            seal_children(&node, |id| {
                #[cfg(test)]
                {
                    self.ownership
                        .get_mut()
                        .expect("root ownership")
                        .get_mut()
                        .child_inspections += 1;
                }
                match self.get(id) {
                    ExprNode::Linear { .. } => {
                        self.ownership
                            .get_mut()
                            .expect("root ownership")
                            .get_mut()
                            .affine_owned
                            .remove(id);
                    }
                    ExprNode::Quadratic(_) => {
                        self.ownership.get_mut().expect("root ownership").get_mut().seal(id);
                    }
                    _ => {}
                }
            });
        }

        let owned_linear = owned_linear && matches!(node, ExprNode::Linear { .. });
        let index =
            self.base.len().checked_add(self.nodes.len()).expect("expression arena overflow");
        let id = ExprId(u32::try_from(index).expect("expression arena overflow"));
        if matches!(node, ExprNode::Quadratic(_)) || owned_linear {
            let prefix = ArenaAccess::len(self);
            self.ownership.get_or_init(|| {
                Box::new(Mutex::new(RootOwnership { prefix, ..RootOwnership::default() }))
            });
            let ownership = self.ownership.get_mut().expect("root ownership").get_mut();
            if owned_linear {
                ownership.affine_owned.insert(id);
            }
            if matches!(node, ExprNode::Quadratic(_)) {
                ownership.quadratic_pending = true;
            }
        }

        self.nodes.push(node);
        id
    }
}

impl ArenaAccess for ForkedExprArena {
    fn reusable_root(&mut self, id: ExprId) -> bool {
        let Some(ownership) = self.ownership.get_mut() else {
            return false;
        };
        let Some(index) = id.index().checked_sub(self.base.len()) else {
            return false;
        };
        let ownership = ownership.get_mut();
        match self.nodes[index] {
            ExprNode::Linear { .. } => ownership.affine_owned.contains(id),
            ExprNode::Quadratic(_) => !ownership.contains(id),
            _ => false,
        }
    }

    fn linear_root(&mut self, id: ExprId) -> bool {
        // See `ExprArena::linear_root`.
        // Ownership membership already implies a live `Linear` node,
        // so no structural re-check is needed.
        id.index() >= self.base.len()
            && self.ownership.get_mut().is_some_and(|o| o.get_mut().affine_owned.contains(id))
    }

    fn accumulate_linear(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        destination: ExprId,
    ) -> Option<ExprId> {
        let index = destination.index().checked_sub(self.base.len())?;
        if (destination != lhs && destination != rhs) || !self.linear_root(destination) {
            return None;
        }
        let result = crate::linear::accumulate_linear_nodes(
            self.get(lhs),
            self.get(rhs),
            destination == lhs,
        )?;

        result.write_into(&mut self.nodes[index]);
        let linear = matches!(self.nodes[index], ExprNode::Linear { .. });
        self.ownership.get_mut()?.get_mut().finish_affine_merge(lhs, rhs, destination, linear);
        *self.classifications.get_mut() = ExprClassCache::default();

        Some(destination)
    }

    fn push_owned_linear(&mut self, node: ExprNode) -> ExprId {
        self.push_internal(node, true)
    }

    fn oversized_affine(&self, id: ExprId) -> bool {
        let Some(ownership) = self.ownership.get() else { return false };
        let ownership = ownership.lock();

        !ownership.oversized_affine.is_empty() && ownership.oversized_affine.contains(&id)
    }

    fn mark_oversized_affine(&mut self, id: ExprId) {
        let prefix = ArenaAccess::len(self);
        self.ownership.get_or_init(|| {
            Box::new(Mutex::new(RootOwnership { prefix, ..RootOwnership::default() }))
        });
        self.ownership.get_mut().expect("root ownership").get_mut().mark_oversized_affine(id);
    }

    fn retire_affine_root(&mut self, id: ExprId) {
        if let Some(ownership) = self.ownership.get_mut() {
            ownership.get_mut().affine_owned.remove(id);
        }
    }

    fn push_immutable(&mut self, node: ExprNode) -> ExprId {
        debug_assert!(!matches!(node, ExprNode::Quadratic(_)));
        if !matches!(
            node,
            ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. }
        ) {
            return self.push_internal(node, false);
        }

        let index =
            self.base.len().checked_add(self.nodes.len()).expect("expression arena overflow");
        let id = ExprId(u32::try_from(index).expect("expression arena overflow"));

        self.nodes.push(node);
        id
    }
    fn quadratic_root(&mut self, id: ExprId) -> bool {
        self.ownership.get_mut().is_some_and(|o| !o.get_mut().contains(id))
            && matches!(self.get(id), ExprNode::Quadratic(_))
    }
    fn seal(&self, id: ExprId) {
        if let Some(ownership) = self.ownership.get() {
            match self.get(id) {
                ExprNode::Linear { .. } => {
                    ownership.lock().affine_owned.remove(id);
                }
                ExprNode::Quadratic(_) => ownership.lock().seal(id),
                _ => {}
            }
        }
    }
    fn accumulate_quadratic(&mut self, lhs: ExprId, rhs: ExprId, scale: f64) -> Option<ExprId> {
        if self.ownership.get_mut()?.get_mut().contains(lhs)
            || !matches!(self.get(lhs), ExprNode::Quadratic(_))
        {
            return None;
        }
        let index = lhs.index().checked_sub(self.base.len())?;
        let slots = self.ownership.get_mut()?.get_mut().slots(lhs);
        if let Some(right) = rhs.index().checked_sub(self.base.len()) {
            crate::quadratic_builder::accumulate_nodes(
                &mut self.nodes,
                index,
                right,
                slots,
                scale,
            )?;
        } else {
            let ExprNode::Quadratic(left) = &mut self.nodes[index] else {
                return None;
            };
            crate::quadratic_builder::merge_unique(
                Arc::get_mut(left)?,
                slots,
                crate::quadratic_builder::numeric_leaf(&self.base.nodes[rhs.index()])?,
                scale,
            )?;
        }
        *self.classifications.get_mut() = ExprClassCache::default();
        Some(lhs)
    }

    fn scale_quadratic(&mut self, id: ExprId, scale: f64) -> Option<ExprId> {
        if self.ownership.get_mut()?.get_mut().contains(id) {
            return None;
        }
        let index = id.index().checked_sub(self.base.len())?;
        let ExprNode::Quadratic(q) = &mut self.nodes[index] else {
            return None;
        };
        crate::quadratic_builder::scale_unique(
            Arc::get_mut(q)?,
            self.ownership.get_mut()?.get_mut().slots(id),
            scale,
        )?;
        *self.classifications.get_mut() = ExprClassCache::default();
        Some(id)
    }
    fn len(&self) -> usize {
        self.base.len() + self.nodes.len()
    }

    fn cached_degree(&self, id: ExprId) -> Option<Degree> {
        self.classifications.borrow_mut().get(id)
    }

    fn cache_degree(&self, id: ExprId, class: Degree) {
        self.classifications.borrow_mut().insert(id, class);
    }

    fn get(&self, id: ExprId) -> &ExprNode {
        let index = id.index();
        if index < self.base.nodes.len() {
            &self.base.nodes[index]
        } else {
            &self.nodes[index - self.base.nodes.len()]
        }
    }

    fn param_value(&self, id: ParamId) -> f64 {
        self.base.param_values[id.index()]
    }

    fn push(&mut self, node: ExprNode) -> ExprId {
        self.push_internal(node, false)
    }
}

struct ActiveFork {
    arena_key: usize,
    arena: ForkedExprArena,
}

thread_local! {
    static ACTIVE_FORKS: RefCell<Vec<ActiveFork>> = const { RefCell::new(Vec::new()) };
    static HELD_WRITE_GUARDS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

const WRITE_REENTRY_MESSAGE: &str = "expression arena write guard re-entered on the same thread";
const BATCH_ACTIVE: u8 = 1 << 0;
const WRITE_GUARD_ACTIVE: u8 = 1 << 1;

struct InstalledWriteGuard {
    arena_key: usize,
}

impl InstalledWriteGuard {
    fn new(arena_key: usize) -> Self {
        HELD_WRITE_GUARDS.with(|guards| {
            let mut guards = guards.borrow_mut();
            assert!(!guards.contains(&arena_key), "{WRITE_REENTRY_MESSAGE}");
            guards.push(arena_key);
        });
        Self { arena_key }
    }
}

impl Drop for InstalledWriteGuard {
    fn drop(&mut self) {
        HELD_WRITE_GUARDS.with(|guards| {
            let mut guards = guards.borrow_mut();
            let position = guards
                .iter()
                .rposition(|&arena_key| arena_key == self.arena_key)
                .expect("expression arena write guard routing state missing");
            guards.remove(position);
        });
    }
}

/// Parallel-safe owner used by [`crate::Expr`] handles.
///
/// Ordinary construction writes the canonical arena. Indexed batch builders
/// temporarily route expression operations to worker-local forks and merge the
/// resulting nodes in deterministic index order.
///
/// Raw-ID consumers publish through `cell.borrow_mut().seal(id)` or
/// `cell.borrow_mut().seal_all()`. With an existing write guard, call these
/// methods on the guard instead.
#[derive(Debug)]
pub struct ExprArenaCell {
    model_id: ModelId,
    inner: Mutex<ExprArena>,
    state: AtomicU8,
    #[cfg(test)]
    access_count: AtomicU32,
}

impl ExprArenaCell {
    pub fn new(arena: ExprArena) -> Self {
        Self {
            model_id: ModelId::fresh(),
            inner: Mutex::new(arena),
            state: AtomicU8::new(0),
            #[cfg(test)]
            access_count: AtomicU32::new(0),
        }
    }

    #[cfg(test)]
    pub(crate) fn reset_access_count(&self) {
        self.access_count.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn access_count(&self) -> u32 {
        self.access_count.load(Ordering::Relaxed)
    }

    #[inline]
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Take a cheap immutable snapshot of the canonical arena.
    ///
    /// The returned value is not a live read lock.
    /// Later writes are isolated by copy-on-write and are not
    /// visible through this snapshot.
    /// Taking the snapshot publishes every current canonical node, permanently
    /// disabling in-place affine/quadratic reuse for those nodes.
    ///
    /// # Panics
    ///
    /// Panics if an indexed batch is active. Snapshot reads cannot observe a
    /// worker-local fork and would otherwise wait on the canonical arena lock
    /// held by the batch.
    pub fn borrow(&self) -> ExprArenaSnapshot<'_> {
        let state = self.state.load(Ordering::Acquire);
        assert!(
            state & BATCH_ACTIVE == 0,
            "expression arena snapshot requested during an active indexed batch"
        );
        self.assert_no_write_reentry(state);
        ExprArenaSnapshot { arena: self.inner.lock().clone(), _cell: PhantomData }
    }

    /// Mutably borrow the canonical expression arena.
    ///
    /// # Panics
    ///
    /// Panics if an indexed batch is active, because its worker-local forks
    /// must see a stable canonical snapshot until they are merged, or if this
    /// thread already holds a write guard for this arena.
    pub fn borrow_mut(&self) -> ExprArenaWriteGuard<'_> {
        let state = self.state.load(Ordering::Acquire);
        assert!(
            state & BATCH_ACTIVE == 0,
            "expression arena accessed outside its indexed worker during an active batch"
        );
        self.assert_no_write_reentry(state);
        let guard = self.inner.lock();
        self.state.fetch_or(WRITE_GUARD_ACTIVE, Ordering::Release);
        ExprArenaWriteGuard { cell: self, _installed: InstalledWriteGuard::new(self.key()), guard }
    }

    fn key(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    #[inline]
    fn assert_no_write_reentry(&self, state: u8) {
        // Avoid touching thread-local state unless a public guard is alive.
        // If another thread owns that guard, the lookup is harmless and we
        // preserve the mutex's normal blocking behavior.
        if state & WRITE_GUARD_ACTIVE != 0 {
            let key = self.key();
            HELD_WRITE_GUARDS.with(|guards| {
                assert!(!guards.borrow().contains(&key), "{WRITE_REENTRY_MESSAGE}");
            });
        }
    }

    /// Number of expression nodes available on this thread, including worker-local nodes.
    #[doc(hidden)]
    #[inline]
    pub fn __node_count(&self) -> usize {
        self.with_ref(|arena| arena.len())
    }

    #[inline]
    pub(crate) fn with_ref<R>(&self, f: impl FnOnce(&dyn ArenaAccess) -> R) -> R {
        #[cfg(test)]
        self.access_count.fetch_add(1, Ordering::Relaxed);
        let state = self.state.load(Ordering::Acquire);

        if state & BATCH_ACTIVE == 0 {
            self.assert_no_write_reentry(state);
            let arena = self.inner.lock();
            return f(&*arena);
        }

        let mut f = Some(f);
        let local = ACTIVE_FORKS.with(|forks| {
            let forks = forks.borrow();
            forks
                .iter()
                .rfind(|fork| fork.arena_key == self.key())
                .map(|fork| f.take().expect("arena callback already consumed")(&fork.arena))
        });
        if let Some(result) = local {
            return result;
        }
        panic!("expression arena accessed outside its indexed worker during an active batch");
    }

    #[inline]
    pub(crate) fn build_expr(
        &self,
        construct: impl FnOnce(&mut dyn ArenaAccess) -> ExprId,
    ) -> ConstructionResult {
        let state = self.state.load(Ordering::Acquire);
        if state & BATCH_ACTIVE == 0 {
            self.assert_no_write_reentry(state);
            #[cfg(test)]
            self.access_count.fetch_add(1, Ordering::Relaxed);
            let mut arena = self.inner.lock();
            let id = construct(&mut *arena);
            return ConstructionResult::from_arena(&mut *arena, id);
        }
        self.with_mut(|arena| {
            let id = construct(arena);
            ConstructionResult::from_arena(arena, id)
        })
    }

    #[inline]
    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(&mut dyn ArenaAccess) -> R) -> R {
        #[cfg(test)]
        self.access_count.fetch_add(1, Ordering::Relaxed);
        let state = self.state.load(Ordering::Acquire);

        if state & BATCH_ACTIVE == 0 {
            self.assert_no_write_reentry(state);
            let mut arena = self.inner.lock();
            return f(&mut *arena);
        }

        let mut f = Some(f);
        let local = ACTIVE_FORKS.with(|forks| {
            let mut forks = forks.borrow_mut();
            forks
                .iter_mut()
                .rfind(|fork| fork.arena_key == self.key())
                .map(|fork| f.take().expect("arena callback already consumed")(&mut fork.arena))
        });
        if let Some(result) = local {
            return result;
        }
        panic!("expression arena accessed outside its indexed worker during an active batch");
    }

    #[doc(hidden)]
    pub fn __begin_batch(&self) -> ExprArenaBatchGuard<'_> {
        let state = self.state.load(Ordering::Acquire);
        self.assert_no_write_reentry(state);
        assert!(
            self.state.fetch_or(BATCH_ACTIVE, Ordering::AcqRel) & BATCH_ACTIVE == 0,
            "nested expression arena batches are not supported"
        );
        ExprArenaBatchGuard { cell: self, arena: self.inner.lock() }
    }

    /// Run a worker callback against a private fork of `base`.
    ///
    /// This is an internal batching primitive. The callback must not start a
    /// nested batch for this arena or access the arena from another thread.
    /// Such access is rejected while the batch is active.
    #[doc(hidden)]
    pub fn __with_fork<R>(&self, base: FrozenExprArena, f: impl FnOnce() -> R) -> ForkOutput<R> {
        let key = self.key();
        ACTIVE_FORKS.with(|forks| {
            let mut forks = forks.borrow_mut();
            assert!(
                !forks.iter().any(|fork| fork.arena_key == key),
                "nested expression forks for one arena are not supported"
            );
            forks.push(ActiveFork { arena_key: key, arena: ForkedExprArena::new(base) });
        });
        let mut installed = InstalledFork { arena_key: key, armed: true };
        let value = f();
        let arena = installed.take();
        ForkOutput { value, base: arena.base, nodes: arena.nodes }
    }
}

impl Default for ExprArenaCell {
    fn default() -> Self {
        Self::new(ExprArena::default())
    }
}

struct InstalledFork {
    arena_key: usize,
    armed: bool,
}

impl InstalledFork {
    fn take(&mut self) -> ForkedExprArena {
        self.armed = false;
        ACTIVE_FORKS.with(|forks| {
            let mut forks = forks.borrow_mut();
            let position = forks
                .iter()
                .rposition(|fork| fork.arena_key == self.arena_key)
                .expect("active expression fork missing");
            forks.remove(position).arena
        })
    }
}

impl Drop for InstalledFork {
    fn drop(&mut self) {
        if self.armed {
            ACTIVE_FORKS.with(|forks| {
                let mut forks = forks.borrow_mut();
                if let Some(position) =
                    forks.iter().rposition(|fork| fork.arena_key == self.arena_key)
                {
                    forks.remove(position);
                }
            });
        }
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ForkOutput<T> {
    pub value: T,
    base: FrozenExprArena,
    nodes: Vec<ExprNode>,
}

#[derive(Copy, Clone, Debug)]
#[doc(hidden)]
pub struct ExprIdRemap {
    local_base: usize,
    local_len: usize,
    global_base: usize,
}

impl ExprIdRemap {
    fn validate(self, id: ExprId) {
        let index = id.index();
        if index >= self.local_base {
            assert!(
                index - self.local_base < self.local_len,
                "expression fork returned an unknown node"
            );
        }
    }

    /// Translate an expression ID produced by one worker fork to its merged ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` refers neither to the fork's canonical base nor to a node
    /// created by that fork, or if the remapped ID exceeds `u32::MAX`.
    pub fn apply(self, id: ExprId) -> ExprId {
        let index = id.index();
        if index < self.local_base {
            return id;
        }
        self.validate(id);
        let local = index - self.local_base;
        ExprId(u32::try_from(self.global_base + local).expect("expression arena overflow"))
    }
}

/// Immutable copy-on-write snapshot of an [`ExprArenaCell`].
pub struct ExprArenaSnapshot<'a> {
    arena: ExprArena,
    _cell: PhantomData<&'a ExprArenaCell>,
}

impl std::fmt::Debug for ExprArenaSnapshot<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.arena.fmt(f)
    }
}

impl Deref for ExprArenaSnapshot<'_> {
    type Target = ExprArena;

    fn deref(&self) -> &Self::Target {
        &self.arena
    }
}

pub struct ExprArenaWriteGuard<'a> {
    // Remove the thread-local marker before unlocking the mutex.
    // There is no user code between field drops.
    cell: &'a ExprArenaCell,
    _installed: InstalledWriteGuard,
    guard: MutexGuard<'a, ExprArena>,
}

impl Drop for ExprArenaWriteGuard<'_> {
    fn drop(&mut self) {
        self.cell.state.fetch_and(!WRITE_GUARD_ACTIVE, Ordering::Release);
    }
}

impl std::fmt::Debug for ExprArenaWriteGuard<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.guard.fmt(f)
    }
}

impl Deref for ExprArenaWriteGuard<'_> {
    type Target = ExprArena;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a> ExprArenaWriteGuard<'a> {
    /// Append a node. Previously issued handles remain immutable.
    pub fn push(&mut self, node: ExprNode) -> ExprId {
        self.guard.push(node)
    }

    pub fn constant(&mut self, value: f64) -> ExprId {
        self.guard.constant(value)
    }

    pub fn var(&mut self, var: VarId) -> ExprId {
        self.guard.var(var)
    }

    pub fn param(&mut self, param: ParamId) -> ExprId {
        self.guard.param(param)
    }

    pub fn linear(&mut self, coeffs: Vec<(VarId, f64)>, constant: f64) -> ExprId {
        self.guard.linear(coeffs, constant)
    }

    pub fn new_param(&mut self, value: f64) -> ParamId {
        self.guard.new_param(value)
    }

    /// Rebind a parameter through the arena's checked write boundary.
    ///
    /// # Errors
    /// Returns the same validation errors as `ExprArena::set_param_value`.
    pub fn set_param_value(&mut self, param: ParamId, value: f64) -> Result<(), crate::ModelError> {
        self.guard.set_param_value(param, value)
    }

    /// Atomically rebind several parameters through the checked arena boundary.
    ///
    /// # Errors
    /// Rejects the entire batch if any parameter or bound assignment is invalid.
    pub fn set_param_values(
        &mut self,
        updates: &[(ParamId, f64)],
    ) -> Result<(), crate::ModelError> {
        self.guard.set_param_values(updates)
    }

    #[doc(hidden)]
    pub fn __set_bound_check(&mut self, variable: VarId, lower: crate::Bound, upper: crate::Bound) {
        self.guard.__set_bound_check(variable, lower, upper);
    }

    #[doc(hidden)]
    pub fn __lock_bound_parameter(&mut self, parameter: ParamId, source: String) {
        self.guard.__lock_bound_parameter(parameter, source);
    }

    #[doc(hidden)]
    pub fn __lock_parameters(&mut self, parameters: impl IntoIterator<Item = ParamId>) {
        self.guard.__lock_parameters(parameters);
    }

    #[doc(hidden)]
    pub fn __reserve_nodes(&mut self, count: usize) {
        self.guard.__reserve_nodes(count);
    }

    #[doc(hidden)]
    pub fn __extend_vars(
        &mut self,
        variables: std::ops::Range<usize>,
    ) -> impl ExactSizeIterator<Item = crate::VariableHandle<'a>> + use<'a> {
        let cell = self.cell;
        let variable_ids = variables.clone();
        self.guard.__extend_vars(variables).zip(variable_ids).map(move |(id, variable)| {
            crate::Expr::from_variable_node(
                ExprId(u32::try_from(id).expect("checked expression count")),
                VarId(u32::try_from(variable).expect("checked variable count")),
                cell,
            )
        })
    }

    #[doc(hidden)]
    pub fn __extend_params<I: ExactSizeIterator<Item = f64>>(
        &mut self,
        values: I,
    ) -> impl ExactSizeIterator<Item = (ParamId, crate::ParameterHandle<'a>)> + use<'a, I> {
        let cell = self.cell;
        let (parameters, expressions) = self.guard.__extend_params(values);
        parameters.zip(expressions).map(move |(parameter, expression)| {
            (
                ParamId(u32::try_from(parameter).expect("checked parameter count")),
                crate::Expr::from_id(
                    ExprId(u32::try_from(expression).expect("checked expression count")),
                    cell,
                ),
            )
        })
    }

    pub fn var_expr(&mut self, var: VarId) -> crate::VariableHandle<'a> {
        let id = self.guard.var(var);
        crate::Expr::from_variable_node(id, var, self.cell)
    }

    pub fn param_expr(&mut self, param: ParamId) -> crate::ParameterHandle<'a> {
        let id = self.guard.param(param);
        crate::Expr::from_id(id, self.cell)
    }
}

#[doc(hidden)]
pub struct ExprArenaBatchGuard<'a> {
    cell: &'a ExprArenaCell,
    arena: MutexGuard<'a, ExprArena>,
}

impl std::fmt::Debug for ExprArenaBatchGuard<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprArenaBatchGuard").field("arena", &self.arena).finish()
    }
}

impl ExprArenaBatchGuard<'_> {
    /// Freeze current nodes, permanently publishing affine/quadratic roots.
    pub fn snapshot(&self) -> FrozenExprArena {
        self.arena.seal_all();
        FrozenExprArena {
            nodes: Arc::clone(&self.arena.nodes),
            param_values: Arc::clone(&self.arena.param_values),
            classifications: self
                .arena
                .classifications
                .lock()
                .as_deref()
                .cloned()
                .unwrap_or_default(),
        }
    }

    /// Append worker-fork nodes in slice order and return each fork's ID map.
    /// Merged affine/quadratic roots are permanently published.
    ///
    /// # Panics
    ///
    /// Panics if a fork was created from a different snapshot or the merged
    /// expression arena would exceed `u32::MAX` nodes.
    pub fn merge<T>(&mut self, forks: &mut [ForkOutput<T>]) -> Vec<ExprIdRemap> {
        let initial_len = self.arena.len();
        let additional = forks.iter().map(|fork| fork.nodes.len()).sum::<usize>();
        let final_len =
            self.arena.len().checked_add(additional).expect("expression arena overflow");
        if final_len > 0 {
            u32::try_from(final_len - 1).expect("expression arena overflow");
        }
        let mut remaps = Vec::with_capacity(forks.len());
        let mut global_base = initial_len;
        for fork in forks.iter() {
            assert!(
                Arc::ptr_eq(&fork.base.nodes, &self.arena.nodes)
                    && Arc::ptr_eq(&fork.base.param_values, &self.arena.param_values),
                "expression fork used a stale arena snapshot"
            );
            remaps.push(ExprIdRemap {
                local_base: fork.base.len(),
                local_len: fork.nodes.len(),
                global_base,
            });
            global_base += fork.nodes.len();
        }

        // Validate every child reference before changing the canonical arena.
        // This keeps merge failure-atomic even if a worker returned an ID that
        // was not part of its fork.
        for (fork, remap) in forks.iter().zip(&remaps) {
            for node in &fork.nodes {
                validate_node(node, *remap);
            }
        }

        // The remaps retain the fork base lengths needed by remap_node and
        // ExprIdRemap::apply. Release the base snapshots before reserving so
        // Arc::make_mut can extend the canonical node vector in place.
        for fork in forks.iter_mut() {
            drop(std::mem::take(&mut fork.base));
        }

        self.arena.__reserve_nodes(additional);
        for (fork, remap) in forks.iter_mut().zip(&remaps) {
            for node in &mut fork.nodes {
                remap_node(node, *remap);
            }
            self.arena.__append_nodes(std::mem::take(&mut fork.nodes));
        }
        remaps
    }
}

fn seal_children(node: &ExprNode, mut seal: impl FnMut(ExprId)) {
    match node {
        ExprNode::Add(c) | ExprNode::Mul(c) | ExprNode::Min(c) | ExprNode::Max(c) => {
            for &id in c {
                seal(id);
            }
        }
        ExprNode::Unary(_, id) => seal(*id),
        ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
            seal(*a);
            seal(*b);
        }
        _ => {}
    }
}

fn validate_node(node: &ExprNode, remap: ExprIdRemap) {
    match node {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => {
            for child in children {
                remap.validate(*child);
            }
        }
        ExprNode::Unary(_, child) => remap.validate(*child),
        ExprNode::Pow(left, right) | ExprNode::Div(left, right) | ExprNode::Atan2(left, right) => {
            remap.validate(*left);
            remap.validate(*right);
        }
        ExprNode::Const(_)
        | ExprNode::Var(_)
        | ExprNode::Param(_)
        | ExprNode::Linear { .. }
        | ExprNode::Quadratic(_) => {}
    }
}

impl Drop for ExprArenaBatchGuard<'_> {
    fn drop(&mut self) {
        self.cell.state.fetch_and(!BATCH_ACTIVE, Ordering::Release);
    }
}

fn remap_node(node: &mut ExprNode, remap: ExprIdRemap) {
    match node {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => {
            for child in children {
                *child = remap.apply(*child);
            }
        }
        ExprNode::Unary(_, child) => *child = remap.apply(*child),
        ExprNode::Pow(left, right) | ExprNode::Div(left, right) | ExprNode::Atan2(left, right) => {
            *left = remap.apply(*left);
            *right = remap.apply(*right);
        }
        ExprNode::Const(_)
        | ExprNode::Var(_)
        | ExprNode::Param(_)
        | ExprNode::Linear { .. }
        | ExprNode::Quadratic(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use crate::ExprClass;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::{Dynamic, Expr, evaluate, extract_linear};

    fn linear_node(coefficient: f64, constant: f64) -> ExprNode {
        ExprNode::Linear { coeffs: vec![(VarId(0), coefficient)], constant }
    }

    #[test]
    fn affine_owned_ids_inline_capacity_dedup_remove_and_clear() {
        let mut ids = AffineOwnedIds::default();
        assert!(ids.is_empty());
        assert!(!ids.spilled());
        assert!(!ids.contains(ExprId(0)));
        assert!(!ids.remove(ExprId(0)));

        for i in 0..4 {
            assert!(ids.insert(ExprId(i)));
            assert!(!ids.insert(ExprId(i)));
            assert_eq!(ids.len(), i as usize + 1);
            assert!(!ids.spilled());
        }

        assert!(!ids.insert(ExprId(0)));
        assert!(!ids.spilled());
        assert!(ids.remove(ExprId(1)));
        assert!(!ids.remove(ExprId(1)));
        assert!(!ids.contains(ExprId(1)));
        for i in [0, 2, 3] {
            assert!(ids.contains(ExprId(i)));
        }

        assert!(ids.insert(ExprId(4)));
        assert_eq!(ids.len(), 4);
        assert!(!ids.spilled());

        ids.clear();
        assert!(ids.is_empty());
        assert!(!ids.spilled());
        for i in 0..4 {
            assert!(!ids.contains(ExprId(i)));
            assert!(ids.insert(ExprId(i)));
            assert!(!ids.spilled());
        }

        for i in 0..4 {
            assert!(ids.remove(ExprId(i)));
        }

        assert!(ids.is_empty());
        assert!(!ids.spilled());
    }

    #[test]
    fn oversized_affine_cache_stays_bounded_and_keeps_the_latest_root() {
        let mut ownership = RootOwnership::default();
        let cache_capacity =
            u32::try_from(MAX_OVERSIZED_AFFINE_ROOTS).expect("cache capacity fits ExprId");
        for index in 0..cache_capacity {
            ownership.mark_oversized_affine(ExprId(index));
        }
        assert_eq!(ownership.oversized_affine.len(), MAX_OVERSIZED_AFFINE_ROOTS);
        assert!(ownership.oversized_affine.contains(&ExprId(0)));

        ownership.mark_oversized_affine(ExprId(cache_capacity));
        assert_eq!(ownership.oversized_affine.len(), 1);
        assert!(!ownership.oversized_affine.contains(&ExprId(0)));
        assert!(ownership.oversized_affine.contains(&ExprId(cache_capacity)));

        for index in 1..10_000 {
            ownership.mark_oversized_affine(ExprId(index));
            assert!(ownership.oversized_affine.len() <= MAX_OVERSIZED_AFFINE_ROOTS);
            assert!(ownership.oversized_affine.contains(&ExprId(index)));
        }
    }

    #[test]
    fn affine_owned_ids_spill_preserves_set_semantics_and_clear_resets_inline() {
        let mut ids = AffineOwnedIds::default();
        for i in 0..256 {
            assert!(ids.insert(ExprId(i)));
            assert_eq!(ids.len(), i as usize + 1);
            assert_eq!(ids.spilled(), i >= 4);
        }

        for i in 0..256 {
            assert!(ids.contains(ExprId(i)));
            assert!(!ids.insert(ExprId(i)));
        }

        assert_eq!(ids.len(), 256);
        for i in (0..256).step_by(2) {
            assert!(ids.remove(ExprId(i)));
            assert!(!ids.remove(ExprId(i)));
        }

        assert_eq!(ids.len(), 128);
        for i in 0..256 {
            assert_eq!(ids.contains(ExprId(i)), i % 2 != 0);
        }

        ids.clear();
        assert!(ids.is_empty());
        assert!(!ids.spilled());
        for i in 0..256 {
            assert!(!ids.contains(ExprId(i)));
            assert!(!ids.remove(ExprId(i)));
        }

        for i in 0..5 {
            assert!(ids.insert(ExprId(i)));
        }

        for i in 0..5 {
            assert!(ids.remove(ExprId(i)));
        }
        assert!(ids.is_empty());
        assert!(ids.spilled(), "removal retains hash storage until clear");

        ids.clear();
        assert!(!ids.spilled());
        assert!(ids.insert(ExprId(0)));
        assert!(!ids.spilled());
    }

    fn check_linear_ownership(arena: &mut dyn ArenaAccess) {
        let before = arena.push(linear_node(1.0, 0.0));
        let owned = arena.push_owned_linear(linear_node(2.0, 1.0));
        let raw = arena.push(linear_node(3.0, 0.0));
        let immutable = arena.push_immutable(linear_node(4.0, 0.0));
        for sealed in [before, raw, immutable] {
            assert!(!arena.linear_root(sealed));
            assert_eq!(arena.accumulate_linear(sealed, owned, sealed), None);
        }

        assert!(arena.linear_root(owned));
        assert_eq!(arena.accumulate_linear(owned, immutable, owned), Some(owned));
        let ExprNode::Linear { coeffs, constant } = arena.get(owned) else {
            panic!("linear root");
        };
        assert_eq!(coeffs, &[(VarId(0), 6.0)]);
        assert_eq!(*constant, 1.0);

        for immutable_parent in [false, true] {
            let child = arena.push_owned_linear(linear_node(1.0, 0.0));
            let parent = ExprNode::Unary(UnaryOp::Sin, child);
            if immutable_parent {
                arena.push_immutable(parent);
            } else {
                arena.push(parent);
            }
            assert!(!arena.linear_root(child));
            assert_eq!(arena.accumulate_linear(child, raw, child), None);
        }

        arena.seal(owned);
        assert!(!arena.linear_root(owned));
    }

    #[test]
    fn linear_ownership_requires_explicit_owned_push_in_canonical_and_fork_arenas() {
        check_linear_ownership(&mut ExprArena::new());
        check_linear_ownership(&mut ForkedExprArena::new(FrozenExprArena::default()));

        let mut arena = ExprArena::new();
        let owned = ArenaAccess::push_owned_linear(&mut arena, linear_node(1.0, 0.0));
        let raw = arena.push(linear_node(2.0, 0.0));
        let builder = arena.linear(vec![(VarId(0), 3.0)], 0.0);
        for sealed in [raw, builder] {
            assert!(!ArenaAccess::linear_root(&mut arena, sealed));
            assert_eq!(ArenaAccess::accumulate_linear(&mut arena, owned, sealed, sealed), None);
        }

        arena.push(ExprNode::Unary(UnaryOp::Sin, owned));
        assert!(!ArenaAccess::linear_root(&mut arena, owned));
    }

    #[test]
    fn immutable_linear_appends_do_not_grow_ownership_bookkeeping() {
        let mut arena = ExprArena::new();
        arena.push(linear_node(1.0, 0.0));
        ArenaAccess::push_immutable(&mut arena, linear_node(1.0, 0.0));
        assert!(arena.ownership.get().is_none());

        let owned = ArenaAccess::push_owned_linear(&mut arena, linear_node(1.0, 0.0));
        for _ in 0..64 {
            arena.push(linear_node(2.0, 0.0));
            ArenaAccess::push(&mut arena, linear_node(3.0, 0.0));
            ArenaAccess::push_immutable(&mut arena, linear_node(4.0, 0.0));
        }

        let ownership = arena.ownership.get_mut().unwrap().get_mut();
        assert_eq!(ownership.affine_owned.len(), 1);
        assert!(!ownership.affine_owned.spilled());
        assert!(ownership.affine_owned.contains(owned));
        assert!(ownership.sealed.is_empty());
        assert!(ownership.slots.is_empty());

        arena.seal(owned);
        let ownership = arena.ownership.get_mut().unwrap().get_mut();
        assert!(ownership.affine_owned.is_empty());
        assert!(ownership.sealed.is_empty());

        let mut fork = ForkedExprArena::new(FrozenExprArena::default());
        fork.push(linear_node(1.0, 0.0));
        fork.push_immutable(linear_node(1.0, 0.0));
        assert!(fork.ownership.get().is_none());

        let owned = fork.push_owned_linear(linear_node(1.0, 0.0));
        for _ in 0..64 {
            fork.push(linear_node(2.0, 0.0));
            fork.push_immutable(linear_node(3.0, 0.0));
        }

        let ownership = fork.ownership.get_mut().unwrap().get_mut();
        assert_eq!(ownership.affine_owned.len(), 1);
        assert!(!ownership.affine_owned.spilled());
        assert!(ownership.affine_owned.contains(owned));
        assert!(ownership.sealed.is_empty());
        assert!(ownership.slots.is_empty());

        fork.seal(owned);
        let ownership = fork.ownership.get_mut().unwrap().get_mut();
        assert!(ownership.affine_owned.is_empty());
        assert!(ownership.sealed.is_empty());
    }

    fn canonical_ownership(arena: &mut ExprArena) -> &mut RootOwnership {
        arena.ownership.get_mut().unwrap().get_mut()
    }

    fn fork_ownership(arena: &mut ForkedExprArena) -> &mut RootOwnership {
        arena.ownership.get_mut().unwrap().get_mut()
    }

    fn check_many_affine_roots<A: ArenaAccess>(
        arena: &mut A,
        ownership: fn(&mut A) -> &mut RootOwnership,
    ) {
        let roots: Vec<_> =
            (0..256).map(|_| arena.push_owned_linear(linear_node(1.0, 0.0))).collect();
        assert_eq!(ownership(arena).affine_owned.len(), roots.len());
        assert!(ownership(arena).affine_owned.spilled());

        for pair in roots.as_chunks::<2>().0 {
            assert!(arena.linear_root(pair[0]));
            assert!(arena.linear_root(pair[1]));
            assert_eq!(arena.accumulate_linear(pair[0], pair[1], pair[0]), Some(pair[0]));
            assert!(!arena.linear_root(pair[1]));
            assert!(arena.linear_root(pair[0]));
        }
        assert_eq!(ownership(arena).affine_owned.len(), roots.len() / 2);

        for pair in roots.as_chunks::<2>().0 {
            arena.push_immutable(ExprNode::Unary(UnaryOp::Sin, pair[0]));
            assert!(!arena.linear_root(pair[0]));
        }
        assert!(ownership(arena).affine_owned.is_empty());
        assert!(!ownership(arena).needs_child_sealing());

        let fresh = arena.push_owned_linear(linear_node(3.0, 0.0));
        assert!(arena.linear_root(fresh));

        let prefix = arena.len();
        ownership(arena).seal_all(prefix);
        assert!(!arena.linear_root(fresh));
        assert!(!ownership(arena).affine_owned.spilled());

        let fresh = arena.push_owned_linear(linear_node(4.0, 0.0));
        assert!(arena.linear_root(fresh));
        assert_eq!(ownership(arena).affine_owned.len(), 1);
        assert!(!ownership(arena).affine_owned.spilled());
    }

    #[test]
    fn many_concurrent_affine_roots_work_in_canonical_and_fork_arenas() {
        check_many_affine_roots(&mut ExprArena::new(), canonical_ownership);
        let base = FrozenExprArena {
            nodes: Arc::new(vec![linear_node(2.0, 1.0)]),
            ..FrozenExprArena::default()
        };

        let mut fork = ForkedExprArena::new(base);
        check_many_affine_roots(&mut fork, fork_ownership);
        assert!(!fork.linear_root(ExprId(0)));
        assert!(matches!(fork.get(ExprId(0)), ExprNode::Linear { constant: 1.0, .. }));
    }

    #[test]
    fn snapshot_resets_spilled_affine_ownership_and_forks_track_ids_independently() {
        let mut arena = ExprArena::new();
        let roots: Vec<_> = (0..16)
            .map(|_| ArenaAccess::push_owned_linear(&mut arena, linear_node(1.0, 0.0)))
            .collect();
        assert!(canonical_ownership(&mut arena).affine_owned.spilled());

        let cell = ExprArenaCell::new(arena);
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        assert!(canonical_ownership(&mut batch.arena).affine_owned.is_empty());
        assert!(!canonical_ownership(&mut batch.arena).affine_owned.spilled());

        for &root in &roots {
            assert!(!ArenaAccess::linear_root(&mut *batch.arena, root));
        }

        let mut first = ForkedExprArena::new(snapshot.clone());
        let mut second = ForkedExprArena::new(snapshot);
        let mut first_roots = Vec::new();
        for i in 0..16 {
            let left = first.push_owned_linear(linear_node(2.0, 0.0));
            let right = second.push_owned_linear(linear_node(3.0, 0.0));
            assert_eq!(left, right);
            assert_eq!(fork_ownership(&mut first).affine_owned.spilled(), i >= 4);
            assert_eq!(fork_ownership(&mut second).affine_owned.spilled(), i >= 4);

            first_roots.push(left);
        }

        for &root in &roots {
            assert!(!first.linear_root(root));
            assert!(!second.linear_root(root));
        }

        for root in first_roots {
            first.seal(root);
            assert!(!first.linear_root(root));
            assert!(second.linear_root(root));
        }
        assert!(fork_ownership(&mut first).affine_owned.is_empty());
        assert_eq!(fork_ownership(&mut second).affine_owned.len(), 16);

        drop(first);
        drop(second);
        let fresh = ArenaAccess::push_owned_linear(&mut *batch.arena, linear_node(4.0, 0.0));
        assert!(ArenaAccess::linear_root(&mut *batch.arena, fresh));
        assert!(!canonical_ownership(&mut batch.arena).affine_owned.spilled());
    }

    fn check_affine_child_scan_gating<A: ArenaAccess>(
        arena: &mut A,
        ownership: fn(&mut A) -> &mut RootOwnership,
    ) {
        let prefix = arena.push_owned_linear(ExprNode::Linear {
            coeffs: (0..32).map(|i| (VarId(i), 1.0)).collect(),
            constant: 0.0,
        });
        let extra = arena.var(VarId(32));
        assert_eq!(arena.accumulate_linear(prefix, extra, prefix), None);

        let mut result = crate::linear::add_with_ownership(arena, prefix, extra, false);
        assert!(!result.reusable);
        let mut sum = result.id;
        assert!(!arena.linear_root(prefix));
        assert!(ownership(arena).affine_owned.is_empty());
        assert!(!ownership(arena).needs_child_sealing());
        assert_eq!(ownership(arena).child_inspections, 2);

        for i in 33..4096 {
            let term = arena.var(VarId(i));
            result = crate::linear::add_with_ownership(arena, sum, term, i % 2 == 0);
            sum = result.id;
        }
        assert_eq!(ownership(arena).child_inspections, 2);

        let first = arena.push_owned_linear(linear_node(1.0, 0.0));
        let second = arena.push_owned_linear(linear_node(2.0, 0.0));
        assert!(ownership(arena).needs_child_sealing());

        arena.push_immutable(ExprNode::Unary(UnaryOp::Sin, first));
        assert!(!arena.linear_root(first));
        assert_eq!(arena.accumulate_linear(first, extra, first), None);
        assert!(arena.linear_root(second));
        assert!(ownership(arena).needs_child_sealing());
        assert_eq!(ownership(arena).child_inspections, 3);

        arena.push(ExprNode::Unary(UnaryOp::Sin, second));
        assert!(!arena.linear_root(second));
        assert_eq!(arena.accumulate_linear(second, extra, second), None);
        assert!(!ownership(arena).needs_child_sealing());
        assert_eq!(ownership(arena).child_inspections, 4);

        arena.push_immutable(ExprNode::Add(smallvec::smallvec![sum, extra]));
        assert_eq!(ownership(arena).child_inspections, 4);
    }

    #[test]
    fn canonical_affine_fallback_scans_only_the_first_parent_and_reactivates() {
        check_affine_child_scan_gating(&mut ExprArena::new(), canonical_ownership);
    }

    #[test]
    fn fork_affine_fallback_scans_only_the_first_parent_and_reactivates() {
        check_affine_child_scan_gating(
            &mut ForkedExprArena::new(FrozenExprArena::default()),
            fork_ownership,
        );
    }

    fn quadratic_node() -> ExprNode {
        ExprNode::Quadratic(Arc::new(
            crate::QuadraticTerms::from_coefficients(
                vec![(VarId(0), VarId(0), 1.0)],
                Vec::new(),
                0.0,
            )
            .unwrap(),
        ))
    }

    fn check_quadratic_child_scan_gating<A: ArenaAccess>(
        arena: &mut A,
        ownership: fn(&mut A) -> &mut RootOwnership,
        quadratic_first: bool,
    ) {
        let first = if quadratic_first { Some(arena.push(quadratic_node())) } else { None };
        let affine = arena.push_owned_linear(linear_node(1.0, 0.0));
        arena.seal(affine);
        assert_eq!(ownership(arena).quadratic_pending, quadratic_first);

        let first = first.unwrap_or_else(|| arena.push(quadratic_node()));
        let second = arena.push(quadratic_node());
        let variable = arena.var(VarId(0));
        assert!(ownership(arena).affine_owned.is_empty());
        assert!(ownership(arena).quadratic_pending);
        assert_eq!(arena.accumulate_quadratic(first, variable, 1.0), Some(first));

        arena.push_immutable(ExprNode::Unary(UnaryOp::Sin, first));
        assert_eq!(arena.accumulate_quadratic(first, variable, 1.0), None);
        assert!(ownership(arena).quadratic_pending, "another quadratic is still live");
        assert!(arena.quadratic_root(second));

        arena.push(ExprNode::Unary(UnaryOp::Sin, second));
        assert_eq!(arena.accumulate_quadratic(second, variable, 1.0), None);
        assert_eq!(ownership(arena).child_inspections, 2);
        assert!(ownership(arena).quadratic_pending, "individual seals remain conservative");

        let unpublished = arena.push(quadratic_node());
        let prefix = arena.len();
        ownership(arena).seal_all(prefix);
        assert!(!ownership(arena).quadratic_pending);
        assert!(!ownership(arena).needs_child_sealing());
        assert_eq!(arena.accumulate_quadratic(unpublished, variable, 1.0), None);

        arena.push_immutable(ExprNode::Unary(UnaryOp::Sin, unpublished));
        assert_eq!(ownership(arena).child_inspections, 2);

        let fresh = arena.push(quadratic_node());
        assert!(ownership(arena).quadratic_pending);
        assert!(arena.quadratic_root(fresh));

        arena.push_immutable(ExprNode::Unary(UnaryOp::Sin, fresh));
        assert_eq!(ownership(arena).child_inspections, 3);
        assert_eq!(arena.accumulate_quadratic(fresh, variable, 1.0), None);
    }

    #[test]
    fn canonical_quadratics_keep_child_sealing_active_until_seal_all() {
        for quadratic_first in [false, true] {
            check_quadratic_child_scan_gating(
                &mut ExprArena::new(),
                canonical_ownership,
                quadratic_first,
            );
        }
    }

    #[test]
    fn fork_quadratics_keep_child_sealing_active_until_seal_all() {
        for quadratic_first in [false, true] {
            check_quadratic_child_scan_gating(
                &mut ForkedExprArena::new(FrozenExprArena::default()),
                fork_ownership,
                quadratic_first,
            );
        }
    }

    #[test]
    fn child_scan_gating_keeps_frozen_base_roots_read_only() {
        let base = FrozenExprArena {
            nodes: Arc::new(vec![linear_node(2.0, 1.0), quadratic_node()]),
            ..FrozenExprArena::default()
        };
        let mut fork = ForkedExprArena::new(base.clone());
        let affine = ExprId(0);
        let quadratic = ExprId(1);
        fork.push_immutable(ExprNode::Add(smallvec::smallvec![affine, quadratic]));
        assert!(fork.ownership.get().is_none());

        let local = fork.push_owned_linear(linear_node(3.0, 2.0));
        assert!(!fork.linear_root(affine));
        assert!(!fork.quadratic_root(quadratic));
        assert_eq!(fork.accumulate_linear(affine, local, affine), None);
        assert_eq!(fork.accumulate_quadratic(quadratic, local, 1.0), None);

        fork.push_immutable(ExprNode::Add(smallvec::smallvec![affine, quadratic, local]));
        assert!(!fork.linear_root(local));
        assert_eq!(fork_ownership(&mut fork).child_inspections, 3);
        assert!(!fork_ownership(&mut fork).needs_child_sealing());

        fork.push(ExprNode::Add(smallvec::smallvec![quadratic, affine]));
        assert_eq!(fork_ownership(&mut fork).child_inspections, 3);
        assert!(Arc::ptr_eq(&base.nodes, &fork.base.nodes));

        let ExprNode::Linear { coeffs, constant } = fork.get(affine) else {
            panic!("frozen affine root");
        };
        assert_eq!(coeffs, &[(VarId(0), 2.0)]);
        assert_eq!(*constant, 1.0);

        let ExprNode::Quadratic(terms) = fork.get(quadratic) else {
            panic!("frozen quadratic root");
        };
        assert_eq!(terms.quadratic, vec![(VarId(0), VarId(0), 1.0)]);
        assert_eq!(terms.linear, []);
        assert_eq!(terms.constant, 0.0);
    }

    #[test]
    fn reusable_root_without_ownership_does_not_inspect_nodes() {
        let mut arena = ExprArena::new();
        let mut fork = ForkedExprArena::new(FrozenExprArena::default());
        assert!(!ArenaAccess::reusable_root(&mut arena, ExprId(u32::MAX)));
        assert!(!fork.reusable_root(ExprId(u32::MAX)));
    }

    #[test]
    fn reusable_roots_keep_affine_and_quadratic_ownership_separate() {
        fn check(arena: &mut dyn ArenaAccess) {
            let affine = arena.push_owned_linear(linear_node(1.0, 0.0));
            let quadratic = arena.push(ExprNode::Quadratic(Arc::new(
                crate::QuadraticTerms::from_coefficients(
                    vec![(VarId(0), VarId(0), 1.0)],
                    Vec::new(),
                    0.0,
                )
                .unwrap(),
            )));
            let immutable = arena.push_immutable(linear_node(2.0, 0.0));
            assert!(arena.reusable_root(affine));
            assert!(arena.reusable_root(quadratic));
            assert!(!arena.reusable_root(immutable));

            arena.seal(affine);
            assert!(!arena.reusable_root(affine));
            assert!(arena.reusable_root(quadratic));

            arena.seal(quadratic);
            assert!(!arena.reusable_root(quadratic));
        }
        check(&mut ExprArena::new());
        check(&mut ForkedExprArena::new(FrozenExprArena::default()));
    }

    #[test]
    fn linear_publication_and_snapshots_permanently_seal_roots() {
        for publication in 0..5 {
            let mut arena = ExprArena::new();
            let root = ArenaAccess::push_owned_linear(&mut arena, linear_node(1.0, 0.0));
            match publication {
                0 => arena.seal(root),
                1 => arena.seal_all(),
                2 => drop(arena.clone()),
                3 => drop(arena.__clone_with_additional_capacity(1)),
                _ => arena.__append_nodes(vec![linear_node(2.0, 0.0)]),
            }
            assert!(!ArenaAccess::linear_root(&mut arena, root));
            assert_eq!(ArenaAccess::accumulate_linear(&mut arena, root, root, root), None);

            let fresh = ArenaAccess::push_owned_linear(&mut arena, linear_node(2.0, 0.0));
            assert!(ArenaAccess::linear_root(&mut arena, fresh));
            assert!(!ArenaAccess::linear_root(&mut arena, root));
        }
    }

    fn check_linear_accumulation(arena: &mut dyn ArenaAccess) {
        let left = arena.push_owned_linear(linear_node(1.0, 2.0));
        let right = arena.push_owned_linear(linear_node(-1.0, 3.0));
        let unrelated = arena.push_owned_linear(linear_node(9.0, 8.0));
        assert_eq!(arena.accumulate_linear(left, right, unrelated), None);
        assert!(arena.linear_root(unrelated));

        arena.cache_degree(right, Degree::One);
        assert_eq!(arena.accumulate_linear(left, right, right), Some(right));
        assert!(matches!(arena.get(right), ExprNode::Const(5.0)));
        assert!(!arena.linear_root(right));
        assert_eq!(arena.cached_degree(right), None);
        assert_eq!(arena.accumulate_linear(left, left, right), None);
        assert!(!arena.linear_root(left), "the consumed source must lose its ownership record");

        let left = arena.push_owned_linear(linear_node(1.0, 2.0));
        assert_eq!(arena.accumulate_linear(left, left, left), Some(left));

        let ExprNode::Linear { coeffs, constant } = arena.get(left) else {
            panic!("linear root");
        };
        assert_eq!(coeffs, &[(VarId(0), 2.0)]);
        assert_eq!(*constant, 4.0);

        let full = arena.push_owned_linear(ExprNode::Linear {
            coeffs: (0..32).map(|i| (VarId(i), 1.0)).collect(),
            constant: 7.0,
        });
        let extra = arena.var(VarId(32));
        arena.cache_degree(full, Degree::One);
        assert_eq!(arena.accumulate_linear(full, extra, full), None);

        let ExprNode::Linear { coeffs, constant } = arena.get(full) else {
            panic!("unchanged linear root");
        };
        assert_eq!(coeffs, &(0..32).map(|i| (VarId(i), 1.0)).collect::<Vec<_>>());
        assert_eq!(*constant, 7.0);
        assert!(arena.linear_root(full));
        assert_eq!(arena.cached_degree(full), Some(Degree::One));

        let unsupported = arena.push_immutable(ExprNode::Unary(UnaryOp::Sin, extra));
        assert_eq!(arena.accumulate_linear(full, unsupported, full), None);
        assert_eq!(arena.cached_degree(full), Some(Degree::One));
    }

    #[test]
    fn linear_accumulation_is_transactional_and_invalidates_classifications() {
        check_linear_accumulation(&mut ExprArena::new());
        check_linear_accumulation(&mut ForkedExprArena::new(FrozenExprArena::default()));
    }

    #[test]
    fn linear_accumulation_requires_unique_storage_and_invalidates_analysis() {
        let mut arena = ExprArena::new();
        let root = ArenaAccess::push_owned_linear(&mut arena, linear_node(1.0, 2.0));
        let shared = Arc::clone(&arena.nodes);
        assert_eq!(ArenaAccess::accumulate_linear(&mut arena, root, root, root), None);
        assert!(matches!(&shared[root.index()], ExprNode::Linear { constant: 2.0, .. }));
        assert!(ArenaAccess::linear_root(&mut arena, root));

        drop(shared);
        arena.__enable_preparation_classifications();
        ArenaAccess::cache_degree(&arena, root, Degree::One);
        assert!(arena.analysis.get().is_some());
        assert_eq!(ArenaAccess::accumulate_linear(&mut arena, root, root, root), Some(root));
        assert!(arena.analysis.get().is_none());
        assert_eq!(ArenaAccess::cached_degree(&arena, root), None);
    }

    #[test]
    fn fork_linear_accumulation_reads_base_but_only_mutates_local_destinations() {
        let mut canonical = ExprArena::new();
        let base_root = ArenaAccess::push_owned_linear(&mut canonical, linear_node(2.0, 1.0));
        let cell = ExprArenaCell::new(canonical);
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        assert!(!ArenaAccess::linear_root(&mut *batch.arena, base_root));

        let mut fork = ForkedExprArena::new(snapshot);
        let local = fork.push_owned_linear(linear_node(3.0, 2.0));
        assert!(!fork.linear_root(base_root));
        assert_eq!(fork.accumulate_linear(local, base_root, base_root), None);
        assert_eq!(fork.accumulate_linear(base_root, local, local), Some(local));

        let ExprNode::Linear { coeffs, constant } = fork.get(local) else {
            panic!("local linear root");
        };
        assert_eq!(coeffs, &[(VarId(0), 5.0)]);
        assert_eq!(*constant, 3.0);

        let ExprNode::Linear { coeffs, constant } = fork.get(base_root) else {
            panic!("immutable base root");
        };
        assert_eq!(coeffs, &[(VarId(0), 2.0)]);
        assert_eq!(*constant, 1.0);

        let mut outputs = [ForkOutput { value: local, base: fork.base, nodes: fork.nodes }];
        let remaps = batch.merge(&mut outputs);
        let merged = remaps[0].apply(local);
        assert!(!ArenaAccess::linear_root(&mut *batch.arena, merged));
        assert_eq!(
            ArenaAccess::accumulate_linear(&mut *batch.arena, merged, base_root, merged),
            None
        );
    }

    #[test]
    fn read_queries_preserve_unique_quadratic_roots() {
        for query in 0..4 {
            let mut arena = ExprArena::new();
            let x = arena.var(VarId(0));
            let root = ArenaAccess::push(
                &mut arena,
                ExprNode::Quadratic(Arc::new(
                    crate::QuadraticTerms::from_hessian(
                        vec![(VarId(0), VarId(0), 2.0)],
                        Vec::new(),
                        0.0,
                    )
                    .unwrap(),
                )),
            );
            match query {
                0 => {
                    let _ = arena.get(root);
                }
                1 => {
                    let _ = arena.nodes();
                }
                2 => {
                    let _ = crate::ExpressionAnalysis::default().known(&arena, root);
                }
                _ => {
                    let _ = crate::extract_quadratic(&arena, root);
                }
            }
            assert_eq!(ArenaAccess::accumulate_quadratic(&mut arena, root, x, 1.0), Some(root));
            let ExprNode::Quadratic(q) = arena.node(root) else { panic!("quadratic root") };
            assert_eq!(q.linear(), [(VarId(0), 1.0)]);
        }
    }

    #[test]
    fn published_residuals_coefficients_and_analysis_protect_quadratic_roots() {
        for publication in 0..5 {
            let mut arena = ExprArena::new();
            let x = arena.var(VarId(0));
            let root = ArenaAccess::push(
                &mut arena,
                ExprNode::Quadratic(Arc::new(
                    crate::QuadraticTerms::from_coefficients(
                        vec![(VarId(0), VarId(0), 1.0)],
                        Vec::new(),
                        0.0,
                    )
                    .unwrap(),
                )),
            );

            match publication {
                0 => arena.seal(root),
                1 => {
                    let _ = crate::ExpressionAnalysis::default().summarize(&arena, root);
                }
                2 => {
                    let _ = crate::extract_quadratic_shared(&arena, root);
                }
                3 => {
                    let _ = crate::split_linear(&arena, root);
                }
                _ => {
                    let _ = crate::split_affine(&arena, root);
                }
            }

            assert!(ArenaAccess::accumulate_quadratic(&mut arena, root, x, 1.0).is_none());
        }
    }

    #[test]
    fn nonquadratic_snapshots_preserve_the_raw_mutation_barrier() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let root = arena.linear(vec![(VarId(0), 1.0)], 0.0);
        let snapshot = arena.clone();
        *arena.get_mut(root) = ExprNode::Quadratic(Arc::new(
            crate::QuadraticTerms::from_hessian(vec![(VarId(0), VarId(0), 2.0)], Vec::new(), 0.0)
                .unwrap(),
        ));
        assert!(matches!(snapshot.get(root), ExprNode::Linear { .. }));
        assert!(ArenaAccess::accumulate_quadratic(&mut arena, root, x, 1.0).is_none());

        let fresh = ArenaAccess::push(
            &mut arena,
            ExprNode::Quadratic(Arc::new(
                crate::QuadraticTerms::from_hessian(
                    vec![(VarId(0), VarId(0), 2.0)],
                    Vec::new(),
                    0.0,
                )
                .unwrap(),
            )),
        );
        assert!(ArenaAccess::accumulate_quadratic(&mut arena, root, x, 1.0).is_none());
        assert_eq!(ArenaAccess::accumulate_quadratic(&mut arena, fresh, x, 1.0), Some(fresh));
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn bulk_metadata_append_preserves_snapshot_ids_values_and_locks() {
        let mut source = ExprArena::new();
        let p = source.new_param(7.0);
        source.param(p);
        source.__lock_parameters([p]);
        let snapshot = source.parameter_snapshot();
        let mut fork = source.clone();

        let expressions = fork.__extend_vars(3..5);
        assert_eq!(expressions, 1..3);
        assert!(matches!(fork.get(ExprId(1)), ExprNode::Var(VarId(3))));
        assert!(matches!(fork.get(ExprId(2)), ExprNode::Var(VarId(4))));

        let (parameters, expressions) = fork.__extend_params([11.0, 13.0].into_iter());
        assert_eq!(parameters, 1..3);
        assert_eq!(expressions, 3..5);
        assert!(matches!(fork.get(ExprId(3)), ExprNode::Param(ParamId(1))));
        assert!(matches!(fork.get(ExprId(4)), ExprNode::Param(ParamId(2))));
        assert_eq!(fork.param_value(ParamId(2)), 13.0);
        assert_eq!(source.num_params(), 1);
        assert_eq!(source.len(), 1);
        assert!(snapshot.same_bits(&source.parameter_snapshot()));
        assert!(fork.set_param_value(p, 8.0).is_err());

        let before = fork.len();
        let invalid = usize::try_from(u32::MAX).unwrap()..usize::try_from(u32::MAX).unwrap() + 2;
        assert!(catch_unwind(AssertUnwindSafe(|| fork.__extend_vars(invalid))).is_err());
        assert_eq!(fork.len(), before);
        assert_eq!(fork.__extend_vars(5..5), before..before);
        assert_eq!(fork.__extend_params(std::iter::empty()), (3..3, before..before));
    }

    #[test]
    fn parameter_locks_survive_cloning_without_locking_the_source() {
        let mut source = ExprArena::new();
        let locked = source.new_param(1.0);
        let unrelated = source.new_param(2.0);
        let mut transformed = source.clone();
        transformed.__lock_parameters([locked]);
        source.set_param_value(locked, 100.0).unwrap();
        for mut copy in [transformed.clone(), transformed.__clone_with_additional_capacity(4)] {
            assert!(copy.set_param_value(locked, 100.0).is_err());
            assert_eq!(copy.param_value(locked), 1.0);
            copy.set_param_value(unrelated, 3.0).unwrap();
            assert_eq!(copy.param_value(unrelated), 3.0);
        }
        assert_eq!(source.param_value(locked), 100.0);
        assert_eq!(transformed.param_value(unrelated), 2.0);
    }

    #[test]
    fn single_write_matches_the_batch_path_for_every_outcome() {
        let mut arena = ExprArena::new();
        let known = arena.new_param(1.0);
        let other = arena.new_param(2.0);
        let locked = arena.new_param(3.0);
        let bound = arena.new_param(0.0);
        let dependent = VarId(0);
        arena.__set_bound_check(
            dependent,
            crate::Bound::Parameter(bound),
            crate::Bound::Constant(10.0),
        );
        arena.__lock_bound_parameter(locked, "generated rows".into());

        let cases = [
            ("unknown", ParamId(999), 1.0),
            ("non-finite", known, f64::INFINITY),
            ("frozen", locked, 4.0),
            ("bound-infeasible", bound, 20.0),
            ("accepted", other, 7.0),
            ("accepted-dependent", bound, 5.0),
        ];

        for (label, p, value) in cases {
            let mut batch = arena.clone();
            let mut single = arena.clone();

            let expected = batch.set_param_values(&[(p, value)]);
            let actual = single.set_param_value(p, value);

            assert_eq!(actual, expected, "outcome diverged for {label}");
            assert_eq!(
                single.parameter_revision(),
                batch.parameter_revision(),
                "revision diverged for {label}"
            );
            for probe in [known, other, locked, bound] {
                assert_eq!(
                    single.param_value(probe).to_bits(),
                    batch.param_value(probe).to_bits(),
                    "value of {probe:?} diverged for {label}"
                );
            }
        }

        // NaN payloads never compare equal.
        let mut batch = arena.clone();
        let mut single = arena.clone();
        assert!(matches!(
            batch.set_param_values(&[(known, f64::NAN)]),
            Err(crate::ModelError::ParamDomain { value, .. }) if value.is_nan()
        ));
        assert!(matches!(
            single.set_param_value(known, f64::NAN),
            Err(crate::ModelError::ParamDomain { value, .. }) if value.is_nan()
        ));
    }

    #[test]
    fn expression_handles_are_send_and_sync() {
        assert_send_sync::<Expr<'static>>();
        assert_send_sync::<ExprArenaCell>();
    }

    #[test]
    fn write_guard_reentry_panics_and_cleans_up_instead_of_blocking() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&cell, VarId(0));

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = cell.borrow_mut();
            std::hint::black_box(x + 1.0);
        }));

        assert!(result.is_err());
        let constant = Expr::constant(&cell, 4.0);
        assert!(matches!(cell.borrow().get(constant.id), ExprNode::Const(4.0)));
    }

    #[test]
    fn forked_nodes_merge_and_remap_in_order() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&cell, VarId(0));
        let y = Expr::from_var(&cell, VarId(1));
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![cell.__with_fork(snapshot.clone(), || {
            let nonlinear = (x.sin() * y.cos()).exp();
            let quotient = nonlinear / (x.abs() + 2.0);
            quotient.powi(2)
        })];
        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let root = remaps[0].apply(forks[0].value.id);
        drop(batch);

        let arena = cell.borrow();
        let values: &[f64] = &[0.5, 0.25];
        let value = evaluate(&arena, root, &values).unwrap();
        let expected = ((0.5_f64.sin() * 0.25_f64.cos()).exp() / 2.5).powi(2);
        assert!((value - expected).abs() < 1e-12);
    }

    #[test]
    fn classification_caches_do_not_alias_worker_local_ids() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&cell, VarId(0));
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![
            cell.__with_fork(snapshot.clone(), || {
                let root = (x.square() + x).as_dynamic();
                for _ in 0..2 {
                    assert_eq!(root.__class(), ExprClass::Quadratic);
                }
                root.id()
            }),
            cell.__with_fork(snapshot.clone(), || {
                let root = x.sin().as_dynamic();
                for _ in 0..2 {
                    assert_eq!(root.__class(), ExprClass::Nonlinear);
                }
                root.id()
            }),
        ];
        assert_eq!(forks[0].value, forks[1].value);
        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let roots: Vec<_> =
            forks.iter().zip(remaps).map(|(fork, remap)| remap.apply(fork.value)).collect();
        drop(batch);
        assert_eq!(Expr::new(roots[0], &cell).__class(), ExprClass::Quadratic);
        assert_eq!(Expr::new(roots[1], &cell).__class(), ExprClass::Nonlinear);
    }

    #[test]
    fn merge_releases_fork_base_before_reserving() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![cell.__with_fork(snapshot.clone(), || Expr::constant(&cell, 1.0))];
        drop(snapshot);
        let nodes = Arc::as_ptr(&batch.arena.nodes);

        let remaps = batch.merge(&mut forks);

        assert_eq!(Arc::as_ptr(&batch.arena.nodes), nodes);
        assert_eq!(remaps[0].local_base, 0);
        assert_eq!(remaps[0].local_len, 1);
    }

    #[test]
    fn invalid_later_fork_does_not_append_earlier_nodes() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![
            cell.__with_fork(snapshot.clone(), || Expr::constant(&cell, 1.0)),
            cell.__with_fork(snapshot, || Expr::constant(&cell, 2.0)),
        ];
        forks[1].nodes.push(ExprNode::Unary(UnaryOp::Neg, ExprId(u32::MAX)));
        assert!(catch_unwind(AssertUnwindSafe(|| batch.merge(&mut forks))).is_err());
        assert_eq!(batch.arena.len(), 0);
        assert_eq!(forks[0].nodes.len(), 1);
        assert_eq!(forks[1].nodes.len(), 2);
    }

    #[test]
    fn merged_linear_node_still_supports_borrowed_extraction() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&cell, VarId(0));
        let mut batch = cell.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![cell.__with_fork(snapshot.clone(), || 3.0 * x + 2.0)];
        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let root = remaps[0].apply(forks[0].value.id);
        drop(batch);

        let arena = cell.borrow();
        let terms = extract_linear(&arena, root).unwrap();
        assert!(matches!(terms.coeffs, std::borrow::Cow::Borrowed(_)));
        assert_eq!(terms.coeffs.as_ref(), &[(VarId(0), 3.0)]);
        assert!((terms.constant - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn consuming_affine_transform_retires_copied_source_roots() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&cell, VarId(0));
        let mut result = None;
        for _ in 0..10_000 {
            result = Some((x + x) * 2.0);
        }

        let arena = cell.inner.lock();
        let ownership = arena.ownership.get().unwrap().lock();
        assert!(ownership.affine_owned.is_empty());
        assert!(!ownership.needs_child_sealing());

        drop(ownership);
        drop(arena);
        let expression = result.unwrap();
        assert_eq!(evaluate(&cell.borrow(), expression.id(), &&[3.0][..]).unwrap(), 12.0);
    }

    #[test]
    fn consuming_sums_retire_copied_affine_roots() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&cell, VarId(0));
        let y = Expr::from_var(&cell, VarId(1));
        let mut result = None;
        for index in 0..10_000 {
            let terms = [(x + y).into_dynamic(), x.square().into_dynamic()];
            result = Some(match index % 3 {
                0 => {
                    let sum: Expr<'_, Dynamic> = terms.into_iter().sum();
                    sum
                }
                1 => Expr::<Dynamic>::__sum_terms(terms.into_iter()).unwrap(),
                _ => Expr::<Dynamic>::__sum_terms_in(&cell, terms.into_iter()),
            });
        }

        let arena = cell.inner.lock();
        let ownership = arena.ownership.get().unwrap().lock();
        assert!(ownership.affine_owned.is_empty());
        assert!(!ownership.affine_owned.spilled());

        drop(ownership);
        drop(arena);
        let expression = result.unwrap();
        assert_eq!(evaluate(&cell.borrow(), expression.id(), &&[3.0, 2.0][..]).unwrap(), 14.0);
    }

    #[test]
    fn panicking_fork_cleans_up_routing_state() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let result = catch_unwind(AssertUnwindSafe(|| {
            let batch = cell.__begin_batch();
            let snapshot = batch.snapshot();
            let _: ForkOutput<()> = cell.__with_fork(snapshot, || panic!("worker failed"));
            drop(batch);
        }));
        assert!(result.is_err());
        let constant = Expr::constant(&cell, 4.0);
        assert!(matches!(cell.borrow().get(constant.id), ExprNode::Const(4.0)));
    }

    #[test]
    #[should_panic(expected = "snapshot requested during an active indexed batch")]
    fn snapshot_read_during_batch_is_rejected_instead_of_blocking() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let _batch = cell.__begin_batch();
        let _snapshot = cell.borrow();
    }

    #[test]
    #[should_panic(expected = "stale arena snapshot")]
    fn merge_rejects_same_length_but_different_snapshot() {
        let cell = ExprArenaCell::new(ExprArena::new());
        let other = ExprArenaCell::new(ExprArena::new());
        let batch = cell.__begin_batch();
        let other_batch = other.__begin_batch();
        let stale = other_batch.snapshot();
        drop(other_batch);
        let mut forks = vec![cell.__with_fork(stale, || Expr::constant(&cell, 1.0))];
        let mut batch = batch;
        let _ = batch.merge(&mut forks);
    }
}
