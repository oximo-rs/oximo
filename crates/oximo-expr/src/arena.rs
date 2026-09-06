use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

impl ExprId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
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

/// Here we use a linear fast-path: `sum(coeff * var) + constant`.
/// Built by the operator overloads when all children are linear,
/// so LP/MILP construction never walks an `Add(Mul(Const, Var), ...)` tree.

#[derive(Clone, Debug)]
pub enum ExprNode {
    Const(f64),
    Var(VarId),
    Param(ParamId),
    Add(Children),
    Mul(Children),
    Neg(ExprId),
    Pow(ExprId, ExprId),
    Div(ExprId, ExprId),
    Sin(ExprId),
    Cos(ExprId),
    Exp(ExprId),
    Log(ExprId),
    Abs(ExprId),
    Linear { coeffs: Vec<(VarId, f64)>, constant: f64 },
}

#[derive(Clone, Debug, Default)]
pub struct ExprArena {
    nodes: Arc<Vec<ExprNode>>,
    param_values: Arc<Vec<f64>>,
}

impl ExprArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Arc::new(Vec::with_capacity(cap)), param_values: Arc::new(Vec::new()) }
    }

    /// Clone this arena while reserving room for nodes that will immediately
    /// be appended to the copy.
    #[doc(hidden)]
    #[must_use]
    pub fn __clone_with_additional_capacity(&self, additional_nodes: usize) -> Self {
        let mut nodes = Vec::with_capacity(self.nodes.len().saturating_add(additional_nodes));
        nodes.extend_from_slice(&self.nodes);
        Self { nodes: Arc::new(nodes), param_values: Arc::new(self.param_values.as_ref().clone()) }
    }

    /// Reserve room for additional expression nodes without changing IDs.
    #[doc(hidden)]
    pub fn __reserve_nodes(&mut self, additional: usize) {
        Arc::make_mut(&mut self.nodes).reserve(additional);
    }

    #[doc(hidden)]
    pub(crate) fn __append_nodes(&mut self, nodes: Vec<ExprNode>) {
        Arc::make_mut(&mut self.nodes).extend(nodes);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// # Panics
    ///
    /// Panics if the number of expressions exceeds `u32::MAX` (expression arena overflow).
    pub fn push(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(u32::try_from(self.nodes.len()).expect("expression arena overflow"));
        Arc::make_mut(&mut self.nodes).push(node);
        id
    }

    #[inline]
    pub fn get(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, id: ExprId) -> &mut ExprNode {
        &mut Arc::make_mut(&mut self.nodes)[id.index()]
    }

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
        let id = ParamId(u32::try_from(self.param_values.len()).expect("parameter arena overflow"));
        Arc::make_mut(&mut self.param_values).push(value);
        id
    }

    #[inline]
    pub fn num_params(&self) -> usize {
        self.param_values.len()
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
    /// # Panics
    ///
    /// Panics if `p` was not allocated by [`Self::new_param`] on this arena.
    #[inline]
    pub fn set_param_value(&mut self, p: ParamId, value: f64) {
        Arc::make_mut(&mut self.param_values)[p.index()] = value;
    }

    pub fn linear(&mut self, coeffs: Vec<(VarId, f64)>, constant: f64) -> ExprId {
        self.push(ExprNode::Linear { coeffs, constant })
    }
}

/// Read/push surface shared by the canonical and worker-local arenas.
pub(crate) trait ArenaAccess {
    fn get(&self, id: ExprId) -> &ExprNode;
    fn param_value(&self, id: ParamId) -> f64;
    fn push(&mut self, node: ExprNode) -> ExprId;

    fn constant(&mut self, value: f64) -> ExprId {
        self.push(ExprNode::Const(value))
    }

    fn var(&mut self, id: VarId) -> ExprId {
        self.push(ExprNode::Var(id))
    }
}

impl ArenaAccess for ExprArena {
    fn get(&self, id: ExprId) -> &ExprNode {
        self.get(id)
    }

    fn param_value(&self, id: ParamId) -> f64 {
        self.param_value(id)
    }

    fn push(&mut self, node: ExprNode) -> ExprId {
        self.push(node)
    }
}

#[derive(Clone, Debug, Default)]
#[doc(hidden)]
pub struct FrozenExprArena {
    nodes: Arc<Vec<ExprNode>>,
    param_values: Arc<Vec<f64>>,
}

impl FrozenExprArena {
    fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug)]
struct ForkedExprArena {
    base: FrozenExprArena,
    nodes: Vec<ExprNode>,
}

impl ForkedExprArena {
    fn new(base: FrozenExprArena) -> Self {
        Self { base, nodes: Vec::new() }
    }
}

impl ArenaAccess for ForkedExprArena {
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
        let index =
            self.base.nodes.len().checked_add(self.nodes.len()).expect("expression arena overflow");
        let id = ExprId(u32::try_from(index).expect("expression arena overflow"));
        self.nodes.push(node);
        id
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
#[derive(Debug, Default)]
pub struct ExprArenaCell {
    inner: Mutex<ExprArena>,
    state: AtomicU8,
}

impl ExprArenaCell {
    pub fn new(arena: ExprArena) -> Self {
        Self { inner: Mutex::new(arena), state: AtomicU8::new(0) }
    }

    /// Take a cheap immutable snapshot of the canonical arena.
    ///
    /// The returned value is not a live read lock.
    /// Later writes are isolated by copy-on-write and are not
    /// visible through this snapshot.
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

    pub(crate) fn with_ref<R>(&self, f: impl FnOnce(&dyn ArenaAccess) -> R) -> R {
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

    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(&mut dyn ArenaAccess) -> R) -> R {
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
    // Remove the thread-local marker before unlocking the mutex. There is no
    // user code between field drops, so another operation cannot observe the
    // short transition.
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

impl DerefMut for ExprArenaWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
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
    pub fn snapshot(&self) -> FrozenExprArena {
        FrozenExprArena {
            nodes: Arc::clone(&self.arena.nodes),
            param_values: Arc::clone(&self.arena.param_values),
        }
    }

    /// Append worker-fork nodes in slice order and return each fork's ID map.
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

fn validate_node(node: &ExprNode, remap: ExprIdRemap) {
    match node {
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            for child in children {
                remap.validate(*child);
            }
        }
        ExprNode::Neg(child)
        | ExprNode::Sin(child)
        | ExprNode::Cos(child)
        | ExprNode::Exp(child)
        | ExprNode::Log(child)
        | ExprNode::Abs(child) => remap.validate(*child),
        ExprNode::Pow(left, right) | ExprNode::Div(left, right) => {
            remap.validate(*left);
            remap.validate(*right);
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {}
    }
}

impl Drop for ExprArenaBatchGuard<'_> {
    fn drop(&mut self) {
        self.cell.state.fetch_and(!BATCH_ACTIVE, Ordering::Release);
    }
}

fn remap_node(node: &mut ExprNode, remap: ExprIdRemap) {
    match node {
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            for child in children {
                *child = remap.apply(*child);
            }
        }
        ExprNode::Neg(child)
        | ExprNode::Sin(child)
        | ExprNode::Cos(child)
        | ExprNode::Exp(child)
        | ExprNode::Log(child)
        | ExprNode::Abs(child) => *child = remap.apply(*child),
        ExprNode::Pow(left, right) | ExprNode::Div(left, right) => {
            *left = remap.apply(*left);
            *right = remap.apply(*right);
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::{Expr, evaluate, extract_linear};

    fn assert_send_sync<T: Send + Sync>() {}

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
        forks[1].nodes.push(ExprNode::Neg(ExprId(u32::MAX)));
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
