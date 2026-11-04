use crate::arena::{ArenaAccess, ExprArena, ExprId, ExprNode, UnaryOp};
use crate::fold::Folder;

/// Snapshot-local symbolic classification work.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassificationStats {
    pub calls: u64,
    pub cache_misses: u64,
    pub visited_nodes: u64,
    pub retained_roots: usize,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct ClassificationCounters {
    pub enabled: std::sync::atomic::AtomicBool,
    pub calls: std::sync::atomic::AtomicU64,
    pub misses: std::sync::atomic::AtomicU64,
    pub visits: std::sync::atomic::AtomicU64,
}

/// Ordinary arenas retain eight roots. Preparation snapshots opt into a larger
/// bounded cache.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExprClassCache {
    entries: smallvec::SmallVec<[(ExprId, Degree); 8]>,
    preparation: Option<Box<PreparationClasses>>,
}

const PREPARATION_ROOTS: usize = 131_072;

#[derive(Clone, Debug)]
struct PreparationClasses {
    entries: rustc_hash::FxHashMap<ExprId, Degree>,
    order: std::collections::VecDeque<ExprId>,
    limit: usize,
}

impl ExprClassCache {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.preparation.as_ref().map_or(self.entries.len(), |p| p.entries.len())
    }

    pub(crate) fn for_preparation() -> Self {
        Self {
            preparation: Some(Box::new(PreparationClasses {
                entries: rustc_hash::FxHashMap::default(),
                order: std::collections::VecDeque::new(),
                limit: PREPARATION_ROOTS,
            })),
            ..Self::default()
        }
    }
    pub(crate) fn get(&mut self, id: ExprId) -> Option<Degree> {
        if let Some(preparation) = &self.preparation {
            return preparation.entries.get(&id).copied();
        }
        let position = self.entries.iter().rposition(|&(root, _)| root == id)?;
        let entry = self.entries[position];
        if position + 1 != self.entries.len() {
            self.entries.remove(position);
            self.entries.push(entry);
        }
        Some(entry.1)
    }

    pub(crate) fn insert(&mut self, id: ExprId, class: Degree) {
        if let Some(preparation) = &mut self.preparation {
            if let Some(value) = preparation.entries.get_mut(&id) {
                *value = class;
                return;
            }
            if preparation.entries.len() == preparation.limit {
                let oldest = preparation.order.pop_front().expect("nonempty bounded cache");
                preparation.entries.remove(&oldest);
            }
            preparation.entries.insert(id, class);
            preparation.order.push_back(id);
            return;
        }
        if let Some(position) = self.entries.iter().position(|&(root, _)| root == id) {
            self.entries.remove(position);
        } else if self.entries.len() == 8 {
            self.entries.remove(0);
        }
        self.entries.push((id, class));
    }
}

/// Highest-degree polynomial class an expression belongs to, ignoring constant
/// folding. Used by backends to pick between linear, quadratic, and general
/// nonlinear translation paths.
///
/// Variants are ordered by increasing degree, so `max` of two classes yields the
/// dominating one (e.g. a model with a quadratic objective and a nonlinear
/// constraint is `Nonlinear`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExprClass {
    Linear,
    Quadratic,
    Nonlinear,
}

/// Polynomial-degree bucket. `Higher` is a saturating sentinel for "anything
/// above quadratic". Both polynomial degree > 2 and transcendentals collapse
/// into it, since neither fits a QP solver's quadratic API.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Degree {
    #[default]
    Zero,
    One,
    Two,
    Higher,
}

impl Degree {
    fn class(self) -> ExprClass {
        match self {
            Self::Zero | Self::One => ExprClass::Linear,
            Self::Two => ExprClass::Quadratic,
            Self::Higher => ExprClass::Nonlinear,
        }
    }

    /// `+` on a sum: take the maximum, saturating at `Higher`.
    pub(crate) fn add(self, other: Degree) -> Degree {
        self.max(other)
    }

    /// `*` on a product: add ordinal degrees, saturating at `Higher`.
    pub(crate) fn mul(self, other: Degree) -> Degree {
        match (self, other) {
            (Degree::Higher, _) | (_, Degree::Higher) => Degree::Higher,
            (Degree::Zero, x) | (x, Degree::Zero) => x,
            (Degree::One, Degree::One) => Degree::Two,
            _ => Degree::Higher,
        }
    }

    /// `^n` on a power: multiply by `n`, saturating at `Higher`.
    pub(crate) fn pow(self, n: u32) -> Degree {
        match (self, n) {
            (_, 0) | (Degree::Zero, _) => Degree::Zero,
            (d, 1) => d,
            (Degree::One, 2) => Degree::Two,
            _ => Degree::Higher,
        }
    }
}

#[cfg(test)]
fn recursive_degree(arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> Degree {
    match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Param(_) => Degree::Zero,
        ExprNode::Var(_) => Degree::One,
        ExprNode::Linear { coeffs, .. } => {
            if coeffs.is_empty() {
                Degree::Zero
            } else {
                Degree::One
            }
        }
        ExprNode::Unary(UnaryOp::Neg, inner) => recursive_degree(arena, *inner),
        ExprNode::Add(children) => {
            let mut d = Degree::Zero;
            for c in children {
                d = d.add(recursive_degree(arena, *c));
                if d == Degree::Higher {
                    return d;
                }
            }
            d
        }
        ExprNode::Mul(children) => {
            let mut d = Degree::Zero;
            for c in children {
                d = d.mul(recursive_degree(arena, *c));
                if d == Degree::Higher {
                    return d;
                }
            }
            d
        }
        ExprNode::Pow(base, exp) => {
            let ExprNode::Const(e) = arena.get(*exp) else { return Degree::Higher };
            if !e.is_finite() || (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
                return Degree::Higher;
            }
            // Bucket the exponent into the only values `Degree::pow` treats
            // distinctly.
            let n = match e.round() {
                v if v < 0.5 => 0,
                v if v < 1.5 => 1,
                v if v < 2.5 => 2,
                _ => 3,
            };
            recursive_degree(arena, *base).pow(n)
        }
        // Transcendentals are always > quadratic. Division is too: `div_into`
        // folds the only degree-preserving case (constant denominator) before a
        // `Div` node is created, so any other `Div` has a non-constant
        // denominator.
        ExprNode::Div(_, _)
        | ExprNode::Atan2(_, _)
        | ExprNode::Min(_)
        | ExprNode::Max(_)
        | ExprNode::Unary(_, _) => Degree::Higher,
    }
}

struct DegreeFolder<'a, A: ?Sized>(&'a A);

enum DegreeOp {
    Add,
    Mul,
    Pow(u32),
}
struct DegreeState<'a> {
    rest: &'a [ExprId],
    value: Degree,
    op: DegreeOp,
}

impl<'a, A: ArenaAccess + ?Sized> crate::fold::Folder for DegreeFolder<'a, A> {
    type Value = Degree;
    type State = DegreeState<'a>;

    fn start(&self, id: ExprId) -> std::ops::ControlFlow<Option<Degree>, Self::State> {
        use std::ops::ControlFlow::{Break, Continue};
        #[cfg(test)]
        self.0.record_classification_visit();

        if matches!(
            self.0.get(id),
            ExprNode::Add(_)
                | ExprNode::Mul(_)
                | ExprNode::Pow(_, _)
                | ExprNode::Unary(UnaryOp::Neg, _)
        ) && let Some(value) = self.0.cached_degree(id)
        {
            return Break(Some(value));
        }
        let (rest, op) = match self.0.get(id) {
            ExprNode::Const(_) | ExprNode::Param(_) => return Break(Some(Degree::Zero)),
            ExprNode::Var(_) => return Break(Some(Degree::One)),
            ExprNode::Linear { coeffs, .. } => {
                return Break(Some(if coeffs.is_empty() { Degree::Zero } else { Degree::One }));
            }
            ExprNode::Add(c) => (c.as_slice(), DegreeOp::Add),
            ExprNode::Mul(c) => (c.as_slice(), DegreeOp::Mul),
            ExprNode::Unary(UnaryOp::Neg, c) => (std::slice::from_ref(c), DegreeOp::Add),
            ExprNode::Pow(base, exp) => {
                let ExprNode::Const(e) = self.0.get(*exp) else {
                    return Break(Some(Degree::Higher));
                };
                if !e.is_finite() || (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
                    return Break(Some(Degree::Higher));
                }
                let n = match e.round() {
                    v if v < 0.5 => 0,
                    v if v < 1.5 => 1,
                    v if v < 2.5 => 2,
                    _ => 3,
                };
                (std::slice::from_ref(base), DegreeOp::Pow(n))
            }
            _ => return Break(Some(Degree::Higher)),
        };
        Continue(DegreeState { rest, value: Degree::Zero, op })
    }

    fn next(&self, state: &mut Self::State) -> std::ops::ControlFlow<Option<Degree>, ExprId> {
        use std::ops::ControlFlow::{Break, Continue};
        while state.value != Degree::Higher {
            let Some((&child, tail)) = state.rest.split_first() else { break };
            state.rest = tail;
            match self.start(child) {
                Break(Some(value)) => self.accept(state, value),
                _ => return Continue(child),
            }
        }
        Break(Some(state.value))
    }

    fn accept(&self, state: &mut Self::State, value: Degree) {
        state.value = match state.op {
            DegreeOp::Add => state.value.add(value),
            DegreeOp::Mul => state.value.mul(value),
            DegreeOp::Pow(n) => value.pow(n),
        };
    }
}

fn degree(arena: &(impl ArenaAccess + ?Sized), mut id: ExprId) -> Degree {
    let root = id;
    while let ExprNode::Unary(UnaryOp::Neg, child) = arena.get(id) {
        id = *child;
    }
    let value =
        crate::fold::fold(arena, id, DegreeFolder(arena)).expect("degree classification is total");
    if matches!(
        arena.get(root),
        ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Pow(_, _)
            | ExprNode::Unary(UnaryOp::Neg, _)
    ) {
        arena.cache_degree(root, value);
    }
    value
}

pub(crate) fn is_constant_access(arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> bool {
    classify_degree(arena, id) == Degree::Zero
}

/// Classify an expression as Linear, Quadratic (polynomial degree <= 2 with at
/// least one degree-2 term), or Nonlinear (transcendentals, non-integer powers,
/// or polynomial degree > 2).
pub fn classify(arena: &ExprArena, id: ExprId) -> ExprClass {
    classify_access(arena, id)
}

pub(crate) fn classify_access(arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> ExprClass {
    classify_degree(arena, id).class()
}

fn classify_degree(arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> Degree {
    let cacheable = matches!(
        arena.get(id),
        ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Pow(_, _)
            | ExprNode::Unary(UnaryOp::Neg, _)
    );
    let cached = if cacheable { arena.cached_degree(id) } else { None };
    #[cfg(test)]
    arena.record_classification(cacheable && cached.is_none());

    if let Some(value) = cached {
        return value;
    }

    // Finish small, shallow expressions without traversal metadata.
    // Remember compound results even when a cached child completes the shallow path.
    let small = match arena.get(id) {
        ExprNode::Add(children) | ExprNode::Mul(children) => children.len() <= 8,
        _ => true,
    };
    if small {
        let folder = DegreeFolder(arena);
        match folder.start(id) {
            std::ops::ControlFlow::Break(Some(value)) => return value,
            std::ops::ControlFlow::Continue(mut state) => {
                if let std::ops::ControlFlow::Break(Some(value)) = folder.next(&mut state) {
                    arena.cache_degree(id, value);
                    return value;
                }
            }
            std::ops::ControlFlow::Break(None) => unreachable!("degree classification is total"),
        }
    }
    // Expensive roots are cached independently in canonical arenas, snapshots,
    // and worker forks.
    if let Some(value) = arena.preparation_degree(id) {
        arena.cache_degree(id, value);
        value
    } else {
        degree(arena, id)
    }
}

/// Bound the probe so preparation retains early nonlinear exits without doing
/// a second full degree traversal before structural summary inference.
pub(crate) fn degree_hint(
    arena: &(impl ArenaAccess + ?Sized),
    id: ExprId,
    budget: &mut usize,
    depth: usize,
) -> Option<Degree> {
    if depth == 8 {
        return None;
    }

    *budget = budget.checked_sub(1)?;
    let value = match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Param(_) => Degree::Zero,
        ExprNode::Var(_) => Degree::One,
        ExprNode::Linear { coeffs, .. } => {
            if coeffs.is_empty() {
                Degree::Zero
            } else {
                Degree::One
            }
        }
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            let multiply = matches!(arena.get(id), ExprNode::Mul(_));
            let mut value = Degree::Zero;
            for &child in children {
                let next = degree_hint(arena, child, budget, depth + 1)?;
                value = if multiply { value.mul(next) } else { value.add(next) };
                if value == Degree::Higher {
                    break;
                }
            }
            value
        }
        ExprNode::Unary(UnaryOp::Neg, child) => degree_hint(arena, *child, budget, depth + 1)?,
        ExprNode::Pow(base, exponent) => {
            let ExprNode::Const(e) = arena.get(*exponent) else { return Some(Degree::Higher) };
            if !e.is_finite() || (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
                return Some(Degree::Higher);
            }
            let n = match e.round() {
                v if v < 0.5 => 0,
                v if v < 1.5 => 1,
                v if v < 2.5 => 2,
                _ => 3,
            };
            degree_hint(arena, *base, budget, depth + 1)?.pow(n)
        }
        _ => Degree::Higher,
    };

    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_retains_101369_roots_without_revisiting_the_graph() {
        let mut arena = ExprArena::new();
        let x = arena.var(crate::VarId(0));
        let roots: Vec<_> =
            (0..101_369).map(|_| arena.push(ExprNode::Mul(smallvec::smallvec![x, x]))).collect();
        arena.__enable_preparation_classifications();
        arena.enable_classification_counters();
        for &root in &roots {
            assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        }

        let before = arena.classification_stats();
        for &root in &roots {
            assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        }

        let after = arena.classification_stats();
        assert_eq!(after.calls - before.calls, 101_369);
        assert_eq!(after.cache_misses, before.cache_misses);
        assert_eq!(after.visited_nodes, before.visited_nodes);
        assert_eq!(after.retained_roots, 101_369);

        let clone = arena.clone();
        assert_eq!(clone.classification_stats().retained_roots, 0);
        assert_eq!(clone.classification_stats().calls, 0);
    }

    #[test]
    fn preparation_degrees_remain_symbolic_under_zero_and_cancellation() {
        let mut arena = ExprArena::new();
        let p = arena.new_param(0.0);
        let parameter = arena.param(p);
        let x = arena.var(crate::VarId(0));
        let q = arena.push(ExprNode::Mul(smallvec::smallvec![parameter, x, x]));
        let neg = arena.push(ExprNode::Unary(UnaryOp::Neg, q));
        let cancel = arena.push(ExprNode::Add(smallvec::smallvec![q, neg]));
        arena.__enable_preparation_classifications();
        for value in [0.0, 1.0, -2.0, 0.0] {
            arena.set_param_value(p, value).unwrap();
            assert_eq!(classify(&arena, q), ExprClass::Quadratic);
            assert_eq!(classify(&arena, cancel), ExprClass::Quadratic);
            assert_eq!(crate::extract_quadratic(&arena, cancel).unwrap().hessian.len(), 0);
            assert!(is_constant_access(&arena, parameter));
            assert!(!is_constant_access(&arena, x));
        }
    }

    #[test]
    fn preparation_cache_evicts_at_its_bound() {
        let mut cache = ExprClassCache::for_preparation();
        cache.preparation.as_mut().unwrap().limit = 4;
        for i in 0..20 {
            cache.insert(ExprId(i), Degree::Two);
        }

        assert_eq!(cache.len(), 4);
        assert_eq!(cache.get(ExprId(0)), None);
        assert_eq!(cache.get(ExprId(19)), Some(Degree::Two));
        assert_eq!(cache.preparation.as_ref().unwrap().order.len(), 4);
    }

    #[test]
    fn iterative_degree_matches_recursive_classification() {
        let (arena, ids) = crate::fold::test_arena();
        for id in ids {
            assert_eq!(degree(&arena, id), recursive_degree(&arena, id), "node {id:?}");
        }
    }
    use crate::arena::{ExprArena, ExprNode, VarId};
    use smallvec::smallvec;

    #[test]
    fn cached_roots_follow_mutation_without_affecting_snapshots() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let y = arena.var(VarId(1));
        let product = arena.push(ExprNode::Mul(smallvec![x, y]));
        let root = arena.push(ExprNode::Add(smallvec![product, product]));
        assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        let snapshot = arena.clone();
        *arena.get_mut(y) = ExprNode::Unary(UnaryOp::Sin, x);
        assert_eq!(classify(&arena, root), ExprClass::Nonlinear);
        assert_eq!(classify(&snapshot, root), ExprClass::Quadratic);
    }

    #[test]
    fn cloned_arenas_cache_divergent_nodes_independently() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let square = arena.push(ExprNode::Mul(smallvec![x, x]));
        let sine = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
        let mut clone = arena.clone();
        let left = arena.push(ExprNode::Add(smallvec![square, x]));
        let right = clone.push(ExprNode::Add(smallvec![sine, x]));
        assert_eq!(left, right);
        for _ in 0..2 {
            assert_eq!(classify(&arena, left), ExprClass::Quadratic);
            assert_eq!(classify(&clone, right), ExprClass::Nonlinear);
        }
    }

    #[test]
    fn classification_cache_is_independent_of_parameter_values() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let p = arena.new_param(0.0);
        let param = arena.param(p);
        let square = arena.push(ExprNode::Mul(smallvec![x, x]));
        let root = arena.push(ExprNode::Mul(smallvec![param, square]));
        assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        arena.set_param_value(p, 3.0).unwrap();
        assert_eq!(classify(&arena, root), ExprClass::Quadratic);
    }

    #[test]
    fn constant_linear_nodes_do_not_raise_product_degree() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let constant = arena.linear(Vec::new(), 2.0);
        let affine = arena.push(ExprNode::Mul(smallvec![constant, x]));
        let quadratic = arena.push(ExprNode::Mul(smallvec![affine, x]));
        assert_eq!(classify(&arena, affine), ExprClass::Linear);
        assert_eq!(classify(&arena, quadratic), ExprClass::Quadratic);
    }

    fn var(arena: &mut ExprArena, i: u32) -> ExprId {
        arena.push(ExprNode::Var(VarId(i)))
    }

    #[test]
    fn linear_var_sum() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let sum = a.push(ExprNode::Add(smallvec![x, y]));
        assert_eq!(classify(&a, sum), ExprClass::Linear);
    }

    #[test]
    fn quadratic_mul_two_vars() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let xy = a.push(ExprNode::Mul(smallvec![x, y]));
        assert_eq!(classify(&a, xy), ExprClass::Quadratic);
    }

    #[test]
    fn quadratic_pow_two() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(x, two));
        assert_eq!(classify(&a, sq), ExprClass::Quadratic);
    }

    #[test]
    fn nonlinear_pow_three() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let three = a.push(ExprNode::Const(3.0));
        let cube = a.push(ExprNode::Pow(x, three));
        assert_eq!(classify(&a, cube), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_div() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let q = a.push(ExprNode::Div(x, y));
        assert_eq!(classify(&a, q), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_sin() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let s = a.push(ExprNode::Unary(UnaryOp::Sin, x));
        assert_eq!(classify(&a, s), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_abs() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let s = a.push(ExprNode::Unary(UnaryOp::Abs, x));
        assert_eq!(classify(&a, s), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_triple_mul() {
        let mut arena = ExprArena::new();
        let x = var(&mut arena, 0);
        let y = var(&mut arena, 1);
        let z = var(&mut arena, 2);
        let prod = arena.push(ExprNode::Mul(smallvec![x, y, z]));
        assert_eq!(classify(&arena, prod), ExprClass::Nonlinear);
    }

    #[test]
    fn linear_promoted_by_const_mul() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let c = a.push(ExprNode::Const(3.0));
        let m = a.push(ExprNode::Mul(smallvec![c, x]));
        assert_eq!(classify(&a, m), ExprClass::Linear);
    }

    #[test]
    fn param_alone_is_linear() {
        let mut a = ExprArena::new();
        let p = a.new_param(4.0);
        let pn = a.param(p);
        assert_eq!(classify(&a, pn), ExprClass::Linear);
    }

    #[test]
    fn param_times_var_is_linear() {
        let mut a = ExprArena::new();
        let p = a.new_param(4.0);
        let pn = a.param(p);
        let x = var(&mut a, 0);
        let m = a.push(ExprNode::Mul(smallvec![pn, x]));
        assert_eq!(classify(&a, m), ExprClass::Linear);
    }

    #[test]
    fn param_times_var_squared_is_quadratic() {
        let mut a = ExprArena::new();
        let p = a.new_param(4.0);
        let pn = a.param(p);
        let x = var(&mut a, 0);
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(x, two));
        let m = a.push(ExprNode::Mul(smallvec![pn, sq]));
        assert_eq!(classify(&a, m), ExprClass::Quadratic);
    }
}
