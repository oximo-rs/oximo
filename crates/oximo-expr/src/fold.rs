//! Streaming, stack-safe folds.
//! Only shared compound nodes retain results.
//! Unshared intermediates are moved into their parent immediately.
use std::ops::ControlFlow;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::arena::{ArenaAccess, ExprId, ExprNode};

/// Lazily allocated sum frames.
pub(crate) type SumFrames<'a> = Option<Box<Vec<(&'a [ExprId], f64)>>>;

/// A conservative absolute-sum bound on contributions to additive folds.
/// If this overflows, original grouping may have overflowed even when a
/// regrouped result is finite.
pub(crate) fn record_sum_magnitude(bound: &std::cell::Cell<f64>, contribution: f64) {
    let previous = bound.get();
    if previous.is_finite() {
        bound.set(previous + contribution.abs());
    }
}

/// Rare-path check before retrying a non-finite regrouped result.
pub(crate) fn finite_inputs(
    arena: &(impl ArenaAccess + ?Sized),
    root: ExprId,
    resolve_params: bool,
) -> bool {
    let mut seen = FxHashSet::default();
    let mut stack = SmallVec::<[ExprId; 16]>::from_slice(&[root]);
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.get(id) {
            ExprNode::Const(c) if !c.is_finite() => return false,
            ExprNode::Param(p) => {
                if !resolve_params || !arena.param_value(*p).is_finite() {
                    return false;
                }
            }
            ExprNode::Linear { coeffs, constant } => {
                if !constant.is_finite() || coeffs.iter().any(|(_, c)| !c.is_finite()) {
                    return false;
                }
            }
            ExprNode::Add(c) | ExprNode::Mul(c) | ExprNode::Min(c) | ExprNode::Max(c) => {
                stack.extend_from_slice(c);
            }
            ExprNode::Unary(_, child) => stack.push(*child),
            ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                stack.extend_from_slice(&[*a, *b]);
            }
            ExprNode::Const(_) | ExprNode::Var(_) => {}
        }
    }
    true
}

pub(crate) trait Folder {
    type Value: Clone;
    type State;
    fn start(&self, id: ExprId) -> ControlFlow<Option<Self::Value>, Self::State>;
    fn next(&self, state: &mut Self::State) -> ControlFlow<Option<Self::Value>, ExprId>;
    fn accept(&self, state: &mut Self::State, value: Self::Value);
    /// Inline an unshared child into the current accumulator, when compatible.
    fn inline(&self, _state: &mut Self::State, _child: ExprId) -> bool {
        false
    }
}

fn is_compound(arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> bool {
    matches!(
        arena.get(id),
        ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Min(_)
            | ExprNode::Max(_)
            | ExprNode::Unary(_, _)
            | ExprNode::Pow(_, _)
            | ExprNode::Div(_, _)
            | ExprNode::Atan2(_, _)
    )
}

fn shared_nodes(arena: &(impl ArenaAccess + ?Sized), root: ExprId) -> FxHashSet<ExprId> {
    let mut seen = FxHashSet::default();
    let mut shared = FxHashSet::default();
    let mut stack = SmallVec::<[ExprId; 16]>::new();
    // Only compound roots can have shared descendants. Leaves need no scan.
    if is_compound(arena, root) {
        seen.insert(root);
        stack.push(root);
    }
    while let Some(id) = stack.pop() {
        match arena.get(id) {
            ExprNode::Add(children)
            | ExprNode::Mul(children)
            | ExprNode::Min(children)
            | ExprNode::Max(children) => {
                for &child in children {
                    if !is_compound(arena, child) {
                        continue;
                    }
                    if seen.insert(child) {
                        stack.push(child);
                    } else {
                        shared.insert(child);
                    }
                }
            }
            ExprNode::Unary(_, inner) => {
                if is_compound(arena, *inner) {
                    if seen.insert(*inner) {
                        stack.push(*inner);
                    } else {
                        shared.insert(*inner);
                    }
                }
            }
            ExprNode::Pow(base, exp) | ExprNode::Div(base, exp) | ExprNode::Atan2(base, exp) => {
                for &child in &[*base, *exp] {
                    if !is_compound(arena, child) {
                        continue;
                    }
                    if seen.insert(child) {
                        stack.push(child);
                    } else {
                        shared.insert(child);
                    }
                }
            }
            ExprNode::Const(_)
            | ExprNode::Var(_)
            | ExprNode::Param(_)
            | ExprNode::Linear { .. } => {}
        }
    }
    shared
}

pub(crate) fn fold<F: Folder>(
    arena: &(impl ArenaAccess + ?Sized),
    root: ExprId,
    folder: F,
) -> Option<F::Value> {
    let mut state = match folder.start(root) {
        ControlFlow::Break(value) => return value,
        ControlFlow::Continue(state) => state,
    };
    // Flat sums are completed without a traversal stack or sharing metadata.
    let mut step = folder.next(&mut state);
    if let ControlFlow::Break(value) = step {
        return value;
    }
    let shared = shared_nodes(arena, root);
    let mut cache = FxHashMap::<ExprId, F::Value>::default();
    let mut stack = vec![(root, state)];
    loop {
        let (id, state) = stack.last_mut()?;
        match step {
            ControlFlow::Continue(child) => {
                if !shared.contains(&child) && folder.inline(state, child) {
                } else if let Some(value) = cache.get(&child) {
                    folder.accept(state, value.clone());
                } else {
                    match folder.start(child) {
                        ControlFlow::Break(value) => folder.accept(state, value?),
                        ControlFlow::Continue(state) => stack.push((child, state)),
                    }
                }
            }
            ControlFlow::Break(value) => {
                let value = value?;
                if shared.contains(id) {
                    cache.insert(*id, value.clone());
                }
                stack.pop();
                match stack.last_mut() {
                    Some((_, parent)) => folder.accept(parent, value),
                    None => return Some(value),
                }
            }
        }
        step = folder.next(&mut stack.last_mut()?.1);
    }
}

#[cfg(test)]
pub(crate) fn test_arena() -> (crate::ExprArena, Vec<ExprId>) {
    use crate::{UnaryOp, VarId};
    use smallvec::smallvec;
    let mut arena = crate::ExprArena::new();
    let mut ids = vec![];
    for c in [-0.0, 0.0, 1.0, 2.0, 3.0, -2.0, 1e16, -1e16, 1e-16] {
        ids.push(arena.push(ExprNode::Const(c)));
    }
    for v in 0..5 {
        ids.push(arena.push(ExprNode::Var(VarId(v))));
    }
    ids.push(arena.push(ExprNode::Linear {
        coeffs: vec![(VarId(2), -0.0), (VarId(0), 1e16), (VarId(0), -1e16)],
        constant: -0.0,
    }));
    let mut seed = 1_234_567_u32;
    let mut previous = 0..ids.len();
    for _ in 0..6 {
        let start = ids.len();
        for i in 0..40 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = ids[previous.start + seed as usize % previous.len()];
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = ids[previous.start + seed as usize % previous.len()];
            let node = match i % 8 {
                0 | 1 => ExprNode::Add(smallvec![a, b, a]),
                2 => ExprNode::Mul(smallvec![a, b]),
                3 => ExprNode::Unary(UnaryOp::Neg, a),
                4 => ExprNode::Pow(a, ids[(i / 8) % 5]),
                5 => ExprNode::Mul(smallvec![ids[6], a, ids[8]]),
                6 => ExprNode::Unary(UnaryOp::Sin, a),
                _ => ExprNode::Add(smallvec![a, ids[6], ids[7], b]),
            };
            ids.push(arena.push(node));
        }
        previous = start..ids.len();
    }
    (arena, ids)
}

#[cfg(test)]
mod tests {
    use crate::{
        ExprArena, ExprClass, ExprNode, UnaryOp, VarId, classify, extract_linear, extract_quadratic,
    };
    use smallvec::smallvec;

    #[test]
    fn deep_and_shared_graphs_are_stack_safe() {
        let mut arena = ExprArena::new();
        let mut root = arena.push(ExprNode::Var(VarId(0)));
        for _ in 0..30_000 {
            root = arena.push(ExprNode::Unary(UnaryOp::Neg, root));
        }
        assert_eq!(extract_linear(&arena, root).unwrap().coeffs.as_ref(), &[(VarId(0), 1.0)]);
        assert_eq!(extract_quadratic(&arena, root).unwrap().linear, vec![(VarId(0), 1.0)]);
        assert_eq!(classify(&arena, root), ExprClass::Linear);
        let zero = arena.push(ExprNode::Const(0.0));
        for _ in 0..12_000 {
            root = arena.push(ExprNode::Add(smallvec![zero, root]));
        }
        assert_eq!(extract_linear(&arena, root).unwrap().coeffs.as_ref(), &[(VarId(0), 1.0)]);
        assert_eq!(extract_quadratic(&arena, root).unwrap().linear, vec![(VarId(0), 1.0)]);
        assert_eq!(classify(&arena, root), ExprClass::Linear);
        for _ in 0..30 {
            root = arena.push(ExprNode::Add(smallvec![root, root]));
        }
        let expected = 2f64.powi(30);
        assert_eq!(extract_linear(&arena, root).unwrap().coeffs.as_ref(), &[(VarId(0), expected)]);
        assert_eq!(extract_quadratic(&arena, root).unwrap().linear, vec![(VarId(0), expected)]);
        assert_eq!(classify(&arena, root), ExprClass::Linear);
    }

    #[test]
    fn shared_cache_is_local_to_each_parameter_snapshot() {
        let mut arena = ExprArena::new();
        let parameter = arena.new_param(2.0);
        let p = arena.param(parameter);
        let x = arena.push(ExprNode::Var(VarId(0)));
        let term = arena.push(ExprNode::Mul(smallvec![p, x]));
        let root = arena.push(ExprNode::Add(smallvec![term, term]));
        let snapshot = arena.clone();
        arena.set_param_value(parameter, 3.0);
        for (source, expected) in [(&snapshot, 4.0), (&arena, 6.0)] {
            assert_eq!(
                extract_linear(source, root).unwrap().coeffs.as_ref(),
                &[(VarId(0), expected)]
            );
            assert_eq!(extract_quadratic(source, root).unwrap().linear, vec![(VarId(0), expected)]);
        }
    }
}
