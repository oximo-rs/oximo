use oximo_expr::{EvalContext, ParamId};
use oximo_expr::{
    ExprArena, ExprId, ExprNode, UnaryOp, VarId, Visitor, evaluate, walk, walk_shared,
};
use smallvec::smallvec;
use std::cell::Cell;

fn deep_neg_chain(arena: &mut ExprArena, depth: usize) -> ExprId {
    let mut root = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..depth {
        root = arena.push(ExprNode::Unary(UnaryOp::Neg, root));
    }
    root
}

#[test]
fn deep_chains_evaluate_without_overflowing() {
    let mut arena = ExprArena::new();
    let x: &[f64] = &[0.7];
    let neg = deep_neg_chain(&mut arena, 30_000);
    // Even depth: value is +x.
    assert_eq!(evaluate(&arena, neg, &x).unwrap(), 0.7);

    let zero = arena.push(ExprNode::Const(0.0));
    let mut root = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..12_000 {
        root = arena.push(ExprNode::Add(smallvec![zero, root]));
    }
    assert_eq!(evaluate(&arena, root, &x).unwrap(), 0.7);
}

#[test]
fn heavily_shared_doubling_evaluates_once_per_node() {
    let mut arena = ExprArena::new();
    let mut root = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..22 {
        root = arena.push(ExprNode::Add(smallvec![root, root]));
    }
    // 2^22 * 0.5
    let x: &[f64] = &[0.5];
    assert_eq!(evaluate(&arena, root, &x).unwrap(), 2f64.powi(22) * 0.5);
}

#[test]
fn shared_sin_fanout_matches_repeated_evaluation() {
    let mut arena = ExprArena::new();
    let x0 = arena.push(ExprNode::Var(VarId(0)));
    let sin = arena.push(ExprNode::Unary(UnaryOp::Sin, x0));
    let children: oximo_expr::Children = std::iter::repeat_n(sin, 128).collect();
    let root = arena.push(ExprNode::Add(children));
    let x: &[f64] = &[0.7];
    assert!((evaluate(&arena, root, &x).unwrap() - 128.0 * 0.7f64.sin()).abs() < 1e-12);
}

struct Counter {
    visits: usize,
}

impl Visitor for Counter {
    fn visit(&mut self, _arena: &ExprArena, _id: ExprId, _node: &ExprNode) {
        self.visits += 1;
    }
}

#[test]
fn walk_is_stack_safe_and_shared_visits_once() {
    let mut arena = ExprArena::new();
    let root = deep_neg_chain(&mut arena, 20_000);
    let mut counter = Counter { visits: 0 };
    walk(&arena, root, &mut counter);
    assert_eq!(counter.visits, 20_001);

    let mut shared = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..10 {
        shared = arena.push(ExprNode::Add(smallvec![shared, shared]));
    }
    let mut once = Counter { visits: 0 };
    walk_shared(&arena, shared, &mut once);
    // 1 var + 10 adds.
    assert_eq!(once.visits, 11);
}

#[derive(Default)]
struct Order(Vec<ExprId>);
impl Visitor for Order {
    fn visit(&mut self, _: &ExprArena, id: ExprId, _: &ExprNode) {
        self.0.push(id);
    }
}

#[test]
fn shared_walk_preserves_first_occurrence_preorder() {
    let mut arena = ExprArena::new();
    let a = arena.var(VarId(0));
    let b = arena.var(VarId(1));
    let repeated = arena.push(ExprNode::Add(smallvec![a, b, a]));
    let branch = arena.push(ExprNode::Add(smallvec![b, a]));
    for root in [
        repeated,
        arena.push(ExprNode::Pow(branch, b)),
        arena.push(ExprNode::Div(branch, b)),
        arena.push(ExprNode::Atan2(branch, b)),
    ] {
        let mut ordinary = Order::default();
        walk(&arena, root, &mut ordinary);
        let mut seen = std::collections::HashSet::new();
        ordinary.0.retain(|id| seen.insert(*id));
        let mut shared = Order::default();
        walk_shared(&arena, root, &mut shared);
        assert_eq!(shared.0, ordinary.0);
    }
}

struct Counted(Cell<usize>);
impl EvalContext for Counted {
    fn var(&self, _: VarId) -> Option<f64> {
        self.0.set(self.0.get() + 1);
        Some(0.5)
    }
    fn param(&self, _: ParamId) -> Option<f64> {
        None
    }
}

#[test]
fn fast_and_fallback_paths_query_shared_leaves_once() {
    let mut arena = ExprArena::new();
    let v = arena.var(VarId(0));
    let duplicate = arena.push(ExprNode::Add(smallvec![v, v]));
    let sin = arena.push(ExprNode::Unary(UnaryOp::Sin, v));
    let shared = arena.push(ExprNode::Add(smallvec![sin, sin]));
    let ctx = Counted(Cell::new(0));
    assert_eq!(evaluate(&arena, duplicate, &ctx).unwrap(), 1.0);
    assert_eq!(ctx.0.replace(0), 1);
    assert_eq!(evaluate(&arena, shared, &ctx).unwrap(), 2.0 * 0.5_f64.sin());
    assert_eq!(ctx.0.replace(0), 1);
    for _ in 0..65_536 {
        arena.push(ExprNode::Const(1.0));
    }
    assert_eq!(evaluate(&arena, shared, &ctx).unwrap(), 2.0 * 0.5_f64.sin());
    assert_eq!(ctx.0.get(), 1);
}

#[test]
fn nested_context_evaluation_and_unwinding_release_scratch() {
    struct Nested<'a>(&'a ExprArena, ExprId);
    impl EvalContext for Nested<'_> {
        fn var(&self, _: VarId) -> Option<f64> {
            let x: &[f64] = &[0.5];
            Some(evaluate(self.0, self.1, &x).unwrap())
        }
        fn param(&self, _: ParamId) -> Option<f64> {
            None
        }
    }
    struct Panics;
    impl EvalContext for Panics {
        fn var(&self, _: VarId) -> Option<f64> {
            panic!("context panic")
        }
        fn param(&self, _: ParamId) -> Option<f64> {
            None
        }
    }
    let mut arena = ExprArena::new();
    let v = arena.var(VarId(0));
    let sin = arena.push(ExprNode::Unary(UnaryOp::Sin, v));
    let root = arena.push(ExprNode::Add(smallvec![sin, sin]));
    assert_eq!(
        evaluate(&arena, root, &Nested(&arena, root)).unwrap(),
        2.0 * (2.0 * 0.5_f64.sin()).sin()
    );
    assert!(std::panic::catch_unwind(|| evaluate(&arena, root, &Panics)).is_err());
    let missing: &[f64] = &[];
    assert!(evaluate(&arena, root, &missing).is_err());
    let x: &[f64] = &[0.5];
    assert_eq!(evaluate(&arena, root, &x).unwrap(), 2.0 * 0.5_f64.sin());
}

#[test]
fn forward_dependencies_and_parameter_updates_are_live() {
    let mut arena = ExprArena::new();
    let root = arena.push(ExprNode::Const(0.0));
    let p = arena.new_param(2.0);
    let leaf = arena.param(p);
    let sin = arena.push(ExprNode::Unary(UnaryOp::Sin, leaf));
    *arena.get_mut(root) = ExprNode::Add(smallvec![sin, sin]);
    let x: &[f64] = &[];
    assert_eq!(evaluate(&arena, root, &x).unwrap(), 2.0 * 2.0_f64.sin());
    arena.set_param_value(p, 3.0);
    assert_eq!(evaluate(&arena, root, &x).unwrap(), 2.0 * 3.0_f64.sin());
}

#[test]
fn negation_fast_path_preserves_ieee_sign_bits() {
    let mut arena = ExprArena::new();
    let v = arena.var(VarId(0));
    let neg = arena.push(ExprNode::Unary(UnaryOp::Neg, v));
    let double = arena.push(ExprNode::Unary(UnaryOp::Neg, neg));
    for value in
        [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::from_bits(0xfff8_0000_0000_0001)]
    {
        let x: &[f64] = &[value];
        assert_eq!(evaluate(&arena, neg, &x).unwrap().to_bits(), (-value).to_bits());
        assert_eq!(evaluate(&arena, double, &x).unwrap().to_bits(), value.to_bits());
    }
}
