mod support;

use oximo_expr::{Expr, ExprArena, ExprArenaCell, ExprNode, UnaryOp, VarId, evaluate};
use smallvec::smallvec;

#[global_allocator]
static ALLOC: support::CountingAllocator = support::CountingAllocator;

fn deep_neg_chain(arena: &mut ExprArena, depth: usize) -> oximo_expr::ExprId {
    let mut root = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..depth {
        root = arena.push(ExprNode::Unary(UnaryOp::Neg, root));
    }
    root
}

fn deep_add_chain(arena: &mut ExprArena, depth: usize) -> oximo_expr::ExprId {
    let zero = arena.push(ExprNode::Const(0.0));
    let mut root = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..depth {
        root = arena.push(ExprNode::Add(smallvec![zero, root]));
    }
    root
}

/// `levels` doublings: `root_{k+1} = root_k + root_k`. Only `levels` distinct
/// nodes, but `2^levels` paths for a naive recursive walk.
fn shared_doubling(arena: &mut ExprArena, levels: usize) -> oximo_expr::ExprId {
    let mut root = arena.push(ExprNode::Var(VarId(0)));
    for _ in 0..levels {
        root = arena.push(ExprNode::Add(smallvec![root, root]));
    }
    root
}

/// One `sin(x0)` shared by `width` summands.
fn shared_sin_fanout(arena: &mut ExprArena, width: usize) -> oximo_expr::ExprId {
    let x0 = arena.push(ExprNode::Var(VarId(0)));
    let sin = arena.push(ExprNode::Unary(UnaryOp::Sin, x0));
    let children: oximo_expr::Children = std::iter::repeat_n(sin, width).collect();
    arena.push(ExprNode::Add(children))
}

fn main() {
    println!("case,nanoseconds,allocations,requested_bytes,peak_live_bytes");
    let mut arena = ExprArena::new();
    let x: &[f64] = &[0.7];

    for depth in [256, 10_000] {
        let neg = deep_neg_chain(&mut arena, depth);
        support::measure(&format!("eval_deep_neg/{depth}"), || evaluate(&arena, neg, &x).unwrap());
        let add = deep_add_chain(&mut arena, depth);
        support::measure(&format!("eval_deep_add/{depth}"), || evaluate(&arena, add, &x).unwrap());
    }

    // Heavily shared DAGs: distinct-node count stays tiny while the path
    // count explodes; the iterative evaluator visits each node once.
    for levels in [14, 20] {
        let root = shared_doubling(&mut arena, levels);
        support::measure(&format!("eval_shared_doubling/{levels}"), || {
            evaluate(&arena, root, &x).unwrap()
        });
    }
    for width in [32, 1024] {
        let root = shared_sin_fanout(&mut arena, width);
        support::measure(&format!("eval_shared_sin/{width}"), || {
            evaluate(&arena, root, &x).unwrap()
        });
    }

    // Wide flat sum for comparison with the extraction benchmarks.
    let vars: Vec<_> = (0..2048).map(|i| arena.push(ExprNode::Var(VarId(i)))).collect();
    let wide_vars: Vec<f64> = (0..2048).map(|i| f64::from(i) * 0.001).collect();
    for n in [32, 2048] {
        let wide = arena.push(ExprNode::Add(vars[..n].iter().copied().collect()));
        support::measure(&format!("eval_wide/{n}"), || {
            evaluate(&arena, wide, &wide_vars.as_slice()).unwrap()
        });
    }
    // Compare repeated tree walks with a one-time compaction, outside timing.
    for n in [32, 4096] {
        let cell = ExprArenaCell::new(ExprArena::new());
        let vars: Vec<_> = (0..n).map(|i| Expr::from_var(&cell, VarId(i))).collect();
        let original = vars.iter().copied().reduce(|a, b| a + b).unwrap();
        let compact = original.compact();
        let snapshot = cell.borrow();
        let values = vec![1.0; usize::try_from(n).unwrap()];
        for (name, expression) in [("evaluate_tree", original), ("evaluate_compact", compact)] {
            support::measure(&format!("{name}/{n}"), || {
                evaluate(&snapshot, expression.id(), &&values[..]).unwrap()
            });
        }
    }
}
