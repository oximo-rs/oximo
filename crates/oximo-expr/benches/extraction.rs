mod support;

use oximo_expr::{ExprArena, ExprNode, VarId, classify, extract_linear, extract_quadratic};
use smallvec::smallvec;

#[global_allocator]
static ALLOC: support::CountingAllocator = support::CountingAllocator;

fn main() {
    println!("case,nanoseconds,allocations,requested_bytes,peak_live_bytes");
    let mut arena = ExprArena::new();
    let vars: Vec<_> = (0..2048).map(|i| arena.var(VarId(i))).collect();
    let p = arena.new_param(1.5);
    let param = arena.param(p);
    for n in [32, 2048] {
        let wide = arena.push(ExprNode::Add(vars[..n].iter().copied().collect()));
        support::measure(&format!("linear_wide/{n}"), || extract_linear(&arena, wide).unwrap());
        support::measure(&format!("quadratic_wide/{n}"), || {
            extract_quadratic(&arena, wide).unwrap()
        });
        support::measure(&format!("classify_wide/{n}"), || classify(&arena, wide));
    }
    let mut root = arena.push(ExprNode::Mul(smallvec![param, vars[0]]));
    for _ in 0..256 {
        root = arena.push(ExprNode::Neg(root));
    }
    support::measure("linear_deep/256", || extract_linear(&arena, root).unwrap());
    support::measure("quadratic_deep/256", || extract_quadratic(&arena, root).unwrap());
    let mut shared = arena.push(ExprNode::Mul(smallvec![param, vars[0]]));
    for _ in 0..14 {
        shared = arena.push(ExprNode::Add(smallvec![shared, shared]));
    }
    support::measure("linear_shared/14", || extract_linear(&arena, shared).unwrap());
    support::measure("quadratic_shared/14", || extract_quadratic(&arena, shared).unwrap());
    support::measure("classify_shared/14", || classify(&arena, shared));
}
