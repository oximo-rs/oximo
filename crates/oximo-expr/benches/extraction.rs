mod support;

use oximo_expr::{
    Expr, ExprArena, ExprArenaCell, ExprNode, VarId, classify, extract_linear, extract_quadratic,
    render_expr, split_linear,
};
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
        root = arena.push(ExprNode::Unary(oximo_expr::UnaryOp::Neg, root));
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
    for depth in [10, 14, 18] {
        let cell = ExprArenaCell::new(ExprArena::new());
        let vars: Vec<_> = (0..64).map(|i| Expr::from_var(&cell, VarId(i))).collect();
        let mut sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
        for _ in 0..depth {
            sum = sum + sum;
        }
        let mixed = sum + vars[0].sin();
        let snapshot = cell.borrow();
        support::measure(&format!("split_shared/{depth}"), || split_linear(&snapshot, mixed.id()));
    }
    for n in [32, 512, 4096, 20_000] {
        let cell = ExprArenaCell::new(ExprArena::new());
        let vars: Vec<_> = (0..n).map(|i| Expr::from_var(&cell, VarId(i))).collect();
        let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
        let nonlinear = sum.sin();
        let compact = sum.compact();
        let snapshot = cell.borrow();
        support::measure(&format!("render_sum/{n}"), || {
            render_expr(&snapshot, nonlinear.id(), &|v| format!("x{}", v.0))
        });
        if n <= 4096 {
            support::measure(&format!("linear_sum/{n}"), || {
                extract_linear(&snapshot, sum.id()).unwrap()
            });
            support::measure(&format!("quadratic_sum/{n}"), || {
                extract_quadratic(&snapshot, sum.id()).unwrap()
            });
        }
        if matches!(n, 32 | 4096) {
            for (name, expression) in [("extract_tree", sum), ("extract_compact", compact)] {
                support::measure(&format!("{name}/{n}"), || {
                    extract_linear(&snapshot, expression.id()).unwrap()
                });
            }
        }
    }
    let cell = ExprArenaCell::new(ExprArena::new());
    let vars: Vec<_> = (0..64).map(|i| Expr::from_var(&cell, VarId(i))).collect();
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let overflow = -1e308 * sum + (1e308 * sum + 1e308 * sum);
    let snapshot = cell.borrow();
    support::measure("linear_hidden_overflow/64", || {
        extract_linear(&snapshot, overflow.id()).unwrap()
    });
    support::measure("quadratic_hidden_overflow/64", || {
        extract_quadratic(&snapshot, overflow.id()).unwrap()
    });
}
