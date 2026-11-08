mod support;

use oximo_expr::{
    Expr, ExprArena, ExprArenaCell, ExprNode, QuadraticTerms, VarId, classify, extract_linear,
    extract_quadratic, render_expr, split_affine, split_linear,
};
use smallvec::smallvec;

#[global_allocator]
static ALLOC: support::CountingAllocator = support::CountingAllocator;

fn main() {
    println!("case,nanoseconds,allocations,requested_bytes,peak_live_bytes");
    split_affine_cases();
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
        let mut sum = vars
            .iter()
            .copied()
            .map(Expr::<oximo_expr::Affine>::from)
            .reduce(|a, b| a + b)
            .unwrap();
        for _ in 0..depth {
            sum = sum.clone() + sum;
        }
        let mixed = sum + vars[0].sin();
        let snapshot = cell.borrow();
        support::measure(&format!("split_shared/{depth}"), || split_linear(&snapshot, mixed.id()));
    }
    for n in [32, 512, 4096, 20_000] {
        let cell = ExprArenaCell::new(ExprArena::new());
        let vars: Vec<_> = (0..n).map(|i| Expr::from_var(&cell, VarId(i))).collect();
        let sum = vars
            .iter()
            .copied()
            .map(Expr::<oximo_expr::Affine>::from)
            .reduce(|a, b| a + b)
            .unwrap();
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
    let sum =
        vars.iter().copied().map(Expr::<oximo_expr::Affine>::from).reduce(|a, b| a + b).unwrap();
    let overflow = -1e308 * &sum + (1e308 * &sum + 1e308 * sum);
    let snapshot = cell.borrow();
    support::measure("linear_hidden_overflow/64", || {
        extract_linear(&snapshot, overflow.id()).unwrap()
    });
    support::measure("quadratic_hidden_overflow/64", || {
        extract_quadratic(&snapshot, overflow.id()).unwrap()
    });
}

fn measure_split(name: &str, arena: &ExprArena, root: oximo_expr::ExprId) {
    support::measure(name, || {
        let (linear, residual) = split_affine(arena, root);
        (linear.into_owned(), residual)
    });
}

fn split_affine_cases() {
    let mut arena = ExprArena::new();
    let variables: Vec<_> = (0..2048).map(|i| arena.var(VarId(i))).collect();
    for count in [32, 2048] {
        let wide = arena.push(ExprNode::Add(variables[..count].iter().copied().collect()));
        measure_split(&format!("split_affine_wide/{count}"), &arena, wide);
        let cell = ExprArenaCell::new(ExprArena::new());
        let sum = (0..count)
            .map(|i| Expr::from_var(&cell, VarId(u32::try_from(i).unwrap())))
            .map(Expr::<oximo_expr::Affine>::from)
            .reduce(|left, right| left + right)
            .unwrap();
        measure_split(&format!("split_affine_built/{count}"), &cell.borrow(), sum.id());
    }

    for count in [8, 128] {
        let terms = (0..count)
            .map(|i| {
                let variable = VarId(i);
                let terms = QuadraticTerms::from_coefficients(
                    vec![(variable, variable, 1.0)],
                    vec![(variable, 2.0)],
                    3.0,
                )
                .unwrap();
                arena.push(ExprNode::Quadratic(std::sync::Arc::new(terms)))
            })
            .collect();
        let root = arena.push(ExprNode::Add(terms));
        measure_split(&format!("split_affine_quadratic_sum/{count}"), &arena, root);
    }

    let shuffled =
        arena.push(ExprNode::Add((0..2048).map(|i| variables[(i * 1151) % 2048]).collect()));
    measure_split("split_affine_shuffled/2048", &arena, shuffled);

    let duplicates =
        arena.push(ExprNode::Add((0..2048).map(|i| variables[(i * 19) % 32]).collect()));
    measure_split("split_affine_duplicates/2048", &arena, duplicates);

    let parameter_id = arena.new_param(1.5);
    let parameter = arena.param(parameter_id);
    let affine = arena.linear((0..32).map(|i| (VarId(i), 1.0)).collect(), 1.0);
    let scaled = arena.push(ExprNode::Mul(smallvec![parameter, affine]));
    let scaled_sum = arena.push(ExprNode::Add((0..2048).map(|_| scaled).collect()));
    measure_split("split_affine_scaled_shared/2048", &arena, scaled_sum);

    let mut level = variables;
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| arena.push(ExprNode::Add(pair.iter().copied().collect())))
            .collect();
    }
    measure_split("split_affine_balanced/2048", &arena, level[0]);

    let mut shared = arena.linear((0..32).map(|i| (VarId(i), 1.0)).collect(), 1.0);
    for _ in 0..40 {
        shared = arena.push(ExprNode::Add(smallvec![shared, shared]));
    }
    measure_split("split_affine_shared/40", &arena, shared);
}
