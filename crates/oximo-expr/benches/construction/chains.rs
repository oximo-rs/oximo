use std::hint::black_box;
use std::time::Instant;

use oximo_expr::{Expr, ExprArena, ExprArenaCell, ExprId, VarId, evaluate, extract_linear};

use crate::allocation;

#[expect(clippy::eq_op, reason = "same-sum cancellation is the regression fixture")]
fn cancellation_fixture(kind: &str, n: u32) -> (ExprArena, ExprId) {
    let cell = ExprArenaCell::new(ExprArena::new());
    let vars: Vec<_> = (0..n).map(|i| Expr::from_var(&cell, VarId(i))).collect();
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let pid = cell.borrow_mut().new_param(2.0);
    let p = Expr::from_param(&cell, pid);
    let input = if kind == "cancel_add" { sum - sum + p } else { (sum - sum) * p };
    let arena = (*cell.borrow()).clone();
    (arena, input.id())
}

fn construct(kind: &str, n: u32, fixture: Option<&(ExprArena, ExprId)>) -> (u128, usize) {
    let start = Instant::now();
    // Retain the original complete-operation scope, using a bounded fresh arena.
    if let Some((snapshot, input)) = fixture {
        let cell = ExprArenaCell::new(snapshot.clone());
        black_box(Expr::new(*input, &cell) * Expr::from_var(&cell, VarId(0)));
        let nodes = cell.borrow().len();
        drop(cell);
        return (start.elapsed().as_nanos(), nodes);
    }
    let cell = ExprArenaCell::new(ExprArena::new());
    if matches!(kind, "sum" | "symbolic_sum") {
        let pid = cell.borrow_mut().new_param(2.0);
        let p = Expr::from_param(&cell, pid);
        let expression = (0..n)
            .map(|i| {
                let x = Expr::from_var(&cell, VarId(i));
                if kind == "symbolic_sum" { (p * x).erase() } else { x.erase() }
            })
            .reduce(|a, b| a + b)
            .unwrap();
        black_box(expression);
        let nodes = cell.borrow().len();
        drop(cell);
        return (start.elapsed().as_nanos(), nodes);
    }
    let x = Expr::from_var(&cell, VarId(0));
    let mut expression = x.erase();
    for _ in 1..n {
        expression = match kind {
            "powers" => expression.powi(2),
            "quadratic" => expression + x * x,
            "product" => expression * x,
            _ => panic!("unknown construction case: {kind}"),
        };
    }
    black_box(expression);
    let nodes = cell.borrow().len();
    drop(cell);
    (start.elapsed().as_nanos(), nodes)
}

pub fn measure_cases(filter: Option<&str>) {
    for (kind, sizes) in [
        ("product", &[32, 500, 1000, 2000, 4000][..]),
        ("powers", &[32, 1000, 4000][..]),
        ("quadratic", &[32, 4096][..]),
        ("sum", &[32, 4096][..]),
        ("symbolic_sum", &[32, 4096][..]),
        ("cancel_add", &[64, 4096][..]),
        ("cancel_scale", &[64, 4096][..]),
    ] {
        for &n in sizes {
            if filter.is_some_and(|value| value != format!("{kind}/{n}")) {
                continue;
            }
            let fixture = kind.starts_with("cancel").then(|| cancellation_fixture(kind, n));
            if let Some((snapshot, input)) = &fixture {
                let cell = ExprArenaCell::new(snapshot.clone());
                let expression = Expr::new(*input, &cell) * Expr::from_var(&cell, VarId(0));
                assert!(expression.try_affine().is_ok());
                assert!(extract_linear(&cell.borrow(), expression.id()).is_some());
                let values = vec![3.0; n as usize];
                let expected = if kind == "cancel_add" { 6.0 } else { 0.0 };
                let actual = evaluate(&cell.borrow(), expression.id(), &&values[..]).unwrap();
                assert!((actual - expected).abs() < 1e-12);
                eprintln!("correct=true");
            }
            let warmup = construct(kind, n, fixture.as_ref()).0.max(1);
            let iterations = (10_000_000 / warmup).clamp(1, 128);
            let mut samples = [0; 7];
            for sample in &mut samples {
                *sample =
                    (0..iterations).map(|_| construct(kind, n, fixture.as_ref()).0).sum::<u128>()
                        / iterations;
            }
            samples.sort_unstable();
            let mut nodes = 0;
            let (calls, bytes, peak) = allocation::allocations(|| {
                nodes = construct(kind, n, fixture.as_ref()).1;
            });
            // Stored-coefficient accounting is specific to the affine matrix.
            println!("{kind},chain,{n},build,{},{calls},{bytes},{peak},{nodes},", samples[3]);
        }
    }
}
