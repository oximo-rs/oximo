use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use oximo_expr::{ExprArena, ExprNode, VarId};
use oximo_solver::prepare::PreparedExpressions;

fn bench(c: &mut Criterion) {
    let mut arena = ExprArena::new();
    let linear = arena.linear((0..64).map(|i| (VarId(i), 1.0)).collect(), 3.0);
    let roots: Vec<_> = (0..256).map(|_| arena.push(ExprNode::Neg(linear))).collect();
    let a = roots[0];
    let shard = |id: oximo_expr::ExprId| id.0.wrapping_mul(0x9e37_79b9) >> 28;
    let b = *roots.iter().find(|&&id| id != a && shard(id) == shard(a)).unwrap();
    let mut group = c.benchmark_group("extraction_cache");
    for (name, sequence, warm) in [
        ("paired", vec![a; 256], false),
        ("interleaved_cold", [a, b].repeat(128), false),
        ("interleaved_warm", [a, b].repeat(128), true),
        ("unique", roots, false),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter_batched(
                || {
                    let prepared = PreparedExpressions::new(arena.clone());
                    if warm {
                        for root in [a, a, b, b] {
                            black_box(prepared.linear(root));
                        }
                    }
                    prepared
                },
                |prepared| {
                    for &root in &sequence {
                        black_box(prepared.linear(black_box(root)));
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2)).sample_size(30);
    targets = bench
}
criterion_main!(benches);
