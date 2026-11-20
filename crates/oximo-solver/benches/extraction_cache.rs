use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};
use oximo_expr::{ExprArena, ExprId, ExprNode, QuadraticTerms, UnaryOp, VarId};
use oximo_solver::prepare::PreparedExpressions;

#[path = "support/allocation.rs"]
mod allocation;

#[global_allocator]
static ALLOCATOR: allocation::CountingAllocator = allocation::CountingAllocator;

fn bench(c: &mut Criterion) {
    let mut arena = ExprArena::new();
    let linear = arena.linear((0..64).map(|i| (VarId(i), 1.0)).collect(), 3.0);
    let roots: Vec<_> =
        (0..256).map(|_| arena.push(ExprNode::Unary(oximo_expr::UnaryOp::Neg, linear))).collect();
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

#[derive(Clone, Copy, Debug)]
enum Shape {
    Direct,
    Shifted,
    ShiftedQuadratic,
    Nested,
    Negated,
}

impl Shape {
    const ALL: [Self; 5] =
        [Self::Direct, Self::Shifted, Self::ShiftedQuadratic, Self::Nested, Self::Negated];

    fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Shifted => "shifted",
            Self::ShiftedQuadratic => "shifted_quadratic",
            Self::Nested => "nested",
            Self::Negated => "negated",
        }
    }
}

// Each row has its own coefficient buffer, matching the measured dense-row workload.
fn scalar_rows(shape: Shape, width: u32) -> (ExprArena, Vec<ExprId>) {
    let mut arena = ExprArena::new();
    let zero = arena.constant(0.0);
    let roots = (0..501)
        .map(|_| {
            let coefficients = (0..width).map(|i| (VarId(i), 1.0)).collect();
            let affine = if matches!(shape, Shape::ShiftedQuadratic) {
                arena.push(ExprNode::Quadratic(Arc::new(
                    QuadraticTerms::from_coefficients(vec![], coefficients, 3.0).unwrap(),
                )))
            } else {
                arena.linear(coefficients, 3.0)
            };
            match shape {
                Shape::Direct => affine,
                Shape::Shifted | Shape::ShiftedQuadratic => {
                    arena.push(ExprNode::Add(vec![affine, zero].into()))
                }
                Shape::Nested => {
                    let inner = arena.push(ExprNode::Add(vec![affine, zero].into()));
                    arena.push(ExprNode::Add(vec![inner, zero].into()))
                }
                Shape::Negated => arena.push(ExprNode::Unary(UnaryOp::Neg, affine)),
            }
        })
        .collect();
    (arena, roots)
}

fn extract_rows(prepared: &PreparedExpressions, roots: &[ExprId]) {
    for &root in roots {
        black_box(prepared.linear(black_box(root)).unwrap());
    }
}

fn scalar_affine(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_affine");
    for width in [1, 64, 1000] {
        for shape in Shape::ALL {
            let (arena, roots) = scalar_rows(shape, width);
            for repeated in [false, true] {
                let name =
                    format!("{}/{}", shape.name(), if repeated { "repeated" } else { "cold" });
                group.bench_with_input(BenchmarkId::new(name, width), &(), |bencher, ()| {
                    bencher.iter_batched_ref(
                        || {
                            let prepared = PreparedExpressions::new(arena.clone());
                            if repeated {
                                // A single reused root fits the cache, including the wide-row case.
                                black_box(prepared.linear(roots[0]));
                                black_box(prepared.linear(roots[0]));
                            }
                            prepared
                        },
                        |prepared| {
                            if repeated {
                                for _ in 0..roots.len() {
                                    black_box(prepared.linear(black_box(roots[0])).unwrap());
                                }
                            } else {
                                extract_rows(prepared, &roots);
                            }
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
        }
    }
    group.finish();
}

fn wide_scalar_sums(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide_scalar_sums");
    for constants in [8, 4096] {
        for (name, affine_children) in [("constants", 0), ("one_affine", 1), ("two_affine", 2)] {
            let mut arena = ExprArena::new();
            let zero = arena.constant(0.0);
            let affine = arena.linear(vec![(VarId(0), 2.0)], 3.0);
            let mut children = vec![zero; constants];
            children.extend(std::iter::repeat_n(affine, affine_children));
            let root = arena.push(ExprNode::Add(children.into()));
            let prepared = PreparedExpressions::new(arena);
            black_box(prepared.linear(root));
            black_box(prepared.linear(root));
            group.bench_function(BenchmarkId::new(name, constants), |bencher| {
                bencher.iter(|| {
                    for _ in 0..501 {
                        black_box(prepared.linear(black_box(root)).unwrap());
                    }
                });
            });
        }
    }
    group.finish();
}

fn alternating_affine_rows(width: u32, stride: usize) -> (ExprArena, [ExprId; 2]) {
    let mut arena = ExprArena::new();
    let coefficients = || (0..width).map(|i| (VarId(i), 1.0)).collect();
    let first = arena.linear(coefficients(), 3.0);
    while arena.len() < stride {
        arena.push(ExprNode::Const(0.0));
    }
    let second = arena.linear(coefficients(), 3.0);
    assert_eq!(second.index() - first.index(), stride);
    let zero = arena.constant(0.0);
    let roots = [first, second].map(|child| arena.push(ExprNode::Add(vec![child, zero].into())));
    (arena, roots)
}

fn eligibility_collisions(c: &mut Criterion) {
    let mut group = c.benchmark_group("eligibility_collisions");
    for width in [64, 1000] {
        for (name, stride) in [("noncolliding", 1), ("colliding", 16)] {
            let (arena, roots) = alternating_affine_rows(width, stride);
            let prepared = PreparedExpressions::new(arena);
            for root in roots {
                black_box(prepared.linear(root).unwrap());
            }
            group.bench_function(BenchmarkId::new(name, width), |bencher| {
                bencher.iter(|| {
                    for _ in 0..501 {
                        for root in roots {
                            black_box(prepared.linear(black_box(root)).unwrap());
                        }
                    }
                });
            });
        }
    }
    group.finish();
}

fn allocation_census() {
    println!("shape,width,rows,calls,requested_bytes,peak_live_bytes");
    for width in [1, 64, 1000] {
        for shape in Shape::ALL {
            let (arena, roots) = scalar_rows(shape, width);
            // Track the complete preparation lifetime, retaining the input arena outside it.
            let (calls, bytes, peak) = allocation::allocations(|| {
                let prepared = PreparedExpressions::new(arena.clone());
                extract_rows(&prepared, &roots);
            });
            println!("{},{width},{},{calls},{bytes},{peak}", shape.name(), roots.len());
        }
    }
    for width in [64, 1000] {
        for (name, stride) in [("noncolliding", 1), ("colliding", 16)] {
            let (arena, roots) = alternating_affine_rows(width, stride);
            let (calls, bytes, peak) = allocation::allocations(|| {
                let prepared = PreparedExpressions::new(arena.clone());
                for _ in 0..501 {
                    extract_rows(&prepared, &roots);
                }
            });
            println!("eligibility_{name},{width},1002,{calls},{bytes},{peak}");
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2)).sample_size(30);
    targets = bench, scalar_affine, wide_scalar_sums, eligibility_collisions
}

fn main() {
    if std::env::var_os("OXIMO_AFFINE_ALLOCATIONS").is_some() {
        allocation_census();
    } else {
        benches();
        Criterion::default().configure_from_args().final_summary();
    }
}
