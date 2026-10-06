//! Solver-free affine construction.
//! Affine timings exclude fixtures and checked handles. Chain timings include the complete
//! construction operation and teardown. Allocation totals include setup and teardown.
#[path = "support/allocation.rs"]
mod allocation;
#[path = "support/cases.rs"]
mod cases;
#[path = "construction/chains.rs"]
mod chains;

use std::hint::black_box;
use std::time::Instant;

use oximo_expr::{
    Affine, AffineBuilder, Expr, ExprArena, ExprArenaCell, ExprNode, VarId, extract_linear,
    extract_quadratic,
};

#[global_allocator]
static ALLOC: allocation::CountingAllocator = allocation::CountingAllocator;

#[derive(Clone, Copy)]
enum Method {
    Left,
    Right,
    Flat,
    Shared,
    Builder,
    BuilderCompensated,
}

fn fixture(n: usize, workload: &str) -> (ExprArenaCell, Vec<oximo_expr::ExprId>) {
    let mut arena = ExprArena::new();
    let param = arena.new_param(2.0);
    let parameter = arena.param(param);
    let terms = (0..n)
        .map(|i| {
            let v = if matches!(workload, "duplicates" | "cancellation") { i % 16 } else { i };
            let id = arena.var(VarId(u32::try_from(v).unwrap()));
            match workload {
                "weighted" | "duplicates" => arena.linear(
                    vec![(VarId(u32::try_from(v).unwrap()), if i % 2 == 0 { 2.0 } else { -1.0 })],
                    0.0,
                ),
                "cancellation" => arena.linear(
                    vec![(
                        VarId(u32::try_from(v).unwrap()),
                        if (i / 16) % 2 == 0 { 1.0 } else { -1.0 },
                    )],
                    0.0,
                ),
                "symbolic" => arena.push(ExprNode::Mul(smallvec::smallvec![parameter, id])),
                _ => id,
            }
        })
        .collect();
    (ExprArenaCell::new(arena), terms)
}

fn build<'a>(
    arena: &'a ExprArenaCell,
    terms: &[Expr<'a, Affine>],
    method: Method,
) -> Expr<'a, Affine> {
    match method {
        Method::Left => terms.iter().copied().reduce(|a, b| a + b).unwrap(),
        Method::Right => terms.iter().rev().copied().reduce(|a, b| b + a).unwrap(),
        Method::Flat => build_flat(terms),
        Method::Builder | Method::BuilderCompensated => {
            let mut builder = if matches!(method, Method::BuilderCompensated) {
                AffineBuilder::with_capacity_compensated(arena, terms.len())
            } else {
                AffineBuilder::with_capacity(arena, terms.len())
            };
            for &term in terms {
                builder.add_term(1.0, term);
            }
            builder.build()
        }
        Method::Shared => {
            let middle = terms.len() / 2;
            let prefix: Expr<'_, Affine> = terms[..middle].iter().copied().sum();
            let left = terms[middle..].iter().copied().fold(prefix, |a, b| a + b);
            let right = terms[middle..].iter().copied().fold(prefix, |a, b| a + b);
            left + right
        }
    }
}

// Keep builder/binary-operator branches out of flat-sum benchmark codegen.
#[inline(never)]
fn build_flat<'a>(terms: &[Expr<'a, Affine>]) -> Expr<'a, Affine> {
    terms.iter().copied().sum()
}

fn run(n: usize, workload: &str, method: Method, phase: &str) -> u128 {
    let (arena, ids) = fixture(n, workload);
    let terms: Vec<_> = ids.iter().map(|&id| Expr::new(id, &arena).try_affine().unwrap()).collect();
    let start = Instant::now();
    let expr = build(&arena, &terms, method);
    if phase == "linear" {
        black_box(extract_linear(&arena.borrow(), expr.id).unwrap());
    } else if phase == "quadratic" {
        black_box(extract_quadratic(&arena.borrow(), expr.id).unwrap());
    }
    black_box(expr);
    start.elapsed().as_nanos()
}

fn main() {
    println!(
        "workload,method,size,phase,median_nanoseconds,setup_allocations,setup_requested_bytes,setup_peak_live_bytes,arena_nodes,stored_coefficients"
    );
    let filter = std::env::var("OXIMO_AFFINE_WORKLOAD").ok();
    let size_filter = std::env::var("OXIMO_AFFINE_SIZE").ok();
    let method_filter = std::env::var("OXIMO_AFFINE_METHOD").ok();
    let phase_filter = std::env::var("OXIMO_AFFINE_PHASE").ok();
    let case_filter = std::env::var("OXIMO_BENCH_CASE").ok();
    for workload in ["distinct", "weighted", "duplicates", "symbolic", "cancellation"] {
        if filter.as_deref().is_some_and(|value| value != workload) {
            continue;
        }
        for n in [32, 256, 1024, 4096] {
            if size_filter.as_deref().is_some_and(|value| value != n.to_string()) {
                continue;
            }
            for (name, method) in [
                ("left", Method::Left),
                ("right", Method::Right),
                ("flat", Method::Flat),
                ("shared", Method::Shared),
                ("builder", Method::Builder),
                ("builder_compensated", Method::BuilderCompensated),
            ] {
                if method_filter.as_deref().is_some_and(|value| value != name) {
                    continue;
                }
                for phase in ["build", "linear", "quadratic"] {
                    if !cases::selected(&format!("{workload}/{name}/{n}/{phase}")) {
                        continue;
                    }
                    if phase_filter.as_deref().is_some_and(|value| value != phase) {
                        continue;
                    }
                    let first = run(n, workload, method, phase).max(1);
                    let iterations = usize::try_from((20_000_000 / first).clamp(1, 128)).unwrap();
                    let mut times = [0; 7];
                    for time in &mut times {
                        *time =
                            (0..iterations).map(|_| run(n, workload, method, phase)).sum::<u128>()
                                / iterations as u128;
                    }
                    times.sort_unstable();
                    let mut nodes = 0;
                    let mut coefficients = 0;
                    let (calls, bytes, peak) = allocation::allocations(|| {
                        let (arena, ids) = fixture(n, workload);
                        let terms: Vec<_> = ids
                            .iter()
                            .map(|&id| Expr::new(id, &arena).try_affine().unwrap())
                            .collect();
                        let expr = build(&arena, &terms, method);
                        let snapshot = arena.borrow();
                        nodes = snapshot.len();
                        coefficients = (0..snapshot.len())
                            .map(|i| {
                                match snapshot.get(oximo_expr::ExprId(u32::try_from(i).unwrap())) {
                                    ExprNode::Linear { coeffs, .. } => coeffs.len(),
                                    _ => 0,
                                }
                            })
                            .sum::<usize>();
                        if phase == "linear" {
                            black_box(extract_linear(&snapshot, expr.id).unwrap());
                        } else if phase == "quadratic" {
                            black_box(extract_quadratic(&snapshot, expr.id).unwrap());
                        }
                    });
                    println!(
                        "{workload},{name},{n},{phase},{},{calls},{bytes},{peak},{nodes},{coefficients}",
                        times[3]
                    );
                }
            }
        }
    }
    if filter.is_some()
        || size_filter.is_some()
        || method_filter.is_some()
        || phase_filter.is_some()
    {
        return;
    }
    for n in [32, 256, 1024, 4096] {
        if cases::selected(&format!("distinct/builder_reuse/{n}/build")) {
            measure_reuse(n);
        }
    }
    chains::measure_cases(case_filter.as_deref());
}

// Reuse one builder and arena. Report per-build time over 32 emissions.
// Allocation totals include the fixture, warmup, and all 32 emissions.
fn measure_reuse(n: usize) {
    let reuse = |track_structure: bool| {
        let (arena, ids) = fixture(n, "distinct");
        let terms: Vec<_> =
            ids.iter().map(|&id| Expr::new(id, &arena).try_affine().unwrap()).collect();
        let mut builder = AffineBuilder::with_capacity(&arena, n);
        for &term in &terms {
            builder.add_term(1.0, term);
        }
        black_box(builder.build());
        let start = Instant::now();
        for _ in 0..32 {
            for &term in &terms {
                builder.add_term(1.0, term);
            }
            black_box(builder.build());
        }
        let elapsed = start.elapsed().as_nanos() / 32;
        let snapshot = arena.borrow();
        let coefficients = if track_structure {
            (0..snapshot.len())
                .map(|i| match snapshot.get(oximo_expr::ExprId(u32::try_from(i).unwrap())) {
                    ExprNode::Linear { coeffs, .. } => coeffs.len(),
                    _ => 0,
                })
                .sum::<usize>()
        } else {
            0
        };
        (elapsed, snapshot.len(), coefficients)
    };
    let mut times = [0; 7];
    for time in &mut times {
        *time = reuse(false).0;
    }
    times.sort_unstable();
    let mut structure = (0, 0, 0);
    let (calls, bytes, peak) = allocation::allocations(|| {
        structure = reuse(true);
    });
    println!(
        "distinct,builder_reuse,{n},build,{},{calls},{bytes},{peak},{},{}",
        times[3], structure.1, structure.2
    );
}
