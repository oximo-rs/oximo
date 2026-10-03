use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oximo_clarabel::{Clarabel, ClarabelOptions, benchmark_support};
use oximo_core::prelude::*;
use oximo_solver::{PersistentSolver, Solver, TerminationStatus};

fn model(n: usize, dense: bool) -> Model {
    let model = Model::new("eigenvalue benchmark");
    variable!(model, t);
    param!(model, p = 1.0);
    let matrix = SymmetricMatrix::from_upper_fn(n, |i, j| {
        if i == j {
            t
        } else {
            let coefficient = if dense {
                1.0 / f64::from(u32::try_from(j - i + 1).unwrap())
            } else {
                f64::from(u8::from(j == i + 1))
            };
            (-p * coefficient).into()
        }
    });
    model.add_psd_constraint("eigenvalue", matrix);
    objective!(model, Min, t);
    model
}

fn bench(c: &mut Criterion) {
    let mut translation = c.benchmark_group("sdp/clarabel_translation");
    for n in [8, 32, 128] {
        for (name, dense) in [("path", false), ("dense", true)] {
            let model = model(n, dense);
            translation.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter(|| black_box(benchmark_support::translate(black_box(&model)).unwrap()));
            });
        }
    }
    translation.finish();

    let mut solves = c.benchmark_group("sdp/clarabel_solve");
    for n in [8, 32] {
        for (name, dense) in [("path", false), ("dense", true)] {
            let model = model(n, dense);
            let p = model.parameter_id("p").unwrap();
            for chordal in [true, false] {
                model.set_param_id(p, 1.0);
                let options = ClarabelOptions::default()
                    .presolve_enable(false)
                    .input_sparse_dropzeros(false)
                    .chordal_decomposition_enable(chordal);
                let mut solver = Clarabel;
                assert_eq!(
                    solver.solve(&model, &options).unwrap().termination,
                    TerminationStatus::Optimal
                );
                solves.bench_function(
                    BenchmarkId::new(format!("{name}/cold/chordal={chordal}"), n),
                    |b| {
                        b.iter(|| black_box(solver.solve(black_box(&model), &options).unwrap()));
                    },
                );
                let mut persistent = solver.persistent();
                persistent.solve(&model, &options).unwrap();
                let mut high = false;
                solves.bench_function(
                    BenchmarkId::new(format!("{name}/persistent/chordal={chordal}"), n),
                    |b| {
                        b.iter(|| {
                            high = !high;
                            model.set_param_id(p, if high { 1.25 } else { 1.0 });
                            black_box(persistent.solve(black_box(&model), &options).unwrap())
                        });
                    },
                );
            }
        }
    }
    solves.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1)).sample_size(20);
    targets = bench
}
criterion_main!(benches);
