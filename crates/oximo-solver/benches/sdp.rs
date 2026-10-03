use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oximo_core::prelude::*;
use oximo_solver::SolutionPoint;
use oximo_solver::prepare::LoweringContext;
use oximo_solver::psd::{PsdTriangleOrder, pack_svec, svec_coordinates, unpack_svec};

fn bench(c: &mut Criterion) {
    let mut coordinates = c.benchmark_group("sdp/coordinates");
    for n in [8, 32, 128] {
        let matrix = SymmetricMatrix::from_upper_fn(n, |i, j| {
            f64::from(u32::try_from(i + 2 * j + 1).unwrap())
        });
        for (name, order) in
            [("upper", PsdTriangleOrder::UpperColumn), ("lower", PsdTriangleOrder::LowerColumn)]
        {
            let packed = pack_svec(&matrix, order);
            coordinates.bench_function(BenchmarkId::new(format!("iterate/{name}"), n), |b| {
                b.iter(|| {
                    let sum = svec_coordinates(black_box(n), order)
                        .fold(0.0, |sum, (k, scale)| sum + matrix.upper_triangle()[k] * scale);
                    black_box(sum)
                });
            });
            coordinates.bench_function(BenchmarkId::new(format!("pack/{name}"), n), |b| {
                b.iter(|| black_box(pack_svec(black_box(&matrix), order)));
            });
            coordinates.bench_function(BenchmarkId::new(format!("unpack/{name}"), n), |b| {
                b.iter(|| black_box(unpack_svec(n, black_box(&packed), order).unwrap()));
            });
        }
    }
    coordinates.finish();

    let mut readback = c.benchmark_group("sdp/readback");
    for n in [8, 32, 128] {
        let model = Model::new("readback");
        let x = model.add_symmetric_variable("X", n);
        param!(model, p = 2.0);
        let affine = x.map(|&entry| p * entry + 1.0);
        let point = SolutionPoint {
            model_id: model.id(),
            primal: x.upper_triangle().iter().map(|entry| (entry.var_id().unwrap(), 3.0)).collect(),
            objective: None,
        };
        for (name, matrix) in [("variables", &x), ("affine", &affine)] {
            readback.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter(|| black_box(point.value_of_matrix(black_box(matrix)).unwrap().unwrap()));
            });
        }
    }
    readback.finish();

    let mut lowering = c.benchmark_group("sdp/lowering");
    for n in [8, 32, 128] {
        let model = Model::new("lowering");
        variable!(model, x);
        variable!(model, y);
        let shared = x + 2.0 * y + 1.0;
        // Distinct direct affine roots expose unnecessary coefficient copies.
        let distinct = SymmetricMatrix::from_upper_fn(n, |i, j| {
            shared + f64::from(u32::try_from(i + j).unwrap())
        });
        let repeated = SymmetricMatrix::from_upper_fn(n, |_, _| shared);
        let distinct_id = model.add_psd_constraint("distinct", distinct);
        let repeated_id = model.add_psd_constraint("repeated", repeated);
        objective!(model, Min, x);
        for (name, id) in [("distinct", distinct_id), ("repeated", repeated_id)] {
            lowering.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter(|| {
                    // Each solve creates a fresh preparation; include that cost.
                    let prepared = LoweringContext::new(black_box(&model)).unwrap();
                    black_box(prepared.explicit_psd(&model.psd_constraints()[id.index()]).unwrap());
                });
            });
        }
    }
    lowering.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1)).sample_size(20);
    targets = bench
}
criterion_main!(benches);
