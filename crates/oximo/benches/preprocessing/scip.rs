#[cfg(feature = "scip")]
use std::hint::black_box;

#[cfg(feature = "scip")]
use criterion::{BenchmarkId, Criterion, Throughput};
#[cfg(feature = "scip")]
use oximo::prelude::*;
#[cfg(feature = "scip")]
use oximo::solvers::Scip;

/// Measure SCIP's public model-support check, which prepares the same
/// solver-facing snapshot used before native problem construction.
#[cfg(feature = "scip")]
pub fn bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/scip_snapshot");
    for rows in [32_u32, 1_024, 8_192] {
        let model = Model::new("scip_snapshot_bench");
        let x = model.__var("x").lb(0.0).build();
        let y = model.__var("y").lb(0.0).build();
        model.__minimize(x + y);
        for i in 0..rows {
            model.__add_constraint_auto((x + 2.0 * y).le(f64::from(i) + 10.0));
        }
        assert!(Scip::new().supports_model(&model));
        group.throughput(Throughput::Elements(u64::from(rows)));
        group.bench_with_input(BenchmarkId::new("lp", rows), &model, |bencher, model| {
            bencher.iter(|| black_box(Scip::new().supports_model(black_box(model))));
        });
    }
    group.finish();
}

#[cfg(not(feature = "scip"))]
pub fn bench(_criterion: &mut criterion::Criterion) {}
