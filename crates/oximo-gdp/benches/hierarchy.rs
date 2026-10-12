//! Construction and reformulation timings for flat and nested numeric GDP.
//! Run with `cargo bench -p oximo-gdp --bench hierarchy`.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oximo_gdp::prelude::*;
use std::hint::black_box;

fn build(groups: usize, depth: usize) -> Model {
    let model = Model::new("hierarchy benchmark");
    for group in 0..groups {
        let x = model.__var(format!("x{group}")).bounds(0.0, 10.0).build();
        let root = model.add_boolean(format!("root{group}"));
        let other = model.add_boolean(format!("other{group}"));
        root.context().add_constraint("capacity", x.le(8.0));
        disjunction!(model, [root, other]);
        let mut parent = root;
        for (level, capacity) in [6.0, 5.0, 4.0, 3.0].into_iter().take(depth).enumerate() {
            let child = model.add_boolean(format!("child{group}_{level}"));
            let sibling = model.add_boolean(format!("sibling{group}_{level}"));
            child.context().add_constraint("capacity", x.le(capacity));
            disjunction!(model, [child, sibling], parent = parent);
            parent = child;
        }
    }
    model
}

fn hierarchy(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("gdp/hierarchy");

    for (case, depth) in [("flat", 0), ("nested", 1), ("deep", 4)] {
        for groups in [100, 1000, 10_000] {
            group.throughput(Throughput::Elements(groups as u64));

            group.bench_function(BenchmarkId::new(format!("{case}/build"), groups), |bencher| {
                bencher.iter_batched(
                    || (),
                    |()| black_box(build(groups, depth)),
                    BatchSize::PerIteration,
                );
            });

            group.bench_function(
                BenchmarkId::new(format!("{case}/reformulate"), groups),
                |bencher| {
                    bencher.iter_batched_ref(
                        || build(groups, depth),
                        |model| black_box(model.reformulate_gdp(BigM::default()).unwrap()),
                        BatchSize::PerIteration,
                    );
                },
            );

            group.bench_function(BenchmarkId::new(format!("{case}/total"), groups), |bencher| {
                bencher.iter_batched(
                    || (),
                    |()| {
                        let model = build(groups, depth);
                        let report = model.reformulate_gdp(BigM::default()).unwrap();

                        black_box((model, report))
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }

    group.finish();
}

criterion_group!(benches, hierarchy);
criterion_main!(benches);
