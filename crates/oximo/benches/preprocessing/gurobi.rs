#[cfg(feature = "_benchmark-proprietary")]
use criterion::Criterion;

#[cfg(feature = "_benchmark-proprietary")]
use super::common::sizes;

#[cfg(feature = "_benchmark-proprietary")]
/// Measure Gurobi preprocessing before serial FFI uploads.
pub fn bench(criterion: &mut Criterion) {
    // Measures complete native model construction, with one Gurobi environment
    // reused across iterations and optimization excluded.
    let env = oximo_gurobi::benchmark_support::environment().unwrap();
    let mut translation = criterion.benchmark_group("preprocessing/gurobi_translation");
    for (kind, degree) in [("lp", 1), ("qcp", 2), ("nlp", 3)] {
        for (size, rows) in sizes(oximo_gurobi::benchmark_support::THRESHOLD) {
            let model = oximo_gurobi::benchmark_support::model(rows, degree);
            translation.throughput(criterion::Throughput::Elements(rows as u64));
            translation.bench_with_input(
                criterion::BenchmarkId::new(format!("{kind}/{size}/{rows}"), "serial"),
                &model,
                |bencher, model| {
                    bencher.iter(|| {
                        std::hint::black_box(
                            oximo_gurobi::benchmark_support::translate(
                                std::hint::black_box(model),
                                &env,
                            )
                            .unwrap(),
                        )
                    });
                },
            );
        }
    }
    translation.finish();
}

#[cfg(not(feature = "_benchmark-proprietary"))]
/// Leave the Gurobi groups absent unless proprietary benchmarks are enabled.
pub fn bench(_criterion: &mut criterion::Criterion) {}
