#[cfg(feature = "_benchmark-proprietary")]
use criterion::Criterion;

#[cfg(feature = "_benchmark-proprietary")]
use super::common::sizes;

#[cfg(feature = "_benchmark-proprietary")]
/// Measure MOSEK preprocessing before serial task uploads.
pub fn bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/mosek_translation");
    for (kind, degree) in [("lp", 1), ("qcp", 2), ("detected_soc", 3), ("explicit_soc", 4)] {
        for (size, rows) in sizes(oximo_mosek::benchmark_support::ROW_THRESHOLD) {
            let model = if degree == 4 {
                oximo_mosek::benchmark_support::explicit_soc_model(rows)
            } else {
                oximo_mosek::benchmark_support::row_model(rows, degree)
            };
            group.throughput(criterion::Throughput::Elements(rows as u64));
            group.bench_with_input(
                criterion::BenchmarkId::new(format!("{kind}/{size}/{rows}"), "serial"),
                &model,
                |bencher, model| {
                    bencher.iter(|| {
                        std::hint::black_box(
                            oximo_mosek::benchmark_support::translate(std::hint::black_box(model))
                                .unwrap(),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "_benchmark-proprietary"))]
/// Leave the MOSEK groups absent unless proprietary benchmarks are enabled.
pub fn bench(_criterion: &mut criterion::Criterion) {}
