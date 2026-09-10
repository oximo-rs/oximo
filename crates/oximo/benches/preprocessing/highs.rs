use criterion::Criterion;
use oximo_highs::benchmark_support;

use super::common::{pair, sizes};

#[expect(clippy::cast_precision_loss)]
/// Measure HiGHS production translation and its rejected result-map candidate.
pub fn bench(criterion: &mut Criterion) {
    // Measures complete RowProblem construction, without solver setup or solve.
    let mut translation_group = criterion.benchmark_group("preprocessing/highs_translation");
    for (size, rows) in sizes(benchmark_support::ROW_THRESHOLD) {
        let model = benchmark_support::row_model(rows);
        translation_group.throughput(criterion::Throughput::Elements(rows as u64));
        translation_group.bench_with_input(
            criterion::BenchmarkId::new(format!("lp/{size}/{rows}"), "serial"),
            &model,
            |bencher, model| {
                bencher.iter(|| {
                    std::hint::black_box(
                        benchmark_support::translate(std::hint::black_box(model)).unwrap(),
                    )
                });
            },
        );
    }
    translation_group.finish();

    // Re-measures the parallel HashMap candidate rejected for result parsing.
    let mut map_group = criterion.benchmark_group("preprocessing/highs_solution_maps_rejected");
    for (size, rows) in sizes(2731) {
        let values: Vec<f64> = (0..rows).map(|i| i as f64).collect();
        pair(&mut map_group, &format!("{size}/{rows}"), rows * 3, &values, |values, parallel| {
            benchmark_support::solution_maps(values, parallel)
        });
    }
    map_group.finish();
}
