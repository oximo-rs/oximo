use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use oximo_pounce::benchmark_support;

use super::common::{pair, sizes, threads};

/// Run stable POUNCE classification, value, and rejected Jacobian candidates.
pub fn bench(criterion: &mut Criterion) {
    nlp_setup(criterion);
    classification(criterion);
    initialization(criterion);
    values(criterion);
    jacobians(criterion);
}

/// Measure the production route and NLP bounds snapshot, excluding oracle
/// construction and optimization.
fn nlp_setup(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/pounce_nlp_setup");
    for rows in [16, 64, 1_024, 8_192] {
        let model = benchmark_support::model(rows, true);
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::from_parameter(rows), |bencher| {
            bencher.iter(|| black_box(benchmark_support::prepare_nlp(&model)));
        });
    }
    group.finish();
}

/// Measure initial constraint-slot classification for QCP and NLP models.
fn classification(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/pounce_classification");
    for (kind, nonlinear) in [("qcp", false), ("nlp", true)] {
        let threshold = if nonlinear {
            benchmark_support::CLASSIFY_THRESHOLD
        } else {
            benchmark_support::QCP_CLASSIFY_THRESHOLD
        };
        for (size, rows) in [
            ("below", threshold / 2),
            ("threshold", threshold),
            ("above", threshold * 2),
            ("large", 1_024),
        ] {
            let model = benchmark_support::model(rows, nonlinear);
            pair(
                &mut group,
                &format!("{kind}/{size}/{rows}"),
                rows,
                &model,
                benchmark_support::classify,
            );
        }
    }
    group.finish();
}

/// Measure complete stable-oracle construction, including sparsity and scatter maps.
fn initialization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/pounce_oracle_initialization");
    for rows in [64, 128, 256, 512, 1_024] {
        let model = benchmark_support::model(rows, false);
        pair(
            &mut group,
            &format!("qcp/crossover/{rows}"),
            rows,
            &model,
            benchmark_support::initialize,
        );
    }
    for rows in [16, 32, 64, 1_024] {
        let model = benchmark_support::model(rows, true);
        pair(
            &mut group,
            &format!("nlp/crossover/{rows}"),
            rows,
            &model,
            benchmark_support::initialize,
        );
    }
    group.finish();
}

/// Measure repeated constraint evaluation using reusable oracle scratch.
fn values(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/pounce_constraint_values");
    for (kind, nonlinear) in [("qcp", false), ("nlp", true)] {
        for (size, rows) in sizes(benchmark_support::VALUE_THRESHOLD) {
            let model = benchmark_support::model(rows, nonlinear);
            let mut serial = benchmark_support::Oracle::new(&model);
            let mut parallel = benchmark_support::Oracle::new(&model);
            let case = format!("{kind}/{size}/{rows}");
            group.throughput(Throughput::Elements(rows as u64));
            group.bench_function(BenchmarkId::new(&case, "serial"), |bencher| {
                bencher.iter(|| black_box(serial.values(false)));
            });
            group.bench_function(
                BenchmarkId::new(&case, format!("parallel-{}t", threads())),
                |bencher| {
                    bencher.iter(|| black_box(parallel.values(true)));
                },
            );
        }
    }
    group.finish();
}

/// Re-measure the sparse and dense Jacobian paths rejected for production use.
fn jacobians(criterion: &mut Criterion) {
    for (name, operation) in [
        (
            "preprocessing/pounce_sparse_jacobian_rejected",
            benchmark_support::Oracle::sparse_jacobian
                as fn(&mut benchmark_support::Oracle, bool) -> f64,
        ),
        (
            "preprocessing/pounce_dense_jacobian_rejected",
            benchmark_support::Oracle::dense_jacobian
                as fn(&mut benchmark_support::Oracle, bool) -> f64,
        ),
    ] {
        let mut group = criterion.benchmark_group(name);
        for (size, rows) in sizes(benchmark_support::JACOBIAN_THRESHOLD) {
            let model = benchmark_support::model(rows, false);
            let mut serial = benchmark_support::Oracle::new(&model);
            let mut parallel = benchmark_support::Oracle::new(&model);
            group.throughput(Throughput::Elements(rows as u64));
            group.bench_function(BenchmarkId::new(format!("{size}/{rows}"), "serial"), |bencher| {
                bencher.iter(|| black_box(operation(&mut serial, false)));
            });
            group.bench_function(
                BenchmarkId::new(format!("{size}/{rows}"), format!("parallel-{}t", threads())),
                |bencher| {
                    bencher.iter(|| black_box(operation(&mut parallel, true)));
                },
            );
        }
        group.finish();
    }

    let mut group = criterion.benchmark_group("preprocessing/pounce_sparse_jacobian_width");
    for n_vars in [256, 2_048] {
        for (size, rows) in sizes(benchmark_support::JACOBIAN_THRESHOLD) {
            let model = benchmark_support::jacobian_model(rows, n_vars);
            let mut serial = benchmark_support::Oracle::new(&model);
            let mut parallel = benchmark_support::Oracle::new(&model);
            group.throughput(Throughput::Elements(rows as u64));
            let case = format!("{n_vars}v/{size}/{rows}");
            group.bench_function(BenchmarkId::new(&case, "serial"), |bencher| {
                bencher.iter(|| black_box(serial.sparse_jacobian(false)));
            });
            group.bench_function(
                BenchmarkId::new(&case, format!("parallel-{}t", threads())),
                |bencher| {
                    bencher.iter(|| black_box(parallel.sparse_jacobian(true)));
                },
            );
        }
    }
    group.finish();
}
