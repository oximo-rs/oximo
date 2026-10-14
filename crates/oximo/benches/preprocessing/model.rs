use criterion::{BenchmarkId, Criterion, Throughput};
use oximo::{Model, SolutionPoint, SolverResult, param, variable};
use oximo_core::model::benchmark_support::{self, IndexedBuildCase};

use super::common::{LARGE, crossover_sizes, pair, sizes};

/// Measure model-kind inference across linear, quadratic, and nonlinear rows.
pub fn bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preprocessing/model_kind");
    for (kind, degree) in [("lp", 1), ("nlp", 3)] {
        for (size, rows) in sizes(benchmark_support::THRESHOLD) {
            let model = benchmark_support::model(rows, degree);
            pair(
                &mut group,
                &format!("{kind}/{size}/{rows}"),
                rows,
                &model,
                benchmark_support::infer,
            );
        }
    }
    for rows in crossover_sizes().into_iter().chain([LARGE]) {
        let model = benchmark_support::model(rows, 2);
        pair(&mut group, &format!("qcp/crossover/{rows}"), rows, &model, benchmark_support::infer);
    }
    for rows in crossover_sizes().into_iter().chain([LARGE]) {
        let model = benchmark_support::soc_model(rows);
        pair(&mut group, &format!("socp/crossover/{rows}"), rows, &model, benchmark_support::infer);
    }
    group.finish();

    let mut build = criterion.benchmark_group("preprocessing/indexed_build");
    for rows in [256, 512, 1_024, 2_048, 8_192] {
        for (name, case) in [
            ("variables", IndexedBuildCase::Variables),
            ("parameters", IndexedBuildCase::Parameters),
            ("algebraic", IndexedBuildCase::Algebraic),
            ("range", IndexedBuildCase::Range),
            ("soc", IndexedBuildCase::Soc),
            ("sos", IndexedBuildCase::Sos),
        ] {
            build.throughput(Throughput::Elements(rows as u64));
            build.bench_with_input(BenchmarkId::new(name, rows), &rows, |bencher, &rows| {
                bencher.iter(|| {
                    std::hint::black_box(benchmark_support::indexed_build(rows, case, false))
                });
            });
            let parallel = format!("{name}_parallel_{}t", super::common::threads());
            build.bench_with_input(BenchmarkId::new(parallel, rows), &rows, |bencher, &rows| {
                bencher.iter(|| {
                    std::hint::black_box(benchmark_support::indexed_build(rows, case, true))
                });
            });
        }
    }
    build.finish();

    let mut scalar = criterion.benchmark_group("preprocessing/scalar_build");
    for rows in [32, 1_024] {
        scalar.throughput(Throughput::Elements(rows as u64));
        scalar.bench_with_input(BenchmarkId::from_parameter(rows), &rows, |bencher, &rows| {
            bencher.iter(|| std::hint::black_box(benchmark_support::scalar_build(rows)));
        });
    }
    scalar.finish();

    let mut provenance = criterion.benchmark_group("model/handle_provenance");

    provenance.bench_function("model_new", |bencher| {
        bencher.iter(|| std::hint::black_box(Model::new("model_id_bench")));
    });

    provenance.bench_function("fix", |bencher| {
        let model = Model::new("fix_provenance_bench");
        variable!(model, -1.0 <= x <= 1.0);
        bencher.iter(|| model.fix(std::hint::black_box(x), 0.0).unwrap());
    });

    provenance.bench_function("set_initial", |bencher| {
        let model = Model::new("initial_provenance_bench");
        variable!(model, x);
        bencher.iter(|| model.set_initial(std::hint::black_box(x), 0.0).unwrap());
    });

    provenance.bench_function("set_param", |bencher| {
        let model = Model::new("param_provenance_bench");
        param!(model, p = 1.0);
        bencher.iter(|| model.set_param(std::hint::black_box(p), 2.0).unwrap());
    });

    provenance.bench_function("solution_point_value_of", |bencher| {
        let model = Model::new("point_provenance_bench");
        variable!(model, x);
        let point = SolutionPoint {
            model_id: model.id(),
            primal: [(x.var_id().unwrap(), 1.0)].into_iter().collect(),
            objective: None,
            parameters: oximo_solver::ParameterSnapshot::default(),
        };
        bencher.iter(|| std::hint::black_box(&point).value_of(std::hint::black_box(x)).unwrap());
    });

    provenance.bench_function("solver_result_value_of", |bencher| {
        let model = Model::new("result_provenance_bench");
        variable!(model, x);
        let point = SolutionPoint {
            model_id: model.id(),
            primal: [(x.var_id().unwrap(), 1.0)].into_iter().collect(),
            objective: None,
            parameters: oximo_solver::ParameterSnapshot::default(),
        };
        let result =
            SolverResult { model_id: model.id(), solutions: vec![point], ..Default::default() };
        bencher.iter(|| std::hint::black_box(&result).value_of(std::hint::black_box(x)).unwrap());
    });

    provenance.finish();
}
