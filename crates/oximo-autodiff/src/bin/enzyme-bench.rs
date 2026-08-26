use std::hint::black_box;
use std::time::{Duration, Instant};

use oximo_autodiff::benchmark_support;

const WARMUP: Duration = Duration::from_millis(250);
const SAMPLE: Duration = Duration::from_millis(100);
const SAMPLES: usize = 15;

fn selected(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|filter| name.contains(filter))
}

fn measure<T>(name: &str, filter: Option<&str>, mut operation: impl FnMut() -> T) {
    if !selected(filter, name) {
        return;
    }

    let warmup_deadline = Instant::now() + WARMUP;
    while Instant::now() < warmup_deadline {
        black_box(operation());
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let deadline = start + SAMPLE;
        let mut iterations = 0u64;
        while Instant::now() < deadline {
            black_box(operation());
            iterations += 1;
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        samples.push(elapsed / iterations.max(1) as f64);
    }

    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    println!("{name:72} median {median:>12.1} ns/op  p95 {p95:>12.1} ns/op");
}

fn case_name(operation: &str, kind: &str, shape: &str, n_vars: usize, rows: usize) -> String {
    format!("autodiff/{operation}/{kind}/{shape}/{n_vars}v/{rows}")
}

fn main() {
    let filter = std::env::args().nth(1);
    println!("oximo-autodiff Enzyme runtime benchmarks");
    if let Some(filter) = filter.as_deref() {
        println!("filter: {filter}");
    }

    for &(kind_name, kind) in &benchmark_support::KINDS {
        for &(shape, n_vars, rows) in &benchmark_support::SHAPES {
            let model = benchmark_support::runtime_model(rows, n_vars, kind);
            let name = case_name("evaluator_initialization", kind_name, shape, n_vars, rows);
            measure(&name, filter.as_deref(), || benchmark_support::build(&model));
        }
    }

    for &(kind_name, kind) in &benchmark_support::KINDS {
        for &(shape, n_vars, rows) in &benchmark_support::SHAPES {
            let model = benchmark_support::runtime_model(rows, n_vars, kind);
            let name = case_name("objective_value", kind_name, shape, n_vars, rows);
            let runtime = benchmark_support::Runtime::new(&model);
            measure(&name, filter.as_deref(), || benchmark_support::objective_value(&runtime));

            let name = case_name("objective_gradient", kind_name, shape, n_vars, rows);
            let mut runtime = benchmark_support::Runtime::new(&model);
            measure(&name, filter.as_deref(), || {
                benchmark_support::objective_gradient(&mut runtime)
            });
        }
    }

    for &(kind_name, kind) in &benchmark_support::KINDS {
        for &(shape, n_vars, rows) in &benchmark_support::SHAPES {
            let model = benchmark_support::runtime_model(rows, n_vars, kind);
            for (operation, parallel) in [
                ("constraint_values/automatic", true),
                ("constraint_values/serial", false),
                ("constraint_values/parallel", true),
            ] {
                let (operation, variant) = operation.split_once('/').unwrap();
                let name =
                    format!("{}/{variant}", case_name(operation, kind_name, shape, n_vars, rows));
                let mut runtime = benchmark_support::Runtime::new(&model);
                measure(&name, filter.as_deref(), || {
                    if variant == "automatic" {
                        benchmark_support::constraint_values_auto(&mut runtime)
                    } else {
                        benchmark_support::constraint_values(&mut runtime, parallel)
                    }
                });
            }

            for (operation, parallel) in [
                ("constraint_jacobian/automatic", true),
                ("constraint_jacobian/serial", false),
                ("constraint_jacobian/parallel", true),
            ] {
                let (operation, variant) = operation.split_once('/').unwrap();
                let name =
                    format!("{}/{variant}", case_name(operation, kind_name, shape, n_vars, rows));
                let mut runtime = benchmark_support::Runtime::new(&model);
                measure(&name, filter.as_deref(), || {
                    if variant == "automatic" {
                        benchmark_support::constraint_jacobian_auto(&mut runtime)
                    } else {
                        benchmark_support::constraint_jacobian(&mut runtime, parallel)
                    }
                });
            }

            let name = format!(
                "{}/automatic",
                case_name("hessian_lagrangian", kind_name, shape, n_vars, rows)
            );
            let mut runtime = benchmark_support::Runtime::new(&model);
            measure(&name, filter.as_deref(), || benchmark_support::hessian(&mut runtime));

            let name = format!(
                "{}/try_refresh",
                case_name("evaluator_refresh", kind_name, shape, n_vars, rows)
            );
            let scale = model.parameter_id("scale").unwrap();
            model.set_param_id(scale, 1.5);
            let mut refresh = benchmark_support::EvaluatorRefresh::new(model);
            measure(&name, filter.as_deref(), || refresh.run());
        }
    }
}
