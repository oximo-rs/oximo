//! Solver-free GAMS and BARON rendering measurements.
//! Each case runs in its own process so a baseline stack overflow is isolated.
use std::hint::black_box;
use std::time::{Duration, Instant};

use oximo_core::{Model, Relate};

fn measure<T>(case: &str, mut operation: impl FnMut() -> T) {
    drop(black_box(operation()));
    let mut iterations = 1_u32;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(operation()));
        }
        if start.elapsed() >= Duration::from_millis(20) || iterations >= 131_072 {
            break;
        }
        iterations *= 2;
    }
    let mut samples = [0_u128; 7];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(operation()));
        }
        *sample = start.elapsed().as_nanos() / u128::from(iterations);
    }
    samples.sort_unstable();
    println!("{case},{},{},{}", samples[3], samples[0], samples[6]);
}

fn fingerprint(output: &[u8]) {
    let hash = output.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    });
    eprintln!("bytes={},fnv1a={hash:016x}", output.len());
}

fn run_case(case: &str) {
    let (consumer, count) = case.split_once('/').unwrap();
    let n: usize = count.parse().unwrap();
    let model = Model::new("affine_consumers");
    let vars: Vec<_> = (0..n).map(|i| model.__var(format!("x{i}")).build()).collect();
    let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
    let expression = if consumer.ends_with("unary") {
        let mut expression = vars[0].exp();
        for _ in 1..n {
            expression = expression.exp();
        }
        expression
    } else {
        sum.exp()
    };
    model.add_constraint("row", expression.le(1.0));
    model.__minimize(expression);
    match consumer {
        "gams_sum" | "gams_unary" => {
            fingerprint(oximo_gams::benchmark_support::render_equations(&model, false).as_bytes());
            measure(case, || oximo_gams::benchmark_support::render_equations(&model, false));
        }
        "baron_sum" | "baron_unary" => {
            fingerprint(
                oximo_baron::benchmark_support::render_equations(&model, false).unwrap().as_bytes(),
            );
            measure(case, || {
                oximo_baron::benchmark_support::render_equations(&model, false).unwrap()
            });
        }
        _ => panic!("unknown consumer: {consumer}"),
    }
}

fn main() {
    if let Ok(case) = std::env::var("OXIMO_BENCH_CASE") {
        run_case(&case);
        return;
    }
    println!("case,median_ns,min_ns,max_ns");
    let executable = std::env::current_exe().expect("cannot locate benchmark executable");
    for size in [32, 512, 4096, 20_000] {
        for kind in ["gams_sum", "gams_unary", "baron_sum", "baron_unary"] {
            let case = format!("{kind}/{size}");
            match std::process::Command::new(&executable).env("OXIMO_BENCH_CASE", &case).status() {
                Ok(status) if status.success() => {}
                Ok(status) => eprintln!("{case}: benchmark process failed with {status}"),
                Err(error) => eprintln!("{case}: could not launch benchmark process: {error}"),
            }
        }
    }
}
