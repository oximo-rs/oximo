//! Solver-free NL export and ASCII/binary roundtrip measurements.
//! Each case runs in its own process so a baseline stack overflow is isolated.
use std::hint::black_box;
use std::time::{Duration, Instant};

use oximo_core::{Model, Relate};
use oximo_io::{NlFormat, WriteOptions, read_nl, to_nl_string, write_nl_with};

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
    let expression = if consumer == "nl_sum" { sum.sin() } else { sum.exp() };
    if consumer == "nl_export_unique" {
        // Distinct affine regions in the row/objective exercise cache misses.
        let row_sum = vars.iter().map(|&x| 2.0 * x).reduce(|a, b| a + b).unwrap();
        model.add_constraint("row", row_sum.exp().le(1.0));
    } else if consumer != "nl_sum" {
        model.add_constraint("row", expression.le(1.0));
    }
    model.__minimize(expression);
    match consumer {
        "nl_sum" | "nl_export" | "nl_export_unique" => {
            fingerprint(to_nl_string(&model).unwrap().as_bytes());
            measure(case, || to_nl_string(&model).unwrap());
        }
        "nl_roundtrip" | "nl_roundtrip_binary" => {
            let options = WriteOptions {
                format: if consumer.ends_with("binary") {
                    NlFormat::Binary
                } else {
                    NlFormat::Ascii
                },
                ..WriteOptions::default()
            };
            measure(case, || {
                let mut bytes = Vec::new();
                write_nl_with(&model, &mut bytes, &options).unwrap();
                read_nl(bytes.as_slice()).unwrap()
            });
        }
        _ => panic!("unknown consumer: {consumer}"),
    }
}

fn main() {
    println!("case,median_ns,min_ns,max_ns");
    if let Ok(case) = std::env::var("OXIMO_BENCH_CASE") {
        run_case(&case);
        return;
    }
    for size in [32, 512, 4096, 20_000] {
        for kind in
            ["nl_sum", "nl_export", "nl_export_unique", "nl_roundtrip", "nl_roundtrip_binary"]
        {
            run_case(&format!("{kind}/{size}"));
        }
    }
}
