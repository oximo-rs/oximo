//! Focused range-registration benchmark.
//! Usage: cargo bench -p oximo-core --bench indexed_ranges -- interval 100000 6 20
//! Cases: interval, split (quadratic body), mixed (alternating), symbolic.
//! Arguments: case, domain size, Rayon threads, measured samples (after 2 warmups).

use std::hint::black_box;
use std::time::Instant;

use oximo_core::prelude::*;

fn sample(case: &str, size: usize) -> u128 {
    let model = Model::new("range_bench");
    let keys = Set::range(0..size);
    let x = model.__indexed_var("x", &keys).build();
    let lo = model.__param("lo", 0.0);
    let start = Instant::now();
    let family = if case == "symbolic" {
        model.__add_range_constraints_over("r", &keys, |i| (x[i] + 1.0, lo, 10.0))
    } else {
        model.__add_range_constraints_over("r", &keys, |i| {
            let body = if case == "split" || (case == "mixed" && i % 2 != 0) {
                x[i].powi(2)
            } else {
                x[i] + 1.0
            };
            (body, 0.0, 10.0)
        })
    };
    black_box(&family);
    let elapsed = start.elapsed().as_nanos();
    assert_eq!(family.len(), size);
    let expected_rows = match case {
        "interval" => size,
        "mixed" => size + size / 2,
        _ => 2 * size,
    };
    assert_eq!(model.num_constraints(), expected_rows);
    for key in [0, size - 1] {
        match family.get(key).unwrap() {
            RangeConstraintIds::Interval(id) => {
                assert_eq!(model.constraint_id(&format!("r[{key}]")), Some(id));
            }
            RangeConstraintIds::Split { lower, upper } => {
                assert_eq!(model.constraint_id(&format!("r[{key}]_lo")), Some(lower));
                assert_eq!(model.constraint_id(&format!("r[{key}]_hi")), Some(upper));
            }
        }
    }
    elapsed
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).filter(|arg| arg != "--bench").collect();
    let case = args.first().map_or("mixed", String::as_str);
    assert!(["interval", "split", "mixed", "symbolic"].contains(&case));
    let size = args.get(1).map_or(100_000, |arg| arg.parse::<usize>().unwrap());
    let threads = args.get(2).map_or(6, |arg| arg.parse::<usize>().unwrap());
    let samples = args.get(3).map_or(20, |arg| arg.parse::<usize>().unwrap());
    assert!(size > 0 && threads > 0 && samples > 0);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
        for _ in 0..2 {
            black_box(sample(case, size));
        }
        println!("case,size,threads,sample,nanoseconds");
        for index in 0..samples {
            println!("{case},{size},{threads},{index},{}", sample(case, size));
        }
    });
}
