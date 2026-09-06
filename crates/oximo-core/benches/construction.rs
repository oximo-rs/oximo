//! Solver-free construction benchmarks. Run with:
//! cargo bench -p oximo-core --features benchmark-support --bench construction -- 6
//! Reports medians of 15 samples, each averaging 20 fresh models/sums.

use std::hint::black_box;
use std::time::Instant;

use oximo_core::model::benchmark_support::{IndexedBuildCase, indexed_build, scalar_build};
use oximo_core::{Model, Set};

fn measure(mut sample: impl FnMut() -> u128) -> u128 {
    for _ in 0..3 {
        black_box(sample());
    }
    let mut times: Vec<_> = (0..15).map(|_| sample()).collect();
    times.sort_unstable();
    times[times.len() / 2] / 20
}

fn main() {
    let threads = std::env::args()
        .skip(1)
        .find(|arg| arg != "--bench")
        .map_or(6, |arg| arg.parse::<usize>().unwrap());
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
        println!("case,size,threads,median_nanoseconds");
        for rows in [32, 8192] {
            for (name, case) in [
                ("variables", IndexedBuildCase::Variables),
                ("parameters", IndexedBuildCase::Parameters),
                ("algebraic", IndexedBuildCase::Algebraic),
            ] {
                let ns = measure(|| {
                    let start = Instant::now();
                    for _ in 0..20 {
                        black_box(indexed_build(rows, case, threads > 1 && rows >= 1024));
                    }
                    start.elapsed().as_nanos()
                });
                println!("{name},{rows},{threads},{ns}");
            }
            let ns = measure(|| {
                let start = Instant::now();
                for _ in 0..20 {
                    black_box(scalar_build(rows));
                }
                start.elapsed().as_nanos()
            });
            println!("scalar,{rows},{threads},{ns}");

            let ns = measure(|| {
                let model = Model::new("sum_bench");
                let keys = Set::range(0..rows);
                let x = model.__indexed_var("x", &keys).build();
                let start = Instant::now();
                for _ in 0..20 {
                    black_box(oximo_core::sum::__sum_over(&keys, |i: usize| x[i]));
                }
                start.elapsed().as_nanos()
            });
            println!("sum,{rows},{threads},{ns}");
        }
    });
}
