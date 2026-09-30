//! Registration benchmarks.
//! Run with `cargo bench -p oximo-core --bench typed_ir`.

use std::hint::black_box;
use std::time::Instant;

use oximo_core::prelude::*;

fn median(mut sample: impl FnMut() -> u128) -> u128 {
    for _ in 0..2 {
        black_box(sample());
    }
    let mut times: Vec<_> = (0..15).map(|_| sample()).collect();
    times.sort_unstable();
    times[times.len() / 2]
}

fn shared_rows(width: usize, dynamic: bool, indexed: bool) -> u128 {
    let model = Model::new("shared");
    variable!(model, x[i in 0..width]);
    let expr = sum!(model, x[i].square() for i in 0..width);
    let keys = Set::range(0..4096usize);
    let start = Instant::now();
    if indexed {
        model.__add_constraints_over("row", &keys, |_: usize| expr.erase().le(10.0));
    } else {
        for _ in 0..256 {
            if dynamic {
                black_box(model.__add_constraint_auto(expr.erase().le(10.0)));
            } else {
                black_box(model.__add_constraint_auto(expr.le(10.0)));
            }
        }
    }
    let elapsed = start.elapsed().as_nanos();
    black_box(model);
    elapsed
}

fn unique_rows(dynamic: bool) -> u128 {
    let model = Model::new("unique");
    variable!(model, x);
    let expressions: Vec<_> = (0..8192).map(|_| x.square()).collect();
    let start = Instant::now();
    for expr in expressions {
        if dynamic {
            black_box(model.__add_constraint_auto(expr.erase().le(10.0)));
        } else {
            black_box(model.__add_constraint_auto(expr.le(10.0)));
        }
    }
    let elapsed = start.elapsed().as_nanos();
    black_box(model);
    elapsed
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap().install(|| {
        println!("case,width,rows,median_nanoseconds");
        for width in [32, 1024, 8192] {
            for (label, dynamic, indexed) in [
                ("shared_static", false, false),
                ("shared_dynamic", true, false),
                ("indexed_shared_dynamic", true, true),
            ] {
                let elapsed = median(|| shared_rows(width, dynamic, indexed));
                let rows = if indexed { 4096 } else { 256 };
                println!("{label},{width},{rows},{elapsed}");
            }
        }
        for (label, dynamic) in [("unique_static", false), ("unique_dynamic", true)] {
            let elapsed = median(|| unique_rows(dynamic));
            println!("{label},1,8192,{elapsed}");
        }
    });
}
