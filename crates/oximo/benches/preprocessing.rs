#[path = "preprocessing/mod.rs"]
mod suite;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

fn configure() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(30)
}

criterion_group! {
    name = benches;
    config = configure();
    targets =
        suite::model::bench,
        suite::io::bench,
        suite::pounce::bench,
        suite::clarabel::bench,
        suite::highs::bench,
        suite::gams::bench,
        suite::baron::bench,
        suite::gurobi::bench,
        suite::mosek::bench,
        suite::scip::bench,
        suite::enzyme::bench
}
criterion_main!(benches);
