//! Optional selection of a single benchmark case.

pub fn selected(name: &str) -> bool {
    std::env::var("OXIMO_BENCH_CASE").map_or(true, |filter| filter == name)
}
