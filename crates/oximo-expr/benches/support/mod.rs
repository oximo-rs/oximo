//! Timing and allocation measurements for standalone benchmarks.
mod allocation;
mod cases;

pub use allocation::CountingAllocator;
use allocation::allocations;
use cases::selected;
use std::hint::black_box;
use std::time::{Duration, Instant};

pub fn measure<T>(name: &str, mut f: impl FnMut() -> T) {
    if !selected(name) {
        return;
    }
    for _ in 0..100 {
        drop(black_box(f()));
    }
    let mut iterations = 1_usize;
    let elapsed = loop {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(f()));
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(20) {
            break elapsed;
        }
        iterations = iterations.checked_mul(2).expect("benchmark iteration overflow");
    };
    let iterations =
        usize::try_from((iterations as u128 * 50_000_000 / elapsed.as_nanos().max(1)).max(1))
            .expect("benchmark iteration overflow");
    let mut times = [0; 7];
    for time in &mut times {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(f()));
        }
        *time = start.elapsed().as_nanos() / iterations as u128;
    }
    times.sort_unstable();
    let (calls, bytes, peak) = allocations(&mut f);
    println!("{name},{},{},{},{}", times[3], calls, bytes, peak);
}
