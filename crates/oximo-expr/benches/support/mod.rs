//! Allocation instrumentation for standalone, single-threaded benchmarks only.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

#[derive(Debug)]
pub struct CountingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: Every operation forwards the unchanged pointer/layout contract to
// System. The counters neither access allocations nor affect their ownership.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACK.load(Relaxed) {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && TRACK.load(Relaxed) {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.load(Relaxed) {
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() && TRACK.load(Relaxed) {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        new
    }
}

pub fn measure<T>(name: &str, mut f: impl FnMut() -> T) {
    for _ in 0..2 {
        drop(black_box(f()));
    }
    let mut times = [0; 7];
    for time in &mut times {
        let start = Instant::now();
        for _ in 0..10 {
            drop(black_box(f()));
        }
        *time = start.elapsed().as_nanos() / 10;
    }
    times.sort_unstable();
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    TRACK.store(true, Relaxed);
    drop(black_box(f()));
    TRACK.store(false, Relaxed);
    assert_eq!(LIVE.load(Relaxed), 0, "benchmark retained measured allocations");
    println!(
        "{name},{},{},{},{}",
        times[3],
        CALLS.load(Relaxed),
        BYTES.load(Relaxed),
        PEAK.load(Relaxed)
    );
}
