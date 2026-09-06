#[path = "../../oximo-expr/benches/support/mod.rs"]
mod support;

#[global_allocator]
static ALLOC: support::CountingAllocator = support::CountingAllocator;

fn main() {
    println!("case,nanoseconds,allocations,requested_bytes,peak_live_bytes");
    for n in [32, 1024] {
        let row = (0..n).map(|i| format!("x{i}")).collect::<Vec<_>>().join(" + ");
        let input = format!("Minimize\n obj: {row}\nSubject To\n c: {row} <= 10\nEnd\n");
        support::measure(&format!("lp_wide/{n}"), || {
            oximo_io::lp::read_lp(input.as_bytes()).unwrap()
        });
    }
}
