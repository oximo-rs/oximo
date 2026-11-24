use oximo_core::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Fake;

impl ConstraintSet for Fake {}

impl SocSet for Fake {
    const KIND: SocKind = SocKind::Standard;
    fn dimension(&self) -> usize {
        2
    }
}

fn main() {}
