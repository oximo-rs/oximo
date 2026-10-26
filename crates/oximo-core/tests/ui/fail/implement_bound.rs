#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;
#[derive(Clone, Copy)]
struct ForeignBound;
impl IntoBound<'_> for ForeignBound {
    fn __into_bound_for_model(self, _: ModelId) -> Result<Bound, ModelError> {
        Ok(Bound::Constant(0.0))
    }
}

fn main() {}
