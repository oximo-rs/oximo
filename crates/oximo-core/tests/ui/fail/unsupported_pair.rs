use oximo_core::prelude::*;
struct CustomFunction;
#[derive(Clone, Debug)]
struct CustomSet;
impl Function for CustomFunction {
    type Owned = ScalarFunctionData<Affine>;
    fn semantic_data(&self) -> Self::Owned { unimplemented!() }
}
impl ConstraintSet for CustomSet {}
fn main() {
    let _ = Constraint::new(CustomFunction, CustomSet);
}
