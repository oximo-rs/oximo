use oximo_core::prelude::*;
struct CustomFunction;
struct CustomSet;
impl Function for CustomFunction {}
impl ConstraintSet for CustomSet {}
fn main() {
    let _ = Constraint::new(CustomFunction, CustomSet);
}
