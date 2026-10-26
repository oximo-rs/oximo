use oximo_core::{ConstraintHandle, ScalarDualQuery};
#[derive(Clone, Copy)]
struct ForeignQuery;
impl ScalarDualQuery for ForeignQuery {
    fn scalar_handle(self) -> ConstraintHandle {
        unreachable!()
    }
}
fn main() {}
