use oximo_core::prelude::*;
use oximo_expr::ExprIdRemap;
use oximo_solver::SolverResult;

#[derive(Clone, Debug)]
struct CustomData {
    model: ModelId,
    root: ExprId,
}
impl SemanticFunction for CustomData {
    fn model_id(&self) -> ModelId {
        self.model
    }
    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.root);
    }
    fn remap(&mut self, remap: ExprIdRemap) {
        self.root = remap.apply(self.root);
    }
    fn rebind_model(&mut self, model: ModelId) {
        self.model = model;
    }
}
impl FunctionInSet<LessThan> for CustomData {}

fn scalar_set_does_not_prove_row_membership(
    result: &SolverResult,
    handle: ConstraintHandle<CustomData, LessThan>,
) {
    let _ = result.dual_of(handle);
}
fn main() {}
