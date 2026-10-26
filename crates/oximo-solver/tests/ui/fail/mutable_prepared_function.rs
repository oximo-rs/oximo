use oximo_core::prelude::*;
use oximo_expr::ExprIdRemap;
use oximo_solver::TranslationContext;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

#[derive(Clone, Debug)]
struct MutableFunction {
    model: ModelId,
    root: Arc<AtomicU32>,
}

impl SemanticFunction for MutableFunction {
    fn model_id(&self) -> ModelId {
        self.model
    }
    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(ExprId(self.root.load(Ordering::Relaxed)));
    }
    fn remap(&mut self, remap: ExprIdRemap) {
        let root = ExprId(self.root.load(Ordering::Relaxed));
        self.root.store(remap.apply(root).0, Ordering::Relaxed);
    }
    fn rebind_model(&mut self, model: ModelId) {
        self.model = model;
    }
}
impl FunctionInSet<LessThan> for MutableFunction {}

fn main() {
    let model = Model::new("mutable root");
    variable!(model, x);
    let function = MutableFunction { model: model.id(), root: Arc::new(AtomicU32::new(x.id().0)) };
    let context = TranslationContext::new(&model).unwrap();
    let _ = context.bind(&function, &LessThan(1.0));
}
