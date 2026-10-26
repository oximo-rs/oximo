#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let _ = ConstraintHandle::<ScalarFunctionData<Affine>, LessThan> {
        id: SemanticConstraintId::Algebraic(ConstraintId(0)),
        model_id: model.id(),
        marker: std::marker::PhantomData,
    };
}
