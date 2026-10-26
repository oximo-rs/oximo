#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;
use oximo_solver::{
    AffineRowTranslator, PreparedConstraint, SolverResult, TranslationContext,
    TypedConstraintTranslator,
};

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let function = ScalarFunctionData::from_expression(x);
    let set = LessThan(1.0);
    let entry;
    {
        let context = TranslationContext::new(&model).unwrap();
        entry = context.bind(&function, &set).unwrap();
    }
    let _ = entry.translate(&AffineRowTranslator);
}
