#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;
use oximo_solver::{
    AffineRowTranslator, PreparedConstraint, SolverResult, TranslationContext,
    TypedConstraintTranslator,
};

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let result = SolverResult::for_model(&model);
    let row = constraint!(model, x <= 1.0);
    let _ = result.psd_dual_of(row);
}
