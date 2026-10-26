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
    let cone = soc_constraint!(model, [x] <= x);
    let _ = result.dual_of(cone);
}
