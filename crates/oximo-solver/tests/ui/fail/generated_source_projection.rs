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
    let _ = oximo_solver::reconstruct::SourceProjection::capture(&model)
        .with_source_variables([GeneratedVarId::from_raw(VarId(0))]);
}
