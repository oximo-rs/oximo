#![allow(unused_imports, unused_variables, dead_code)]

use oximo_core::prelude::*;
use oximo_solver::{
    AffineRowTranslator, PreparedConstraint, SolverResult, TranslationContext,
    TypedConstraintTranslator,
};

// References belong to the original inputs.
fn references<'ctx, F, S>(
    entry: PreparedConstraint<'ctx, F, S>,
) -> (&'ctx F, &'ctx S, &'ctx TranslationContext) {
    (entry.function(), entry.set(), entry.context())
}

fn main() {
    let model = Model::new("contract");
    variable!(model, x);
    let function = ScalarFunctionData::from_expression(x);
    let set = LessThan(1.0);
    let context = TranslationContext::new(&model).unwrap();
    let entry = context.bind(&function, &set).unwrap();
    let (bound_function, bound_set, bound_context) = references(entry);
    assert_eq!(bound_function.expression(), function.expression());
    assert_eq!(bound_set.0, set.0);
    assert_eq!(bound_context.model_id(), model.id());
    let _ = entry.translate(&AffineRowTranslator).unwrap();
    let row = constraint!(model, x <= 1.0);
    let result = SolverResult::for_model(&model);
    let _ = result.dual_of(row).unwrap();
    let checked = model.require_scalar_dual_handle(row.erase()).unwrap();
    let _ = result.dual_of(checked).unwrap();
    let cone = soc_constraint!(model, [x] <= x);
    let _ = result.soc_dual_of(cone).unwrap();
    let psd = psd_constraint!(model, SymmetricMatrix::from_upper_triangle(1, [x]));
    let _ = result.psd_dual_of(psd).unwrap();
    for handles in [constraint!(model, 0 <= x <= 1), constraint!(model, 0 <= x.square() <= 1)] {
        match handles {
            RangeConstraintHandles::Interval(row) => {
                let _ = result.dual_of(row).unwrap();
            }
            RangeConstraintHandles::Split { lower, upper } => {
                let _ = result.dual_of(lower).unwrap();
                let _ = result.dual_of(upper).unwrap();
            }
        }
    }
}
