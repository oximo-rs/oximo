use oximo_core::function_set::{PreparedScalarConstraint, ScalarConstraintInput};
use oximo_core::prelude::*;

struct ForgedSoc<'a>(Constraint<ScalarAffineFunction<'a>, LessThan>);

impl ScalarConstraintInput for ForgedSoc<'_> {
    type Function = oximo_core::VectorAffineData;
    type Set = SecondOrderCone;

    fn into_scalar_input(self) -> PreparedScalarConstraint {
        self.0.into_scalar_input()
    }
}

fn main() {
    let model = Model::new("forged SOC handle");
    variable!(model, x);
    let handle: SocConstraintHandle = model.__add_constraint("forged", ForgedSoc(x.le(1.0)));
    let _ = handle.soc_id();
}
