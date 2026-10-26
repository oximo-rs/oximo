use oximo_clarabel::{Clarabel, ClarabelOptions};
use oximo_core::ScalarFunctionData;
use oximo_core::prelude::*;
use oximo_solver::{Capability, ConstraintSignature, Solver};

#[test]
fn detected_soc_is_validated_without_changing_its_declared_quadratic_identity() {
    let model = Model::new("detected SOC semantic validation");

    variable!(model, x >= 3.0);
    variable!(model, y >= 4.0);
    variable!(model, t >= 0.0);

    let cone = constraint!(model, cone, x.square() + y.square() <= t.square());
    objective!(model, Min, t);

    let options = ClarabelOptions::default();
    let signature = ConstraintSignature::of::<ScalarFunctionData<Quadratic>, LessThan>();

    assert_eq!(Clarabel.constraint_support(signature, &options), Capability::Unsupported);
    assert!(Clarabel.validate_model(&model, &options).is_supported());
    assert!(
        model
            .semantic_constraint(cone)
            .unwrap()
            .unwrap()
            .pair
            .function
            .signature()
            .is::<ScalarFunctionData<Quadratic>>()
    );

    let result = Clarabel.solve(&model, &options).unwrap();
    assert!((result.value_of(t).unwrap().unwrap() - 5.0).abs() < 1e-6);
}

#[derive(Clone, Debug)]
struct OpaqueData {
    model: ModelId,
    root: oximo_expr::ExprId,
}

impl SemanticFunction for OpaqueData {
    fn model_id(&self) -> ModelId {
        self.model
    }
    fn rebind_model(&mut self, model: ModelId) {
        self.model = model;
    }
    fn visit_expressions(&self, visitor: &mut dyn FnMut(oximo_expr::ExprId)) {
        visitor(self.root);
    }
    fn remap(&mut self, remap: oximo_expr::ExprIdRemap) {
        self.root = remap.apply(self.root);
    }
}

struct Opaque<'a>(Expr<'a>);
impl Function for Opaque<'_> {
    type Owned = OpaqueData;
    fn semantic_data(&self) -> Self::Owned {
        OpaqueData { model: self.0.model_id(), root: self.0.id() }
    }
}
impl FunctionInSet<LessThan> for Opaque<'_> {}
impl<'a> LowerConstraint<LessThan> for Opaque<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
    fn lower(self, set: LessThan) -> Self::Ir {
        (2.0 * self.0).le(2.0 * set.0).into_ir()
    }
}

#[derive(Clone, Debug)]
struct CustomUpper(f64);
impl oximo_core::ConstraintSet for CustomUpper {}
impl FunctionInSet<CustomUpper> for ScalarAffineFunction<'_> {}
impl<'a> LowerConstraint<CustomUpper> for ScalarAffineFunction<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
    fn lower(self, set: CustomUpper) -> Self::Ir {
        self.expression().le(set.0).into_ir()
    }
}

#[test]
fn explicit_opaque_and_custom_set_lowerings_validate_and_solve_in_native_coordinates() {
    use oximo_solver::PersistentSolver;
    let model = Model::new("explicit native lowerings");
    variable!(model, x);
    variable!(model, y);
    objective!(model, Max, x + y);
    let opaque = model.add_lowered_constraint(
        "opaque",
        Constraint::new(Opaque(x.erase()), LessThan(2.0)).into_ir(),
    );
    let custom = model.add_lowered_constraint(
        "custom_set",
        Constraint::new(y.into_function(), CustomUpper(2.0)).into_ir(),
    );
    let options = ClarabelOptions::default();
    assert_eq!(
        Clarabel.constraint_support(ConstraintSignature::of::<OpaqueData, LessThan>(), &options),
        Capability::Unsupported
    );
    assert_eq!(
        Clarabel.constraint_support(
            ConstraintSignature::of::<ScalarFunctionData<Affine>, CustomUpper>(),
            &options
        ),
        Capability::Unsupported
    );
    assert!(Clarabel.validate_model(&model, &options).is_supported());
    let mut persistent = Clarabel.persistent();
    assert!(persistent.validate_model(&model, &options).is_supported());
    for result in
        [Clarabel.solve(&model, &options).unwrap(), persistent.solve(&model, &options).unwrap()]
    {
        assert!((result.value_of(x).unwrap().unwrap() - 2.0).abs() < 1e-6);
        assert!((result.value_of(y).unwrap().unwrap() - 2.0).abs() < 1e-6);
        assert_eq!(
            result.dual_of(model.require_scalar_dual_handle(opaque).unwrap()).unwrap(),
            None
        );
        assert_eq!(
            result.dual_of(model.require_scalar_dual_handle(custom).unwrap()).unwrap(),
            None
        );
    }
    assert!(
        model
            .semantic_constraint(opaque)
            .unwrap()
            .unwrap()
            .pair
            .function
            .signature()
            .is::<OpaqueData>()
    );
    assert!(
        model
            .semantic_constraint(custom)
            .unwrap()
            .unwrap()
            .pair
            .set
            .signature()
            .is::<CustomUpper>()
    );
}

#[test]
fn explicitly_lowered_opaque_quadratic_can_use_native_soc_detection() {
    let model = Model::new("opaque SOC lowering");
    variable!(model, x >= 3.0);
    variable!(model, y >= 4.0);
    variable!(model, t >= 0.0);
    let source = model.add_lowered_constraint(
        "opaque_soc",
        Constraint::new(Opaque((x.square() + y.square() - t.square()).erase()), LessThan(0.0))
            .into_ir(),
    );
    objective!(model, Min, t);
    let options = ClarabelOptions::default();
    assert!(Clarabel.validate_model(&model, &options).is_supported());
    let result = Clarabel.solve(&model, &options).unwrap();
    assert!((result.value_of(t).unwrap().unwrap() - 5.0).abs() < 1e-6);
    assert_eq!(result.dual_of(source).unwrap(), None);
}
