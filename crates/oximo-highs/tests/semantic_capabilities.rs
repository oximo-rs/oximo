use oximo_core::prelude::*;
use oximo_core::{FunctionRepresentationRef, ScalarFunctionData};
use oximo_highs::{Highs, HighsOptions};
use oximo_solver::{Capability, ConstraintSignature, Solver};

struct Scaled<'a>(Expr<'a, Affine>);
impl Function for Scaled<'_> {
    type Owned = ScalarFunctionData<Affine>;
    fn semantic_data(&self) -> Self::Owned {
        ScalarFunctionData::from_expression(self.0)
    }
}
impl FunctionInSet<LessThan> for Scaled<'_> {}
impl<'a> LowerConstraint<LessThan> for Scaled<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
    fn lower(self, set: LessThan) -> Self::Ir {
        (2.0 * self.0).le(2.0 * set.0).into_ir()
    }
}

#[test]
fn individually_supported_domains_and_objectives_do_not_admit_miqp() {
    for domain in [Domain::Integer, Domain::SemiContinuous { threshold: 1.0 }] {
        let model = Model::new("mixed quadratic");
        let x = model.__var("x").domain(domain).build();
        objective!(model, Min, x.square());
        let options = HighsOptions::default();
        assert_eq!(Highs.domain_support(domain), Capability::Native);
        assert_eq!(
            Highs.objective_support(
                oximo_core::SemanticType::of::<ScalarFunctionData<Quadratic>>(),
                ObjectiveSense::Minimize,
                &options
            ),
            Capability::Native
        );
        assert_eq!(model.kind(), ModelKind::MIQP);
        assert!(!Highs.validate_model(&model, &options).is_supported());
    }
}

#[test]
fn compilation_can_narrow_without_changing_the_declared_function() {
    let model = Model::new("quadratic upper bound");

    variable!(model, x >= 0.0);
    let function = ScalarQuadraticFunction::from(x);
    let row =
        model.add_constraint("declared_quadratic", Constraint::new(function, GreaterThan(1.0)));
    objective!(model, Min, x);

    let options = HighsOptions::default();
    assert_eq!(
        Highs.constraint_support(
            ConstraintSignature::of::<ScalarFunctionData<Quadratic>, GreaterThan>(),
            &options
        ),
        Capability::Unsupported
    );
    assert!(Highs.validate_model(&model, &options).is_supported());

    let result = Highs.solve(&model, &options).unwrap();
    assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-8);

    let record = model.semantic_constraint(row).unwrap().unwrap();
    assert!(record.pair.function.signature().is::<ScalarFunctionData<Quadratic>>());
    assert!(matches!(record.pair.function.representation(), FunctionRepresentationRef::Scalar(_)));
}

#[test]
fn affine_intervals_with_quadratic_declarations_narrow_without_bridges() {
    use oximo_solver::PersistentSolver;

    let model = Model::new("conservative interval degree");

    variable!(model, x);
    let row = model.add_constraint(
        "band",
        Constraint::new(ScalarQuadraticFunction::from(x), Interval { lower: 1.0, upper: 2.0 }),
    );
    objective!(model, Min, x);

    let options = HighsOptions::default();
    assert_eq!(
        Highs.constraint_support(
            ConstraintSignature::of::<ScalarFunctionData<Quadratic>, Interval>(),
            &options
        ),
        Capability::Unsupported
    );
    assert!(Highs.validate_model(&model, &options).is_supported());

    let mut resident = Highs.persistent();
    for result in
        [Highs.solve(&model, &options).unwrap(), resident.solve(&model, &options).unwrap()]
    {
        assert!((result.value_of(x).unwrap().unwrap() - 1.0).abs() < 1e-8);
    }

    let record = model.semantic_constraint(row).unwrap().unwrap();
    assert!(record.pair.function.signature().is::<ScalarFunctionData<Quadratic>>());
    assert!(record.pair.set.signature().is::<Interval>());
}

#[test]
fn inactive_soc_does_not_block_lp_solving_or_persistent_snapshots() {
    use oximo_solver::PersistentSolver;
    let model = Model::new("disabled cone LP");

    variable!(model, 3.0 <= fixed_x <= 3.0);
    variable!(model, bound >= 0.0);
    let cone = model.add_soc_constraint("disabled", [fixed_x], bound);
    objective!(model, Min, bound);

    model.set_constraint_active(cone, false).unwrap();
    assert_eq!(model.kind(), ModelKind::LP);

    let options = HighsOptions::default();
    assert!(Highs.validate_model(&model, &options).is_supported());

    let mut resident = Highs.persistent();
    for result in
        [Highs.solve(&model, &options).unwrap(), resident.solve(&model, &options).unwrap()]
    {
        assert!(result.value_of(bound).unwrap().unwrap().abs() < 1e-9);
    }
}

#[test]
fn alternative_lowering_updates_fail_before_changing_the_native_optimum() {
    let model = Model::new("scaled scalar lowering");

    variable!(model, 0.0 <= x <= 10.0);
    objective!(model, Max, x);
    model.add_constraints([("row".into(), Constraint::new(Scaled(x), LessThan(2.0)).into_ir())]);

    let handle =
        model.typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>("row").unwrap();
    assert!(model.set_constraint_set(handle, LessThan(3.0)).is_err());
    assert!(model.set_constraint_function(handle, (2.0 * x).into_function()).is_err());

    model.set_constraint_active(handle, false).unwrap();
    model.set_constraint_active(handle, true).unwrap();
    let result = Highs.solve(&model, &HighsOptions::default()).unwrap();
    assert!((result.value_of(x).unwrap().unwrap() - 2.0).abs() < 1e-9);

    let direct_model = Model::new("direct semantic registration");
    variable!(direct_model, 0.0 <= y <= 10.0);
    objective!(direct_model, Max, y);
    let direct = direct_model.add_constraint("row", Constraint::new(Scaled(y), LessThan(2.0)));
    direct_model.set_constraint_set(direct, LessThan(3.0)).unwrap();
    let result = Highs.solve(&direct_model, &HighsOptions::default()).unwrap();
    assert!((result.value_of(y).unwrap().unwrap() - 3.0).abs() < 1e-9);
}

#[test]
fn alternative_lowering_duals_do_not_inherit_native_row_coordinates() {
    use oximo_solver::{DualStatus, PersistentSolver};

    let model = Model::new("direct and scaled multipliers");
    variable!(model, 0.0 <= x <= 10.0);
    variable!(model, 0.0 <= y <= 10.0);
    param!(model, cost = 1.0);
    objective!(model, Max, cost * (x + y));

    let direct = model.add_constraint("direct", x.le(2.0));
    model.add_constraints([("scaled".into(), Constraint::new(Scaled(y), LessThan(2.0)).into_ir())]);
    let scaled =
        model.typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>("scaled").unwrap();
    let erased = model.constraint_handle("scaled").unwrap();
    let options = HighsOptions::default();
    let mut resident = Highs.persistent();

    for coefficient in [1.0, 2.0] {
        model.set_param(cost, coefficient).unwrap();
        for result in
            [Highs.solve(&model, &options).unwrap(), resident.solve(&model, &options).unwrap()]
        {
            assert!((result.value_of(x).unwrap().unwrap() - 2.0).abs() < 1e-9);
            assert!((result.value_of(y).unwrap().unwrap() - 2.0).abs() < 1e-9);
            assert_eq!(result.dual_status, DualStatus::FeasiblePoint);
            assert!((result.dual_of(direct).unwrap().unwrap() - coefficient).abs() < 1e-9);
            assert_eq!(result.dual_of(scaled).unwrap(), None);
            assert_eq!(
                result.dual_of(model.require_scalar_dual_handle(erased).unwrap()).unwrap(),
                None
            );
            assert!((result.dual.get(scaled.row_id()).unwrap() - coefficient / 2.0).abs() < 1e-9);

            let report = result.report(&model).unwrap().to_string();
            assert!(report.contains(&format!("direct  dual = {coefficient}")), "{report}");
            assert!(report.contains("scaled  dual = n/a"), "{report}");
        }
    }
}
