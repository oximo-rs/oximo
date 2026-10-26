use oximo_core::ScalarFunctionData;
use oximo_core::prelude::*;
use oximo_solver::reconstruct::finalize_result;
use oximo_solver::{DualStatus, SolverResult};

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
fn explicit_ir_conversion_preserves_semantic_evaluation_and_dual_defaults() {
    let model = Model::new("explicit lowering coordinates");

    variable!(model, x);
    let handle =
        model.__add_constraint("scaled", Constraint::new(Scaled(x), LessThan(2.0)).into_ir());
    let mut result = SolverResult::for_model(&model);

    result.dual_status = DualStatus::FeasiblePoint;
    result.dual.insert(handle.row_id(), 0.5);
    result.add_solution([(x.var_id().unwrap(), 1.5)].into_iter().collect(), None);
    let result = finalize_result(result, &model, model.num_variables());
    assert_eq!(result.dual_of(handle).unwrap(), None);
    assert_eq!(result.dual[&handle.row_id()], 0.5);

    let evaluation = result.constraint_evaluation(&model, handle).unwrap().unwrap();
    assert_eq!(evaluation.activity, 1.5);
    assert_eq!(evaluation.upper_slack, Some(0.5));
}

#[test]
fn finalization_uses_saved_parameters_and_preserves_explicit_semantic_mappings() {
    let model = Model::new("model-aware result finalization");

    variable!(model, x);
    param!(model, cost = 2.0);
    objective!(model, Min, cost * x);
    let scaled =
        model.__add_constraint("scaled", Constraint::new(Scaled(x), LessThan(2.0)).into_ir());
    let parameters = model.arena().parameter_snapshot();
    let num_variables = model.num_variables();
    model.set_param(cost, 4.0).unwrap();
    model.__var("later").build();

    for mapped in [false, true] {
        let mut result = SolverResult {
            model_id: model.id(),
            dual_status: DualStatus::FeasiblePoint,
            parameters: parameters.clone(),
            ..Default::default()
        };
        result.add_solution([(x.var_id().unwrap(), 1.5)].into_iter().collect(), Some(3.0));
        result.dual.insert(scaled.row_id(), 0.5);
        if mapped {
            result.semantic_dual.insert(scaled.id(), Some(1.0));
        }
        let result = finalize_result(result, &model, num_variables);
        assert_eq!(result.dual_of(scaled).unwrap(), mapped.then_some(1.0));
        assert_eq!(result.dual[&scaled.row_id()], 0.5);
        assert!(result.parameters.same_bits(&parameters));
        assert!(result.best().unwrap().parameters.same_bits(&parameters));
        assert_eq!(result.value_of(cost * x).unwrap(), Some(3.0));
    }
}

#[test]
#[should_panic(expected = "dual provenance belongs to a different model")]
fn finalization_rejects_a_foreign_model() {
    let source = Model::new("source");
    let foreign = Model::new("foreign");

    finalize_result(SolverResult::for_model(&source), &foreign, 0);
}

#[test]
fn finalization_keeps_unavailable_dual_markers_after_clearing_uncertified_multipliers() {
    let model = Model::new("unavailable semantic multipliers");

    variable!(model, x);
    let scaled =
        model.__add_constraint("scaled", Constraint::new(Scaled(x), LessThan(2.0)).into_ir());
    let mut result = SolverResult::for_model(&model);
    result.dual.insert(scaled.row_id(), 0.5);
    result.semantic_dual.insert(scaled.id(), Some(1.0));
    let result = finalize_result(result, &model, 1);

    assert!(result.dual.is_empty());
    assert_eq!(result.semantic_dual.get(&scaled.id()), Some(&None));
    assert!(result.report(&model).unwrap().to_string().contains("scaled  dual = n/a"));
}

#[test]
fn alternative_dual_defaults_survive_normalization_and_preserve_explicit_mappings() {
    let model = Model::new("dual coordinates");

    variable!(model, x);
    variable!(model, y);
    let direct = model.add_constraint("direct", x.le(2.0));
    model.add_constraints([("scaled".into(), Constraint::new(Scaled(y), LessThan(2.0)).into_ir())]);
    let scaled =
        model.typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>("scaled").unwrap();

    let assembled = SolverResult {
        model_id: model.id(),
        parameters: model.arena().parameter_snapshot(),
        ..Default::default()
    }
    .with_semantic_dual_defaults(&model);

    for mut result in [SolverResult::for_model(&model), assembled] {
        result.dual_status = DualStatus::FeasiblePoint;
        result.dual.insert(direct.row_id(), 1.0);
        result.dual.insert(scaled.row_id(), 0.5);
        let result = finalize_result(result, &model, model.num_variables());

        assert_eq!(result.dual_of(direct).unwrap(), Some(1.0));
        assert_eq!(result.dual_of(scaled).unwrap(), None);
        assert_eq!(result.dual[&scaled.row_id()], 0.5);
        let report = result.report(&model).unwrap().to_string();
        assert!(report.contains("direct  dual = 1"), "{report}");
        assert!(report.contains("scaled  dual = n/a"), "{report}");

        let mut mapped = result;
        mapped.semantic_dual.insert(scaled.id(), Some(1.0));
        let mapped = finalize_result(mapped.with_semantic_dual_defaults(&model), &model, 2);

        assert_eq!(mapped.dual_of(scaled).unwrap(), Some(1.0));
        assert_eq!(mapped.dual[&scaled.row_id()], 0.5);
        let report = mapped.report(&model).unwrap().to_string();
        assert!(report.contains("scaled  dual = 1"), "{report}");
    }
}

#[derive(Clone, Debug)]
struct OpaqueData {
    model: oximo_core::ModelId,
    root: oximo_expr::ExprId,
}

impl oximo_core::SemanticFunction for OpaqueData {
    fn model_id(&self) -> oximo_core::ModelId {
        self.model
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(oximo_expr::ExprId)) {
        visitor(self.root);
    }

    fn remap(&mut self, remap: oximo_expr::ExprIdRemap) {
        self.root = remap.apply(self.root);
    }

    fn rebind_model(&mut self, model: oximo_core::ModelId) {
        self.model = model;
    }
}

struct OpaqueScaled<'a>(Expr<'a, Affine>);

impl Function for OpaqueScaled<'_> {
    type Owned = OpaqueData;

    fn semantic_data(&self) -> Self::Owned {
        OpaqueData { model: self.0.model_id(), root: self.0.id() }
    }
}

impl FunctionInSet<LessThan> for OpaqueScaled<'_> {}

impl<'a> LowerConstraint<LessThan> for OpaqueScaled<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;

    fn lower(self, set: LessThan) -> Self::Ir {
        (2.0 * self.0).le(2.0 * set.0).into_ir()
    }
}

#[test]
fn evaluation_requires_a_complete_semantic_scalar_pair() {
    let model = Model::new("semantic evaluation");

    variable!(model, x);
    variable!(model, y);
    model.add_constraints([("scalar".into(), Constraint::new(Scaled(x), LessThan(2.0)).into_ir())]);
    model.add_constraints([(
        "opaque".into(),
        Constraint::new(OpaqueScaled(y), LessThan(2.0)).into_ir(),
    )]);
    let scalar =
        model.typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>("scalar").unwrap();
    let opaque = model.typed_constraint_handle::<OpaqueData, LessThan>("opaque").unwrap();
    let mut result = SolverResult::for_model(&model);
    result.add_solution(
        [(x.var_id().unwrap(), 1.5), (y.var_id().unwrap(), 1.5)].into_iter().collect(),
        None,
    );

    for active in [true, false] {
        model.set_constraint_active(scalar, active).unwrap();
        model.set_constraint_active(opaque, active).unwrap();
        let evaluation = result.constraint_evaluation(&model, scalar).unwrap().unwrap();
        assert_eq!(evaluation.activity, 1.5);
        assert_eq!(evaluation.upper_slack, Some(0.5));
        assert_eq!(evaluation.violation, 0.0);
        assert_eq!(result.constraint_evaluation(&model, opaque).unwrap(), None);
        assert_eq!(result.constraint_evaluation_at(&model, opaque, 0).unwrap(), None);
        assert_eq!(
            result.constraint_evaluation_with_current_parameters(&model, opaque).unwrap(),
            None
        );
        assert_eq!(
            result.constraint_evaluation_at_with_current_parameters(&model, opaque, 0).unwrap(),
            None
        );
    }
}

struct SquaredBound<'a>(Expr<'a, Affine>);
impl Function for SquaredBound<'_> {
    type Owned = ScalarFunctionData<Affine>;
    fn semantic_data(&self) -> Self::Owned {
        ScalarFunctionData::from_expression(self.0)
    }
}
impl FunctionInSet<LessThan> for SquaredBound<'_> {}
impl<'a> LowerConstraint<LessThan> for SquaredBound<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
    fn lower(self, set: LessThan) -> Self::Ir {
        self.0.square().le(set.0 * set.0).into_ir()
    }
}

#[test]
fn native_lowering_is_checked_even_when_the_source_signature_is_supported() {
    use oximo_solver::{Solver, SolverError};
    struct LinearOnly;
    impl Solver for LinearOnly {
        type Options = ();
        fn name(&self) -> &str {
            "linear-only"
        }
        fn supports(&self, kind: ModelKind) -> bool {
            kind == ModelKind::LP
        }
        fn solve(&mut self, _: &Model, (): &()) -> Result<SolverResult, SolverError> {
            unreachable!()
        }
    }
    let model = Model::new("unsupported selected representation");
    variable!(model, x >= 0.0);
    objective!(model, Max, x);
    let row = model.add_lowered_constraint(
        "squared",
        Constraint::new(SquaredBound(x), LessThan(2.0)).into_ir(),
    );
    let validation = LinearOnly.validate_model(&model, &());
    let diagnostic = validation.errors.iter().find(|d| d.constraint == Some(row.id())).unwrap();
    assert!(diagnostic.reason.contains("unsupported function/set pair"));
    assert!(
        diagnostic
            .reason
            .contains(oximo_core::SemanticType::of::<ScalarFunctionData<Affine>>().name())
    );
    assert!(diagnostic.reason.contains(oximo_core::SemanticType::of::<LessThan>().name()));
}

#[test]
fn projection_defaults_cannot_restore_generated_alternative_lowerings() {
    use oximo_solver::reconstruct::{SourceCoordinates, SourceProjection};

    for interleaved in [false, true] {
        for certified in [false, true] {
            let model = Model::new("projected semantic defaults");

            variable!(model, x);
            let original = model.add_lowered_constraint(
                "original",
                Constraint::new(Scaled(x), LessThan(2.0)).into_ir(),
            );
            let coordinates = SourceCoordinates::capture(&model);
            let generated = model.add_lowered_constraint(
                "generated",
                Constraint::new(Scaled(x), LessThan(3.0)).into_ir(),
            );
            let later = interleaved.then(|| {
                model.add_lowered_constraint(
                    "later",
                    Constraint::new(Scaled(x), LessThan(4.0)).into_ir(),
                )
            });
            let mut result = SolverResult::for_model(&model);

            result.add_solution([(x.var_id().unwrap(), 1.0)].into_iter().collect(), Some(1.0));
            if certified {
                result.dual_status = DualStatus::FeasiblePoint;
            }

            result.dual.insert(original.row_id(), 0.5);
            result.dual.insert(generated.row_id(), 7.0);
            result.semantic_dual.clear();
            result.semantic_dual.insert(generated.id(), Some(7.0));
            let mut projection = SourceProjection::with_coordinates(&model, coordinates);
            if let Some(later) = later {
                result.dual.insert(later.row_id(), 0.5);
                projection = projection.with_source_rows([original.row_id(), later.row_id()]);
            }
            let projected = projection.project(result).unwrap();
            assert_eq!(projected.semantic_dual.get(&original.id()), Some(&None));
            assert_eq!(projected.dual_of(original).unwrap(), None);
            assert!(!projected.semantic_dual.contains_key(&generated.id()));
            assert!(!projected.dual.contains_key(&generated.row_id()));
            assert_eq!(projected.dual_of(generated).unwrap(), None);

            if let Some(later) = later {
                assert_eq!(projected.semantic_dual.get(&later.id()), Some(&None));
                assert_eq!(projected.dual_of(later).unwrap(), None);
            }
        }
    }
}
