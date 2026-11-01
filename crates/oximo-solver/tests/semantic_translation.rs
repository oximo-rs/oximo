use oximo_core::prelude::*;
use oximo_core::{ModelId, ScalarFunctionData, SemanticFunction, SemanticType};
use oximo_expr::{ExprClass, ExprId, ExprIdRemap};
use oximo_solver::{
    AffineRowTranslator, ConstraintSignature, DualStatus, RequirementSupport, Solver, SolverError,
    SolverResult, TranslationContext,
};

#[derive(Debug)]
struct BoundRows;

impl Solver for BoundRows {
    type Options = ();

    fn name(&self) -> &str {
        "bound rows"
    }

    fn profile(&self, _opts: &Self::Options) -> std::borrow::Cow<'_, oximo_solver::SolverProfile> {
        std::borrow::Cow::Owned({
            oximo_solver::SolverProfile::new(self.name())
                .with_scalar_constraints([ExprClass::Linear])
                .with_scalar_objectives([ExprClass::Linear, ExprClass::Quadratic])
                .with_kinds(oximo_solver::MODEL_KINDS.into_iter().filter(|&kind| {
                    matches!(kind, ModelKind::LP | ModelKind::MILP | ModelKind::QP)
                }))
                .with_domains([
                    oximo_solver::VariableDomain::Real,
                    oximo_solver::VariableDomain::Integer,
                    oximo_solver::VariableDomain::Binary,
                ])
                .with_operators(oximo_solver::ExpressionOperator::ALL.iter().copied())
                .with_support(
                    oximo_solver::Requirement::Constraint(ConstraintSignature::of::<
                        ScalarFunctionData<Affine>,
                        Interval,
                    >()),
                    RequirementSupport::Unsupported,
                )
        })
    }

    fn solve(&mut self, model: &Model, opts: &()) -> Result<SolverResult, SolverError> {
        self.validate_model(model, opts).into_result()?;
        let mut result = SolverResult::for_model(model);
        result.dual_status = DualStatus::FeasiblePoint;
        let primal = oximo_solver::SolutionValues::from_dense(vec![1.0; model.num_variables()]);
        result.dual = oximo_solver::SolutionValues::missing(model.constraints().algebraic().len());
        result.add_solution(primal, Some(1.0));
        for record in model.semantic_constraints().into_iter().filter(|c| c.active) {
            if let oximo_core::SemanticConstraintId::Algebraic(row) = record.id {
                result.dual.set(
                    row,
                    Some(if record.pair.set.signature().is::<GreaterThan>() { 1.0 } else { 0.0 }),
                );
            }
        }
        Ok(result)
    }
}

#[test]
fn pair_support_does_not_imply_supported_combinations() {
    let solver = BoundRows;
    let m = Model::new("MIQP combination");

    variable!(m, x, Integer);
    objective!(m, Min, x.square());
    assert_eq!(solver.profile(&()).domain_support(Domain::Integer), RequirementSupport::Supported);
    assert_eq!(
        solver.profile(&()).objective_support(
            SemanticType::of::<ScalarFunctionData<Quadratic>>(),
            ObjectiveSense::Minimize
        ),
        RequirementSupport::Supported
    );

    let report = solver.validate_model(&m, &());
    assert!(!report.is_supported());
    assert!(report.errors.iter().any(|d| d.reason.contains("combination")));
}

#[derive(Clone, Debug)]
struct CustomData {
    model: ModelId,
    root: ExprId,
}

impl SemanticFunction for CustomData {
    fn model_id(&self) -> ModelId {
        self.model
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.root);
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.root = remap.apply(self.root);
    }

    fn rebind_model(&mut self, model: ModelId) {
        self.model = model;
    }
}

struct CustomFunction(CustomData);

impl Function for CustomFunction {
    type Owned = CustomData;

    fn semantic_data(&self) -> CustomData {
        self.0.clone()
    }
}

#[derive(Clone, Debug)]
struct CustomSet;

impl ConstraintSet for CustomSet {}
impl FunctionInSet<CustomSet> for CustomFunction {}
impl FunctionInSet<oximo_core::SpecialOrderedSet> for CustomFunction {}

#[derive(Debug)]
struct Sos1Rows;

impl Solver for Sos1Rows {
    type Options = ();

    fn name(&self) -> &str {
        "SOS1 rows"
    }

    fn profile(&self, _opts: &Self::Options) -> std::borrow::Cow<'_, oximo_solver::SolverProfile> {
        std::borrow::Cow::Owned({
            oximo_solver::SolverProfile::new(self.name())
                .with_scalar_constraints([ExprClass::Linear])
                .with_scalar_objectives([ExprClass::Linear])
                .with_kinds(
                    oximo_solver::MODEL_KINDS
                        .into_iter()
                        .filter(|&kind| matches!(kind, ModelKind::LP | ModelKind::MILP)),
                )
                .with_domains([
                    oximo_solver::VariableDomain::Real,
                    oximo_solver::VariableDomain::Integer,
                    oximo_solver::VariableDomain::Binary,
                ])
                .with_operators(oximo_solver::ExpressionOperator::ALL.iter().copied())
                .with_sos([SosType::Sos1])
                .with_indicators(true)
        })
    }

    fn solve(&mut self, model: &Model, opts: &()) -> Result<SolverResult, SolverError> {
        self.validate_model(model, opts).into_result()?;
        Ok(SolverResult::for_model(model))
    }
}

#[derive(Clone, Debug)]
struct Described<T: SemanticFunction>(T);

impl<T: SemanticFunction> SemanticFunction for Described<T> {
    fn model_id(&self) -> ModelId {
        self.0.model_id()
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        self.0.visit_expressions(visitor);
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.0.remap(remap);
    }

    fn rebind_model(&mut self, model: ModelId) {
        self.0.rebind_model(model);
    }

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        self.0.representation()
    }
}

impl FunctionInSet<oximo_core::SpecialOrderedSet> for Described<oximo_core::VariableVectorData> {}
impl FunctionInSet<oximo_core::IndicatorSet> for Described<oximo_core::IndicatorFunctionData> {}
impl FunctionInSet<LessThan> for Described<ScalarFunctionData<Affine>> {}

#[test]
fn custom_function_diagnostics_retain_native_alternatives_and_live_origins() {
    use oximo_solver::{Requirement, SolverProfile, VariableDomain};
    let model = Model::new("custom scalar alternatives");

    variable!(model, x);
    let row = model.add_constraint(
        "custom",
        Constraint::new(Described(ScalarFunctionData::from_expression(x)), LessThan(2.0)),
    );
    objective!(model, Min, x);
    let source = Requirement::Constraint(ConstraintSignature::of::<
        Described<ScalarFunctionData<Affine>>,
        LessThan,
    >());

    let native =
        Requirement::Constraint(ConstraintSignature::of::<ScalarFunctionData<Affine>, LessThan>());

    let profile = SolverProfile::new("no scalar rows")
        .with_scalar_objectives([ExprClass::Linear])
        .with_kinds([ModelKind::LP])
        .with_domains([VariableDomain::Real])
        .with_operators(oximo_solver::ExpressionOperator::ALL.iter().copied());

    let report = profile.check_model(&model);
    assert_eq!(report.errors.len(), 1);

    let mut diagnostic = report.errors[0].clone();
    assert_eq!(diagnostic.constraint(), Some(row.id()));
    assert_eq!(diagnostic.location(), "constraint \"custom\"");
    assert_eq!(
        diagnostic.alternatives.iter().map(|a| a.requirement.clone()).collect::<Vec<_>>(),
        vec![source.clone(), native.clone()]
    );
    assert!(diagnostic.alternatives.iter().all(|a| a.support == RequirementSupport::Unsupported));

    diagnostic.origins[0].location = "renamed row".into();
    diagnostic.origins[0].entity = oximo_solver::RequirementEntity::Objective;
    assert_eq!(diagnostic.location(), "renamed row");
    assert_eq!(diagnostic.constraint(), None);

    diagnostic.origins.clear();
    assert_eq!(diagnostic.location(), "");

    let detail = "native scalar data needs a local check";
    let support = RequirementSupport::Conditional(detail);
    let report = profile.with_support(native.clone(), support.clone()).check_model(&model);
    assert!(report.is_supported() && !report.is_confirmed());
    assert_eq!(report.conditions.len(), 1);

    let diagnostic = &report.conditions[0];
    assert_eq!(diagnostic.requirement, native);
    assert_eq!(diagnostic.alternatives[0].requirement, source);
    assert_eq!(diagnostic.alternatives[0].support, RequirementSupport::Unsupported);
    assert_eq!(diagnostic.alternatives[1].support, support);
    assert!(report.to_string().contains(detail));
}

#[test]
fn custom_structured_representations_are_classified_without_erasing_their_identity() {
    let model = Model::new("described native constraints");

    variable!(model, x);
    variable!(model, trigger, Bin);
    objective!(model, Feasibility);
    let sos = model.add_constraint(
        "sos",
        Constraint::new(
            Described(oximo_core::VariableVectorData {
                model_id: model.id(),
                variables: vec![x.var_id().unwrap()],
            }),
            oximo_core::SpecialOrderedSet { kind: SosType::Sos1, weights: vec![1.0] },
        ),
    );
    let indicator = model.add_constraint(
        "indicator",
        Constraint::new(
            Described(oximo_core::IndicatorFunctionData {
                model_id: model.id(),
                trigger: trigger.var_id().unwrap(),
                consequent: oximo_core::FunctionValue::new(ScalarFunctionData::from_expression(x)),
            }),
            oximo_core::IndicatorSet {
                active_value: true,
                consequent: oximo_core::SetValue::new(LessThan(2.0)),
            },
        ),
    );

    assert!(Sos1Rows.validate_model(&model, &()).is_supported());
    assert!(Sos1Rows.solve(&model, &()).is_ok());
    assert!(sos.try_sos_id().is_some());
    assert!(indicator.try_indicator_id().is_some());
    let records = model.semantic_constraints_ref();
    for record in records.iter() {
        assert_eq!(record.pair.function.signature().function_kind(), FunctionKind::Extension);
        assert!(!matches!(
            record.pair.function.representation(),
            FunctionRepresentationRef::Opaque
        ));
    }
}

#[test]
fn opaque_pairs_are_rejected_before_backend_validation() {
    let model = Model::new("custom SOS representation");

    variable!(model, x);
    objective!(model, Feasibility);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.add_constraint(
            "custom",
            Constraint::new(
                CustomFunction(CustomData { model: model.id(), root: x.id() }),
                oximo_core::SpecialOrderedSet { kind: SosType::Sos1, weights: vec![1.0] },
            ),
        );
    }));
    assert!(error.is_err());

    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.add_constraint(
            "custom",
            Constraint::new(
                CustomFunction(CustomData { model: model.id(), root: x.id() }),
                CustomSet,
            ),
        );
    }));
    assert!(error.is_err());
    assert_eq!(model.num_constraints(), 0);
    assert!(model.constraint_handle("custom").is_none());

    let native = model.add_sos_constraint("native", SosType::Sos1, [(x, 1.0)]);
    assert!(Sos1Rows.validate_model(&model, &()).is_supported());
    assert!(Sos1Rows.solve(&model, &()).is_ok());

    model
        .set_constraint_set(
            native,
            oximo_core::SpecialOrderedSet { kind: SosType::Sos2, weights: vec![1.0] },
        )
        .unwrap();
    assert!(!Sos1Rows.validate_model(&model, &()).is_supported());
}

#[test]
fn typed_translation_checks_snapshot_ownership_and_numeric_data() {
    let m = Model::new("typed translator");

    variable!(m, x);
    param!(m, p = 2.0);
    let function = Function::semantic_data(&(p * x + 1.0).into_function());
    let context = TranslationContext::new(&m).unwrap();
    p.set_param_value(4.0).unwrap();
    let row =
        context.bind(&function, &LessThan(5.0)).unwrap().translate(&AffineRowTranslator).unwrap();
    assert_eq!(row.terms.coeffs[0].1, 2.0);
    assert_eq!(row.upper, 4.0);
    assert!(
        context
            .bind(&function, &LessThan(f64::NAN))
            .unwrap()
            .translate(&AffineRowTranslator)
            .is_err()
    );

    let other = m.fork();
    let foreign = Function::semantic_data(&other.__constant(1.0).into_function());
    assert!(context.bind(&foreign, &LessThan(1.0)).is_err());
}

#[test]
fn newer_function_roots_require_a_fresh_translation_context() {
    let model = Model::new("new expression after snapshot");

    variable!(model, x);
    let context = TranslationContext::new(&model).unwrap();
    let function = Function::semantic_data(&(2.0 * x + 1.0).into_function());
    let error = context.bind(&function, &LessThan(5.0)).unwrap_err();
    assert!(error.to_string().contains("refresh TranslationContext"));

    let fresh = TranslationContext::new(&model).unwrap();
    assert!(fresh.bind(&function, &LessThan(5.0)).unwrap().translate(&AffineRowTranslator).is_ok());
}

#[derive(Debug)]
struct RestrictedRows(RequirementSupport);

impl Solver for RestrictedRows {
    type Options = ();

    fn name(&self) -> &str {
        "restricted rows"
    }

    fn profile(&self, _opts: &Self::Options) -> std::borrow::Cow<'_, oximo_solver::SolverProfile> {
        std::borrow::Cow::Owned({
            oximo_solver::SolverProfile::new(self.name())
                .with_scalar_constraints([ExprClass::Linear])
                .with_scalar_objectives([ExprClass::Linear, ExprClass::Quadratic])
                .with_kinds(
                    oximo_solver::MODEL_KINDS
                        .into_iter()
                        .filter(|&kind| BoundRows.profile(&()).supports_kind(kind)),
                )
                .with_domains([
                    oximo_solver::VariableDomain::Real,
                    oximo_solver::VariableDomain::Integer,
                    oximo_solver::VariableDomain::Binary,
                ])
                .with_operators(oximo_solver::ExpressionOperator::ALL.iter().copied())
                .with_support(
                    oximo_solver::Requirement::Constraint(ConstraintSignature::of::<
                        ScalarFunctionData<Affine>,
                        LessThan,
                    >()),
                    self.0.clone(),
                )
                .with_support(
                    oximo_solver::Requirement::Objective {
                        function: SemanticType::of::<ScalarFunctionData<Affine>>(),
                        sense: ObjectiveSense::Minimize,
                    },
                    self.0.clone(),
                )
                .with_support(
                    oximo_solver::Requirement::Domain(oximo_solver::VariableDomain::Real),
                    self.0.clone(),
                )
        })
    }

    fn solve(&mut self, model: &Model, opts: &()) -> Result<SolverResult, SolverError> {
        self.validate_model(model, opts).into_result()?;
        BoundRows.solve(model, opts)
    }
}

#[test]
fn validation_preserves_capability_reasons() {
    let model = Model::new("capability diagnostics");

    variable!(model, x);
    let cap = constraint!(model, cap, x <= 2.0);
    objective!(model, Min, x);
    assert!(
        RestrictedRows(RequirementSupport::Supported).validate_model(&model, &()).is_supported()
    );

    let detail = "custom data check is required";
    let validation =
        RestrictedRows(RequirementSupport::Conditional(detail)).validate_model(&model, &());
    assert_eq!(validation.conditions.len(), 3);
    assert!(validation.conditions.iter().all(|error| error.reason.contains(detail)));
    assert!(validation.conditions.iter().any(|error| error.constraint() == Some(cap.id())));
    assert!(validation.conditions.iter().any(|error| error.location() == "objective"));
    assert!(validation.conditions.iter().any(|error| error.location() == "variable \"x\""));
    assert!(!validation.is_confirmed());
    assert!(validation.into_result().is_ok());

    assert!(
        !RestrictedRows(RequirementSupport::Unsupported).validate_model(&model, &()).is_supported()
    );
}

#[test]
fn prepared_entries_keep_their_snapshot_when_parameters_and_contexts_change() {
    let model = Model::new("prepared entries");

    variable!(model, x);
    param!(model, coefficient = 2.0);
    let function = ScalarFunctionData::from_expression(coefficient * x);
    let set = LessThan(10.0);
    let old = TranslationContext::new(&model).unwrap();
    let entry = old.bind(&function, &set).unwrap();

    coefficient.set_param_value(4.0).unwrap();
    let fresh = TranslationContext::new(&model).unwrap();
    let old_row = entry.translate(&AffineRowTranslator).unwrap();
    let fresh_row = fresh.bind(&function, &set).unwrap().translate(&AffineRowTranslator).unwrap();
    assert_eq!(old_row.terms.coeffs.as_ref(), [(x.var_id().unwrap(), 2.0)]);
    assert_eq!(fresh_row.terms.coeffs.as_ref(), [(x.var_id().unwrap(), 4.0)]);
}

#[test]
fn scalar_view_checks_preserve_identity_and_reject_foreign_or_conic_handles() {
    let model = Model::new("scalar views");

    variable!(model, x);
    let row = constraint!(model, x <= 1.0);
    let cone = soc_constraint!(model, [x] <= x);
    let checked = model.require_scalar_dual_handle(row.erase()).unwrap();
    assert_eq!(checked.id(), row.id());
    assert_eq!(checked.model_id(), model.id());
    assert!(model.scalar_dual_handle(cone).unwrap().is_none());
    assert_eq!(
        model.require_scalar_dual_handle(cone),
        Err(oximo_core::ScalarDualHandleError::MissingScalarView(cone.id()))
    );

    let other = Model::new("foreign scalar view");
    let error = other.scalar_dual_handle(row).unwrap_err();
    assert_eq!(error, oximo_core::ModelMismatchError::new(other.id(), model.id()));
    assert_eq!(
        other.require_scalar_dual_handle(row),
        Err(oximo_core::ScalarDualHandleError::ModelMismatch(error))
    );
    let foreign_result = SolverResult::for_model(&other);
    assert!(foreign_result.dual_of(checked).is_err());
}

// Plain owned data explicitly opts into the shared-access immutability contract.
impl oximo_core::ImmutableSemanticFunction for CustomData {}
impl FunctionInSet<LessThan> for CustomData {}

#[derive(Debug)]
struct CustomRootTranslator;

impl oximo_solver::TypedConstraintTranslator<CustomData, LessThan> for CustomRootTranslator {
    type Output = ExprId;

    fn translate(
        &self,
        entry: oximo_solver::PreparedConstraint<'_, CustomData, LessThan>,
    ) -> Result<Self::Output, SolverError> {
        Ok(entry.function().root)
    }
}

#[test]
fn immutable_custom_functions_remain_supported_by_prepared_translation() {
    let model = Model::new("immutable custom function");

    variable!(model, x);
    let function = CustomData { model: model.id(), root: x.id() };
    let set = LessThan(1.0);
    let context = TranslationContext::new(&model).unwrap();
    let entry = context.bind(&function, &set).unwrap();
    assert_eq!(entry.translate(&CustomRootTranslator).unwrap(), x.id());
}
