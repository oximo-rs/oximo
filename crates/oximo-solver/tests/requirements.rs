use oximo_core::prelude::*;
use oximo_expr::{ExprClass, UnaryOp};
use oximo_solver::{
    ExpressionOperator, ModelRequirements, Requirement, RequirementEntity, RequirementSupport,
    Solver, SolverError, SolverProfile, SolverResult, VariableDomain,
};

fn lp_profile() -> SolverProfile {
    SolverProfile::new("LP only")
        .with_scalar_constraints([ExprClass::Linear])
        .with_scalar_objectives([ExprClass::Linear])
        .with_kinds([ModelKind::LP])
        .with_domains([VariableDomain::Real])
        .with_operators(ExpressionOperator::ALL.iter().copied())
}

#[test]
fn individual_interfaces_and_model_combinations_are_declared_independently() {
    use oximo_core::{ScalarFunctionData, SemanticType, VectorAffineData};
    use oximo_solver::ConstraintSignature;
    let affine = ConstraintSignature::of::<ScalarFunctionData<Affine>, LessThan>();
    let soc = ConstraintSignature::of::<VectorAffineData, SecondOrderCone>();
    let only_kinds = SolverProfile::new("kinds only").with_kinds(oximo_solver::MODEL_KINDS);
    assert_eq!(only_kinds.constraint_support(affine), RequirementSupport::Unsupported);
    assert_eq!(only_kinds.constraint_support(soc), RequirementSupport::Unsupported);
    assert_eq!(
        only_kinds.objective_support(
            SemanticType::of::<ScalarFunctionData<Affine>>(),
            ObjectiveSense::Minimize,
        ),
        RequirementSupport::Unsupported
    );

    let profile = SolverProfile::new("MINLP interface")
        .with_kinds([ModelKind::MINLP])
        .with_scalar_constraints([ExprClass::Linear, ExprClass::Nonlinear])
        .with_scalar_objectives([ExprClass::Linear])
        .with_domains([VariableDomain::Integer])
        .with_operators(ExpressionOperator::ALL.iter().copied());
    assert!(!profile.supports_kind(ModelKind::LP));
    assert_eq!(profile.constraint_support(affine), RequirementSupport::Supported);
    assert_eq!(profile.constraint_support(soc), RequirementSupport::Unsupported);
    assert_eq!(
        profile.clone().with_soc(true).constraint_support(soc),
        RequirementSupport::Supported
    );

    let model = Model::new("affine interface inside MINLP");

    variable!(model, x, Integer);
    constraint!(model, affine, x <= 2.0);
    let nonlinear = constraint!(model, nonlinear, x.exp() <= 8.0);
    objective!(model, Min, x);
    assert!(profile.check_model(&model).is_confirmed());

    model.set_constraint_active(nonlinear, false).unwrap();
    let report = profile.check_model(&model);
    assert_eq!(report.errors.len(), 1);
    assert!(matches!(report.errors[0].requirement,
        Requirement::Combination(features) if features.kind == ModelKind::MILP));
}

#[test]
fn independent_failures_accumulate_with_all_origins_and_stable_order() {
    let model = Model::new("requirements");

    variable!(model, x, Integer);
    variable!(model, y, Integer);
    let shared = x.tan();
    let first = constraint!(model, first, &shared + &shared <= 1.0);
    let second = constraint!(model, second, shared <= 2.0);
    objective!(model, Min, x.sin());

    let profile = lp_profile()
        .with_support(
            Requirement::Operator(ExpressionOperator::Unary(UnaryOp::Tan)),
            RequirementSupport::Unsupported,
        )
        .with_support(
            Requirement::Operator(ExpressionOperator::Unary(UnaryOp::Sin)),
            RequirementSupport::Unsupported,
        );
    let inferred = ModelRequirements::infer(&model);
    let report = profile.check(&inferred);
    assert!(!report.is_supported());

    let integers = report
        .errors
        .iter()
        .find(|d| d.requirement == Requirement::Domain(VariableDomain::Integer))
        .unwrap();
    assert_eq!(integers.origins.len(), 2);
    assert_eq!(integers.origins[0].entity, RequirementEntity::Variable(x.var_id().unwrap()));
    assert_eq!(integers.origins[1].entity, RequirementEntity::Variable(y.var_id().unwrap()));

    let tangent = report
        .errors
        .iter()
        .find(|d| d.requirement == Requirement::Operator(ExpressionOperator::Unary(UnaryOp::Tan)))
        .unwrap();
    assert_eq!(tangent.origins.len(), 2);
    assert_eq!(tangent.origins[0].entity, RequirementEntity::Constraint(first.id()));
    assert_eq!(tangent.origins[1].entity, RequirementEntity::Constraint(second.id()));
    assert!(report.errors.iter().any(|d| matches!(d.requirement, Requirement::Combination(_))));

    let text = report.to_string();
    assert!(
        text.contains("LP only")
            && text.contains("tan")
            && text.contains("first")
            && text.contains("second")
    );
    assert_eq!(text, profile.check(&ModelRequirements::infer(&model)).to_string());
    assert!(
        matches!(report.into_result(), Err(SolverError::IncompatibleModel(report)) if report.errors.len() >= 5)
    );
}

#[test]
fn missing_objective_does_not_hide_constraint_requirements() {
    let model = Model::new("missing objective");

    variable!(model, x);
    constraint!(model, nonlinear, x.exp() <= 1.0);
    let report = lp_profile().check(&ModelRequirements::infer(&model));
    assert!(report.errors.iter().any(|d| d.requirement == Requirement::ObjectiveDeclared));
    assert!(report.errors.iter().any(|d| matches!(d.requirement, Requirement::Constraint(_))));

    objective!(model, Feasibility);

    let report = lp_profile().check(&ModelRequirements::infer(&model));
    assert!(!report.errors.iter().any(|d| d.requirement == Requirement::ObjectiveDeclared));
}

#[test]
fn inference_observes_activity_and_ignores_unreachable_nodes() {
    let model = Model::new("activity");

    variable!(model, x);
    let _unused = x.atan2(x);
    let row = constraint!(model, nonlinear, x.tan() <= 1.0);
    objective!(model, Min, x);

    model.set_constraint_active(row, false).unwrap();
    let requirements = ModelRequirements::infer(&model);
    assert!(lp_profile().check(&requirements).is_confirmed());
    assert!(!requirements.entries.iter().any(|e| {
        e.alternatives.iter().any(|r| matches!(r, Requirement::Operator(ExpressionOperator::Atan2)))
    }));

    model.set_constraint_active(row, true).unwrap();
    assert!(!lp_profile().check(&ModelRequirements::infer(&model)).is_supported());
    assert!(lp_profile().check(&requirements).is_confirmed());
}

#[test]
fn parameter_changes_produce_fresh_classification() {
    let model = Model::new("parameters");

    variable!(model, -1.0 <= x <= 1.0);
    variable!(model, t >= 0.0);
    param!(model, p = 1.0);
    constraint!(model, cone, p * x.square() <= t.square());
    objective!(model, Min, t);

    let profile = lp_profile().with_kinds([ModelKind::SOCP]).with_soc(true);
    let original = ModelRequirements::infer(&model);
    assert!(profile.check(&original).is_supported());

    model.set_param(p, -1.0).unwrap();
    assert!(!profile.check(&ModelRequirements::infer(&model)).is_supported());
    assert!(profile.check(&original).is_supported());
}

#[test]
fn operators_are_required_before_constant_folding_and_powers_are_distinct() {
    let model = Model::new("operators");

    variable!(model, x);
    param!(model, p = 1.0);
    objective!(model, Min, p.tan() * x + x.pow(p) + x.pow(x));

    let requirements = ModelRequirements::infer(&model);
    for operator in [
        ExpressionOperator::Unary(UnaryOp::Tan),
        ExpressionOperator::ConstantPower,
        ExpressionOperator::VariablePower,
    ] {
        assert!(
            requirements
                .entries
                .iter()
                .any(|entry| entry.alternatives.contains(&Requirement::Operator(operator)))
        );
    }
}

#[test]
fn every_expression_operator_requires_an_explicit_declaration() {
    for &operator in ExpressionOperator::ALL {
        let requirement = Requirement::Operator(operator);
        assert_eq!(
            SolverProfile::new("empty").support(&requirement),
            RequirementSupport::Unsupported
        );
        assert_eq!(lp_profile().support(&requirement), RequirementSupport::Supported);
    }
}

#[test]
fn sos_kind_is_resolved_without_treating_partial_support_as_permission() {
    let profile = lp_profile().with_kinds([ModelKind::MILP]).with_sos([SosType::Sos1]);
    for (kind, expected) in [(SosType::Sos1, true), (SosType::Sos2, false)] {
        let model = Model::new("specific SOS requirement");

        variable!(model, x);
        variable!(model, y);
        model.add_sos_constraint("choice", kind, [(x, 1.0), (y, 2.0)]);
        objective!(model, Feasibility);

        let report = profile.check(&ModelRequirements::infer(&model));
        assert_eq!(report.is_supported(), expected);
        assert!(report.conditions.is_empty());
        if !expected {
            assert!(report.errors.iter().any(|d| d.requirement == Requirement::Sos(kind)));
            assert!(report.to_string().contains("Model::reformulate_sos"));
        }
    }
}

#[test]
fn sos_denials_cannot_be_bypassed_by_generic_interface_support() {
    use oximo_core::{SpecialOrderedSet, VariableVectorData};
    use oximo_solver::ConstraintSignature;
    let generic =
        Requirement::Constraint(ConstraintSignature::of::<VariableVectorData, SpecialOrderedSet>());

    for denied in [SosType::Sos1, SosType::Sos2] {
        let profile = lp_profile()
            .with_kinds([ModelKind::MILP])
            .with_sos([SosType::Sos1, SosType::Sos2])
            .with_support(Requirement::Sos(denied), RequirementSupport::Unsupported);
        assert_eq!(profile.sos_support(denied), RequirementSupport::Unsupported);
        assert!(!profile.supports_sos(denied));
        assert_eq!(profile.support(&generic), RequirementSupport::Unsupported);

        let profile = profile.with_support(generic.clone(), RequirementSupport::Supported);
        for kind in [SosType::Sos1, SosType::Sos2] {
            let model = Model::new("SOS override");

            variable!(model, x);
            let row = model.add_sos_constraint("choice", kind, [(x, 1.0)]);
            objective!(model, Feasibility);

            for report in
                [profile.check_model(&model), profile.check(&ModelRequirements::infer(&model))]
            {
                assert_eq!(report.is_confirmed(), kind != denied);
                if kind == denied {
                    assert_eq!(report.errors.len(), 1);
                    assert_eq!(report.errors[0].requirement, Requirement::Sos(kind));
                    assert_eq!(report.errors[0].constraint(), Some(row.id()));
                }
            }
        }
    }
}

#[test]
fn sos_conditions_survive_generic_interface_support() {
    use oximo_core::{SpecialOrderedSet, VariableVectorData};
    use oximo_solver::ConstraintSignature;
    let signature = ConstraintSignature::of::<VariableVectorData, SpecialOrderedSet>();
    let detail = "SOS data must meet an external restriction";

    for kind in [SosType::Sos1, SosType::Sos2] {
        let support = RequirementSupport::Conditional(detail);
        let profile = lp_profile()
            .with_kinds([ModelKind::MILP])
            .with_sos([SosType::Sos1, SosType::Sos2])
            .with_support(Requirement::Sos(kind), support.clone());
        assert_eq!(profile.sos_support(kind), support);
        assert!(!profile.supports_sos(kind));
        assert_eq!(profile.constraint_support(signature), support);

        let profile =
            profile.with_support(Requirement::Constraint(signature), RequirementSupport::Supported);

        let model = Model::new("conditional SOS override");
        variable!(model, x);
        model.add_sos_constraint("choice", kind, [(x, 1.0)]);
        objective!(model, Feasibility);

        for report in
            [profile.check_model(&model), profile.check(&ModelRequirements::infer(&model))]
        {
            assert!(report.is_supported());
            assert!(!report.is_confirmed());
            assert_eq!(report.conditions.len(), 1);

            let diagnostic = &report.conditions[0];
            assert_eq!(diagnostic.requirement, Requirement::Sos(kind));
            assert_eq!(diagnostic.reason, detail);
            assert_eq!(diagnostic.alternatives.len(), 1);
            assert_eq!(diagnostic.alternatives[0].support, support);
        }
    }
}

#[test]
fn failed_detected_cones_retain_scalar_and_soc_alternatives() {
    use oximo_core::{ScalarFunctionData, VectorAffineData};
    use oximo_solver::ConstraintSignature;

    let model = Model::new("unsupported cone representations");

    variable!(model, x);
    variable!(model, t >= 0.0);
    let row = constraint!(model, cone, x.square() <= t.square());
    objective!(model, Min, t);
    let profile = lp_profile().with_kinds([ModelKind::SOCP]);
    let report = profile.check_model(&model);
    assert_eq!(report.errors.len(), 1);

    let diagnostic = &report.errors[0];
    assert_eq!(diagnostic.constraint(), Some(row.id()));
    assert_eq!(
        diagnostic.alternatives.iter().map(|a| a.requirement.clone()).collect::<Vec<_>>(),
        vec![
            Requirement::Constraint(ConstraintSignature::of::<
                ScalarFunctionData<Quadratic>,
                LessThan,
            >()),
            Requirement::Constraint(ConstraintSignature::of::<VectorAffineData, SecondOrderCone>()),
        ]
    );
    assert!(diagnostic.alternatives.iter().all(|a| a.support == RequirementSupport::Unsupported));

    let text = report.to_string();
    assert!(text.contains("SecondOrderCone") && text.contains("ScalarFunctionData"));
}

#[test]
fn external_uncertainty_is_visible_and_permits_a_solve_attempt() {
    let model = Model::new("conditional");

    variable!(model, x);
    objective!(model, Min, x);

    let requirements = ModelRequirements::infer(&model);
    let profile = lp_profile().with_external_uncertainty("custom external solver");
    let report = profile.check(&requirements);
    assert!(report.is_supported());
    assert!(!report.is_confirmed());
    assert_eq!(report.conditions.len(), 1);
    assert!(report.to_string().contains("custom external solver"));
    assert!(report.into_result().is_ok());
}

#[test]
fn combination_rules_retain_features_hidden_by_the_dominant_kind() {
    let model = Model::new("quadratic SDP");

    variable!(model, x);
    psd_constraint!(model, SymmetricMatrix::from_upper_triangle(1, [x]));
    objective!(model, Min, x.square());

    let profile = lp_profile()
        .with_kinds([ModelKind::SDP, ModelKind::QP])
        .with_scalar_objectives([ExprClass::Quadratic])
        .with_psd(true)
        .reject_combination(
            |f| f.has_psd && f.objective_class != oximo_expr::ExprClass::Linear,
            "affine SDP objectives only",
        );

    let report = profile.check(&ModelRequirements::infer(&model));
    assert!(
        report.errors.iter().any(
            |d| matches!(d.requirement, Requirement::Combination(f) if f.kind == ModelKind::SDP)
                && d.reason.contains("affine")
        )
    );
}

#[test]
fn deep_expression_walk_is_iterative() {
    let model = Model::new("deep");

    variable!(model, x);
    let mut expression = x.erase();
    for _ in 0..8_000 {
        expression = expression.sin().erase();
    }
    objective!(model, Min, expression);

    let requirements = ModelRequirements::infer(&model);
    let entry = requirements
        .entries
        .iter()
        .find(|e| {
            e.alternatives.contains(&Requirement::Operator(ExpressionOperator::Unary(UnaryOp::Sin)))
        })
        .unwrap();
    assert_eq!(entry.origins.len(), 1);
}

#[derive(Debug)]
struct LocalRestriction;
impl Solver for LocalRestriction {
    type Options = ();
    fn name(&self) -> &str {
        "local"
    }

    fn profile(&self, (): &()) -> std::borrow::Cow<'_, SolverProfile> {
        std::borrow::Cow::Owned(lp_profile())
    }

    fn validate_requirements(
        &self,
        _: &Model,
        (): &(),
        _: &ModelRequirements,
    ) -> Vec<oximo_solver::RequirementDiagnostic> {
        vec![oximo_solver::RequirementDiagnostic::adapter(
            "local bounds",
            "a local data restriction",
        )]
    }

    fn solve(&mut self, model: &Model, opts: &()) -> Result<SolverResult, SolverError> {
        self.validate_model(model, opts).into_result()?;
        unreachable!()
    }
}

#[test]
fn local_validation_is_combined_with_independent_profile_failures() {
    let model = Model::new("local and structural");

    variable!(model, x, Integer);
    objective!(model, Min, x);

    let report = LocalRestriction.validate_model(&model, &());
    assert!(report.errors.iter().any(|d| matches!(d.requirement, Requirement::Domain(_))));
    assert!(
        report
            .errors
            .iter()
            .any(|d| matches!(d.requirement, Requirement::AdapterRestriction("local bounds")))
    );
}

#[test]
fn cached_structure_still_checks_options_and_local_state_each_time() {
    use std::cell::Cell;
    struct Configurable {
        blocked: Cell<bool>,
        calls: Cell<usize>,
    }
    impl Solver for Configurable {
        type Options = bool;
        fn name(&self) -> &str {
            "configurable"
        }

        fn profile(&self, enabled: &bool) -> std::borrow::Cow<'_, SolverProfile> {
            std::borrow::Cow::Owned({
                if *enabled { lp_profile() } else { SolverProfile::new(self.name()) }
            })
        }

        fn validate_requirements(
            &self,
            _: &Model,
            _: &bool,
            _: &ModelRequirements,
        ) -> Vec<oximo_solver::RequirementDiagnostic> {
            self.calls.set(self.calls.get() + 1);
            if self.blocked.get() {
                vec![oximo_solver::RequirementDiagnostic::adapter("live restriction", "blocked")]
            } else {
                Vec::new()
            }
        }

        fn solve(&mut self, _: &Model, _: &bool) -> Result<SolverResult, SolverError> {
            unreachable!()
        }
    }

    let solver = Configurable { blocked: Cell::new(false), calls: Cell::new(0) };
    let model = Model::new("live options and state");

    variable!(model, x);
    objective!(model, Min, x);
    assert!(solver.validate_model(&model, &true).is_confirmed());

    solver.blocked.set(true);
    assert!(!solver.validate_model(&model, &true).is_supported());

    solver.blocked.set(false);
    assert!(!solver.validate_model(&model, &false).is_supported());
    assert!(solver.validate_model(&model, &true).is_confirmed());
    assert_eq!(solver.calls.get(), 4);
}

#[test]
fn cold_structural_matching_and_full_validation_return_the_same_named_failures() {
    let model = Model::new("lazy diagnostic names");

    variable!(model, x, Integer);
    variable!(model, y, Integer);
    let shared = x.tan();
    constraint!(model, first_named_row, &shared <= 1.0);
    constraint!(model, second_named_row, &shared <= 2.0);
    objective!(model, Min, shared + y);

    let profile = lp_profile().with_support(
        Requirement::Operator(ExpressionOperator::Unary(UnaryOp::Tan)),
        RequirementSupport::Unsupported,
    );
    let cold = profile.check_model(&model);
    let full = profile.check(&ModelRequirements::infer(&model));
    assert_eq!(cold.to_string(), full.to_string());

    let operator = cold
        .errors
        .iter()
        .find(|diagnostic| {
            diagnostic.requirement == Requirement::Operator(ExpressionOperator::Unary(UnaryOp::Tan))
        })
        .unwrap();
    assert_eq!(operator.origins.len(), 3);

    let text = cold.to_string();
    for name in ["x", "y", "first_named_row", "second_named_row"] {
        assert!(text.contains(name), "missing {name}: {text}");
    }
    assert!(cold.errors.iter().flat_map(|d| &d.origins).all(|o| !o.location.is_empty()));
}
