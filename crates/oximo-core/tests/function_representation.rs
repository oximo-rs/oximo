use oximo_core::prelude::*;
use oximo_core::{
    FunctionValue, IndicatorFunctionData, IndicatorSet, SetValue, SpecialOrderedSet,
    SymmetricAffineData, VariableVectorData, VectorAffineData,
};
use oximo_expr::{ExprId, ExprIdRemap};

#[derive(Clone, Debug)]
struct Variables(VariableVectorData);

impl SemanticFunction for Variables {
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

impl FunctionInSet<SpecialOrderedSet> for Variables {}

#[derive(Clone, Debug)]
struct Indicator(IndicatorFunctionData);

impl SemanticFunction for Indicator {
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

impl FunctionInSet<IndicatorSet> for Indicator {}

#[test]
fn custom_variable_representations_register_update_and_fork_without_losing_identity() {
    let model = Model::new("custom variable representation");
    variable!(model, x);
    variable!(model, y);
    let function = |variable| {
        Variables(VariableVectorData { model_id: model.id(), variables: vec![variable] })
    };
    let handle = model.add_constraint(
        "custom",
        Constraint::new(
            function(x.var_id().unwrap()),
            SpecialOrderedSet { kind: SosType::Sos1, weights: vec![1.0] },
        ),
    );
    assert!(handle.try_sos_id().is_some());
    model.set_constraint_function(handle, function(y.var_id().unwrap())).unwrap();
    model
        .set_constraint_set(handle, SpecialOrderedSet { kind: SosType::Sos2, weights: vec![2.0] })
        .unwrap();
    let rows = model.sos_constraints();
    assert_eq!(rows[handle.index()].members[0].variable, y.var_id().unwrap());
    assert_eq!(rows[handle.index()].members[0].weight, 2.0);
    assert_eq!(rows[handle.index()].sos_type, SosType::Sos2);
    drop(rows);
    let fork = model.fork();
    let rebound = fork.typed_constraint_handle::<Variables, SpecialOrderedSet>("custom").unwrap();
    let record = fork.semantic_constraint(rebound).unwrap().unwrap();
    assert!(record.pair.function.signature().is::<Variables>());
    assert_eq!(record.pair.function.model_id(), fork.id());
    let FunctionRepresentationRef::Variables(ids) = record.pair.function.representation() else {
        panic!("missing variable representation")
    };
    assert_eq!(ids.iter().collect::<Vec<_>>(), vec![y.var_id().unwrap()]);
    let native = model.add_sos_constraint("native", SosType::Sos1, [(x, 1.0), (y, 2.0)]);
    let declarations = model.semantic_constraints_ref();
    let view = declarations.iter().find(|c| c.id == native.id()).unwrap();
    let FunctionRepresentationRef::Variables(ids) = view.pair.function.representation() else {
        panic!("native view lost variable structure")
    };
    assert_eq!(ids.iter().collect::<Vec<_>>(), vec![x.var_id().unwrap(), y.var_id().unwrap()]);
}

#[test]
fn custom_indicator_representations_share_registration_and_update_validation() {
    let model = Model::new("custom indicator representation");
    variable!(model, x);
    variable!(model, y);
    variable!(model, trigger, Bin);
    let function = |root| {
        Indicator(IndicatorFunctionData {
            model_id: model.id(),
            trigger: trigger.var_id().unwrap(),
            consequent: FunctionValue::new(ScalarFunctionData::from_expression(root)),
        })
    };
    let handle = model.add_constraint(
        "custom",
        Constraint::new(
            function(x),
            IndicatorSet { active_value: true, consequent: SetValue::new(LessThan(3.0)) },
        ),
    );
    assert!(handle.try_indicator_id().is_some());
    model.set_constraint_function(handle, function(y)).unwrap();
    model
        .set_constraint_set(
            handle,
            IndicatorSet { active_value: false, consequent: SetValue::new(GreaterThan(1.0)) },
        )
        .unwrap();
    let rows = model.indicator_constraints();
    assert_eq!(rows[handle.index()].lhs, y.id());
    assert!(!rows[handle.index()].active_value);
    assert_eq!(rows[handle.index()].lower, 1.0);
    drop(rows);
    let foreign = Model::new("foreign consequent");
    variable!(foreign, z);
    let invalid = Indicator(IndicatorFunctionData {
        model_id: model.id(),
        trigger: trigger.var_id().unwrap(),
        consequent: FunctionValue::new(ScalarFunctionData::from_expression(z)),
    });
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || model.set_constraint_function(handle, invalid)
        ))
        .is_err()
    );
    assert_eq!(model.indicator_constraints()[handle.index()].lhs, y.id());
    let fork = model.fork();
    let rebound = fork.typed_constraint_handle::<Indicator, IndicatorSet>("custom").unwrap();
    let record = fork.semantic_constraint(rebound).unwrap().unwrap();
    let FunctionRepresentationRef::Indicator { consequent, .. } =
        record.pair.function.representation()
    else {
        panic!("missing indicator representation")
    };
    assert_eq!(consequent.model_id(), fork.id());
    assert!(
        matches!(consequent.representation(), FunctionRepresentationRef::Scalar(root) if root == y.id())
    );
}

#[test]
fn objective_views_remain_consistent_after_replacement_and_independent_forks() {
    let model = Model::new("objective view");
    variable!(model, x);
    objective!(model, Min, x.square());
    let fork = model.fork();
    fork.maximize(ScalarFunctionData::from_expression(fork.variable_handle(x.var_id().unwrap())));
    for m in [&model, &fork] {
        let semantic = m.semantic_objective();
        let semantic = semantic.as_ref().unwrap();
        let algebraic = m.objective();
        let algebraic = algebraic.as_ref().unwrap();
        assert_eq!(semantic.sense, algebraic.sense);
        assert_eq!(semantic.function.model_id(), m.id());
        assert!(
            matches!(semantic.function.representation(), FunctionRepresentationRef::Scalar(root) if root == algebraic.expr)
        );
    }
    assert_eq!(model.objective().as_ref().unwrap().sense, ObjectiveSense::Minimize);
    assert_eq!(fork.objective().as_ref().unwrap().sense, ObjectiveSense::Maximize);
}

#[derive(Clone, Debug)]
struct Wrapped<F>(F);

impl<F: SemanticFunction> SemanticFunction for Wrapped<F> {
    fn model_id(&self) -> ModelId {
        self.0.model_id()
    }
    fn rebind_model(&mut self, model: ModelId) {
        self.0.rebind_model(model);
    }
    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        self.0.visit_expressions(visitor);
    }
    fn remap(&mut self, remap: ExprIdRemap) {
        self.0.remap(remap);
    }
    fn representation(&self) -> FunctionRepresentationRef<'_> {
        self.0.representation()
    }
}

impl FunctionInSet<SecondOrderCone> for Wrapped<VectorAffineData> {}
impl FunctionInSet<PositiveSemidefiniteCone> for Wrapped<SymmetricAffineData> {}

impl<F: SemanticFunction> ObjectiveFunction for Wrapped<F> {}

#[test]
fn custom_objective_payloads_survive_guarded_forks_and_builtin_replacement() {
    let model = Model::new("custom objective");

    variable!(model, x);
    param!(model, p = 2.0);
    let function = Wrapped(ScalarFunctionData::from_expression(p * x));
    model.minimize(function);
    let key = model.__update_key();
    let semantic = model.semantic_objective();
    assert!(
        semantic.as_ref().unwrap().function.signature().is::<Wrapped<ScalarFunctionData<Affine>>>()
    );

    let fork = model.fork();
    let copied = fork.semantic_objective();
    assert!(
        copied.as_ref().unwrap().function.signature().is::<Wrapped<ScalarFunctionData<Affine>>>()
    );
    assert_eq!(copied.as_ref().unwrap().function.model_id(), fork.id());
    assert_eq!(semantic.as_ref().unwrap().function.model_id(), model.id());
    assert_eq!(model.__update_key(), key);
    assert_eq!(
        model.parameter_dependencies().parameters_for(oximo_core::ParameterTarget::Objective),
        &[p.param_id().unwrap()]
    );

    drop(copied);
    drop(semantic);
    objective!(model, Max, x);
    assert!(
        model
            .semantic_objective()
            .as_ref()
            .unwrap()
            .function
            .signature()
            .is::<ScalarFunctionData<Affine>>()
    );
    assert!(
        fork.semantic_objective()
            .as_ref()
            .unwrap()
            .function
            .signature()
            .is::<Wrapped<ScalarFunctionData<Affine>>>()
    );
}

#[test]
fn cone_lookup_handles_match_the_exact_semantic_signature() {
    let model = Model::new("cone handle signatures");
    variable!(model, x);
    variable!(model, t);
    let vector = VectorAffineData { model_id: model.id(), components: vec![t.id(), x.id()].into() };
    let matrix = SymmetricMatrix::from_upper_triangle(2, [t.id(), x.id(), t.id()]);
    let soc = model.add_constraint(
        "wrapped_soc",
        Constraint::new(Wrapped(vector), SecondOrderCone { dimension: 2 }),
    );
    let psd = model.add_constraint(
        "wrapped_psd",
        Constraint::new(
            Wrapped(SymmetricAffineData { model_id: model.id(), matrix }),
            PositiveSemidefiniteCone { side_dimension: 2 },
        ),
    );
    assert!(model.soc_constraint_handle("wrapped_soc").is_none());
    assert!(model.soc_constraint_handle_from_id(soc.try_soc_id().unwrap()).is_none());
    assert!(model.psd_constraint_handle("wrapped_psd").is_none());
    assert!(model.psd_constraint_handle_from_id(psd.try_psd_id().unwrap()).is_none());
    assert!(
        model.try_typed_handle::<VectorAffineData, SecondOrderCone>(soc.erase()).unwrap().is_none()
    );
    assert!(
        model
            .try_typed_handle::<SymmetricAffineData, PositiveSemidefiniteCone>(psd.erase())
            .unwrap()
            .is_none()
    );
    let checked_soc = model
        .try_typed_handle::<Wrapped<VectorAffineData>, SecondOrderCone>(soc.erase())
        .unwrap()
        .unwrap();
    let checked_psd = model
        .try_typed_handle::<Wrapped<SymmetricAffineData>, PositiveSemidefiniteCone>(psd.erase())
        .unwrap()
        .unwrap();
    model.set_constraint_set(checked_soc, SecondOrderCone { dimension: 2 }).unwrap();
    model.set_constraint_set(checked_psd, PositiveSemidefiniteCone { side_dimension: 2 }).unwrap();

    let native_soc = model.add_soc_constraint("native_soc", [x], t);
    let native_psd =
        model.add_psd_constraint("native_psd", SymmetricMatrix::from_upper_fn(2, |_, _| t));
    for handle in [
        model.soc_constraint_handle("native_soc").unwrap(),
        model.soc_constraint_handle_from_id(native_soc.try_soc_id().unwrap()).unwrap(),
    ] {
        assert_eq!(handle, native_soc);
        model.set_constraint_set(handle, SecondOrderCone { dimension: 2 }).unwrap();
    }
    for handle in [
        model.psd_constraint_handle("native_psd").unwrap(),
        model.psd_constraint_handle_from_id(native_psd.try_psd_id().unwrap()).unwrap(),
    ] {
        assert_eq!(handle, native_psd);
        model.set_constraint_set(handle, PositiveSemidefiniteCone { side_dimension: 2 }).unwrap();
    }
    assert!(model.soc_constraint_handle_from_id(oximo_core::SocConstraintId(u32::MAX)).is_none());
    assert!(model.psd_constraint_handle_from_id(oximo_core::PsdConstraintId(u32::MAX)).is_none());
    let fork = model.fork();
    assert!(fork.soc_constraint_handle("wrapped_soc").is_none());
    assert!(fork.psd_constraint_handle("wrapped_psd").is_none());
    assert!(
        fork.typed_constraint_handle::<Wrapped<VectorAffineData>, SecondOrderCone>("wrapped_soc")
            .is_some()
    );
    assert!(
        fork.typed_constraint_handle::<Wrapped<SymmetricAffineData>, PositiveSemidefiniteCone>(
            "wrapped_psd"
        )
        .is_some()
    );
    assert_eq!(fork.soc_constraint_handle("native_soc").unwrap().model_id(), fork.id());
    assert_eq!(fork.psd_constraint_handle("native_psd").unwrap().model_id(), fork.id());
}
