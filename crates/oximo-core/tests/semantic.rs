use oximo_core::prelude::*;
use oximo_core::{
    AnyScalarFunctionData, ConstraintHandle, ConstraintOrigin, Erased, FunctionRepresentationRef,
    FunctionValue, ModelId, ScalarFunctionData, SemanticFunction,
};
use oximo_expr::{ExprId, ExprIdRemap};

#[derive(Clone, Debug)]
struct CustomData {
    model: ModelId,
    root: ExprId,
}

impl SemanticFunction for CustomData {
    fn model_id(&self) -> ModelId {
        self.model
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model = model_id;
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.root);
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.root = remap.apply(self.root);
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
struct CustomSet(f64);

impl ConstraintSet for CustomSet {}
impl FunctionInSet<CustomSet> for CustomFunction {}
impl FunctionInSet<LessThan> for CustomFunction {}
impl ObjectiveFunction for CustomFunction {}

#[test]
fn identities_survive_shared_scalar_views_and_explicit_erasure() {
    let m = Model::new("semantic identities");
    variable!(m, x);
    let affine: ConstraintHandle<ScalarFunctionData<Affine>, LessThan> =
        constraint!(m, a, x <= 2.0);
    let quadratic = m.add_constraint("q", x.square().le(2.0));
    let nonlinear = constraint!(m, n, x.sin() >= 0.0);
    let band = m.add_constraint(
        "band",
        Constraint::new(x.into_function(), Interval { lower: 2.0, upper: 2.0 }),
    );
    for (handle, expected) in [
        (affine.erase(), "a"),
        (quadratic.erase(), "q"),
        (nonlinear.erase(), "n"),
        (band.erase(), "band"),
    ] {
        let record = m.semantic_constraint(handle).unwrap().unwrap();
        assert_eq!(record.name, expected);
        assert!(matches!(
            record.pair.function.representation(),
            FunctionRepresentationRef::Scalar(_)
        ));
    }
    assert!(
        m.semantic_constraint(quadratic)
            .unwrap()
            .unwrap()
            .pair
            .function
            .signature()
            .is::<ScalarFunctionData<Quadratic>>()
    );
    assert!(m.semantic_constraint(band).unwrap().unwrap().pair.set.signature().is::<Interval>());
    assert_eq!(
        m.typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>("a"),
        Some(affine)
    );
    assert!(m.typed_constraint_handle::<ScalarFunctionData<Quadratic>, LessThan>("a").is_none());
    let _: ConstraintHandle<Erased, Erased> = affine.erase();
}

#[test]
fn opaque_pairs_are_rejected_without_registering_names_or_ids() {
    let m = Model::new("deferred extension");
    variable!(m, x);
    let before = m.to_string();

    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.add_constraint(
            "custom",
            Constraint::new(
                CustomFunction(CustomData { model: m.id(), root: x.id() }),
                CustomSet(3.0),
            ),
        );
    }));
    assert!(error.is_err());
    assert_eq!(m.to_string(), before);
    assert_eq!(m.num_constraints(), 0);
    assert!(m.semantic_constraints_ref().is_empty());
    assert!(m.constraint_handle("custom").is_none());

    let row = m.add_constraint("custom", x.le(3.0));
    assert_eq!(row.row_id().index(), 0);
    let fork = m.fork();
    assert_eq!(fork.num_constraints(), 1);
    assert!(fork.constraint_handle("custom").is_some());
}

#[test]
fn opaque_functions_require_a_representation_even_with_builtin_sets() {
    let m = Model::new("opaque scalar bound");
    variable!(m, x);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.add_constraint(
            "custom",
            Constraint::new(
                CustomFunction(CustomData { model: m.id(), root: x.id() }),
                LessThan(3.0),
            ),
        );
    }));
    assert!(error.is_err());
    assert!(m.constraints().algebraic().is_empty());
    assert!(m.semantic_constraints_ref().is_empty());
}

#[test]
fn semantic_data_does_not_freeze_parameters_and_objective_degrees() {
    let m = Model::new("parameters");
    variable!(m, x);
    param!(m, p = 2.0);
    let handle = constraint!(m, row, p * x <= 3.0);
    objective!(m, Min, x.square());
    let before = m.semantic_constraint(handle).unwrap().unwrap();
    p.set_param_value(4.0);
    let after = m.semantic_constraint(handle).unwrap().unwrap();
    assert_eq!(before.pair.function.expressions(), after.pair.function.expressions());
    assert!(
        m.semantic_objective()
            .as_ref()
            .unwrap()
            .function
            .signature()
            .is::<ScalarFunctionData<Quadratic>>()
    );
    let arena = m.arena();
    let terms = oximo_expr::extract_linear(&arena, after.pair.function.expressions()[0]).unwrap();
    assert_eq!(terms.coeffs[0].1, 4.0);
}

#[test]
fn typed_updates_and_activity_keep_semantics_and_views_consistent() {
    let m = Model::new("modifications");
    variable!(m, x);
    let row = constraint!(m, row, x <= 1.0);
    m.set_constraint_set(row, LessThan(5.0)).unwrap();
    m.set_constraint_function(row, Function::semantic_data(&(2.0 * x).into_function())).unwrap();
    let original = m.semantic_constraint(row).unwrap().unwrap().pair.function.expressions();
    m.set_constraint_active(row, false).unwrap();
    assert!(!m.constraints().algebraic()[row.index()].active);
    assert_eq!(m.semantic_constraint(row).unwrap().unwrap().pair.function.expressions(), original);
    m.set_constraint_set(row, LessThan(7.0)).unwrap();
    assert_eq!(m.constraints().algebraic()[row.index()].upper, f64::INFINITY);
    m.set_constraint_active(row, true).unwrap();
    assert_eq!(m.constraints().algebraic()[row.index()].upper, 7.0);
    assert_eq!(m.constraints().algebraic()[row.index()].lhs, original[0]);
    let foreign = Model::new("other");
    assert!(foreign.set_constraint_set(row, LessThan(4.0)).is_err());
}

#[test]
fn indexed_worker_remapping_retains_function_and_set_signatures() {
    let m = Model::new("indexed semantics");
    variable!(m, x);
    let rows =
        constraint!(m, rows[i in 0..2048], x + f64::from(u32::try_from(i).unwrap()) <= 3000.0);
    let records = m.semantic_constraints();
    assert_eq!(records.len(), 2048);
    for (key, handle) in rows.iter() {
        let record = &records[handle.index()];
        assert_eq!(record.id, handle.id());
        assert!(record.pair.function.signature().is::<ScalarFunctionData<Affine>>());
        assert!(record.pair.set.signature().is::<LessThan>());
        assert_eq!(
            record.pair.function.expressions()[0],
            m.constraints().algebraic()[handle.index()].lhs
        );
        assert_eq!(record.name, format!("rows[{key}]"));
    }
}

#[test]
fn failed_registration_does_not_add_semantic_records() {
    let m = Model::new("bad input");
    variable!(m, x);
    let before = m.semantic_constraints().len();
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.add_constraint("bad", x.le(f64::NAN));
    }));
    assert!(error.is_err());
    assert_eq!(m.semantic_constraints().len(), before);
    assert_eq!(m.num_constraints(), 0);
    let value = FunctionValue::new(CustomData { model: m.id(), root: x.id() });
    assert!(value.downcast_ref::<ScalarFunctionData<Affine>>().is_none());
    let row = constraint!(m, row, x <= 2.0);
    m.set_constraint_origin(row.id(), ConstraintOrigin::Range("range".into()));
    assert!(matches!(
        m.semantic_constraint(row).unwrap().unwrap().origin,
        ConstraintOrigin::Range(_)
    ));
}

#[test]
fn opaque_functions_validate_recognized_scalar_bounds_before_registration() {
    for bound in [f64::NAN, f64::NEG_INFINITY] {
        let model = Model::new("opaque invalid bounds");
        variable!(model, x);
        let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.add_constraint(
                "bad",
                Constraint::new(
                    CustomFunction(CustomData { model: model.id(), root: x.id() }),
                    LessThan(bound),
                ),
            );
        }));

        assert!(error.is_err());
        assert!(model.semantic_constraints_ref().is_empty());
        assert!(!model.contains_constraint_name("bad"));
        assert_eq!(model.num_constraints(), 0);
    }
}

#[test]
fn owned_pairs_register_in_their_native_registries() {
    use oximo_core::{
        IndicatorFunctionData, IndicatorSet, SetValue, SpecialOrderedSet, VariableVectorData,
    };

    let model = Model::new("owned pairs");
    variable!(model, x);
    variable!(model, b, Binary);
    let dynamic = x.erase().into_function().semantic_data();
    model.add_constraint("lt", Constraint::new(dynamic.clone(), LessThan(2.0)));
    model.add_constraint("gt", Constraint::new(dynamic.clone(), GreaterThan(0.0)));
    model.add_constraint("eq", Constraint::new(dynamic.clone(), EqualTo(1.0)));
    model.add_constraint("band", Constraint::new(dynamic, Interval { lower: 0.0, upper: 2.0 }));

    let sos = model.add_constraint(
        "members",
        Constraint::new(
            VariableVectorData { model_id: model.id(), variables: vec![x.var_id().unwrap()] },
            SpecialOrderedSet { kind: SosType::Sos1, weights: vec![1.0] },
        ),
    );
    let indicator = model.add_constraint(
        "switch",
        Constraint::new(
            IndicatorFunctionData {
                model_id: model.id(),
                trigger: b.var_id().unwrap(),
                consequent: FunctionValue::new(x.into_function().semantic_data()),
            },
            IndicatorSet { active_value: true, consequent: SetValue::new(LessThan(2.0)) },
        ),
    );

    assert_eq!(model.constraints().algebraic().len(), 4);
    assert_eq!(model.num_constraints(), 6);
    assert!(sos.try_sos_id().is_some());
    assert!(indicator.try_indicator_id().is_some());
    assert_eq!(model.sos_constraints()[0].members[0].variable, x.var_id().unwrap());
    assert_eq!(model.indicator_constraints()[0].lhs, x.id());
    assert_eq!(model.indicator_constraints()[0].upper, 2.0);
    model.set_constraint_active(indicator, false).unwrap();
    model
        .set_constraint_set(
            indicator,
            IndicatorSet { active_value: false, consequent: SetValue::new(GreaterThan(-1.0)) },
        )
        .unwrap();
    model.set_constraint_active(indicator, true).unwrap();
    assert!(!model.indicator_constraints()[0].active_value);
    assert_eq!(model.indicator_constraints()[0].lower, -1.0);
}

#[test]
fn typed_recovery_uses_identity_when_registry_names_overlap() {
    let model = Model::new("shared name");
    variable!(model, x);
    variable!(model, t);
    let row = model.add_constraint("same", x.le(2.0));
    let cone = model.add_soc_constraint("same", [x], t);

    assert_eq!(
        model.try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(row.erase()).unwrap(),
        Some(row)
    );
    assert_eq!(
        model
            .try_typed_handle::<oximo_core::VectorAffineData, SecondOrderCone>(cone.erase())
            .unwrap(),
        Some(cone)
    );
    assert!(
        model
            .try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(cone.erase())
            .unwrap()
            .is_none()
    );
    assert!(
        Model::new("foreign")
            .try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(row.erase())
            .is_err()
    );
}

#[test]
fn mixed_registry_positions_survive_batches_updates_and_forks() {
    for threads in [1, 6] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let model = Model::new("mixed semantic positions");
            variable!(model, x);
            variable!(model, t);
            variable!(model, enabled, Binary);
            let first_row = model.add_constraint("first", x.le(10.0));
            let cone = model.add_soc_constraint("shared", [x], t);
            let row = model.add_constraint("shared", x.le(2.0));
            let psd =
                model.add_psd_constraint("psd", SymmetricMatrix::from_upper_triangle(1, [t]));
            let sos = sos_constraint!(model, members, SOS1, [x, t]);
            let indicator = indicator_constraint!(model, guard, enabled == 1 => x <= 2.0);
            let first = [
                first_row.erase(),
                cone.erase(),
                row.erase(),
                psd.erase(),
                sos.erase(),
                indicator.erase(),
            ];

            constraint!(model, family[i in 0..1024], x + f64::from(u32::try_from(i).unwrap()) <= 4.0);
            constraint!(model, bands[i in 0..1024], 0.0 <= x + f64::from(u32::try_from(i).unwrap()) <= 5.0);
            let last = model.add_constraint("last", t.le(3.0));

            let records = model.semantic_constraints();
            assert_eq!(records.len(), 2055);
            assert_eq!(records.last().unwrap().id, last.id());
            for (record, handle) in records.iter().zip(first) {
                assert_eq!(record.id, handle.id());
                assert_eq!(model.semantic_constraint(handle).unwrap().unwrap().name, record.name);
            }
            for name in ["family[0]", "family[1023]", "bands[0]", "bands[1023]"] {
                let handle = model.constraint_handle(name).unwrap();
                assert_eq!(model.semantic_constraint(handle).unwrap().unwrap().name, name);
            }

            model.set_constraint_set(row, LessThan(8.0)).unwrap();
            model.set_constraint_active(cone, false).unwrap();
            let fork = model.fork();
            for record in &records {
                let handle = fork.semantic_constraint_handle(record.id).unwrap();
                let recovered = fork.semantic_constraint(handle).unwrap().unwrap();
                assert_eq!(recovered.name, record.name);
                assert_eq!(recovered.pair.function.model_id(), fork.id());
            }
            let fork_row = fork
                .try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(
                    fork.semantic_constraint_handle(row.id()).unwrap(),
                )
                .unwrap()
                .unwrap();
            fork.set_constraint_set(fork_row, LessThan(9.0)).unwrap();
            assert_eq!(model.constraints().algebraic()[row.index()].upper, 8.0);
            assert_eq!(fork.constraints().algebraic()[row.index()].upper, 9.0);
            assert!(
                model
                    .semantic_constraint_handle(oximo_core::ConstraintId(u32::MAX).into())
                    .is_none()
            );
            assert!(model.semantic_constraint_handle(oximo_core::SocConstraintId(u32::MAX).into()).is_none());
        });
    }
}

#[test]
fn invalid_bound_and_dimension_updates_return_errors_without_mutation() {
    use oximo_core::{ConstraintUpdateError, VectorAffineData};

    let model = Model::new("fallible updates");
    variable!(model, x);
    variable!(model, t);
    let row = model.add_constraint("row", x.le(2.0));
    assert!(matches!(
        model.set_constraint_set(row, LessThan(f64::NAN)),
        Err(ConstraintUpdateError::InvalidBounds(_))
    ));
    assert_eq!(model.constraints().algebraic()[row.index()].upper, 2.0);
    let cone = model.add_soc_constraint("cone", [x], t);
    assert!(matches!(
        model.set_constraint_set(cone, SecondOrderCone { dimension: 3 }),
        Err(ConstraintUpdateError::DimensionMismatch { expected: 2, actual: 3 })
    ));
    assert!(matches!(
        model.set_constraint_function(
            cone,
            VectorAffineData {
                model_id: model.id(),
                components: vec![t.id(), x.id(), x.id()].into()
            }
        ),
        Err(ConstraintUpdateError::DimensionMismatch { expected: 2, actual: 3 })
    ));
    assert_eq!(model.soc_constraints()[0].terms, [x.id()]);
    assert_eq!(
        model
            .semantic_constraint(cone)
            .unwrap()
            .unwrap()
            .pair
            .set
            .downcast_ref::<SecondOrderCone>()
            .unwrap()
            .dimension,
        2
    );
}

#[test]
fn unique_functions_remap_in_place_and_shared_functions_remain_independent() {
    use oximo_core::VectorAffineData;

    let model = Model::new("worker roots");
    variable!(model, x);
    let mut batch = x.arena.__begin_batch();
    let snapshot = batch.snapshot();
    let mut forks: Vec<_> = [1.0, 2.0]
        .into_iter()
        .map(|offset| {
            x.arena.__with_fork(snapshot.clone(), || {
                FunctionValue::new(VectorAffineData {
                    model_id: model.id(),
                    components: vec![(x + offset).id()].into(),
                })
            })
        })
        .collect();
    drop(snapshot);
    let remaps = batch.merge(&mut forks);
    drop(batch);

    let before = forks[0].value.downcast_ref::<VectorAffineData>().unwrap().components.as_ptr();
    forks[0].value.remap(remaps[0]);
    let foreign = Model::new("new owner");
    forks[0].value.rebind_model(foreign.id());
    assert_eq!(
        forks[0].value.downcast_ref::<VectorAffineData>().unwrap().components.as_ptr(),
        before
    );
    assert_eq!(forks[0].value.model_id(), foreign.id());

    let original = forks[1].value.clone();
    let old_roots = original.expressions();
    forks[1].value.remap(remaps[1]);
    forks[1].value.rebind_model(foreign.id());
    assert_eq!(original.model_id(), model.id());
    assert_eq!(original.expressions(), old_roots);
    assert_eq!(
        forks[1].value.expressions(),
        old_roots.iter().map(|&root| remaps[1].apply(root)).collect::<Vec<_>>()
    );
    assert_ne!(forks[1].value.expressions(), old_roots);
}

#[test]
fn cone_views_and_forks_share_buffers_but_updates_remain_independent() {
    use oximo_core::{SymmetricAffineData, VectorAffineData};

    let model = Model::new("shared cone storage");
    variable!(model, x);
    variable!(model, t);
    let soc = model.add_soc_constraint("soc", [x; 5], t);
    let matrix = SymmetricMatrix::from_upper_triangle(2, [t, x, t]);
    let psd = model.add_psd_constraint("psd", &matrix);
    let soc_record = model.semantic_constraint(soc).unwrap().unwrap();
    let psd_record = model.semantic_constraint(psd).unwrap().unwrap();
    let components =
        &soc_record.pair.function.downcast_ref::<VectorAffineData>().unwrap().components;
    let triangle = psd_record
        .pair
        .function
        .downcast_ref::<SymmetricAffineData>()
        .unwrap()
        .matrix
        .upper_triangle();

    assert_eq!(components[1..].as_ptr(), model.soc_constraints()[0].terms.as_ptr());
    assert_eq!(triangle.as_ptr(), model.psd_constraints()[0].matrix.upper_triangle().as_ptr());

    let fork = model.fork();
    let fork_soc =
        fork.typed_constraint_handle::<VectorAffineData, SecondOrderCone>("soc").unwrap();
    let fork_psd = fork
        .typed_constraint_handle::<SymmetricAffineData, PositiveSemidefiniteCone>("psd")
        .unwrap();
    let fork_record = fork.semantic_constraint(fork_soc).unwrap().unwrap();
    let fork_function = fork_record.pair.function.downcast_ref::<VectorAffineData>().unwrap();
    assert_eq!(fork_function.model_id, fork.id());
    assert_eq!(fork_function.components.as_ptr(), components.as_ptr());
    assert_eq!(triangle.as_ptr(), fork.psd_constraints()[0].matrix.upper_triangle().as_ptr());

    fork.set_constraint_function(
        fork_soc,
        VectorAffineData {
            model_id: fork.id(),
            components: vec![x.id(), t.id(), t.id(), t.id(), t.id(), t.id()].into(),
        },
    )
    .unwrap();
    fork.set_constraint_function(
        fork_psd,
        SymmetricAffineData {
            model_id: fork.id(),
            matrix: SymmetricMatrix::from_upper_triangle(2, [x.id(), t.id(), x.id()]),
        },
    )
    .unwrap();

    assert_eq!(model.soc_constraints()[0].terms, [x.id(); 5]);
    assert_eq!(fork.soc_constraints()[0].terms, [t.id(); 5]);
    assert_eq!(model.psd_constraints()[0].matrix.upper_triangle(), &[t.id(), x.id(), t.id()]);
    assert_eq!(fork.psd_constraints()[0].matrix.upper_triangle(), &[x.id(), t.id(), x.id()]);
    assert_eq!(soc_record.pair.function.model_id(), model.id());
    assert_eq!(components.as_ref(), &[t.id(), x.id(), x.id(), x.id(), x.id(), x.id()]);
}

#[test]
fn small_cone_views_keep_bound_prefixes_and_independent_updates() {
    use oximo_core::VectorAffineData;

    for width in 1..=5 {
        let model = Model::new("small cone components");
        variable!(model, x);
        variable!(model, t);
        let cone = model.add_soc_constraint("cone", std::iter::repeat_n(x, width), t);
        let snapshot = model.semantic_constraint(cone).unwrap().unwrap();
        let roots = snapshot.pair.function.expressions();
        assert_eq!(roots[0], t.id());
        assert_eq!(&roots[1..], vec![x.id(); width]);
        let mut borrowed_roots = Vec::new();
        model
            .semantic_constraints_ref()
            .get(0)
            .unwrap()
            .pair
            .function
            .visit_expressions(&mut |root| borrowed_roots.push(root));
        assert_eq!(borrowed_roots, roots);

        let fork = model.fork();
        let handle =
            fork.typed_constraint_handle::<VectorAffineData, SecondOrderCone>("cone").unwrap();
        fork.set_constraint_function(
            handle,
            VectorAffineData {
                model_id: fork.id(),
                components: std::iter::once(x.id())
                    .chain(std::iter::repeat_n(t.id(), width))
                    .collect(),
            },
        )
        .unwrap();
        assert_eq!(fork.soc_constraints()[0].bound, x.id());
        assert_eq!(&*fork.soc_constraints()[0].terms, vec![t.id(); width]);
        assert_eq!(model.soc_constraints()[0].bound, t.id());
        assert_eq!(&*model.soc_constraints()[0].terms, vec![x.id(); width]);
        assert_eq!(snapshot.pair.function.expressions(), roots);
    }
}

#[test]
fn owned_cones_still_validate_roots_affinity_dimensions_and_ownership() {
    use oximo_core::{SymmetricAffineData, VectorAffineData};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let model = Model::new("checked owned cones");
    variable!(model, x);
    let nonlinear = x.square().id();
    let foreign = Model::new("foreign");

    for (owner, root, dimension) in [
        (model.id(), nonlinear, 2),
        (model.id(), ExprId(u32::MAX), 2),
        (model.id(), x.id(), 3),
        (foreign.id(), x.id(), 2),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                model.add_constraint(
                    "invalid soc",
                    Constraint::new(
                        VectorAffineData { model_id: owner, components: vec![x.id(), root].into() },
                        SecondOrderCone { dimension },
                    ),
                );
            }))
            .is_err()
        );
    }

    for root in [nonlinear, ExprId(u32::MAX)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                model.add_constraint(
                    "invalid psd",
                    Constraint::new(
                        SymmetricAffineData {
                            model_id: model.id(),
                            matrix: SymmetricMatrix::from_upper_triangle(1, [root]),
                        },
                        PositiveSemidefiniteCone { side_dimension: 1 },
                    ),
                );
            }))
            .is_err()
        );
    }

    assert_eq!(model.num_constraints(), 0);
}

#[test]
fn symmetric_worker_remapping_detaches_shared_components() {
    use oximo_core::SymmetricAffineData;

    let model = Model::new("shared symmetric worker roots");
    variable!(model, x);
    let mut batch = x.arena.__begin_batch();
    let snapshot = batch.snapshot();
    let mut workers: Vec<_> = [1.0, 2.0]
        .into_iter()
        .map(|offset| {
            x.arena.__with_fork(snapshot.clone(), || {
                FunctionValue::new(SymmetricAffineData {
                    model_id: model.id(),
                    matrix: SymmetricMatrix::from_upper_fn(2, |_, _| (x + offset).id()),
                })
            })
        })
        .collect();
    drop(snapshot);
    let remaps = batch.merge(&mut workers);
    drop(batch);
    let original = workers[1].value.clone();
    let before = original.expressions();
    workers[1].value.remap(remaps[1]);

    assert_eq!(original.expressions(), before);
    assert_eq!(
        workers[1].value.expressions(),
        before.iter().map(|&root| remaps[1].apply(root)).collect::<Vec<_>>()
    );
    assert_ne!(workers[1].value.expressions(), before);
}

#[test]
fn inline_sets_keep_exact_signatures_and_custom_sets_keep_their_values() {
    use oximo_core::SetValue;

    let built_in = SetValue::new(LessThan(3.0_f64));
    let generic = SetValue::new(LessThan(4.0_f32));
    let custom = SetValue::new(CustomSet(5.0));
    assert_eq!(built_in.downcast_ref::<LessThan>().unwrap().0, 3.0);
    assert_eq!(generic.downcast_ref::<LessThan<f32>>().unwrap().0, 4.0);
    assert!(generic.downcast_ref::<LessThan>().is_none());
    assert_eq!(custom.clone().downcast_ref::<CustomSet>().unwrap().0, 5.0);
    assert_eq!(built_in.scalar_bounds().unwrap().1, 3.0);
    assert_eq!(generic.scalar_bounds(), None);
}

#[test]
fn opaque_objectives_are_rejected_without_replacing_the_solve_direction() {
    for initial in 0..3 {
        let m = Model::new("opaque objective");
        variable!(m, x);
        match initial {
            0 => {}
            1 => {
                objective!(m, Feasibility);
            }
            _ => {
                objective!(m, Min, x);
            }
        }
        let before = m.to_string();
        let declared = m.ensure_objective_declared().is_ok();
        let feasibility = m.is_feasibility();
        let objective = m.objective().as_ref().map(|o| (o.expr, o.sense));
        let signature = m.semantic_objective().as_ref().map(|o| o.function.signature());

        for minimize in [true, false] {
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let function = CustomFunction(CustomData { model: m.id(), root: x.id() });
                if minimize {
                    m.minimize(function);
                } else {
                    m.maximize(function);
                }
            }));
            assert!(error.is_err());
            assert_eq!(m.to_string(), before);
            assert_eq!(m.ensure_objective_declared().is_ok(), declared);
            assert_eq!(m.is_feasibility(), feasibility);
            assert_eq!(m.objective().as_ref().map(|o| (o.expr, o.sense)), objective);
            assert_eq!(m.semantic_objective().as_ref().map(|o| o.function.signature()), signature);
        }
        let fork = m.fork();
        assert_eq!(fork.ensure_objective_declared().is_ok(), declared);
        assert_eq!(fork.is_feasibility(), feasibility);
    }
}

#[test]
fn special_constraint_updates_preserve_typed_identity_and_nested_sets() {
    let m = Model::new("special modifications");
    variable!(m, x);
    variable!(m, y);
    variable!(m, z);
    variable!(m, binary, Binary);
    variable!(m, other_binary, Binary);
    let choice: ConstraintHandle<oximo_core::VariableVectorData, oximo_core::SpecialOrderedSet> =
        sos_constraint!(m, choice, SOS1, [x, y]);
    m.set_constraint_set(
        choice,
        oximo_core::SpecialOrderedSet { kind: SosType::Sos2, weights: vec![3.0, 4.0] },
    )
    .unwrap();
    m.set_constraint_function(
        choice,
        oximo_core::VariableVectorData {
            model_id: m.id(),
            variables: vec![y.var_id().unwrap(), z.var_id().unwrap()],
        },
    )
    .unwrap();
    assert_eq!(m.sos_constraints()[choice.index()].sos_type, SosType::Sos2);
    assert_eq!(m.sos_constraints()[choice.index()].members[0].variable, y.var_id().unwrap());
    let guard = indicator_constraint!(m, guard, binary == 1 => x <= 2.0);
    let original = m.semantic_constraint(guard).unwrap().unwrap();
    let original_set = original.pair.set.downcast_ref::<oximo_core::IndicatorSet>().unwrap();
    assert!(original_set.consequent.signature().is::<LessThan>());
    m.set_constraint_set(
        guard,
        oximo_core::IndicatorSet {
            active_value: false,
            consequent: oximo_core::SetValue::new(GreaterThan(3.0)),
        },
    )
    .unwrap();
    m.set_constraint_function(
        guard,
        oximo_core::IndicatorFunctionData {
            model_id: m.id(),
            trigger: other_binary.var_id().unwrap(),
            consequent: FunctionValue::new(Function::semantic_data(&(2.0 * x).into_function())),
        },
    )
    .unwrap();
    assert_eq!(m.indicator_constraints()[guard.index()].trigger, other_binary.var_id().unwrap());
    assert!(!m.indicator_constraints()[guard.index()].active_value);
    assert_eq!(m.indicator_constraints()[guard.index()].lower, 3.0);
    assert_eq!(m.semantic_constraint(guard).unwrap().unwrap().id, guard.id());
    let foreign = Model::new("foreign");
    assert!(choice.reformulate(&foreign, SosReformulationOptions::default()).is_err());
}

#[test]
fn semantic_display_retains_original_function_when_view_is_inactive() {
    let m = Model::new("semantic rendering");
    variable!(m, x);
    let row = constraint!(m, row, x <= 2.0);
    m.set_constraint_active(row, false).unwrap();
    assert_eq!(m.display_constraint(row).unwrap().to_string(), "row: x <= 2 (inactive)");
    let other = Model::new("other");
    assert!(other.display_constraint(row).is_err());
}

#[test]
fn disabled_sources_do_not_lock_bounds_but_reformulated_sources_reject_updates() {
    let m = Model::new("constraint lifecycle");
    variable!(m, 0.0 <= x <= 10.0);
    variable!(m, 0.0 <= y <= 10.0);
    variable!(m, 0.0 <= z <= 10.0);
    variable!(m, binary, Binary);
    let guard = indicator_constraint!(m, guard, binary == 1 => x <= 2.0);
    let choice = sos_constraint!(m, choice, SOS1, [x, y]);
    m.set_constraint_active(guard, false).unwrap();
    m.set_constraint_active(choice, false).unwrap();
    m.fix(x, 3.0).unwrap();
    m.unfix_var(x.var_id().unwrap(), 0.0, 10.0);
    m.set_constraint_active(guard, true).unwrap();
    m.set_constraint_active(choice, true).unwrap();
    guard.reformulate(&m, IndicatorReformulationOptions::default()).unwrap();
    choice.reformulate(&m, SosReformulationOptions::default()).unwrap();
    let before = m.to_string();
    assert!(
        m.set_constraint_set(
            guard,
            oximo_core::IndicatorSet {
                active_value: true,
                consequent: oximo_core::SetValue::new(LessThan(8.0)),
            }
        )
        .is_err()
    );
    assert!(
        m.set_constraint_function(
            guard,
            oximo_core::IndicatorFunctionData {
                model_id: m.id(),
                trigger: binary.var_id().unwrap(),
                consequent: FunctionValue::new(Function::semantic_data(&z.into_function())),
            }
        )
        .is_err()
    );
    assert!(
        m.set_constraint_function(
            choice,
            oximo_core::VariableVectorData {
                model_id: m.id(),
                variables: vec![y.var_id().unwrap(), z.var_id().unwrap()],
            }
        )
        .is_err()
    );
    assert!(
        m.set_constraint_set(
            choice,
            oximo_core::SpecialOrderedSet { kind: SosType::Sos1, weights: vec![2.0, 3.0] }
        )
        .is_err()
    );
    assert!(m.set_constraint_active(guard, true).is_err());
    assert_eq!(m.to_string(), before);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.fix(x, 1.0))).is_err());
}

#[test]
fn typed_updates_accept_modeling_functions_and_native_views_are_fallible() {
    let m = Model::new("ergonomic semantic updates");
    variable!(m, x);
    variable!(m, t >= 0.0);
    let row = constraint!(m, row, x <= 3.0);
    m.set_constraint_function(row, (2.0 * x).into_function()).unwrap();
    let cone = m.add_soc_constraint("cone", [x], t);
    assert_eq!(row.try_row_id(), Some(row.row_id()));
    assert_eq!(cone.try_soc_id(), Some(cone.soc_id()));
    assert!(row.try_soc_id().is_none());
    assert!(cone.try_row_id().is_none());
    assert!(row.try_psd_id().is_none());
    assert!(row.try_sos_id().is_none());
    assert!(row.try_indicator_id().is_none());
    let records = m.semantic_constraints_ref();
    let same_records = m.semantic_constraints_ref();
    assert_eq!(records.len(), same_records.len());
    let mut roots = Vec::new();
    let record = records.get(1).unwrap();
    record.pair.function.visit_expressions(&mut |root| roots.push(root));
    let oximo_core::FunctionRepresentationRef::VectorAffine(components) =
        record.pair.function.representation()
    else {
        panic!("vector representation");
    };
    assert_eq!(components, roots);
}

#[test]
fn borrowed_declarations_match_owned_snapshots_across_registries() {
    let model = Model::new("canonical source views");
    variable!(model, x >= 0.0);
    variable!(model, t >= 0.0);
    variable!(model, b, Bin);
    constraint!(model, affine, x <= 2.0);
    constraint!(model, quadratic, x.square() >= 1.0);
    model.add_soc_constraint("soc", [x], t);
    model.add_psd_constraint("psd", SymmetricMatrix::from_upper_triangle(2, [t, x, t]));
    model.add_sos_constraint("sos", SosType::Sos1, [(x, 1.0), (t, 2.0)]);
    model.add_indicator_constraint("indicator", b, true, x.le(3.0));
    model.add_constraint(
        "interval",
        Constraint::new(x.into_function(), Interval { lower: 0.0, upper: 5.0 }),
    );

    let owned = model.semantic_constraints();
    let borrowed = model.semantic_constraints_ref();
    assert_eq!(owned.len(), borrowed.len());
    for (view, snapshot) in borrowed.iter().zip(&owned) {
        assert_eq!(view.id, snapshot.id);
        assert_eq!(view.name, snapshot.name.as_str());
        assert_eq!(view.active, snapshot.active);
        assert_eq!(view.pair.function.signature(), snapshot.pair.function.signature());
        assert_eq!(view.pair.set.signature(), snapshot.pair.set.signature());
        assert_eq!(view.pair.function.model_id(), model.id());
        let mut roots = Vec::new();
        view.pair.function.visit_expressions(&mut |root| roots.push(root));
        assert_eq!(roots, snapshot.pair.function.expressions());
        assert_eq!(view.pair.set.scalar_bounds(), snapshot.pair.set.scalar_bounds());
    }
}

#[test]
fn inactive_canonical_scalar_edits_preserve_the_source_until_reactivation() {
    let model = Model::new("inactive source storage");
    variable!(model, x);
    let row = constraint!(model, row, x <= 2.0);
    model.set_constraint_active(row, false).unwrap();
    model.set_constraint_set(row, LessThan(7.0)).unwrap();
    model.set_constraint_function(row, (3.0 * x).into_function()).unwrap();

    let source = model.semantic_constraint(row).unwrap().unwrap();
    assert!(!source.active);
    assert_eq!(source.pair.set.scalar_bounds().unwrap().1, 7.0);
    let root = source.pair.function.expressions()[0];
    assert_ne!(root, model.constraints().algebraic()[row.index()].lhs);

    let sibling = model.fork();
    let sibling_row = sibling
        .try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(
            sibling.semantic_constraint_handle(row.id()).unwrap(),
        )
        .unwrap()
        .unwrap();
    sibling.set_constraint_active(sibling_row, true).unwrap();
    assert_eq!(sibling.constraints().algebraic()[row.index()].lhs, root);
    assert_eq!(sibling.constraints().algebraic()[row.index()].upper, 7.0);
    assert!(!model.constraints().algebraic()[row.index()].active);

    model.set_constraint_active(row, true).unwrap();
    assert_eq!(model.constraints().algebraic()[row.index()].lhs, root);
}

#[test]
fn fork_first_edits_detach_names_bounds_parameters_and_provenance() {
    let model = Model::new("fork lifecycle");
    variable!(model, 0.0 <= x <= 10.0);
    param!(model, p = 2.0);
    let row = constraint!(model, row, x <= p);
    let snapshot = model.semantic_constraint(row).unwrap().unwrap();
    let fork = model.fork();
    let fork_row = fork.semantic_constraint_handle(row.id()).unwrap();
    fork.fix_var(x.var_id().unwrap(), 4.0);
    fork.set_param_id(p.param_id().unwrap(), 8.0);
    fork.set_constraint_origin(fork_row.id(), ConstraintOrigin::Range("fork range".into()));
    variable!(fork, y >= 0.0);
    constraint!(fork, added, y <= 3.0);

    assert_eq!(model.variables()[0].ub, 10.0);
    assert_eq!(model.param_value_of(p).unwrap(), Some(2.0));
    assert!(!model.contains_constraint_name("added"));
    assert!(matches!(snapshot.origin, ConstraintOrigin::Declared));
    assert!(matches!(
        model.semantic_constraint(row).unwrap().unwrap().origin,
        ConstraintOrigin::Declared
    ));
    assert!(matches!(
        fork.semantic_constraint(fork_row).unwrap().unwrap().origin,
        ConstraintOrigin::Range(_)
    ));

    // Forking is also valid while a caller holds read guards over source storage.
    let views = model.semantic_constraints_ref();
    let guarded_fork = model.fork();
    assert_eq!(guarded_fork.num_constraints(), views.len());
    assert_eq!(guarded_fork.semantic_constraints()[0].pair.function.model_id(), guarded_fork.id());
}

#[test]
fn range_registration_preserves_static_and_dynamic_function_identities() {
    let model = Model::new("range signatures");
    variable!(model, x);
    model.__add_range("static", x, 1.0, 2.0);
    model.__add_range("dynamic", x.erase(), 1.0, 2.0);
    assert!(
        model.typed_constraint_handle::<ScalarFunctionData<Affine>, Interval>("static").is_some()
    );
    let dynamic =
        model.typed_constraint_handle::<AnyScalarFunctionData, Interval>("dynamic").unwrap();
    model.set_constraint_function(dynamic, x.square().erase().into_function()).unwrap();
    let updated = model.semantic_constraint(dynamic).unwrap().unwrap();
    assert_eq!(
        updated.pair.function.downcast_ref::<AnyScalarFunctionData>().unwrap().class,
        oximo_expr::ExprClass::Quadratic
    );
}

#[test]
fn indexed_interval_provenance_survives_mixed_registration_and_fork_edits() {
    rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap().install(|| {
        for cone_first in [false, true] {
            let model = Model::new("indexed interval provenance");
            variable!(model, x);
            if cone_first {
                model.add_soc_constraint("cone", [x], x + 4.0);
            } else {
                constraint!(model, prior, x >= 0.0);
            }

            model.__add_range_constraints_over("bands", &Set::range(0..1024), |i: usize| {
                (x + 1.0, 0.0, f64::from(u32::try_from(i).unwrap()) + 2.0)
            });
            let fork = model.fork();

            for i in [0, 511, 1023] {
                let name = format!("bands[{i}]");
                let source = model.constraint_handle(&name).unwrap();
                let copied = fork.constraint_handle(&name).unwrap();
                for (owner, handle) in [(&model, source), (&fork, copied)] {
                    let record = owner.semantic_constraint(handle).unwrap().unwrap();
                    assert!(matches!(
                        record.origin,
                        ConstraintOrigin::Range(ref range) if range.as_str() == name
                    ));
                    assert_eq!(record.pair.function.model_id(), owner.id());
                    assert!(record.pair.function.signature().is::<ScalarFunctionData<Affine>>());
                    assert!(record.pair.set.signature().is::<Interval>());
                }

                fork.set_constraint_origin(copied.id(), ConstraintOrigin::Declared);
                assert!(matches!(
                    model.semantic_constraint(source).unwrap().unwrap().origin,
                    ConstraintOrigin::Range(ref range) if range.as_str() == name
                ));
                assert!(matches!(
                    fork.semantic_constraint(copied).unwrap().unwrap().origin,
                    ConstraintOrigin::Declared
                ));
            }
        }
    });
}

#[test]
fn homogeneous_declarations_keep_identity_when_one_signature_or_origin_changes() {
    let model = Model::new("homogeneous declarations");
    variable!(model, x);
    let rows: Vec<_> = (0..64)
        .map(|i| {
            model.add_constraint(
                format!("row{i}"),
                Constraint::new(x.erase().into_function(), LessThan(f64::from(i))),
            )
        })
        .collect();
    let fork = model.fork();

    model.set_constraint_function(rows[31], x.square().erase().into_function()).unwrap();
    let cone = model.add_soc_constraint("mixed registry", [x], x);
    model.set_constraint_origin(rows[63].id(), ConstraintOrigin::Range("edited origin".into()));
    assert_eq!(model.semantic_constraint(cone).unwrap().unwrap().id, cone.id());

    for (i, row) in rows.iter().copied().enumerate() {
        let source = model.semantic_constraint(row).unwrap().unwrap();
        let previous = fork
            .semantic_constraint(fork.semantic_constraint_handle(row.id()).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(source.id, previous.id);
        assert_eq!(source.name, previous.name);
        assert_eq!(
            source.pair.set.downcast_ref::<LessThan>().unwrap().0,
            f64::from(u32::try_from(i).unwrap())
        );
        let function = source.pair.function.downcast_ref::<AnyScalarFunctionData>().unwrap();
        assert_eq!(
            function.class,
            if i == 31 { oximo_expr::ExprClass::Quadratic } else { oximo_expr::ExprClass::Linear }
        );
        assert_eq!(
            previous.pair.function.downcast_ref::<AnyScalarFunctionData>().unwrap().class,
            oximo_expr::ExprClass::Linear
        );
        assert!(matches!(previous.origin, ConstraintOrigin::Declared));
        assert_eq!(matches!(source.origin, ConstraintOrigin::Range(_)), i == 63);
    }
    let declarations = model.semantic_constraints_ref();
    assert_eq!(
        declarations.ids().take(64).collect::<Vec<_>>(),
        rows.iter().map(|h| h.id()).collect::<Vec<_>>()
    );
    assert_eq!(declarations.ids().next_back(), Some(cone.id()));
    assert_eq!(fork.num_constraints(), 64);
}

#[test]
fn custom_lowering_cannot_hide_invalid_source_bounds() {
    struct IgnoresSet<'a>(Expr<'a, Affine>);
    impl Function for IgnoresSet<'_> {
        type Owned = CustomData;
        fn semantic_data(&self) -> CustomData {
            CustomData { model: self.0.model_id(), root: self.0.id() }
        }
    }
    impl FunctionInSet<LessThan> for IgnoresSet<'_> {}
    impl<'a> LowerConstraint<LessThan> for IgnoresSet<'a> {
        type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
        fn lower(self, _: LessThan) -> Self::Ir {
            self.0.le(0.0).into_ir()
        }
    }

    let model = Model::new("source bound validation");
    variable!(model, x);
    for value in [f64::NAN, f64::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Constraint::new(IgnoresSet(x), LessThan(value)).into_ir();
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.add_constraints([(
                    "invalid".into(),
                    Constraint::new(IgnoresSet(x), LessThan(value)),
                )]);
            }))
            .is_err()
        );
        assert_eq!(model.num_constraints(), 0);
        assert!(model.constraint_handle("invalid").is_none());
    }
}

#[test]
fn indicator_edits_can_change_nested_representations_without_losing_sources() {
    use oximo_core::{FunctionRepresentationRef, IndicatorFunctionData, SetValue};

    #[derive(Clone, Debug)]
    struct NamedAffine(CustomData);
    impl SemanticFunction for NamedAffine {
        fn model_id(&self) -> ModelId {
            self.0.model
        }
        fn rebind_model(&mut self, model_id: ModelId) {
            self.0.model = model_id;
        }
        fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
            visitor(self.0.root);
        }
        fn representation(&self) -> FunctionRepresentationRef<'_> {
            FunctionRepresentationRef::Scalar(self.0.root)
        }
        fn remap(&mut self, remap: ExprIdRemap) {
            self.0.root = remap.apply(self.0.root);
        }
    }

    let model = Model::new("nested source promotion");
    variable!(model, x);
    variable!(model, b, Bin);
    let row = model.add_indicator_constraint("row", b, true, x.le(2.0));
    model
        .set_constraint_set(
            row,
            oximo_core::IndicatorSet {
                active_value: false,
                consequent: SetValue::new(GreaterThan(5.0)),
            },
        )
        .unwrap();
    model
        .set_constraint_function(
            row,
            IndicatorFunctionData {
                model_id: model.id(),
                trigger: b.var_id().unwrap(),
                consequent: FunctionValue::new(NamedAffine(CustomData {
                    model: model.id(),
                    root: x.id(),
                })),
            },
        )
        .unwrap();
    let source = model.semantic_constraint(row).unwrap().unwrap();
    let function = source.pair.function.downcast_ref::<IndicatorFunctionData>().unwrap();
    assert!(function.consequent.downcast_ref::<NamedAffine>().is_some());
    assert_eq!(model.indicator_constraints()[row.index()].lower, 5.0);
    assert!(!model.indicator_constraints()[row.index()].active_value);

    let fork = model.fork();
    let record = fork.semantic_constraints().remove(0);
    let function = record.pair.function.downcast_ref::<IndicatorFunctionData>().unwrap();
    assert_eq!(function.consequent.model_id(), fork.id());
    assert_eq!(function.consequent.downcast_ref::<NamedAffine>().unwrap().0.root, x.id());
}

#[derive(Clone, Copy)]
struct ScaledLowering<'a>(Expr<'a, Affine>);

#[derive(Clone, Debug)]
struct OptionalScalarView {
    model: ModelId,
    root: ExprId,
    available: bool,
}

impl SemanticFunction for OptionalScalarView {
    fn model_id(&self) -> ModelId {
        self.model
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model = model_id;
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.root);
    }

    fn representation(&self) -> oximo_core::FunctionRepresentationRef<'_> {
        if self.available {
            oximo_core::FunctionRepresentationRef::Scalar(self.root)
        } else {
            oximo_core::FunctionRepresentationRef::Opaque
        }
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.root = remap.apply(self.root);
        // An extension may invalidate an optional cached algebraic view on remap.
        self.available = false;
    }
}

struct OptionalScalarLowering<'a>(Expr<'a, Affine>);

impl Function for OptionalScalarLowering<'_> {
    type Owned = OptionalScalarView;

    fn semantic_data(&self) -> Self::Owned {
        OptionalScalarView { model: self.0.model_id(), root: self.0.id(), available: true }
    }
}

impl FunctionInSet<LessThan> for OptionalScalarLowering<'_> {}

impl<'a> LowerConstraint<LessThan> for OptionalScalarLowering<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;

    fn lower(self, set: LessThan) -> Self::Ir {
        self.0.le(set.0).into_ir()
    }
}

#[test]
fn batch_view_detection_inspects_extensions_after_worker_remapping() {
    for threads in [1, 6] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let model = Model::new("remapped optional views");
            variable!(model, x);
            model.__add_constraints_over("rows", &Set::range(0..1024), |i: usize| {
                Constraint::new(
                    OptionalScalarLowering(x + f64::from(u32::try_from(i).unwrap())),
                    LessThan(2.0),
                )
            });
            let handle = model
                .typed_constraint_handle::<OptionalScalarView, LessThan>("rows[1023]")
                .unwrap();
            let updated = model.set_constraint_set(handle, LessThan(8.0));
            if threads == 1 {
                updated.unwrap();
                assert_eq!(model.constraints().algebraic()[handle.index()].upper, 8.0);
            } else {
                assert!(matches!(
                    updated,
                    Err(oximo_core::ConstraintUpdateError::AlternativeLowering(_))
                ));
                assert_eq!(model.constraints().algebraic()[handle.index()].upper, 2.0);
            }
        });
    }
}

impl Function for ScaledLowering<'_> {
    type Owned = ScalarFunctionData<Affine>;

    fn semantic_data(&self) -> Self::Owned {
        ScalarFunctionData::from_expression(self.0)
    }
}

impl FunctionInSet<LessThan> for ScaledLowering<'_> {}

impl<'a> LowerConstraint<LessThan> for ScaledLowering<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;

    fn lower(self, set: LessThan) -> Self::Ir {
        (2.0 * self.0).le(2.0 * set.0).into_ir()
    }
}

#[test]
fn scalar_ir_conversions_preserve_sources_and_reject_unmapped_edits() {
    let model = Model::new("scalar conversion provenance");
    variable!(model, x);
    let first: oximo_core::function_set::AlgebraicConstraintIr =
        Constraint::new(ScaledLowering(x), LessThan(2.0)).into_ir();
    let explicit = Constraint::new(ScaledLowering(x), LessThan(2.0)).into_ir();

    for (name, ir) in [("first", first), ("explicit", explicit)] {
        let handle = model.add_lowered_constraint(name, ir);
        let record = model.semantic_constraint(handle).unwrap().unwrap();
        assert_eq!(record.pair.function.expressions(), [x.id()]);
        assert_eq!(record.pair.set.scalar_bounds().unwrap().1, 2.0);
        assert_eq!(model.constraints().algebraic()[handle.index()].upper, 4.0);

        let typed = model
            .try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(handle)
            .unwrap()
            .unwrap();
        assert!(matches!(
            model.set_constraint_set(typed, LessThan(3.0)),
            Err(oximo_core::ConstraintUpdateError::AlternativeLowering(_))
        ));
        assert!(matches!(
            model.set_constraint_function(typed, (3.0 * x).into_function()),
            Err(oximo_core::ConstraintUpdateError::AlternativeLowering(_))
        ));
        model.set_constraint_active(handle, false).unwrap();
        model.set_constraint_active(handle, true).unwrap();
        assert_eq!(model.constraints().algebraic()[handle.index()].upper, 4.0);
    }
}

#[test]
fn scalar_views_preserve_source_coordinates_and_inactive_bounds() {
    let model = Model::new("scalar inspection");
    variable!(model, x);
    let direct = constraint!(model, direct, x <= 2.0);
    model.add_constraints([(
        "scaled".into(),
        Constraint::new(ScaledLowering(x), LessThan(2.0)).into_ir(),
    )]);
    let scaled = model.constraint_handle("scaled").unwrap();

    for active in [true, false, true] {
        for handle in [direct.erase(), scaled] {
            model.set_constraint_active(handle, active).unwrap();
            let view = model.scalar_constraint_view(handle).unwrap().unwrap();
            assert_eq!(view.lhs, x.id());
            assert_eq!((view.lower, view.upper), (f64::NEG_INFINITY, 2.0));
        }
    }

    let foreign = Model::new("foreign scalar inspection");
    assert!(foreign.scalar_constraint_view(direct).is_err());
}

#[test]
fn alternative_scalar_lowerings_reject_edits_and_restore_their_exact_view() {
    for rows in [1, 1024] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let model = Model::new("alternative lowering");
            variable!(model, x);
            if rows == 1 {
                model.add_constraints([(
                    "scaled[0]".into(),
                    Constraint::new(ScaledLowering(x), LessThan(2.0)).into_ir(),
                )]);
            } else {
                model.__add_constraints_over("scaled", &Set::range(0..rows), |_: usize| {
                    Constraint::new(ScaledLowering(x), LessThan(2.0)).into_ir()
                });
            }

            for index in [0, rows - 1] {
                let handle = model
                    .typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>(&format!(
                        "scaled[{index}]"
                    ))
                    .unwrap();
                let before = model.constraints().algebraic()[handle.index()].clone();
                let semantic = model.semantic_constraint(handle).unwrap().unwrap();
                assert_eq!(semantic.pair.function.expressions(), [x.id()]);
                assert_eq!(semantic.pair.set.scalar_bounds().unwrap().1, 2.0);
                assert_eq!(before.upper, 4.0);

                assert!(matches!(
                    model.set_constraint_set(handle, LessThan(3.0)),
                    Err(oximo_core::ConstraintUpdateError::AlternativeLowering(_))
                ));
                assert!(matches!(
                    model.set_constraint_function(handle, (3.0 * x).into_function()),
                    Err(oximo_core::ConstraintUpdateError::AlternativeLowering(_))
                ));

                model.set_constraint_active(handle, false).unwrap();
                model.set_constraint_active(handle, true).unwrap();
                let restored = model.constraints().algebraic()[handle.index()].clone();
                assert_eq!(restored.lhs, before.lhs);
                assert_eq!((restored.lower, restored.upper), (before.lower, before.upper));
                assert_eq!(
                    model
                        .semantic_constraint(handle)
                        .unwrap()
                        .unwrap()
                        .pair
                        .set
                        .scalar_bounds()
                        .unwrap()
                        .1,
                    2.0
                );

                let fork = model.fork();
                let fork_handle = fork
                    .typed_constraint_handle::<ScalarFunctionData<Affine>, LessThan>(&format!(
                        "scaled[{index}]"
                    ))
                    .unwrap();
                assert!(fork.set_constraint_set(fork_handle, LessThan(3.0)).is_err());
            }
        });
    }
}

#[test]
fn alternatively_lowered_indicator_consequents_reject_typed_edits() {
    let model = Model::new("alternative indicator consequent");
    variable!(model, x);
    variable!(model, enabled, Binary);
    let lowered: oximo_core::function_set::AlgebraicConstraintIr =
        Constraint::new(ScaledLowering(x), LessThan(2.0)).into_ir();
    let guard = model.add_indicator_constraint("guard", enabled, true, lowered.into_affine());

    assert!(matches!(
        model.set_constraint_set(
            guard,
            oximo_core::IndicatorSet {
                active_value: true,
                consequent: oximo_core::SetValue::new(LessThan(8.0)),
            }
        ),
        Err(oximo_core::ConstraintUpdateError::AlternativeLowering(_))
    ));
    assert!(
        model
            .set_constraint_function(
                guard,
                oximo_core::IndicatorFunctionData {
                    model_id: model.id(),
                    trigger: enabled.var_id().unwrap(),
                    consequent: FunctionValue::new(ScalarFunctionData::from_expression(x)),
                }
            )
            .is_err()
    );

    model.set_constraint_active(guard, false).unwrap();
    model.set_constraint_active(guard, true).unwrap();
    assert_eq!(model.indicator_constraints()[guard.index()].upper, 4.0);
    let record = model.semantic_constraint(guard).unwrap().unwrap();
    let set = record.pair.set.downcast_ref::<oximo_core::IndicatorSet>().unwrap();
    assert_eq!(set.consequent.scalar_bounds().unwrap().1, 2.0);
}
