use super::classify::PAR_KIND_THRESHOLD;
use super::indexed::{format_index_name, format_index_name_small};
use super::{ConstraintRef, Model, ModelKind};
use crate::constraint::RangeConstraintIds;
use crate::constraint::{ConstraintId, RangeConstraintHandles, Relate};
use crate::set::{IndexKey, Set};
use crate::sos::SosType;
use oximo_expr::{ExprClass, ModelMismatchError, classify, extract_linear};

use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn metadata_batches_preserve_names_ids_and_atomicity_on_fresh_and_populated_forks() {
    use super::variables::{PendingParam, PendingVar};
    use crate::Domain;

    for populated in [false, true] {
        let source = Model::new("metadata");
        if populated {
            source.__var("existing_x").build();
            source.__param("existing_p", 5.0);
        }
        let model = source.fork();
        let first = usize::from(populated);
        let vars = [
            PendingVar { name: "x".into(), lb: -1.0, ub: 4.0 },
            PendingVar { name: "y".into(), lb: 0.0, ub: 8.0 },
        ];
        let params = [
            PendingParam { name: "p".into(), value: 11.0 },
            PendingParam { name: "q".into(), value: 13.0 },
        ];
        let handles = model.register_vars_batch(&vars, Domain::Integer);
        let parameters = model.register_params_batch(&params);
        for (offset, (variable, parameter)) in handles.iter().zip(&parameters).enumerate() {
            assert_eq!(variable.var_id().unwrap().index(), first + offset);
            assert_eq!(parameter.param_id().unwrap().index(), first + offset);
            assert_eq!(model.variable_id(&vars[offset].name), variable.var_id());
            assert_eq!(model.parameter_id(&params[offset].name), parameter.param_id());
            assert_eq!(model.param_value_of(*parameter).unwrap(), Some(params[offset].value));
            assert_eq!(model.variables()[first + offset].domain, Domain::Integer);
        }
        assert_eq!(source.num_variables(), first);
        assert!(source.variable_id("x").is_none());
        assert!(source.parameter_id("p").is_none());

        let before = model.arena().len();
        assert!(
            catch_unwind(AssertUnwindSafe(|| model.register_vars_batch(&vars, Domain::Real)))
                .is_err()
        );
        assert!(catch_unwind(AssertUnwindSafe(|| model.register_params_batch(&params))).is_err());
        assert_eq!(model.num_variables(), first + 2);
        assert_eq!(model.arena().len(), before);
    }

    let model = Model::new("duplicate_batch");
    let vars = [
        PendingVar { name: "duplicate".into(), lb: 0.0, ub: 1.0 },
        PendingVar { name: "duplicate".into(), lb: 0.0, ub: 1.0 },
    ];
    let params = [
        PendingParam { name: "duplicate".into(), value: 1.0 },
        PendingParam { name: "duplicate".into(), value: 2.0 },
    ];
    assert!(
        catch_unwind(AssertUnwindSafe(|| model.register_vars_batch(&vars, Domain::Real))).is_err()
    );
    assert!(catch_unwind(AssertUnwindSafe(|| model.register_params_batch(&params))).is_err());
    assert_eq!(model.num_variables(), 0);
    assert_eq!(model.arena().len(), 0);
    assert!(model.variable_id("duplicate").is_none());
    assert!(model.parameter_id("duplicate").is_none());
}

#[test]
fn inline_indexed_variable_names_match_existing_rendering() {
    let keys = [
        IndexKey::Int(-123),
        IndexKey::Str("áβ🙂".into()),
        IndexKey::Tuple(vec![IndexKey::Int(7), IndexKey::Str("a,b".into())].into()),
    ];
    for base in ["x", "αβ", "very_long_indexed_variable_name_beyond_inline_capacity"] {
        for key in &keys {
            assert_eq!(format_index_name_small(base, key).as_str(), format_index_name(base, key));
        }
    }
}

#[test]
fn model_kind_display_uses_standard_ascii_acronyms() {
    let kinds = [
        ModelKind::LP,
        ModelKind::MILP,
        ModelKind::QP,
        ModelKind::MIQP,
        ModelKind::QCP,
        ModelKind::MIQCP,
        ModelKind::SOCP,
        ModelKind::MISOCP,
        ModelKind::NLP,
        ModelKind::MINLP,
    ];
    let labels: Vec<_> = kinds.into_iter().map(|kind| kind.to_string()).collect();
    assert_eq!(
        labels,
        ["LP", "MILP", "QP", "MIQP", "QCP", "MIQCP", "SOCP", "MISOCP", "NLP", "MINLP"]
    );
    assert!(labels.iter().all(|label| label.is_ascii()));
}

#[test]
fn param_times_var_keeps_model_linear() {
    let m = Model::new("p");
    let param = m.__param("param", 4.0);
    let x = m.__var("x").lb(0.0).build();
    m.__minimize(param * x);
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn param_coeff_resolves_and_rebinds() {
    let m = Model::new("p");
    let param = m.__param("param", 4.0);
    let x = m.__var("x").lb(0.0).build();
    let obj = param * x;

    let coeff = |m: &Model| {
        let arena = m.arena();
        extract_linear(&arena, obj.id).expect("linear").coeffs[0].1
    };
    assert!((coeff(&m) - 4.0).abs() < f64::EPSILON);

    m.set_param(param, 9.0).unwrap();
    assert!((coeff(&m) - 9.0).abs() < f64::EPSILON);
    assert_eq!(m.parameter_id("param"), Some(param.param_id().unwrap()));
}

#[test]
fn param_value_reads_live_arena_value() {
    let m = Model::new("p");
    let param = m.__param("param", 4.0);
    let id = param.param_id().unwrap();
    assert!((m.param_value(id) - 4.0).abs() < f64::EPSILON);
    assert!((m.param_value_of(param).unwrap().unwrap() - 4.0).abs() < f64::EPSILON);

    m.set_param(param, 7.5).unwrap();
    assert!((m.param_value(id) - 7.5).abs() < f64::EPSILON);

    let x = m.__var("x").build();
    assert!(m.param_value_of(x).unwrap().is_none());
}

#[test]
fn handle_mutations_reject_foreign_models_without_changing_colliding_slots() {
    let source = Model::new("source");
    let target = Model::new("target");
    let source_x = source.__var("x").build();
    let target_x = target.__var("x").lb(-1.0).ub(1.0).build();
    let source_p = source.__param("p", 7.0);
    let target_p = target.__param("p", 3.0);

    let expected = ModelMismatchError::new(target.id(), source.id());
    assert_eq!(target.fix(source_x, 0.5), Err(expected));
    assert_eq!(target.set_initial(source_x, 0.5), Err(expected));
    assert_eq!(target.set_param(source_p, 9.0), Err(expected));

    let variable = &target.variables()[target_x.var_id().unwrap().index()];
    assert_eq!((variable.lb, variable.ub, variable.initial), (-1.0, 1.0, None));
    assert_eq!(target.param_value_of(target_p).unwrap(), Some(3.0));
    assert_ne!(source.id(), target.id());
    assert_eq!(source_x.model_id(), source.id());
}

#[test]
fn arena_snapshot_is_not_live_after_parameter_rebind() {
    let m = Model::new("snapshot");
    let param = m.__param("p", 1.0);
    let id = param.param_id().unwrap();
    let snapshot = m.arena();
    m.set_param(param, 2.0).unwrap();
    assert!((snapshot.param_value(id) - 1.0).abs() < f64::EPSILON);
    assert!((m.param_value(id) - 2.0).abs() < f64::EPSILON);
}

#[test]
#[should_panic(expected = "different model")]
fn indexed_constraints_reject_foreign_expression_arenas() {
    let target = Model::new("target");
    let foreign = Model::new("foreign");
    let x = foreign.__var("x").build();
    let keys = Set::range(0..1024);
    target.add_constraints_over_with("c", &keys, &|_| x.le(1.0), Some(true));
}

#[test]
fn set_param_invalidates_kind_cache() {
    let m = Model::new("p");
    let p = m.__param("p", 1.0);
    let x = m.__var("x").lb(0.0).build();
    m.__add_constraint("c", (p * x).le(10.0));
    assert_eq!(m.kind(), ModelKind::LP);
    m.set_param(p, 2.0).unwrap();
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn unified_constraints_include_inactive_entries() {
    let m = Model::new("inactive");
    let x = m.__var("x").build();
    let t = m.__var("t").lb(0.0).build();
    m.__add_constraint("row", x.le(1.0));
    m.add_soc_constraint("cone", [x], t);
    m.add_sos_constraint("sos", SosType::Sos1, [(x, 1.0)]);
    m.constraints.borrow_mut()[0].active = false;
    m.soc_constraints.borrow_mut()[0].active = false;
    m.sos_constraints.borrow_mut()[0].active = false;

    let constraints = m.constraints();
    assert_eq!(constraints.len(), 3);
    assert!(constraints.iter().all(|entry| match entry {
        ConstraintRef::PositiveSemidefinite { constraint, .. } => !constraint.active,
        ConstraintRef::Algebraic { constraint, .. } => !constraint.active,
        ConstraintRef::SecondOrderCone { constraint, .. } => !constraint.active,
        ConstraintRef::SpecialOrderedSet { constraint, .. } => !constraint.active,
        ConstraintRef::Indicator { constraint, .. } => !constraint.active,
    }));
}

#[test]
fn uncached_kind_inference_leaves_kind_cache_empty() {
    let m = Model::new("uncached_kind");
    let x = m.__var("x").build();
    m.__minimize(x);
    for _ in 0..PAR_KIND_THRESHOLD {
        m.__add_constraint_auto(x.powi(2).le(1.0));
    }

    assert_eq!(m.infer_kind(false), ModelKind::QCP);
    assert_eq!(m.cached_kind.get(), None);
    assert_eq!(m.infer_kind(true), ModelKind::QCP);
    assert_eq!(m.cached_kind.get(), None);

    assert_eq!(m.kind(), ModelKind::QCP);
    assert_eq!(m.cached_kind.get(), Some(ModelKind::QCP));
}

#[test]
#[should_panic(expected = "parameter name \"dup\" is already registered")]
fn duplicate_param_name_panics() {
    let m = Model::new("p");
    let _a = m.__param("dup", 1.0);
    let _b = m.__param("dup", 2.0);
}

#[test]
fn indexed_param_dense_value_and_per_key_rebind() {
    let m = Model::new("ip");
    let items = Set::range(0..3);
    let data = [10.0, 20.0, 30.0];
    let cost = m.__indexed_param("cost", &items, |i: usize| data[i]);

    assert!(cost.is_dense());
    assert_eq!(cost.len(), 3);
    assert_eq!(m.num_parameters(), 3);
    assert!(m.parameter_id("cost[0]").is_some());
    assert!(m.parameter_id("cost[2]").is_some());
    assert!((m.param_value_idx(&cost, 1usize).unwrap().unwrap() - 20.0).abs() < f64::EPSILON);

    let x = m.__var("x").lb(0.0).build();
    let obj = cost.at([1]) * x;
    let coeff = |m: &Model| {
        let arena = m.arena();
        extract_linear(&arena, obj.id).expect("linear").coeffs[0].1
    };
    assert!((coeff(&m) - 20.0).abs() < f64::EPSILON);

    m.set_param_idx(&cost, 1usize, 99.0).unwrap();
    assert!((coeff(&m) - 99.0).abs() < f64::EPSILON);
    assert!((m.param_value_idx(&cost, 1usize).unwrap().unwrap() - 99.0).abs() < f64::EPSILON);
    assert!((m.param_value_idx(&cost, 0usize).unwrap().unwrap() - 10.0).abs() < f64::EPSILON);
}

#[test]
fn set_param_idx_rejects_foreign_family() {
    let a = Model::new("a");
    let b = Model::new("b");
    let items = Set::range(0..2);
    let pa = a.__indexed_param("p", &items, |_i: usize| 1.0);
    let pb = b.__indexed_param("p", &items, |_i: usize| 2.0);
    assert_eq!(b.set_param_idx(&pa, 0usize, 5.0), Err(ModelMismatchError::new(b.id(), a.id())));
    assert_eq!(b.param_value_idx(&pb, 0usize).unwrap(), Some(2.0));
}

#[test]
fn indexed_param_sparse_string_keyed() {
    let m = Model::new("ips");
    let plants = Set::strings(["a", "b"]);
    let price = m.__indexed_param("price", &plants, |p: String| if p == "a" { 1.5 } else { 2.5 });
    assert!(!price.is_dense());
    assert_eq!(price.len(), 2);
    assert!((m.param_value_idx(&price, "a").unwrap().unwrap() - 1.5).abs() < f64::EPSILON);
    assert!((m.param_value_idx(&price, "b").unwrap().unwrap() - 2.5).abs() < f64::EPSILON);
    assert!(m.param_value_idx(&price, "z").unwrap().is_none());
}

#[test]
fn kind_forced_serial_and_parallel_classification_agree() {
    let qcp = Model::new("qcp");
    let x = qcp.__var("x").build();
    let y = qcp.__var("y").build();
    qcp.__minimize(x + y);
    for i in 0..PAR_KIND_THRESHOLD + 3 {
        qcp.__add_constraint_auto((x.powi(2) + y).le(i as f64 + 1.0));
    }
    assert_eq!(qcp.infer_kind(false), ModelKind::QCP);
    assert_eq!(qcp.infer_kind(false), qcp.infer_kind(true));
    assert_eq!(qcp.infer_kind(false), qcp.infer_kind_with(None));

    let socp = Model::new("socp");
    let x = socp.__var("x").build();
    let t = socp.__var("t").lb(0.0).build();
    socp.__minimize(t);
    for _ in 0..PAR_KIND_THRESHOLD + 3 {
        socp.__add_constraint_auto((x.powi(2) - t.powi(2)).le(0.0));
    }
    assert_eq!(socp.infer_kind(false), ModelKind::SOCP);
    assert_eq!(socp.infer_kind(false), socp.infer_kind(true));
    assert_eq!(socp.infer_kind(false), socp.infer_kind_with(None));

    let nlp = Model::new("nlp");
    let x = nlp.__var("x").build();
    let y = nlp.__var("y").build();
    let z = nlp.__var("z").build();
    nlp.__minimize(x + y + z);
    for _ in 0..PAR_KIND_THRESHOLD + 3 {
        nlp.__add_constraint_auto((x * y * z).le(1.0));
    }
    assert_eq!(nlp.infer_kind(false), ModelKind::NLP);
    assert_eq!(nlp.infer_kind(false), nlp.infer_kind(true));
}

#[test]
fn arena_snapshot_allows_nested_model_reads() {
    let model = Model::new("nested_reads");
    let x = model.__var("x").build();
    model.__minimize(x);

    let arena = model.arena();
    assert_eq!(model.kind(), ModelKind::LP);
    assert!(matches!(arena.get(x.id), oximo_expr::ExprNode::Var(_)));
}

#[test]
fn indexed_constraint_handles_match_in_serial_and_parallel() {
    #[derive(Debug, PartialEq, Eq)]
    struct Handles {
        ordinary: Vec<(usize, ConstraintId)>,
        ranges: Vec<(usize, RangeConstraintIds)>,
    }

    fn build(parallel: bool, keys: &Set<usize>) -> Handles {
        let model = Model::new("family_id_parity");
        let x = model.__var("x").build();
        model.__add_constraint("prior", x.ge(0.0));
        let ordinary = model.add_constraints_over_with("c", keys, &|_| x.le(5.0), Some(parallel));
        let ranges = model.add_range_constraints_over_with(
            "r",
            keys,
            &|i| (if i % 2 == 0 { (x + 1.0).erase() } else { x.powi(2) }, 0.0, 10.0),
            Some(parallel),
        );
        for (key, id) in ordinary.iter() {
            assert_eq!(ordinary.get(key), Some(id));
            assert_eq!(model.constraint_handle(&format!("c[{key}]")), Some(id.erase()));
        }
        for (key, ids) in ranges.iter() {
            assert_eq!(ranges.get(key), Some(ids));
            match ids {
                RangeConstraintHandles::Interval(id) => {
                    assert_eq!(model.constraint_handle(&format!("r[{key}]")), Some(id.erase()));
                    let row = &model.constraints.borrow()[id.index()];
                    assert_eq!((row.lower, row.upper), (0.0, 10.0));
                    assert_eq!(classify(&model.arena(), row.lhs), ExprClass::Linear);
                }
                RangeConstraintHandles::Split { lower, upper } => {
                    assert_eq!(model.constraint_handle(&format!("r[{key}]_lo")), Some(lower));
                    assert_eq!(model.constraint_handle(&format!("r[{key}]_hi")), Some(upper));
                    let rows = model.constraints.borrow();
                    assert_eq!(rows[lower.index()].lower.to_bits(), 0.0_f64.to_bits());
                    assert_eq!(rows[upper.index()].upper.to_bits(), 10.0_f64.to_bits());
                    assert_eq!(
                        classify(&model.arena(), rows[lower.index()].lhs),
                        ExprClass::Quadratic
                    );
                }
            }
        }
        Handles {
            ordinary: ordinary.iter().map(|(key, handle)| (key, handle.row_id())).collect(),
            ranges: ranges.iter().map(|(key, handles)| (key, handles.ids())).collect(),
        }
    }

    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| {
        for keys in [Set::range(3..1030), Set::from_ints([8, 1, 6, 3, 4])] {
            assert_eq!(build(false, &keys), build(true, &keys));
        }
    });
}

#[test]
fn indexed_build_forced_parallel_matches_serial_for_every_family() {
    #[derive(Debug, PartialEq)]
    struct Digest {
        variables: Vec<String>,
        parameters: Vec<(String, f64)>,
        rows: Vec<String>,
        typed: Vec<String>,
        arena_len: usize,
    }

    fn digest(parallel: bool) -> Digest {
        let model = Model::new("indexed_parity");
        let keys = Set::range(0..128usize);
        let x = model
            .__indexed_var("x", &keys)
            .lb_by(|i: usize| -(i as f64))
            .ub_by(|i: usize| i as f64 + 10.0)
            .parallel_for_benchmark(parallel)
            .build();
        let y = model.__indexed_var("y", &keys).parallel_for_benchmark(parallel).build();
        let values = |i: usize| i as f64 + 0.5;
        let p = model.indexed_param_with("p".to_owned(), &keys, &values, Some(parallel));

        let algebraic = |i: usize| (2.0 * x[i] + p[i]).le(i as f64 + 20.0);
        model.add_constraints_over_with("c", &keys, &algebraic, Some(parallel));
        let ranges = |i: usize| (y[i] + 1.0, -(i as f64), i as f64 + 5.0);
        model.add_range_constraints_over_with("r", &keys, &ranges, Some(parallel));
        let symbolic_ranges = |i: usize| (x[i] + 2.0, -p[i], p[i] + 3.0);
        model.add_range_constraints_over_with("sr", &keys, &symbolic_ranges, Some(parallel));
        let t = model.__var("t").lb(0.0).build();
        let cones = |i: usize| ([x[i] + 1.0, y[i] - 1.0], t + p[i]);
        model.add_soc_constraints_over_with("q", &keys, &cones, Some(parallel));
        let explicit_sos = |i: usize| [(x[i], 1.0), (y[i], 2.0)];
        model.add_sos_constraints_over_with(
            "s",
            &keys,
            SosType::Sos1,
            &explicit_sos,
            Some(parallel),
        );
        let auto_sos = |i: usize| [x[i], y[i]];
        model.add_sos_constraints_over_auto_weights_with(
            "a",
            &keys,
            SosType::Sos2,
            &auto_sos,
            Some(parallel),
        );

        let variables = model
            .variables()
            .iter()
            .map(|variable| {
                format!(
                    "{}:{:?}:{}:{}:{}",
                    variable.name, variable.id, variable.lb, variable.ub, variable.domain
                )
            })
            .collect();
        let parameters = model
            .parameters()
            .iter()
            .map(|parameter| (parameter.name.to_string(), model.param_value(parameter.id)))
            .collect();
        let arena = model.arena();
        let constraints = model.constraints();
        let rows = constraints
            .algebraic()
            .iter()
            .map(|constraint| {
                let terms = extract_linear(&arena, constraint.lhs).unwrap();
                format!(
                    "{}:{:?}:{}:{}:{}",
                    constraint.name,
                    terms.coeffs,
                    terms.constant,
                    constraint.lower,
                    constraint.upper
                )
            })
            .collect();
        let mut typed = constraints
            .second_order_cones()
            .iter()
            .map(|constraint| {
                let terms: Vec<_> = constraint
                    .terms
                    .iter()
                    .map(|&term| extract_linear(&arena, term).unwrap().into_owned())
                    .collect();
                let bound = extract_linear(&arena, constraint.bound).unwrap().into_owned();
                format!("{}:{terms:?}:{bound:?}", constraint.name)
            })
            .collect::<Vec<_>>();
        typed.extend(
            constraints.special_ordered_sets().iter().map(|constraint| format!("{constraint:?}")),
        );
        let arena_len = arena.len();
        Digest { variables, parameters, rows, typed, arena_len }
    }

    assert_eq!(digest(false), digest(true));
}

#[test]
fn parallel_duplicate_name_failure_does_not_mutate_model_or_arena() {
    let model = Model::new("indexed_atomicity");
    let x = model.__var("x").build();
    let duplicate_keys = Set::from_ints([0usize, 0]);
    let arena_len = model.arena().len();
    let rule = |i: usize| (x + i as f64).le(1.0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        model.add_constraints_over_with("dup", &duplicate_keys, &rule, Some(true));
    }));
    assert!(result.is_err());
    assert_eq!(model.num_constraints(), 0);
    assert_eq!(model.arena().len(), arena_len);

    model.__add_constraint("after", x.le(2.0));
    assert_eq!(model.constraint_id("after"), Some(ConstraintId(0)));
}

#[test]
fn parallel_range_failure_does_not_mutate_model_or_arena() {
    for duplicate_name in [false, true] {
        let model = Model::new("range_batch_atomicity");
        let x = model.__var("x").build();
        let prior = model.__add_constraint("r[17]_hi", x.le(2.0));
        let arena_len = model.arena().len();
        let keys = Set::range(0..128usize);
        let rule = |i: usize| {
            let body = if i.is_multiple_of(2) { (x + 1.0).erase() } else { x.powi(2) };
            assert!(duplicate_name || i != 17, "deliberate range callback panic");
            (body, 0.0, 10.0)
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            model.add_range_constraints_over_with("r", &keys, &rule, Some(true));
        }));
        assert!(result.is_err());
        assert_eq!(model.num_constraints(), 1);
        assert_eq!(model.constraint_handle("r[17]_hi"), Some(prior.erase()));
        assert_eq!(model.arena().len(), arena_len);

        // The failed batch must also release the arena for subsequent builds.
        let family =
            model.add_range_constraints_over_with("after", &keys, &|_| (x, 0.0, 1.0), Some(true));
        assert_eq!(family.get(0).unwrap().ids(), RangeConstraintIds::Interval(ConstraintId(1)));
        assert_eq!(model.num_constraints(), 129);
    }
}

#[test]
fn parallel_callback_panic_does_not_mutate_model_or_arena() {
    let model = Model::new("indexed_callback_atomicity");
    let x = model.__var("x").build();
    let keys = Set::range(0..128usize);
    let arena_len = model.arena().len();
    let rule = |i: usize| {
        let constraint = (x + i as f64).le(1.0);
        assert_ne!(i, 17, "deliberate callback panic");
        constraint
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        model.add_constraints_over_with("panic", &keys, &rule, Some(true));
    }));
    assert!(result.is_err());
    assert_eq!(model.num_constraints(), 0);
    assert_eq!(model.arena().len(), arena_len);

    model.__add_constraint("after", x.le(2.0));
    assert_eq!(model.constraint_id("after"), Some(ConstraintId(0)));
}
