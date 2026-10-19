use oximo_core::prelude::*;

#[test]
fn bare_parameter_bounds_are_symbolic_and_every_write_path_checks_ordering() {
    let m = Model::new("symbolic bounds");

    param!(m, lo = 1.0);
    param!(m, hi = 4.0);
    variable!(m,lo<=x<=hi);
    assert_eq!(m.variables()[0].lb, Bound::Parameter(lo.param_id().unwrap()));
    assert!(m.set_param(lo, 5.0).is_err());
    assert!(lo.set_param_value(5.0).is_err());
    assert_eq!(m.param_value(lo.param_id().unwrap()), 1.0);

    m.set_params(&[(lo, 6.0), (hi, 8.0)]).unwrap();
    assert_eq!(m.variables()[0].resolved_bounds(&m.arena()), (6.0, 8.0));
    assert!(m.set_params(&[(lo, 2.0), (hi, f64::NAN)]).is_err());
    assert_eq!(m.param_value(lo.param_id().unwrap()), 6.0);

    m.fix(x, 7.0).unwrap();
    lo.set_param_value(99.0).unwrap();
    m.unfix_var(x.var_id().unwrap(), 0.0, hi).unwrap();
    assert!(hi.set_param_value(-1.0).is_err());
}

#[test]
fn indexed_and_parallel_bound_inputs_retain_parameters() {
    let m = Model::new("indexed bounds");

    param!(m,cap[i in 0..600]=10.0);
    variable!(m,0.0<=x[i in 0..600]<=cap[i]);
    m.set_param_idx(&cap, 399, 20.0).unwrap();

    assert_eq!(m.variables()[399].ub, Bound::Parameter(cap[399].param_id().unwrap()));
    assert_eq!(m.resolved_variables()[399].ub, 20.0);
    assert_eq!(x.len(), 600);
}

#[test]
fn declaration_checks_identity_composition_domains_and_nonfinite_data() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let m = Model::new("bounds");
    let other = Model::new("foreign");
    param!(m, p = 2.0);
    param!(other, q = 3.0);

    assert!(catch_unwind(AssertUnwindSafe(|| m.__var("foreign").ub(q))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| m.__var("composed").ub(2.0 * p + 1.0))).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.__var("binary").ub(p).domain(Domain::Binary).build()))
            .is_err()
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(m.set_param(p, bad).is_err());
    }
    assert!(m.set_param_id(ParamId(99), 1.0).is_err());
    assert!(
        m.set_param_ids(&[(p.param_id().unwrap(), 1.0), (p.param_id().unwrap(), 3.0)]).is_err()
    );
}

#[test]
fn reformulation_locks_bound_parameters_without_locking_the_source_copy() {
    let source = Model::new("SOS bound locks");
    param!(source, cap = 5.0);
    variable!(source,0.0<=x<=cap);
    variable!(source,0.0<=y<=cap);
    sos_constraint!(source, choice, SOS1, [x, y]);

    let transformed = source.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    assert!(transformed.set_param_id(cap.param_id().unwrap(), 6.0).is_err());

    source.set_param(cap, 6.0).unwrap();
    source.reformulate_sos(SosReformulationOptions::default()).unwrap();
    assert!(cap.set_param_value(7.0).is_err());
}

#[test]
fn expression_writes_invalidate_parameter_specific_soc_classification() {
    let model = Model::new("bound-dependent cone recognition");

    param!(model, floor = 0.0);
    variable!(model, x);
    variable!(model, t >= floor);
    constraint!(model, x.square() - t.square() <= 0.0);
    objective!(model, Min, t);
    assert_eq!(model.kind(), ModelKind::SOCP);

    floor.set_param_value(-1.0).unwrap();
    assert_eq!(model.kind(), ModelKind::QCP);

    floor.set_param_value(0.0).unwrap();
    assert_eq!(model.kind(), ModelKind::SOCP);
}
