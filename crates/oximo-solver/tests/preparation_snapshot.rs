use oximo_core::prelude::*;
use oximo_solver::prepare::LoweringContext;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn concurrent_parameter_writes_keep_prepared_bounds_coefficients_and_kind_consistent() {
    let model = Model::new("consistent preparation");
    param!(model, floor = 0.0);
    variable!(model, x);
    variable!(model, t[i in 0..16] >= floor);
    constraint!(model, x.square() <= t[0].square());
    objective!(model, Min, floor * t[0]);
    let stop = AtomicBool::new(false);
    let start = Barrier::new(2);

    std::thread::scope(|scope| {
        let stop_ref = &stop;
        let start_ref = &start;
        let writer = scope.spawn(move || {
            start_ref.wait();
            while !stop_ref.load(Ordering::Relaxed) {
                floor.set_param_value(-1.0).unwrap();
                floor.set_param_value(0.0).unwrap();
            }
        });
        start.wait();
        let consistent = (0..512).all(|_| {
            let prepared = LoweringContext::new(&model).unwrap();
            let value = prepared.arena().param_value(floor.param_id().unwrap());
            let expected_kind = if value >= 0.0 { ModelKind::SOCP } else { ModelKind::QCP };
            prepared.variables().iter().skip(1).all(|v| v.lb.to_bits() == value.to_bits())
                && prepared.kind() == expected_kind
        });
        stop.store(true, Ordering::Relaxed);
        writer.join().unwrap();
        assert!(consistent, "preparation mixed parameter assignments");
    });
}

#[test]
fn preparation_retains_bounds_and_classification_after_later_updates() {
    let model = Model::new("saved preparation");
    param!(model, floor = 0.0);
    variable!(model, x);
    variable!(model, t >= floor);
    constraint!(model, x.square() <= t.square());
    objective!(model, Min, floor * t);

    let prepared = LoweringContext::new(&model).unwrap();
    floor.set_param_value(-1.0).unwrap();
    assert_eq!(prepared.variables()[1].lb, 0.0);
    assert_eq!(prepared.arena().param_value(floor.param_id().unwrap()), 0.0);
    assert_eq!(prepared.kind(), ModelKind::SOCP);
    let next = LoweringContext::new(&model).unwrap();
    assert_eq!(next.variables()[1].lb, -1.0);
    assert_eq!(next.kind(), ModelKind::QCP);
}
