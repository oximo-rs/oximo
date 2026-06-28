//! Indexed parameters built with `param!(m, name[i in dom[, ...]][ if cond] = value)`.

#![allow(clippy::float_cmp)]

use oximo_core::prelude::*;

#[test]
fn dense_indexed_param_used_in_model() {
    let m = Model::new("ip");
    let items = Set::range(0..3);
    let cap = [10.0, 20.0, 30.0];
    param!(m, c[i in items] = cap[i]);
    variable!(m, x[i in 0..3] >= 0.0);

    assert!(c.is_dense());
    assert_eq!(c.len(), 3);
    assert_eq!(m.num_parameters(), 3);

    constraint!(m, lim[i in 0..3], c[i] * x[i] <= 100.0);
    objective!(m, Max, sum!(c[i] * x[i] for i in 0..3));
    assert_eq!(m.kind(), ModelKind::LP);
    assert_eq!(m.num_constraints(), 3);

    assert!((m.param_value_idx(&c, 2usize).unwrap() - 30.0).abs() < f64::EPSILON);
    m.set_param_idx(&c, 2usize, 5.0);
    assert!((m.param_value_idx(&c, 2usize).unwrap() - 5.0).abs() < f64::EPSILON);
}

#[test]
fn multi_index_param_is_dense() {
    let m = Model::new("ipm");
    let w = [[1.0, 2.0], [3.0, 4.0]];
    param!(m, weight[i in 0..2, j in 0..2] = w[i][j]);
    assert!(weight.is_dense());
    assert_eq!(weight.shape().as_deref(), Some(&[2usize, 2][..]));
    assert_eq!(weight.len(), 4);
    assert!((m.param_value_idx(&weight, (1usize, 1usize)).unwrap() - 4.0).abs() < f64::EPSILON);
}

#[test]
fn string_keyed_indexed_param_is_sparse() {
    let m = Model::new("ips");
    let plants = Set::strings(["a", "b", "c"]);
    let prices = [1.0, 2.0, 3.0];
    param!(m, cost[p: String in plants] = price_for(&p, &prices));
    assert!(!cost.is_dense());
    assert_eq!(cost.shape(), None);
    assert_eq!(cost.len(), 3);
    assert!((m.param_value_idx(&cost, "b").unwrap() - 2.0).abs() < f64::EPSILON);
    assert!(m.param_value_idx(&cost, "z").is_none());
}

#[test]
fn constant_indexed_param_does_not_reference_index() {
    let m = Model::new("ipc");
    param!(m, k[i in 0..3] = 7.0);
    assert_eq!(k.len(), 3);
    assert!((m.param_value_idx(&k, 0usize).unwrap() - 7.0).abs() < f64::EPSILON);
}

#[test]
fn filtered_indexed_param_is_sparse() {
    let m = Model::new("ipf");
    let v = [0.0, 1.0, 2.0, 3.0, 4.0];
    param!(m, even[i in 0..5 if i % 2 == 0] = v[i]);
    assert!(!even.is_dense());
    assert_eq!(even.len(), 3);
    assert!(m.param_value_idx(&even, 1usize).is_none());
    assert!((m.param_value_idx(&even, 4usize).unwrap() - 4.0).abs() < f64::EPSILON);
}

fn price_for(p: &str, prices: &[f64; 3]) -> f64 {
    match p {
        "a" => prices[0],
        "b" => prices[1],
        _ => prices[2],
    }
}
