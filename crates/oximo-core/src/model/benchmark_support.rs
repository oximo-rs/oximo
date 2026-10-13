//! Benchmark helpers for model classification and construction.

use super::classify::PAR_KIND_THRESHOLD;
use super::{Model, ModelKind};
use crate::constraint::Relate;
use crate::set::Set;
use crate::sos::SosType;

pub const THRESHOLD: usize = PAR_KIND_THRESHOLD;

pub fn model(rows: usize, degree: usize) -> Model {
    let model = Model::new("kind_bench");
    let x = model.__var("x").build();
    let y = model.__var("y").build();
    let z = model.__var("z").build();
    model.__minimize(x + y + z);
    for i in 0..rows {
        let lhs = match degree {
            1 => (x + 2.0 * y - z).erase(),
            2 => x.powi(2) + y,
            _ => (x * y * z).erase(),
        };
        model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
    }
    model
}

pub fn soc_model(rows: usize) -> Model {
    let model = Model::new("kind_soc_bench");
    let x = model.__var("x").build();
    let y = model.__var("y").build();
    let t = model.__var("t").lb(0.0).build();
    model.__minimize(t);
    for _ in 0..rows {
        model.__add_constraint_auto((x.powi(2) + y.powi(2) - t.powi(2)).le(0.0));
    }
    model
}

pub fn infer(model: &Model, parallel: bool) -> ModelKind {
    model.infer_kind(parallel)
}

#[derive(Copy, Clone, Debug)]
pub enum IndexedBuildCase {
    Variables,
    Parameters,
    Algebraic,
    Range,
    Soc,
    Sos,
}

/// Build and immediately inspect a fresh indexed model so Criterion measures
/// construction rather than solver translation.
pub fn indexed_build(rows: usize, case: IndexedBuildCase, parallel: bool) -> usize {
    let model = Model::new("indexed_build_bench");
    let keys = Set::range(0..rows);
    match case {
        IndexedBuildCase::Variables => {
            let x = model
                .__indexed_var("x", &keys)
                .lb_by(|i: usize| -(i as f64))
                .ub_by(|i: usize| i as f64 + 1.0)
                .parallel_for_benchmark(parallel)
                .build();
            std::hint::black_box(x.len());
        }
        IndexedBuildCase::Parameters => {
            let value = |i: usize| i as f64 + 1.0;
            let p = model.indexed_param_with("p".to_owned(), &keys, &value, Some(parallel));
            std::hint::black_box(p.len());
        }
        IndexedBuildCase::Algebraic => {
            let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
            let rule = |i: usize| (2.0 * x[i] + 1.0).le(i as f64 + 10.0);
            model.add_constraints_over_with("c", &keys, &rule, Some(parallel));
        }
        IndexedBuildCase::Range => {
            let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
            let rule = |i: usize| (x[i] + 1.0, -(i as f64), i as f64 + 10.0);
            model.add_range_constraints_over_with("r", &keys, &rule, Some(parallel));
        }
        IndexedBuildCase::Soc => {
            let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
            let y = model.__indexed_var("y", &keys).parallel_for_benchmark(parallel).build();
            let t = model.__var("t").lb(0.0).build();
            let rule = |i: usize| ([x[i] + 1.0, y[i] - 1.0], t + i as f64 + 1.0);
            model.add_soc_constraints_over_with("q", &keys, &rule, Some(parallel));
        }
        IndexedBuildCase::Sos => {
            let x = model.__indexed_var("x", &keys).parallel_for_benchmark(parallel).build();
            let y = model.__indexed_var("y", &keys).parallel_for_benchmark(parallel).build();
            let rule = |i: usize| [(x[i], 1.0), (y[i], 2.0)];
            model.add_sos_constraints_over_with("s", &keys, SosType::Sos1, &rule, Some(parallel));
        }
    }
    model.num_variables() + model.num_parameters() + model.num_constraints() + model.arena().len()
}

/// Scalar construction guardrail for the parallel-safe arena migration.
pub fn scalar_build(rows: usize) -> usize {
    let model = Model::new("scalar_build_bench");
    for i in 0..rows {
        let x = model.__var(format!("x{i}")).build();
        model.__add_constraint(format!("c{i}"), (2.0 * x + 1.0).le(i as f64 + 10.0));
    }
    model.num_variables() + model.num_constraints() + model.arena().len()
}
