//! Benchmark helpers for model classification and construction.

use super::classify::PAR_KIND_THRESHOLD;
use super::{Model, ModelKind};
use crate::constraint::Relate;
use crate::set::Set;
use crate::sos::SosType;
use std::cell::RefCell;
use std::time::{Duration, Instant};

/// Wall-clock phases of forced-parallel scalar family construction.
#[derive(Clone, Copy, Debug)]
pub enum IndexedPhase {
    Setup,
    WorkerPreparation,
    NameValidation,
    ArenaMerge,
    RootRemapping,
    Registration,
    HandleAssembly,
}

impl IndexedPhase {
    pub const ALL: [Self; 7] = [
        Self::Setup,
        Self::WorkerPreparation,
        Self::NameValidation,
        Self::ArenaMerge,
        Self::RootRemapping,
        Self::Registration,
        Self::HandleAssembly,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::WorkerPreparation => "worker_preparation",
            Self::NameValidation => "name_validation",
            Self::ArenaMerge => "arena_merge",
            Self::RootRemapping => "root_remapping",
            Self::Registration => "registration",
            Self::HandleAssembly => "handle_assembly",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IndexedBuildPhases {
    elapsed: [Duration; 7],
}

impl IndexedBuildPhases {
    pub fn elapsed(&self, phase: IndexedPhase) -> Duration {
        self.elapsed[phase as usize]
    }
}

thread_local! {
    static PHASES: RefCell<Option<IndexedBuildPhases>> = const { RefCell::new(None) };
}

pub(super) struct PhaseTimer(Option<Instant>);

impl PhaseTimer {
    pub(super) fn new() -> Self {
        Self(PHASES.with_borrow(Option::is_some).then(Instant::now))
    }

    pub(super) fn mark(&mut self, phase: IndexedPhase) {
        if let Some(previous) = self.0.as_mut() {
            let now = Instant::now();
            PHASES.with_borrow_mut(|slot| {
                slot.as_mut().expect("phase recording ended early").elapsed[phase as usize] +=
                    now.duration_since(*previous);
            });
            *previous = now;
        }
    }
}

struct PhaseRecording;

impl Drop for PhaseRecording {
    fn drop(&mut self) {
        PHASES.with_borrow_mut(|slot| *slot = None);
    }
}

/// Profile the same family pipeline used by indexed scalar constraints.
/// Fixture creation and destruction are excluded from phase durations. Setup
/// starts after key collection; worker preparation includes scheduling and names.
/// Clock and recording overhead is included in these diagnostic durations.
///
/// # Panics
/// Panics for zero rows or terms, or nested recording on this thread.
pub fn profile_indexed_scalar_build(
    rows: usize,
    terms: usize,
    range: bool,
) -> (usize, IndexedBuildPhases) {
    PHASES.with_borrow_mut(|slot| {
        assert!(slot.is_none(), "nested indexed phase recording");
        *slot = Some(IndexedBuildPhases::default());
    });

    let _recording = PhaseRecording;
    let checksum = indexed_scalar_build(rows, terms, range);
    let phases = PHASES.with_borrow_mut(Option::take).expect("missing indexed phase recording");

    (checksum, phases)
}

/// Profile forced-parallel SOC or SOS registration using the ordinary indexed fixture.
/// Variable fixture construction is included in total timings, outside phase clocks.
///
/// # Panics
/// Panics for nested recording on this thread.
pub fn profile_indexed_build(rows: usize, case: IndexedBuildCase) -> (usize, IndexedBuildPhases) {
    PHASES.with_borrow_mut(|slot| {
        assert!(slot.is_none(), "nested indexed phase recording");
        *slot = Some(IndexedBuildPhases::default());
    });

    let _recording = PhaseRecording;
    let checksum = indexed_build(rows, case, true);
    let phases = PHASES.with_borrow_mut(Option::take).expect("missing indexed phase recording");
    (checksum, phases)
}

/// Whole-model fixture for phase/scaling comparisons with 1, 2, 4 or 6 workers.
/// The forced-parallel family path is used even with one worker. Increasing
/// `terms` increases affine expression work without changing row count or IDs.
///
/// # Panics
/// Panics for zero rows or terms.
pub fn indexed_scalar_build(rows: usize, terms: usize, range: bool) -> usize {
    assert!(rows > 0 && terms > 0, "phase fixture dimensions must be positive");
    let model = Model::new("indexed scalar phase benchmark");
    let keys = Set::range(0..rows);
    let x = model.__indexed_var("x", &keys).parallel_for_benchmark(true).build();
    let expression = |i: usize| {
        let mut lhs = x[i] + 1.0;

        for term in 1..terms {
            lhs = lhs + (term as f64 + 1.0) * x[(i + term) % rows];
        }
        lhs
    };

    if range {
        let rule = |i: usize| (expression(i), -(i as f64), i as f64 + 10.0);
        model.add_range_constraints_over_with("r", &keys, &rule, Some(true));
    } else {
        let rule = |i: usize| expression(i).le(i as f64 + 10.0);
        model.add_constraints_over_with("c", &keys, &rule, Some(true));
    }

    model.num_variables() + model.num_constraints() + model.arena().len()
}

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
