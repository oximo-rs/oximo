//! Criterion timings and allocation counts for semantic storage.
//! Run `cargo bench -p oximo-core --bench semantic_structure`.

#[path = "../../oximo-expr/benches/support/allocation.rs"]
mod allocation;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use oximo_core::prelude::*;
use oximo_core::{
    FunctionRepresentationRef, FunctionValue, SemanticPair, SetValue, SymmetricAffineData,
    VectorAffineData,
};

#[global_allocator]
static ALLOC: allocation::CountingAllocator = allocation::CountingAllocator;

#[derive(Clone, Copy, Debug)]
enum Structure {
    Soc(usize),
    Psd(usize),
}

impl Structure {
    fn pair(self, model: &Model, root: oximo_expr::ExprId) -> SemanticPair {
        match self {
            Self::Soc(width) => SemanticPair {
                function: FunctionValue::new(VectorAffineData {
                    model_id: model.id(),
                    components: vec![root; width].into(),
                }),
                set: SetValue::new(SecondOrderCone { dimension: width }),
            },
            Self::Psd(side) => SemanticPair {
                function: FunctionValue::new(SymmetricAffineData {
                    model_id: model.id(),
                    matrix: SymmetricMatrix::from_upper_fn(side, |_, _| root),
                }),
                set: SetValue::new(PositiveSemidefiniteCone { side_dimension: side }),
            },
        }
    }
}

fn structured_model(structure: Structure, fork: bool) -> Model {
    let model = Model::new("wide semantic function");
    variable!(model, x);
    model.__register_semantic_pair("wide", structure.pair(&model, x.id()));

    if fork { model.fork() } else { model }
}

fn scalar_model(rows: usize) -> Model {
    let model = Model::new("indexed scalar family");
    variable!(model, x);
    constraint!(model, rows[i in 0..rows], x + f64::from(u32::try_from(i).unwrap()) <= 1.0);
    model
}

fn worker_remapping(structure: Structure, shared: bool) {
    let model = Model::new("worker remapping");
    variable!(model, x);
    let mut batch = x.arena.__begin_batch();
    let snapshot = batch.snapshot();
    let mut forks: Vec<_> = (0..4)
        .map(|i| {
            x.arena
                .__with_fork(snapshot.clone(), || structure.pair(&model, (x + f64::from(i)).id()))
        })
        .collect();
    drop(snapshot);
    let remaps = batch.merge(&mut forks);
    drop(batch);

    for (mut worker, remap) in forks.into_iter().zip(remaps) {
        let original = shared.then(|| worker.value.clone());
        worker.value.remap(remap);
        black_box((worker, original));
    }
}

fn inspected_roots(function: &FunctionValue) -> usize {
    match function.representation() {
        FunctionRepresentationRef::VectorAffine(roots) => roots.len(),
        FunctionRepresentationRef::SymmetricAffine(matrix) => matrix.upper_triangle().len(),
        _ => 0,
    }
}

fn allocation_record<T>(
    output: &mut String,
    case: &str,
    input: &str,
    operation: impl FnOnce() -> T,
) {
    use std::fmt::Write;

    let (calls, bytes, peak) = allocation::allocations(operation);
    writeln!(output, "{case},{input},{calls},{bytes},{peak}").unwrap();
}

fn main() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        worker_remapping(Structure::Soc(2), false);
        let mut criterion = Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
            .configure_from_args();
        let mut allocations =
            String::from("case,input,allocation_calls,allocated_bytes,peak_live_bytes\n");

        let mut scalar = criterion.benchmark_group("semantic_scalars");
        for rows in [2048, 4096, 8192] {
            allocation_record(&mut allocations, "registration", &rows.to_string(), || {
                scalar_model(rows)
            });
            scalar.bench_with_input(BenchmarkId::new("registration", rows), &rows, |b, &rows| {
                b.iter(|| black_box(scalar_model(rows)));
            });
        }
        scalar.finish();

        let mut wide = criterion.benchmark_group("semantic_structure");
        for structure in [
            Structure::Soc(32),
            Structure::Soc(1024),
            Structure::Soc(8192),
            Structure::Psd(4),
            Structure::Psd(32),
            Structure::Psd(128),
        ] {
            let label = format!("{structure:?}");
            for fork in [false, true] {
                let case = if fork { "model_fork" } else { "registration" };
                allocation_record(&mut allocations, case, &label, || {
                    structured_model(structure, fork)
                });
                wide.bench_function(BenchmarkId::new(case, &label), |b| {
                    b.iter(|| black_box(structured_model(structure, fork)));
                });
            }
            for shared in [false, true] {
                let case = if shared { "shared_worker_remap" } else { "unique_worker_remap" };
                allocation_record(&mut allocations, case, &label, || {
                    worker_remapping(structure, shared);
                });
                wide.bench_function(BenchmarkId::new(case, &label), |b| {
                    b.iter(|| worker_remapping(structure, shared));
                });
            }

            let model = structured_model(structure, false);
            let inspect = || {
                let records = model.semantic_constraints_ref();
                let mut count = 0;
                for record in records.iter() {
                    record.pair.function.visit_expressions(&mut |_| count += 1);
                }
                count
            };
            allocation_record(&mut allocations, "borrowed_registry", &label, inspect);
            wide.bench_function(BenchmarkId::new("borrowed_registry", &label), |b| {
                b.iter(|| black_box(inspect()));
            });
            let record = model.semantic_constraints_ref().get(0).unwrap().to_owned();
            allocation_record(&mut allocations, "borrowed_shape", &label, || {
                inspected_roots(&record.pair.function)
            });
            wide.bench_function(BenchmarkId::new("borrowed_shape", &label), |b| {
                b.iter(|| black_box(inspected_roots(black_box(&record.pair.function))));
            });
        }
        wide.finish();
        criterion.final_summary();

        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("semantic_structure_allocations.csv");
        std::fs::write(&path, allocations).unwrap();
        println!("Allocation counts: {}", path.display());
    });
}
