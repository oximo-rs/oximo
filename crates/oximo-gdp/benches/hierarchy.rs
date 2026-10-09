//! Construction and reformulation timings for flat and nested numeric GDP.
//! Run with `cargo bench -p oximo-gdp --bench hierarchy`.

use oximo_gdp::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn build(groups: usize, depth: usize) -> Model {
    let model = Model::new("hierarchy benchmark");
    for group in 0..groups {
        let x = model.__var(format!("x{group}")).bounds(0.0, 10.0).build();
        let root = model.add_boolean(format!("root{group}"));
        let other = model.add_boolean(format!("other{group}"));
        root.context().add_constraint("capacity", x.le(8.0));
        disjunction!(model, [root, other]);
        let mut parent = root;
        for (level, capacity) in [6.0, 5.0, 4.0, 3.0].into_iter().take(depth).enumerate() {
            let child = model.add_boolean(format!("child{group}_{level}"));
            let sibling = model.add_boolean(format!("sibling{group}_{level}"));
            child.context().add_constraint("capacity", x.le(capacity));
            disjunction!(model, [child, sibling], parent = parent);
            parent = child;
        }
    }
    model
}

fn median(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    println!("case,groups,depth,samples,build_ns,reformulate_ns,total_ns,new_rows,new_nodes");
    for (case, depth) in [("flat", 0), ("nested", 1), ("deep", 4)] {
        for groups in [100, 1000, 10_000] {
            let mut construction = Vec::new();
            let mut reformulation = Vec::new();
            let mut total = Vec::new();
            let mut generated = (0, 0);
            for sample in 0..22 {
                let start = Instant::now();
                let model = black_box(build(groups, depth));
                let built = start.elapsed().as_nanos();
                let before = (model.num_constraints(), model.arena().len());
                let reformulate = Instant::now();
                black_box(model.reformulate_gdp(BigM::default()).unwrap());
                let reformed = reformulate.elapsed().as_nanos();
                generated = (model.num_constraints() - before.0, model.arena().len() - before.1);
                if sample > 0 {
                    construction.push(built);
                    reformulation.push(reformed);
                    total.push(built + reformed);
                }
                black_box(model);
            }
            println!(
                "{case},{groups},{depth},21,{},{},{},{},{}",
                median(&mut construction),
                median(&mut reformulation),
                median(&mut total),
                generated.0,
                generated.1,
            );
        }
    }
}
