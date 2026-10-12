//! Run `cargo bench -p oximo-gdp --bench hull`.
//! Set `OXIMO_GDP_BENCH_MEMORY=1` to also report artifact counts and Linux peak
//! memory in a fresh process for each selected case.

use criterion::{BatchSize, Criterion};
use oximo_gdp::prelude::*;
use std::hint::black_box;

fn tree(
    groups: usize,
    depth: usize,
    width: usize,
    mixed: bool,
) -> (Model, GdpReformulationOptions) {
    let model = Model::new("Hull benchmark");

    let mut options = GdpReformulationOptions::new(Hull::default());
    for group in 0..groups {
        let x = model.__var(format!("x{group}")).bounds(-2.0, 10.0).build();
        let mut parent = None;
        for level in 0..depth {
            let branches: Vec<_> = (0..width)
                .map(|branch| model.add_boolean(format!("b{group}_{level}_{branch}")))
                .collect();
            branches[0].context().add_constraint("capacity", x.le(5.0));
            let first = branches[0];
            let choice = if let Some(p) = parent {
                disjunction!(model, branches, parent = p)
            } else {
                disjunction!(model, branches)
            };
            if mixed && level % 2 == 1 {
                options = options.with_method(choice, BigM::default());
            }
            parent = Some(first);
        }
    }
    (model, options)
}

fn distinct_coordinates(width: usize) -> (Model, GdpReformulationOptions) {
    let model = Model::new("one coordinate per branch");

    let branches: Vec<_> = (0..width)
        .map(|i| {
            let x = model.__var(format!("x{i}")).bounds(-2.0, 2.0).build();
            let branch = model.add_boolean(format!("b{i}"));
            branch.context().add_constraint("limit", x.le(1.0));
            branch
        })
        .collect();
    disjunction!(model, branches);
    (model, Hull::default().into())
}

fn shared(rows: usize) -> (Model, GdpReformulationOptions) {
    let model = Model::new("shared Hull DAG");

    variable!(model, 0.0 <= x <= 1.0);
    let mut body = x.erase();
    for _ in 0..20 {
        body = (body * body + 0.1).erase();
    }
    let active = model.add_boolean("active");
    for i in 0..rows {
        active.context().add_constraint(format!("r{i}"), body.le(2.0));
    }
    (model, Hull::default().into())
}

fn inclusive(width: usize) -> (Model, GdpReformulationOptions) {
    let model = Model::new("enumerated Hull");

    variable!(model, 0.0 <= x <= 10.0);
    let branches: Vec<_> = (0..width)
        .map(|i| {
            let b = model.add_boolean(format!("b{i}"));
            b.context().add_constraint("limit", x.le(5.0));
            b
        })
        .collect();
    disjunction!(model, branches, AtLeastOne);
    (
        model,
        Hull::default()
            .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 1023 })
            .into(),
    )
}

fn nested_inclusive(groups: usize) -> (Model, GdpReformulationOptions) {
    let model = Model::new("inclusive ancestors with exclusive children");

    for group in 0..groups {
        let x = model.__var(format!("x{group}")).bounds(0.0, 10.0).build();
        let parents: Vec<_> = (0..3)
            .map(|branch| {
                let parent = model.add_boolean(format!("p{group}_{branch}"));
                let low = model.add_boolean(format!("low{group}_{branch}"));
                let high = model.add_boolean(format!("high{group}_{branch}"));
                low.context().add_constraint("limit", x.le(4.0));
                high.context().add_constraint("limit", x.ge(6.0));
                disjunction!(model, [low, high], parent = parent);
                parent
            })
            .collect();
        disjunction!(model, parents, AtLeastOne);
    }
    (
        model,
        Hull::default()
            .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 7 })
            .into(),
    )
}

fn inclusive_depth(depth: usize) -> (Model, GdpReformulationOptions) {
    let model = Model::new("nested inclusive expansion");

    variable!(model, -2.0 <= x <= 4.0);
    let mut parent = None;
    for level in 0..depth {
        let left = model.add_boolean(format!("left{level}"));
        let right = model.add_boolean(format!("right{level}"));
        left.context().add_constraint("limit", x.le(3.0));
        if let Some(parent) = parent {
            disjunction!(model, [left, right], parent = parent, AtLeastOne);
        } else {
            disjunction!(model, [left, right], AtLeastOne);
        }
        parent = Some(left);
    }
    (
        model,
        Hull::default()
            .with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 })
            .into(),
    )
}

fn build_case(name: &str) -> (Model, GdpReformulationOptions) {
    match name {
        "wide_1000" => tree(1, 1, 1000, false),
        "deep_500" => tree(1, 500, 2, false),
        "shared_1000" => shared(1000),
        "inclusive_8" => inclusive(8),
        "inclusive_depth_6" => inclusive_depth(6),
        _ => {
            let (kind, count) = name.rsplit_once('_').expect("case count");
            let count = count.parse().expect("numeric count");
            match kind {
                "distinct" => distinct_coordinates(count),
                "flat" => tree(count, 1, 2, false),
                "nested" => tree(count, 4, 2, false),
                "mixed" => tree(count, 4, 2, true),
                "nested_inclusive" => nested_inclusive(count),
                _ => panic!("unknown case {name}"),
            }
        }
    }
}

fn memory_kib(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix(field)
                    .and_then(|value| value.split_whitespace().next()?.parse().ok())
            })
        })
        .unwrap_or(0)
}

fn memory_sample(name: &str) {
    let (model, options) = build_case(name);

    let before = (model.num_variables(), model.num_constraints(), model.arena().len());
    let source_rss = memory_kib("VmRSS:");
    let report = black_box(model.reformulate_gdp(options).unwrap());

    println!(
        "{name},{},{},{},{source_rss},{},{},{}",
        model.num_variables() - before.0,
        model.num_constraints() - before.1,
        model.arena().len() - before.2,
        memory_kib("VmHWM:"),
        report.planning.row_expansions,
        report.planning.private_expression_nodes
    );
}

fn report_memory(name: &str) {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--memory-case", name])
        .output()
        .expect("benchmark memory child");

    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    print!("{}", String::from_utf8(output.stdout).unwrap());
}

fn hull(criterion: &mut Criterion) {
    let memory = std::env::var_os("OXIMO_GDP_BENCH_MEMORY").is_some();
    let mut cases = Vec::new();

    for groups in [100, 1000, 10_000] {
        for kind in ["flat", "nested", "mixed"] {
            cases.push(format!("{kind}_{groups}"));
        }
    }
    for case in [
        "wide_1000",
        "distinct_1000",
        "distinct_2000",
        "distinct_4000",
        "deep_500",
        "shared_1000",
        "inclusive_8",
        "inclusive_depth_6",
        "nested_inclusive_100",
        "nested_inclusive_1000",
    ] {
        cases.push(case.to_owned());
    }

    if memory {
        println!(
            "case,new_variables,new_rows,new_nodes,source_rss_kib,peak_process_kib,row_expansions,private_expression_nodes"
        );
    }

    let mut group = criterion.benchmark_group("gdp/hull");

    for name in cases {
        let mut memory_reported = false;

        group.bench_function(&name, |bencher| {
            if memory && !memory_reported {
                report_memory(&name);
                memory_reported = true;
            }

            bencher.iter_batched(
                || build_case(&name),
                |(model, options)| {
                    let report = model.reformulate_gdp(options).unwrap();

                    black_box((model, report))
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

fn main() {
    let mut args = std::env::args().skip(1);

    if args.next().is_some_and(|arg| arg == "--memory-case") {
        memory_sample(&args.next().expect("memory benchmark case"));
        return;
    }

    let mut criterion = Criterion::default().configure_from_args();
    hull(&mut criterion);
    criterion.final_summary();
}
