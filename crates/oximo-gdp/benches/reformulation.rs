//! Run with `cargo bench -p oximo-gdp --bench reformulation`.

use oximo_expr::ExprNode;
use oximo_gdp::prelude::*;
use oximo_solver::prepare::PreparedExpressions;
use std::hint::black_box;
use std::time::Instant;

fn selection(width: usize, nested: bool) -> Model {
    let model = Model::new("selection");
    let parent = nested.then(|| model.add_boolean("parent"));
    let branches: Vec<_> = (0..width).map(|i| model.add_boolean(format!("b{i}"))).collect();
    if let Some(parent) = parent {
        disjunction!(model, branches, parent = parent);
    } else {
        disjunction!(model, branches);
    }
    model
}

fn predicate(width: usize, kind: &str) -> Model {
    let model = Model::new("assertion");
    let branches: Vec<_> = (0..width).map(|i| model.add_boolean(format!("b{i}"))).collect();
    let formula = match kind {
        "exactly" => exactly(width / 2, branches),
        "and" => logical_and(branches),
        _ => logical_or(branches),
    };
    logical_constraint!(model, formula);
    model
}

fn shared(rows: usize, depth: usize) -> Model {
    let model = Model::new("shared expression");
    variable!(model, 0.0 <= x <= 1.0);
    let indicator = model.add_boolean("active");
    let mut arena = model.__sum_context().borrow_mut();
    let mut source = x.id;
    for _ in 0..depth {
        source = arena.push(ExprNode::Unary(UnaryOp::Neg, source));
    }
    drop(arena);
    let expression: Expr = Expr::new(source, model.__sum_context());
    for i in 0..rows {
        indicator.context().add_constraint(format!("r{i}"), expression.le(2.0));
    }
    model
}

fn shared_shifted(rows: usize, depth: usize) -> Model {
    let model = Model::new("shared expression with distinct roots");

    variable!(model, 0.0 <= x <= 1.0);
    let indicator = model.add_boolean("active");

    let mut arena = model.__sum_context().borrow_mut();
    let mut source = x.id;
    for _ in 0..depth {
        source = arena.push(ExprNode::Unary(UnaryOp::Neg, source));
    }
    drop(arena);
    for i in 0..rows {
        let offset = model.__param(format!("p{i}"), 1.0);
        let root = model
            .__sum_context()
            .borrow_mut()
            .push(ExprNode::Add([source, offset.id].into_iter().collect()));
        let expression: Expr = Expr::new(root, model.__sum_context());
        indicator.context().add_constraint(format!("r{i}"), expression.le(2.0));
    }
    model
}

fn history(count: usize) -> Model {
    let model = Model::new("history");

    for i in 0..count {
        let boolean = model.add_boolean(format!("b{i}"));
        logical_constraint!(model, boolean);
    }

    model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
    model
}

fn sparse_arena(unrelated: usize) -> Model {
    let model = Model::new("sparse GDP arena");

    variable!(model, 0.0 <= x <= 10.0);
    let active = model.add_boolean("active");

    let mut arena = model.__sum_context().borrow_mut();
    for _ in 0..unrelated {
        arena.push(ExprNode::Const(0.0));
    }
    let source = arena.push(ExprNode::Unary(UnaryOp::Neg, x.id));
    drop(arena);
    let expression: Expr = Expr::new(source, model.__sum_context());
    active.context().add_constraint("row", expression.ge(-2.0));
    model
}

fn compound_logic(count: usize, kind: &str) -> Model {
    let model = Model::new("compound logic");

    let booleans: Vec<_> = (0..=count).map(|i| model.add_boolean(format!("b{i}"))).collect();
    for pair in booleans.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let formula = match kind {
            "implies" => implies(a, b),
            "iff" => iff(a, b),
            _ => a ^ b,
        };
        logical_constraint!(model, formula);
    }

    model
}

fn selected(case: &str) -> bool {
    std::env::args()
        .skip(1)
        .find(|argument| argument != "--bench")
        .is_none_or(|filter| case.contains(&filter))
}

fn measure(case: &str, size: usize, repetitions: usize, setup: impl Fn() -> Model) {
    if !selected(case) {
        return;
    }
    let mut samples = Vec::new();
    let mut generated = (0, 0, 0);
    for sample in 0..6 {
        let model = setup();
        let before = (model.num_variables(), model.num_constraints(), model.arena().len());
        let start = Instant::now();
        for _ in 0..repetitions {
            black_box(model.reformulate_gdp(GdpReformulationOptions::default()).unwrap());
        }
        let elapsed = start.elapsed().as_nanos();
        generated = (
            model.num_variables() - before.0,
            model.num_constraints() - before.1,
            model.arena().len() - before.2,
        );
        if sample > 0 {
            samples.push(elapsed);
        }
        black_box(model);
    }
    samples.sort_unstable();
    println!(
        "{case},{size},{repetitions},{},{},{},{}",
        samples[2], generated.0, generated.1, generated.2
    );
}

fn measure_preparation(rows: usize, extraction: bool) {
    let case = if extraction { "shared_preparation_extract" } else { "shared_preparation_kind" };
    if !selected(case) {
        return;
    }
    let mut samples = Vec::new();
    for sample in 0..6 {
        let model = shared(rows, 1000);
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        let expressions = PreparedExpressions::new((*model.arena()).clone());
        let constraints = model.constraints();
        let start = Instant::now();
        if extraction {
            for row in constraints.algebraic() {
                black_box(expressions.linear(row.lhs).unwrap());
            }
        } else {
            black_box(model.kind());
        }
        if sample > 0 {
            samples.push(start.elapsed().as_nanos());
        }
    }
    samples.sort_unstable();
    println!("{case},{rows},1,{},0,0,0", samples[2]);
}

fn measure_incremental(history: usize) {
    let case = "incremental_history";
    if !selected(case) {
        return;
    }
    let mut samples = Vec::new();
    for sample in 0..6 {
        let model = Model::new(case);
        let b = model.add_boolean("b");
        for _ in 0..history {
            logical_constraint!(model, b);
        }
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        // We can exclude the one-time vector growth immediately after a large initial
        // batch, so this measures steady incremental traversal rather than realloc.
        logical_constraint!(model, b);
        model.reformulate_gdp(GdpReformulationOptions::default()).unwrap();
        let start = Instant::now();
        for _ in 0..20 {
            logical_constraint!(model, b);
            black_box(model.reformulate_gdp(GdpReformulationOptions::default()).unwrap());
        }
        if sample > 0 {
            samples.push(start.elapsed().as_nanos());
        }
    }
    samples.sort_unstable();
    println!("{case},{history},20,{},0,20,20", samples[2]);
}

fn main() {
    println!("case,size,repetitions,median_nanoseconds,new_variables,new_rows,new_nodes");
    for width in [1000, 16_000] {
        measure("selection", width, 1, || selection(width, false));
        measure("nested_selection", width, 1, || selection(width, true));
    }
    for kind in ["exactly", "and", "or"] {
        measure(kind, 1000, 1, || predicate(1000, kind));
    }
    for rows in [100, 1000] {
        measure("shared_depth_1000", rows, 1, || shared(rows, 1000));
        measure("shared_shifted_depth_1000", rows, 1, || shared_shifted(rows, 1000));
    }
    measure("no_op_history", 10_000, 20, || history(10_000));
    for kind in ["implies", "iff", "xor"] {
        measure(kind, 1000, 1, || compound_logic(1000, kind));
    }
    for rows in [100, 1000] {
        measure_preparation(rows, false);
        measure_preparation(rows, true);
    }
    for unrelated in [0, 200_000, 1_000_000] {
        measure("sparse_arena", unrelated, 1, || sparse_arena(unrelated));
    }
    for count in [1000, 100_000] {
        measure_incremental(count);
    }
}
