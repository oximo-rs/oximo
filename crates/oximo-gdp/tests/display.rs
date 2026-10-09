use std::panic::{AssertUnwindSafe, catch_unwind};

use oximo_gdp::prelude::*;

#[test]
fn pending_model_displays_named_sources_and_debug_counts() {
    let model = Model::new("selection");

    variable!(model, 0.0 <= flow <= 10.0);
    let small = disjunct!(model, small, |d| {
        constraint!(d, capacity, flow <= 3.0);
    });
    let large = disjunct!(model, large, |d| {
        constraint!(d, capacity, flow <= 8.0);
    });
    let choice = disjunction!(model, unit, [small, large]);
    let logic = logical_constraint!(model, implication, implies(small, !large));
    objective!(model, Max, flow);

    assert_eq!(small.to_string(), "small");
    assert_eq!(
        model.display_boolean(small).to_string(),
        "small: Boolean (__oximo_gdp_boolean0) (open)"
    );
    assert_eq!(
        model.display_disjunction(choice).to_string(),
        "unit: exactly_one [small, large] (pending)"
    );
    assert_eq!(
        model.display_logical_constraint(logic).to_string(),
        "implication: (small -> !large) (pending)"
    );

    let source = concat!(
        "gdp\n",
        "  small: Boolean (__oximo_gdp_boolean0) (open)\n",
        "  large: Boolean (__oximo_gdp_boolean1) (open)\n",
        "  small::capacity: small -> flow <= 3 (pending)\n",
        "  large::capacity: large -> flow <= 8 (pending)\n",
        "  unit: exactly_one [small, large] (pending)\n",
        "  implication: (small -> !large) (pending)\n",
    );

    assert_eq!(model.display_gdp().to_string(), source);
    assert_eq!(
        model.to_string(),
        format!(
            "Model 'selection' (MILP)\n\
             GDP: 2 Boolean decisions, 2 conditional rows, 1 disjunctions, 1 logical constraints (pending reformulation)\n\
             maximize flow\n\
             {source}\
             vars\n\
             \x20 0 <= flow <= 10\n\
             \x20 0 <= __oximo_gdp_boolean0 <= 1, binary\n\
             \x20 0 <= __oximo_gdp_boolean1 <= 1, binary\n"
        )
    );
    let debug = format!("{model:?}");
    for field in [
        "gdp_booleans: 2",
        "gdp_rows: 2",
        "gdp_disjunctions: 1",
        "gdp_logical_constraints: 1",
        "gdp_pending: true",
    ] {
        assert!(debug.contains(field), "{debug}");
    }

    let view = model.gdp();
    assert_eq!(view.rows.len(), 2);
    assert_eq!(model.display_gdp().to_string(), source);
}

#[test]
fn conditional_rows_render_all_sides_and_live_parameters() {
    let model = Model::new("rows");

    variable!(model, 0.0 <= x <= 10.0);
    param!(model, coefficient = 2.0);
    boolean_variable!(model, active);
    let upper = disjunct_constraint!(model, active, upper, coefficient * x <= 4.0);
    let lower = disjunct_constraint!(model, active, lower, x >= 1.0);
    let equality = disjunct_constraint!(model, active, equality, x == 2.0);
    let range = disjunct_constraint!(model, active, range, 2.0 <= x <= 8.0);
    let GdpRangeHandles::Interval(range) = range else { panic!("constant interval") };
    let nonlinear = disjunct_constraint!(model, active, nonlinear, x.exp() <= 10.0);
    let free = active.context().add_constraint(
        "free",
        Constraint::new(
            x.into_function(),
            Interval { lower: f64::NEG_INFINITY, upper: f64::INFINITY },
        ),
    );
    for (handle, expected) in [
        (upper, "active::upper: active -> 2 x <= 4 (pending)"),
        (lower, "active::lower: active -> x >= 1 (pending)"),
        (equality, "active::equality: active -> x = 2 (pending)"),
        (range, "active::range: active -> 2 <= x <= 8 (pending)"),
        (nonlinear, "active::nonlinear: active -> exp(x) <= 10 (pending)"),
        (free, "active::free: active -> x free (pending)"),
    ] {
        assert_eq!(model.display_disjunct_constraint(handle).to_string(), expected);
    }
    coefficient.set_param_value(3.0);
    assert_eq!(
        model.display_disjunct_constraint(upper.id()).to_string(),
        "active::upper: active -> 3 x <= 4 (pending)"
    );
}

#[test]
fn nested_sources_show_parent_guards_and_inclusive_selection() {
    let model = Model::new("nested");

    boolean_variable!(model, present);
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    let choice = disjunction!(model, technology, [a, b], AtLeastOne, parent = present);
    let logic = logical_constraint!(present.context(), required, a | b);
    assert_eq!(
        model.display_disjunction(choice).to_string(),
        "present::technology: present -> at_least_one [a, b] (pending)"
    );
    assert_eq!(
        model.display_boolean(a).to_string(),
        "a: Boolean (__oximo_gdp_boolean1), parent = present (open)"
    );
    assert_eq!(
        model.display_logical_constraint(logic).to_string(),
        "present::required: present -> (a | b) (pending)"
    );
}

#[test]
fn logical_display_preserves_operators_order_and_cardinality_multiplicity() {
    let model = Model::new("logic");

    boolean_variable!(model, a);
    boolean_variable!(model, b);
    let cases = [
        (LogicalExpr::from(true), "true"),
        (LogicalExpr::from(false), "false"),
        (LogicalExpr::from(a), "a"),
        (!a, "!a"),
        (!(a & b), "!(a & b)"),
        (a & b, "(a & b)"),
        (a | b, "(a | b)"),
        (a ^ b, "(a ^ b)"),
        (implies(a, b), "(a -> b)"),
        (iff(a, b), "(a <-> b)"),
        (exactly(2, [a, a, b]), "exactly(2, [a, a, b])"),
        (at_most(1, [a, b]), "at_most(1, [a, b])"),
        (at_least(1, [a, b]), "at_least(1, [a, b])"),
        (exactly(0, Vec::<BooleanHandle<'_>>::new()), "exactly(0, [])"),
        (logical_and(Vec::<BooleanHandle<'_>>::new()), "true"),
        (logical_or(Vec::<BooleanHandle<'_>>::new()), "false"),
        (logical_and([a.into(), a | !b]), "(a & (a | !b))"),
    ];

    for (expression, expected) in cases {
        assert_eq!(model.display_logical_expr(&expression).to_string(), expected);
    }

    assert_eq!((a | !b).to_string(), "(boolean[0] | !boolean[1])");
    let foreign = Model::new("foreign");
    boolean_variable!(foreign, a);
    let expression = LogicalExpr::from(a);
    assert!(
        catch_unwind(AssertUnwindSafe(|| { model.display_logical_expr(&expression).to_string() }))
            .is_err()
    );
}

#[test]
fn transformed_model_retains_named_sources_and_prints_generated_rows() {
    let model = Model::new("transformation");

    variable!(model, 0.0 <= x <= 10.0);
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    let row = disjunct_constraint!(model, a, capacity, x <= 2.0);
    let choice = disjunction!(model, choice, [a, b]);
    let logic = logical_constraint!(model, required, a | b);
    let transformed = model.to_reformulated_gdp_model(GdpReformulationOptions::default()).unwrap();

    assert_eq!(
        transformed.display_disjunct_constraint(row.id()).to_string(),
        "a::capacity: a -> x <= 2 (reformulated)"
    );
    assert_eq!(
        transformed.display_disjunction(choice.id()).to_string(),
        "choice: exactly_one [a, b] (reformulated)"
    );
    assert_eq!(
        transformed.display_logical_constraint(logic.id()).to_string(),
        "required: (a | b) (reformulated)"
    );

    let output = transformed.to_string();
    assert_eq!(output, transformed.model().to_string());
    assert!(output.contains("a: Boolean (__oximo_gdp_boolean0) (sealed)"), "{output}");
    assert!(output.contains("s.t.\n  __oximo_gdp_row"), "{output}");
    assert!(!output.contains("pending"), "{output}");
    assert!(format!("{transformed:?}").contains("gdp_pending: false"));
    assert!(model.to_string().contains("pending reformulation"));

    let expression = &transformed.gdp_snapshot().logical_constraints[0].expression;
    assert_eq!(transformed.display_logical_expr(expression).to_string(), "(a | b)");
}

#[test]
fn empty_and_boolean_only_models_print_without_extra_empty_sections() {
    let model = Model::new("empty");

    assert_eq!(model.display_gdp().to_string(), "");
    assert!(!model.to_string().contains("GDP:"));
    let boolean = model.add_boolean("standalone");
    assert_eq!(boolean.to_string(), "standalone");
    let output = model.to_string();
    assert!(output.contains("GDP: 1 Boolean decisions, 0 conditional rows"), "{output}");
    assert!(
        output.contains("gdp\n  standalone: Boolean (__oximo_gdp_boolean0) (open)"),
        "{output}"
    );
}

#[test]
fn deep_logical_sources_print_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let model = Model::new("deep display");
            let boolean = model.add_boolean("active");
            let mut expression = LogicalExpr::from(boolean);
            for _ in 0..20_000 {
                expression = !expression;
            }
            let expected = format!("{}active", "!".repeat(20_000));
            assert_eq!(model.display_logical_expr(&expression).to_string(), expected);
            assert_eq!(expression.to_string(), format!("{}boolean[0]", "!".repeat(20_000)));
            logical_constraint!(model, deep, expression);
            assert!(model.to_string().contains(&format!("deep: {expected} (pending)")));
            assert!(format!("{model:?}").contains("gdp_logical_constraints: 1"));
        })
        .unwrap()
        .join()
        .unwrap();
}
