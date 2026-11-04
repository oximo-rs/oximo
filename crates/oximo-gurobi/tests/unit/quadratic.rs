#![allow(clippy::float_cmp)] // Canonical coefficients and native attributes are exact here.

use super::*;
use oximo_core::prelude::*;
use oximo_solver::resident::{PersistentPreparation, PreparationChange};

fn quiet_env() -> Env {
    let mut env = default_env().unwrap();
    env.set(gurobi_rs::param::OutputFlag, 0).unwrap();
    env
}

fn quadratic_coefficients(
    terms: impl IntoIterator<Item = (gurobi_rs::Var, gurobi_rs::Var, f64)>,
    variables: &[gurobi_rs::Var],
) -> FxHashMap<(usize, usize), f64> {
    let mut coefficients = FxHashMap::default();
    for (row, column, coefficient) in terms {
        let row = variables.iter().position(|v| *v == row).unwrap();
        let column = variables.iter().position(|v| *v == column).unwrap();
        *coefficients.entry((row.min(column), row.max(column))).or_insert(0.0) += coefficient;
    }
    coefficients
}

#[test]
fn sparse_upload_matches_polynomial_values_at_fixed_points() {
    let model = Model::new("sparse evaluation");
    variable!(model, -2.0 <= x <= 2.0);
    variable!(model, -2.0 <= y <= 2.0);
    let root = 2.0 * (x + 2.0 * y + 1.0).square() + x * y - 3.0 * y * y + 2.0 * x - 7.0;
    objective!(model, Min, root);
    let row = constraint!(model, row, root <= 20.0);

    let env = quiet_env();
    let mut built = build(&model, &GurobiOptions::default(), &env).unwrap();
    built.model.update().unwrap();
    let expected = FxHashMap::from_iter([((0, 0), 2.0), ((0, 1), 9.0), ((1, 1), 5.0)]);
    assert_eq!(
        quadratic_coefficients(built.model.get_qobjective_terms().unwrap(), &built.vars),
        expected,
    );

    let [GurobiRow::Quad(native_row)] = built.constrs[row.index()].rows.as_slice() else {
        panic!("native quadratic row")
    };
    let terms = built.model.get_qconstr_terms(native_row).unwrap();
    assert_eq!(quadratic_coefficients(terms.quadratic, &built.vars), expected);
    assert_eq!(
        terms.linear.into_iter().collect::<FxHashMap<_, _>>(),
        FxHashMap::from_iter([(built.vars[0], 6.0), (built.vars[1], 8.0)]),
    );
    assert_eq!(built.model.get_obj_attr(attr::QCRHS, native_row).unwrap(), 25.0);
    assert_eq!(built.model.get_attr(attr::ObjCon).unwrap(), -5.0);

    built.model.set_param(gurobi_rs::param::NonConvex, 2).unwrap();
    for point in [[0.0, 0.0], [1.0, -0.5], [-1.25, 0.75]] {
        for (variable, value) in built.vars.iter().zip(point) {
            built.model.set_obj_attr(gurobi_rs::attr::LB, variable, value).unwrap();
            built.model.set_obj_attr(gurobi_rs::attr::UB, variable, value).unwrap();
        }
        built.model.optimize().unwrap();
        let expected = 2.0 * (point[0] + 2.0 * point[1] + 1.0_f64).powi(2) + point[0] * point[1]
            - 3.0 * point[1] * point[1]
            + 2.0 * point[0]
            - 7.0;
        assert!((built.model.get_attr(gurobi_rs::attr::ObjVal).unwrap() - expected).abs() < 1e-10);
    }
}

#[test]
fn sparse_affine_batches_keep_duplicates_constants_ranges_and_source_slots() {
    let model = Model::new("sparse affine batches");
    variable!(model, -10.0 <= x <= 10.0);
    variable!(model, -10.0 <= y <= 10.0);
    let arena = model.__sum_context();
    let x_id = x.var_id().unwrap();
    let y_id = y.var_id().unwrap();
    let root =
        arena.borrow_mut().linear(vec![(x_id, 3.0), (x_id, -2.0), (y_id, 5.0), (y_id, -5.0)], 4.0);
    let expression = oximo_expr::Expr::new(root, arena).try_affine().unwrap();
    let upper = constraint!(model, upper, expression <= 5.0);
    let lower = constraint!(model, lower, expression >= 3.0);
    let equality = constraint!(model, equality, expression == 4.0);
    let range = constraint!(model, range, -2.0 <= expression <= 4.0);
    let constant = constraint!(model, constant, 0.0 * x + 7.0 <= 8.0);
    let free = constraint!(model, free, f64::NEG_INFINITY <= expression <= f64::INFINITY);

    let oximo_core::RangeConstraintHandles::Interval(free) = free else {
        panic!("affine free interval")
    };
    let oximo_core::RangeConstraintHandles::Interval(range) = range else {
        panic!("affine interval")
    };

    let inactive = constraint!(model, inactive, expression <= -100.0);
    model.set_constraint_active(inactive, false).unwrap();
    let arena = model.__sum_context();
    let root = arena.borrow_mut().linear(vec![(x_id, 3.0), (x_id, -2.0)], 2.0);
    objective!(model, Min, oximo_expr::Expr::new(root, arena));

    let mut built = build(&model, &GurobiOptions::default(), &quiet_env()).unwrap();
    built.model.update().unwrap();
    assert_eq!(built.model.get_attr(attr::NumConstrs).unwrap(), 5);
    assert_eq!(built.model.get_attr(attr::NumVars).unwrap(), 3);
    assert!(built.constrs[free.index()].rows.is_empty());
    assert!(built.constrs[inactive.index()].rows.is_empty());

    for (source, sense, rhs) in [
        (upper.index(), gurobi_rs::ConstrSense::Less, 1.0),
        (lower.index(), gurobi_rs::ConstrSense::Greater, -1.0),
        (equality.index(), gurobi_rs::ConstrSense::Equal, 0.0),
    ] {
        let [GurobiRow::Lin(row)] = built.constrs[source].rows.as_slice() else {
            panic!("native affine row")
        };
        assert_eq!(built.model.get_coeff(&built.vars[0], row).unwrap(), 1.0);
        assert_eq!(built.model.get_coeff(&built.vars[1], row).unwrap(), 0.0);
        assert_eq!(built.model.get_obj_attr(attr::Sense, row).unwrap(), sense);
        assert_eq!(built.model.get_obj_attr(attr::RHS, row).unwrap(), rhs);
    }
    let [GurobiRow::Lin(row)] = built.constrs[constant.index()].rows.as_slice() else {
        panic!("constant row")
    };

    assert_eq!(built.model.get_obj_attr(attr::RHS, row).unwrap(), 1.0);

    let [slack] = built.constrs[range.index()].variables.as_slice() else { panic!("range slack") };
    assert_eq!(built.model.get_obj_attr(attr::LB, slack).unwrap(), 0.0);
    assert_eq!(built.model.get_obj_attr(attr::UB, slack).unwrap(), 6.0);
    assert_eq!(built.model.get_obj_attr(attr::Obj, &built.vars[0]).unwrap(), 1.0);
    assert_eq!(built.model.get_attr(attr::ObjCon).unwrap(), 2.0);
}

#[test]
fn polynomial_refresh_clears_quadratics_without_losing_small_replacement_coefficients() {
    let model = Model::new("absolute objective refresh");
    variable!(model, -1.0 <= x <= 1.0);
    param!(model, p = 1.0);
    objective!(model, Min, p * x * x + (p - 1.0) * x + 3.0 * p);

    let env = quiet_env();
    let mut preparation = PersistentPreparation::default();
    let mut built = {
        let prepared = preparation.prepare(&model).unwrap();
        build_prepared(&model, &GurobiOptions::default(), &env, &prepared).unwrap()
    };
    preparation.commit();

    for coefficient in [2.0, 0.0, -1.0, 1e16, 1.0] {
        model.set_param(p, coefficient).unwrap();
        let prepared = preparation.prepare(&model).unwrap();
        assert!(built.refresh(&prepared, preparation.change()).unwrap());
        let native =
            quadratic_coefficients(built.model.get_qobjective_terms().unwrap(), &built.vars);
        let expected = if coefficient == 0.0 {
            FxHashMap::default()
        } else {
            FxHashMap::from_iter([((0, 0), coefficient)])
        };
        assert_eq!(native, expected);
        assert_eq!(built.model.get_obj_attr(attr::Obj, &built.vars[0]).unwrap(), coefficient - 1.0);
        assert_eq!(built.model.get_attr(attr::ObjCon).unwrap(), 3.0 * coefficient);
        assert!(built.objective_generated.is_empty() && built.objective_variables.is_empty());
        drop(prepared);
        preparation.commit();
    }
}

#[test]
fn batched_refresh_survives_quadratic_recreation_and_bound_updates() {
    let model = Model::new("mixed batched refresh");
    param!(model, p = 1.0);
    variable!(model, p <= x <= 10.0);
    variable!(model, -10.0 <= y <= 10.0);
    let first = constraint!(model, first, p * x + y + p <= 8.0);
    let second = constraint!(model, second, x - p * y >= -2.0);
    let quadratic = constraint!(model, quadratic, p * x * x <= 16.0);
    objective!(model, Min, p * x * x + 2.0 * p * y + 3.0);

    let env = quiet_env();
    let mut preparation = PersistentPreparation::default();
    let mut built = {
        let prepared = preparation.prepare(&model).unwrap();
        build_prepared(&model, &GurobiOptions::default(), &env, &prepared).unwrap()
    };
    preparation.commit();
    model.set_param(p, 2.0).unwrap();
    let prepared = preparation.prepare(&model).unwrap();
    assert!(built.refresh(&prepared, preparation.change()).unwrap());
    assert_eq!(built.model.get_obj_attr(attr::LB, &built.vars[0]).unwrap(), 2.0);
    assert_eq!(built.model.get_obj_attr(attr::UB, &built.vars[0]).unwrap(), 10.0);

    for (source, x_coefficient, y_coefficient, rhs) in
        [(first.index(), 2.0, 1.0, 6.0), (second.index(), 1.0, -2.0, -2.0)]
    {
        let [GurobiRow::Lin(row)] = built.constrs[source].rows.as_slice() else {
            panic!("linear row")
        };
        assert_eq!(built.model.get_coeff(&built.vars[0], row).unwrap(), x_coefficient);
        assert_eq!(built.model.get_coeff(&built.vars[1], row).unwrap(), y_coefficient);
        assert_eq!(built.model.get_obj_attr(attr::RHS, row).unwrap(), rhs);
    }
    let [GurobiRow::Quad(row)] = built.constrs[quadratic.index()].rows.as_slice() else {
        panic!("quadratic row")
    };
    assert_eq!(
        quadratic_coefficients(built.model.get_qconstr_terms(row).unwrap().quadratic, &built.vars),
        FxHashMap::from_iter([((0, 0), 2.0)]),
    );
    assert_eq!(built.model.get_obj_attr(attr::Obj, &built.vars[1]).unwrap(), 4.0);
}

#[test]
fn shared_operator_proofs_preserve_numeric_and_hidden_operator_checks() {
    let model = Model::new("zero denominator");
    variable!(model, x);
    param!(model, p = 1.0);
    objective!(model, Min, x / p);

    for value in [1.0, 0.0] {
        p.set_param_value(value).unwrap();
        let prepared = LoweringContext::new(&model).unwrap();
        crate::PROFILE.check_prepared(&model, &prepared).unwrap().into_result().unwrap();
        let result = validate_model_operators(&prepared);
        if value == 0.0 {
            assert!(result.unwrap_err().to_string().contains("division by zero"));
        } else {
            result.unwrap();
        }
    }
    let hidden = x.asin().powi(0);
    objective!(model, Min, hidden);

    let prepared = LoweringContext::new(&model).unwrap();
    let report = crate::PROFILE.check_prepared(&model, &prepared).unwrap();
    assert!(report.into_result().is_err());
    assert!(validate_model_operators(&prepared).is_err());
}

#[test]
fn indicator_divisions_are_validated_before_native_rows() {
    let model = Model::new("indicator zero denominator");
    variable!(model, b, Binary);
    variable!(model, x);
    param!(model, p = 1.0);
    let lhs = (x + (x / p).powi(0)).try_affine().unwrap();
    let row = indicator_constraint!(model, row, b == 1 => lhs <= 1.0);
    objective!(model, Min, x);

    for value in [1.0, 0.0] {
        p.set_param_value(value).unwrap();
        let prepared = LoweringContext::new(&model).unwrap();
        let result = validate_prepared(&model, &prepared);
        if value == 0.0 {
            assert!(
                matches!(result, Err(SolverError::Backend(message)) if message.contains("division by zero"))
            );
        } else {
            result.unwrap();
        }
    }

    model.set_constraint_active(row, false).unwrap();
    let prepared = LoweringContext::new(&model).unwrap();
    validate_prepared(&model, &prepared).unwrap();
}

#[test]
fn sparse_conversion_matches_generic_coefficients_and_evaluations() {
    let model = Model::new("quadratic coefficient convention");
    variable!(model, -2.0 <= x <= 2.0);
    variable!(model, -2.0 <= y <= 2.0);
    param!(model, p = 2.0);
    let shared = x + 2.0 * y + 1.0;
    let root = p * shared * shared + x * y - 3.0 * y * y + 2.0 * x - 7.0;
    objective!(model, Min, root);

    let env = quiet_env();
    let prepared = LoweringContext::new(&model).unwrap();
    let mut native = gurobi_rs::Model::with_env("coefficients", &env).unwrap();
    let vars = add_variables(&mut native, prepared.variables()).unwrap();
    let q = prepared.quadratic(root.id()).unwrap();
    let actual = quadratic_expression(&q, &vars);
    let mut ctx = LoweringCtx::new(&mut native, &vars, 0);
    let LoweredExpr::Quadratic(expected) = lower(prepared.arena(), root.id(), &mut ctx).unwrap()
    else {
        panic!("quadratic reference")
    };
    assert!(ctx.generated.is_empty());

    let (aq, al) = actual.into_parts();
    let (eq, el) = expected.into_parts();
    assert_eq!(aq, eq);
    assert_eq!(al.into_parts(), el.into_parts());

    for point in [[0.0, 0.0], [1.0, -0.5], [-1.25, 0.75]] {
        let expected = 2.0 * (point[0] + 2.0 * point[1] + 1.0_f64).powi(2) + point[0] * point[1]
            - 3.0 * point[1] * point[1]
            + 2.0 * point[0]
            - 7.0;
        let actual = q.constant
            + q.linear.iter().map(|&(v, c)| c * point[v.index()]).sum::<f64>()
            + q.hessian
                .iter()
                .map(|&(v, w, c)| {
                    (if v == w { c / 2.0 } else { c }) * point[v.index()] * point[w.index()]
                })
                .sum::<f64>();
        assert!((expected - actual).abs() < 1e-12);
    }
}

#[test]
fn converter_preserves_small_nonzero_coefficients_and_cancellation() {
    let env = quiet_env();
    let mut native = gurobi_rs::Model::with_env("exact zeros", &env).unwrap();
    let x = native.add_var("x", Continuous, 0.0, -1.0, 1.0, []).unwrap();
    let q = quadratic_expression(
        &QuadraticTerms {
            hessian: vec![(VarId(0), VarId(0), 2e-100), (VarId(0), VarId(0), 0.0)],
            linear: vec![(VarId(0), 1e-100), (VarId(0), 0.0)],
            constant: 3.0,
        },
        &[x],
    );
    let (quadratic, linear) = q.into_parts();
    assert_eq!(quadratic[&(x, x)], 1e-100);

    let (linear, constant) = linear.into_parts();
    assert_eq!(linear[&x], 1e-100);
    assert_eq!(constant, 3.0);

    let model = Model::new("cancelled quadratic");
    variable!(model, -1.0 <= y <= 1.0);
    let square = y * y;
    let cancelled = constraint!(model, cancel, square - square + y <= 1.0);
    objective!(model, Max, square - square + 3.0);

    let mut built = build(&model, &GurobiOptions::default(), &env).unwrap();
    assert!(matches!(built.constrs[cancelled.index()].rows.as_slice(), [GurobiRow::Lin(_)]));
    assert_eq!(built.model.get_qobjective_terms().unwrap().len(), 0);

    built.model.optimize().unwrap();
    assert!((built.model.get_attr(attr::ObjVal).unwrap() - 3.0).abs() < 1e-8);
    assert_eq!(built.model.get_attr(attr::NumGenConstrs).unwrap(), 0);
}

#[test]
fn quadratic_roots_do_not_hide_unsupported_active_expressions() {
    let model = Model::new("quadratic plus unsupported operator");
    variable!(model, x >= 1.0);
    objective!(model, Min, x * x);
    constraint!(model, unsupported, x.cbrt() <= 2.0);

    let env = quiet_env();
    let result = build(&model, &GurobiOptions::default(), &env);
    let Err(SolverError::IncompatibleModel(report)) = result else { panic!("compatibility error") };
    assert!(report.errors.iter().any(|diagnostic| {
        diagnostic.origins.iter().any(|origin| origin.location.contains("unsupported"))
    }));
}

#[test]
fn missing_declarations_retain_compatibility_diagnostics_before_native_setup() {
    let model = Model::new("missing objective");
    let error = solve(&model, &GurobiOptions::default()).unwrap_err();
    assert!(matches!(error, SolverError::IncompatibleModel(_)));

    let mut persistent = crate::GurobiPersistent::new();
    let error = oximo_solver::Solver::solve(&mut persistent, &model, &GurobiOptions::default())
        .unwrap_err();
    assert!(matches!(error, SolverError::IncompatibleModel(_)));
}

#[test]
fn constant_division_by_zero_is_rejected_before_native_setup() {
    let model = Model::new("zero denominator preflight");
    variable!(model, x >= 1.0);
    param!(model, denominator = 0.0);
    objective!(model, Min, x * x / denominator);

    let prepared = LoweringContext::new(&model).unwrap();
    let error = validate_prepared(&model, &prepared).unwrap_err();
    assert!(error.to_string().contains("division by zero"));

    let error = solve(&model, &GurobiOptions::default()).unwrap_err();
    assert!(error.to_string().contains("division by zero"));
}

#[test]
fn deep_prepared_quadratics_are_stack_safe() {
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(|| {
            let model = Model::new("deep direct quadratic");
            variable!(model, -1.0 <= x <= 1.0);
            let cell = model.__sum_context();
            let mut root = (x * x).id();
            {
                let mut arena = cell.borrow_mut();
                let one = arena.constant(1.0);
                for _ in 0..20_000 {
                    root = arena.push(oximo_expr::ExprNode::Add(vec![root, one].into()));
                }
            }
            model.__minimize(oximo_expr::Expr::new(root, cell));
            let env = quiet_env();
            let mut built = build(&model, &GurobiOptions::default(), &env).unwrap();
            built.model.update().unwrap();
            assert_eq!(built.model.get_attr(attr::NumQNZs).unwrap(), 1);
            assert_eq!(built.model.get_attr(attr::ObjCon).unwrap(), 20_000.0);
            assert_eq!(built.model.get_attr(attr::NumGenConstrs).unwrap(), 0);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn direct_quadratic_rows_keep_iis_provenance() {
    let model = Model::new("quadratic IIS provenance");
    variable!(model, -1.0 <= x <= 1.0);
    let row = constraint!(model, impossible, x * x <= -1.0);
    objective!(model, Min, x * x + 3.0);

    let env = quiet_env();
    let mut built = build(&model, &GurobiOptions::default(), &env).unwrap();
    built.model.optimize().unwrap();
    let iis = compute_iis_resident(&mut built).unwrap();
    assert!(iis.constraints.contains(&row.row_id()));
    assert!(built.constrs[row.index()].generated.is_empty());
}

#[test]
fn quadratic_rows_keep_bounds_names_constants_and_source_slots() {
    let model = Model::new("quadratic bound forms");
    variable!(model, -2.0 <= x <= 2.0);
    objective!(model, Min, x * x + 3.0);

    let env = quiet_env();
    let root = x * x + 5.0;
    for (i, (lower, upper, active)) in [
        (5.0, 5.0, true),
        (5.0, f64::INFINITY, true),
        (f64::NEG_INFINITY, 9.0, true),
        (5.0, 9.0, true),
        (f64::NEG_INFINITY, f64::INFINITY, true),
        (5.0, 9.0, false),
    ]
    .into_iter()
    .enumerate()
    {
        let row = model.add_constraint(
            format!("row{i}"),
            Constraint::new(root.into_function(), Interval { lower, upper }),
        );
        model.set_constraint_active(row, active).unwrap();
    }
    let prepared = LoweringContext::new(&model).unwrap();
    let mut native = gurobi_rs::Model::with_env("bound forms", &env).unwrap();
    let vars = add_variables(&mut native, prepared.variables()).unwrap();
    let handles = add_constraints(
        &prepared,
        prepared.constraints().algebraic(),
        &mut native,
        &vars,
        &mut 0,
        true,
    )
    .unwrap();
    native.update().unwrap();
    assert_eq!(handles.iter().map(|h| h.rows.len()).collect::<Vec<_>>(), [1, 1, 1, 2, 0, 0]);
    assert!(handles.iter().all(|h| h.variables.is_empty() && h.generated.is_empty()));
    assert_eq!(native.get_attr(attr::NumVars).unwrap(), 1);
    assert_eq!(native.get_attr(attr::NumGenConstrs).unwrap(), 0);

    let mut names = Vec::new();
    let mut bounds = Vec::new();
    for handle in handles {
        for row in handle.rows {
            let GurobiRow::Quad(row) = row else { panic!("direct quadratic row") };
            names.push(native.get_obj_attr(attr::QCName, &row).unwrap());
            bounds.push(native.get_obj_attr(attr::QCRHS, &row).unwrap());
        }
    }
    assert_eq!(names, ["row0", "row1", "row2", "row3_0", "row3_1"]);
    assert_eq!(bounds, [0.0, 0.0, 4.0, 0.0, 4.0]);
}

#[test]
fn objectives_preserve_sense_offset_and_have_no_nonlinear_auxiliaries() {
    let env = quiet_env();
    for integer in [false, true] {
        for maximize in [false, true] {
            let model = Model::new("quadratic objective");
            let x = if integer {
                model.__var("x").integer().lb(-1.0).ub(1.0).build()
            } else {
                model.__var("x").lb(-1.0).ub(1.0).build()
            };
            if maximize {
                model.__maximize(3.0 - x * x);
            } else {
                model.__minimize(3.0 + x * x);
            }
            let mut built = build(&model, &GurobiOptions::default(), &env).unwrap();
            built.model.update().unwrap();
            assert_eq!(built.model.get_attr(attr::NumVars).unwrap(), 1);
            assert_eq!(built.model.get_attr(attr::NumGenConstrs).unwrap(), 0);
            assert_eq!(built.model.get_attr(attr::ObjCon).unwrap(), 3.0);
            assert!(built.objective_variables.is_empty() && built.objective_generated.is_empty());
            assert_eq!(built.obj_constant, 0.0);

            built.model.optimize().unwrap();
            assert!((built.model.get_attr(attr::ObjVal).unwrap() - 3.0).abs() < 1e-8);
        }
    }
}

#[test]
fn parameter_refresh_replaces_quadratics_and_clears_cancelled_terms() {
    let model = Model::new("parameterized quadratic recreation");
    variable!(model, -1.0 <= x <= 1.0);
    param!(model, p = 1.0);
    let row = constraint!(model, row, p * x * x <= 1.0);
    objective!(model, Min, p * x * x + 3.0);

    let env = quiet_env();
    let mut preparation = PersistentPreparation::default();
    let mut built = {
        let prepared = preparation.prepare(&model).unwrap();
        build_prepared(&model, &GurobiOptions::default(), &env, &prepared).unwrap()
    };
    preparation.commit();
    for coefficient in [2.0, 0.0, 1.0] {
        model.set_param(p, coefficient).unwrap();
        let prepared = preparation.prepare(&model).unwrap();
        assert!(matches!(preparation.change(), PreparationChange::Selected(_)));
        assert!(built.refresh(&prepared, preparation.change()).unwrap());
        assert_eq!(built.model.get_attr(attr::NumVars).unwrap(), 1);
        assert_eq!(built.model.get_attr(attr::NumGenConstrs).unwrap(), 0);
        assert!(built.constrs[row.index()].generated.is_empty());
        assert_eq!(built.constrs[row.index()].variables, Vec::<gurobi_rs::Var>::new());
        assert_eq!(built.model.get_attr(attr::ObjCon).unwrap(), 3.0);
        assert_eq!(built.model.get_attr(attr::NumQNZs).unwrap(), i32::from(coefficient != 0.0));

        built.model.optimize().unwrap();
        assert!((built.model.get_attr(attr::ObjVal).unwrap() - 3.0).abs() < 1e-8);

        drop(prepared);
        preparation.commit();
    }
}

#[test]
fn persistent_parameter_powers_transition_between_polynomial_and_nonlinear_paths() {
    use oximo_solver::Solver;
    let model = Model::new("parameter power transitions");
    variable!(model, 1.0 <= x <= 2.0);
    param!(model, power = 2.0);
    constraint!(model, row, x.pow(power) <= 4.0);
    objective!(model, Max, x.pow(power));

    let mut solver = crate::GurobiPersistent::new();
    for (value, expected) in [(2.0, 4.0), (3.0, 4.0), (1.0, 2.0), (2.0, 4.0)] {
        model.set_param(power, value).unwrap();
        let result = solver.solve(&model, &GurobiOptions::default()).unwrap();
        assert!((result.objective().unwrap() - expected).abs() < 1e-6);
    }
}
