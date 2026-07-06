//! End-to-end checks that the macros expand to the same model the
//! builder API would produce.

use oximo_core::prelude::*;

#[test]
fn scalar_variables_bounds_and_objective() {
    let m = Model::new("scalar");
    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 10.0);
    variable!(m, z, Bin);

    constraint!(m, cap, x + y + z <= 10.0);
    objective!(m, Max, x + 2.0 * y);

    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.num_constraints(), 1);
    assert_eq!(m.kind(), ModelKind::MILP);
}

#[test]
fn variable_bounds_apply() {
    let m = Model::new("bounds");
    variable!(m, 1.5 <= x <= 4.0);
    objective!(m, Min, x);
    let vars = m.variables();
    let v = &vars[0];
    assert!((v.lb - 1.5).abs() < f64::EPSILON);
    assert!((v.ub - 4.0).abs() < f64::EPSILON);
}

#[test]
fn indexed_variable_sum_and_family() {
    let m = Model::new("indexed");
    let assets = Set::range(0..3);
    variable!(m, 0.0 <= w[i in assets] <= 1.0);

    constraint!(m, budget, sum!(w[i] for i in assets) == 1);
    constraint!(m, ub[i in 0..3], w[i] <= 1.0);
    objective!(m, Min, sum!(w[i] for i in assets));

    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.num_constraints(), 1 + 3);
    assert_eq!(m.kind(), ModelKind::LP);
    assert!(m.constraint_id("budget").is_some());
    assert!(m.constraint_id("ub[0]").is_some());
    assert!(m.constraint_id("ub[2]").is_some());
}

#[test]
fn anonymous_constraints_are_auto_named() {
    let m = Model::new("anon");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, x + y >= 1.0);
    constraint!(m, x - y <= 2.0);
    assert_eq!(m.num_constraints(), 2);
    assert!(m.constraint_id("_c0").is_some());
    assert!(m.constraint_id("_c1").is_some());
}

#[test]
fn nested_sum_is_quadratic() {
    let m = Model::new("qp");
    let n = Set::range(0..2);
    let sigma = [[1.0, 0.2], [0.2, 1.0]];
    variable!(m, w[i in n] >= 0.0);
    constraint!(m, budget, sum!(w[i] for i in n) == 1);
    objective!(m, Min, sum!(sigma[i][j] * w[i] * w[j] for i in n, j in n));
    assert_eq!(m.kind(), ModelKind::QP);
}

#[test]
fn filtered_sum_skips_nonmatching_keys() {
    let m = Model::new("filter");
    let items = Set::range(0..5);
    variable!(m, x[i in items] >= 0.0);
    objective!(m, Max, sum!(x[i] for i in items if i % 2 == 0));
    constraint!(m, evens, sum!(x[i] for i in items if i % 2 == 0) <= 3.0);

    let arena = m.arena();
    let obj = m.try_objective().unwrap();
    let terms = oximo_expr::extract_linear(&arena, obj.expr).expect("linear");
    assert_eq!(terms.coeffs.len(), 3);
}

#[test]
fn large_sum_builds_correctly() {
    const N: usize = 2000;
    let m = Model::new("bigsum");
    let c: Vec<f64> = (0..N).map(|i| [1.0, 2.0, 3.0, 4.0, 5.0][i % 5]).collect();
    variable!(m, x[i in 0..N] >= 0.0);
    objective!(m, Min, sum!(c[i] * x[i] for i in 0..N));

    let arena = m.arena();
    let obj = m.try_objective().unwrap();
    let terms = oximo_expr::extract_linear(&arena, obj.expr).expect("linear");
    assert_eq!(terms.coeffs.len(), N);
    let total: f64 = terms.coeffs.iter().map(|(_, v)| v).sum();
    let expected: f64 = c.iter().sum();
    assert!((total - expected).abs() < 1e-6);
}

#[test]
fn large_filtered_sum_builds_correctly() {
    const N: usize = 1000;
    let m = Model::new("bigfilter");
    variable!(m, x[i in 0..N] >= 0.0);
    objective!(m, Max, sum!(x[i] for i in 0..N if i % 2 == 0));

    let arena = m.arena();
    let obj = m.try_objective().unwrap();
    let terms = oximo_expr::extract_linear(&arena, obj.expr).expect("linear");
    assert_eq!(terms.coeffs.len(), N / 2);
}

// --- Two-sided range constraints: `lo <= e <= hi` lowers to `_lo` + `_hi` rows.

#[test]
fn range_constraint_with_const_bounds_collapses_to_one_row() {
    let m = Model::new("range");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, band, 1.0 <= x + y <= 3.0);

    assert_eq!(m.num_constraints(), 1);
    let cons = m.constraints();
    let band = &cons[m.constraint_id("band").expect("range row").index()];
    assert!(band.is_range());
    assert!((band.lower - 1.0).abs() < f64::EPSILON);
    assert!((band.upper - 3.0).abs() < f64::EPSILON);
}

#[test]
fn range_constraint_ge_form_is_equivalent() {
    let m = Model::new("rangege");
    variable!(m, x >= 0.0);
    constraint!(m, b, 3.0 >= x >= 1.0);

    assert_eq!(m.num_constraints(), 1);
    let cons = m.constraints();
    let b = &cons[m.constraint_id("b").unwrap().index()];
    assert!(b.is_range());
    assert!((b.lower - 1.0).abs() < f64::EPSILON);
    assert!((b.upper - 3.0).abs() < f64::EPSILON);
}

#[test]
fn range_constraint_with_expr_bounds_falls_back_to_two_rows() {
    let m = Model::new("rangeexpr");
    variable!(m, x >= 0.0);
    variable!(m, lo >= 0.0);
    variable!(m, hi >= 0.0);
    constraint!(m, band, lo <= x <= hi);

    assert_eq!(m.num_constraints(), 2);
    let cons = m.constraints();
    let lo_row = &cons[m.constraint_id("band_lo").expect("lo row").index()];
    let hi_row = &cons[m.constraint_id("band_hi").expect("hi row").index()];
    assert_eq!(lo_row.as_single().map(|(s, _)| s), Some(Sense::Ge));
    assert_eq!(hi_row.as_single().map(|(s, _)| s), Some(Sense::Le));
}

#[test]
fn anonymous_range_makes_one_auto_row() {
    let m = Model::new("anonr");
    variable!(m, x >= 0.0);
    constraint!(m, 0.0 <= x <= 5.0);
    assert_eq!(m.num_constraints(), 1);
    let c = &m.constraints()[m.constraint_id("_c0").expect("auto row").index()];
    assert!(c.is_range());
}

#[test]
fn family_range_const_bounds_makes_one_row_per_element() {
    let m = Model::new("famr");
    let lo = [1.0, 2.0, 3.0];
    let hi = [4.0, 5.0, 6.0];
    variable!(m, x[i in 0..3] >= 0.0);
    constraint!(m, cap[i in 0..3], lo[i] <= x[i] <= hi[i]);

    assert_eq!(m.num_constraints(), 3);
    let cons = m.constraints();
    let c = &cons[m.constraint_id("cap[1]").expect("range row").index()];
    assert!(c.is_range());
    assert!((c.lower - 2.0).abs() < f64::EPSILON);
    assert!((c.upper - 5.0).abs() < f64::EPSILON);
}

#[test]
fn inverted_range_stays_a_range_not_an_equality() {
    let m = Model::new("inv");
    variable!(m, x);
    constraint!(m, c, 5.0 <= x <= 1.0);
    assert_eq!(m.num_constraints(), 1);
    let c = &m.constraints()[m.constraint_id("c").unwrap().index()];
    assert!(c.is_range());
    assert_eq!(c.as_single(), None);
}

#[test]
fn nonlinear_range_falls_back_to_two_rows() {
    let m = Model::new("nlr");
    variable!(m, x);
    constraint!(m, c, 1.0 <= x * x <= 4.0);
    assert_eq!(m.num_constraints(), 2);
    let cons = m.constraints();
    let lo = &cons[m.constraint_id("c_lo").expect("lo row").index()];
    let hi = &cons[m.constraint_id("c_hi").expect("hi row").index()];
    assert!(!lo.is_range());
    assert_eq!(lo.as_single().map(|(s, _)| s), Some(Sense::Ge));
    assert_eq!(hi.as_single().map(|(s, _)| s), Some(Sense::Le));
}

#[test]
fn computed_name_range_collapses_to_one_row() {
    let m = Model::new("crange");
    variable!(m, x >= 0.0);
    let tag = "band";
    constraint!(m, name = format!("{tag}"), 1.0 <= x <= 2.0);
    assert_eq!(m.num_constraints(), 1);
    let c = &m.constraints()[m.constraint_id("band").expect("range row").index()];
    assert!(c.is_range());
}

#[test]
fn semicontinuous_domain_sets_threshold() {
    let m = Model::new("semi");
    variable!(m, x <= 10.0, SemiCont(2.0));
    objective!(m, Min, x);
    let vars = m.variables();
    assert_eq!(vars[0].domain, Domain::SemiContinuous { threshold: 2.0 });
    assert!((vars[0].ub - 10.0).abs() < f64::EPSILON);
}

#[test]
fn semi_integer_domain_sets_threshold() {
    let m = Model::new("semii");
    variable!(m, y <= 5.0, SemiInteger(1.0));
    objective!(m, Min, y);
    let vars = m.variables();
    assert_eq!(vars[0].domain, Domain::SemiInteger { threshold: 1.0 });
    assert!(vars[0].domain.is_integer());
}

#[test]
fn keyword_bounds_match_relational() {
    let m = Model::new("kw_bounds");
    variable!(m, x, lb = 1.5, ub = 4.0);
    objective!(m, Min, x);
    let v = &m.variables()[0];
    assert!((v.lb - 1.5).abs() < f64::EPSILON);
    assert!((v.ub - 4.0).abs() < f64::EPSILON);
}

#[test]
fn keyword_domain_and_mixing() {
    let m = Model::new("kw_domain");
    variable!(m, va, lb = 0.0, domain = Int);
    variable!(m, vb, lb = 0.0, ub = 10.0, Int);
    variable!(m, vc, domain = SemiCont(2.0), ub = 10.0);
    objective!(m, Min, va + vb + vc);
    let vars = m.variables();
    assert_eq!(vars[0].domain, Domain::Integer);
    assert_eq!(vars[1].domain, Domain::Integer);
    assert!((vars[1].ub - 10.0).abs() < f64::EPSILON);
    assert_eq!(vars[2].domain, Domain::SemiContinuous { threshold: 2.0 });
    assert!((vars[2].ub - 10.0).abs() < f64::EPSILON);
}

#[test]
fn keyword_initial_and_fix() {
    let m = Model::new("kw_init_fix");
    variable!(m, p, lb = 0.0, initial = 3.0);
    variable!(m, q, fix = 5.0);
    objective!(m, Min, p + q);
    let v = m.variables();
    assert_eq!(v[0].initial, Some(3.0));
    assert!((v[1].lb - 5.0).abs() < f64::EPSILON);
    assert!((v[1].ub - 5.0).abs() < f64::EPSILON);
}

#[test]
fn keyword_indexed_bound_infers_key() {
    let m = Model::new("kw_indexed");
    let items = Set::range(0..3);
    let cap = [2.0, 4.0, 6.0];
    variable!(m, w[i in items], lb = 0.0, ub = cap[i]);
    assert_eq!(w.len(), 3);
    let vars = m.variables();
    assert!((vars[2].ub - 6.0).abs() < f64::EPSILON);
}

#[test]
fn domain_aliases_map_correctly() {
    let m = Model::new("aliases");
    variable!(m, va, Bin);
    variable!(m, vb, Binary);
    variable!(m, vc, Int);
    variable!(m, vd, Integer);
    variable!(m, ve, Real);
    variable!(m, vf, Cont);
    variable!(m, vg, Continuous);
    objective!(m, Min, va + vb + vc + vd + ve + vf + vg);

    let v = m.variables();
    assert_eq!(v[0].domain, Domain::Binary);
    assert_eq!(v[1].domain, Domain::Binary);
    assert_eq!(v[2].domain, Domain::Integer);
    assert_eq!(v[3].domain, Domain::Integer);
    assert_eq!(v[4].domain, Domain::Real);
    assert_eq!(v[5].domain, Domain::Real);
    assert_eq!(v[6].domain, Domain::Real);
}

#[test]
fn objective_sense_aliases_map_correctly() {
    use ObjectiveSense::{Maximize, Minimize};

    let m = Model::new("o_min_long");
    variable!(m, x >= 0.0);
    objective!(m, Minimize, x);
    assert_eq!(m.objective().as_ref().unwrap().sense, Minimize);

    let m = Model::new("o_min_lower");
    variable!(m, x >= 0.0);
    objective!(m, min, x);
    assert_eq!(m.objective().as_ref().unwrap().sense, Minimize);

    let m = Model::new("o_max_long");
    variable!(m, x >= 0.0);
    objective!(m, Maximize, x);
    assert_eq!(m.objective().as_ref().unwrap().sense, Maximize);

    let m = Model::new("o_max_lower");
    variable!(m, x >= 0.0);
    objective!(m, max, x);
    assert_eq!(m.objective().as_ref().unwrap().sense, Maximize);
}

#[test]
fn feasibility_declares_a_problem_without_objective() {
    fn check(m: &Model, spelling: &str) {
        assert!(m.is_feasibility(), "{spelling}: expected feasibility");
        assert!(m.objective().is_none(), "{spelling}: no objective expr");
        assert!(m.ensure_objective_declared().is_ok(), "{spelling}");
        assert_eq!(m.kind(), ModelKind::LP, "{spelling}");
    }

    // All three spellings lower to `__feasibility()`.
    let m = Model::new("feas");
    variable!(m, 0.0 <= x <= 1.0);
    constraint!(m, c, x >= 0.5);
    objective!(m, Feasibility);
    check(&m, "Feasibility");

    let m = Model::new("feas");
    variable!(m, 0.0 <= x <= 1.0);
    constraint!(m, c, x >= 0.5);
    objective!(m, feasibility);
    check(&m, "feasibility");

    let m = Model::new("feas");
    variable!(m, 0.0 <= x <= 1.0);
    constraint!(m, c, x >= 0.5);
    objective!(m, feas);
    check(&m, "feas");
}

#[test]
fn unset_objective_is_not_declared() {
    let m = Model::new("unset");
    variable!(m, x >= 0.0);
    constraint!(m, c, x <= 1.0);
    assert!(!m.is_feasibility());
    assert!(m.ensure_objective_declared().is_err());
}

#[test]
fn optimize_after_feasibility_clears_feasibility() {
    let m = Model::new("switch");
    variable!(m, x >= 0.0);
    objective!(m, Feasibility);
    assert!(m.is_feasibility());
    objective!(m, Min, x);
    assert!(!m.is_feasibility());
    assert!(m.objective().is_some());
    assert!(m.ensure_objective_declared().is_ok());
}

#[test]
fn param_handle_keeps_model_linear() {
    let m = Model::new("param");
    param!(m, rate = 0.05);
    variable!(m, x >= 0.0);
    constraint!(m, c, rate * x <= 1.0);
    objective!(m, Max, rate * x);
    assert_eq!(m.kind(), ModelKind::LP);
    assert!((m.param_value_of(rate).unwrap() - 0.05).abs() < f64::EPSILON);
}

#[test]
fn infers_string_key_without_annotation() {
    let m = Model::new("strkey");
    let plants = Set::strings(["a", "b", "c"]);
    variable!(m, x[p in plants] >= 0.0);
    constraint!(m, total, sum!(x[p] for p in plants) <= 1.0);
    objective!(m, Max, sum!(x[p] for p in plants));
    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn infers_tuple_key_without_annotation() {
    let m = Model::new("tuplekey");
    let plants = Set::strings(["p1", "p2"]);
    let times = Set::range(0..3); // Set<usize>
    let pt = &plants * &times; // Set<(String, usize)>
    variable!(m, b[(p, t) in pt] >= 0.0);
    constraint!(m, lim[(p, t) in pt], b[(p, t)] <= 10.0);
    assert_eq!(m.num_variables(), 6);
    assert_eq!(m.num_constraints(), 6);
}

#[test]
fn range_literal_defaults_usize_for_array_index() {
    let m = Model::new("arr");
    let cost = [1.0, 2.0, 3.0];
    variable!(m, x[i in 0..3] >= 0.0);
    // `i` defaults to usize, so it can index `cost` directly.
    objective!(m, Min, sum!(cost[i] * x[i] for i in 0..3));
    assert_eq!(m.num_variables(), 3);
}

#[test]
fn named_integer_set_infers_usize() {
    let m = Model::new("intset");
    let days = Set::range(0..4); // Set<usize>
    let demand = [5.0, 3.0, 8.0, 2.0];
    variable!(m, y[d in days] >= 0.0);
    constraint!(m, meet[d in days], y[d] >= demand[d]);
    assert_eq!(m.num_constraints(), 4);
}

#[test]
fn index_dependent_bound_infers_key() {
    let m = Model::new("bound");
    let items = Set::range(0..3);
    let cap = [2.0, 4.0, 6.0];
    variable!(m, 0.0 <= w[i in items] <= cap[i]);
    assert_eq!(w.len(), 3);
    let vars = m.variables();
    assert!((vars[2].ub - 6.0).abs() < f64::EPSILON);
}

// --- Multi-index access sugar: `q[i, j, k]` == `q[(&i, &j, &k)]`.

#[test]
fn multi_index_sugar_builds_the_family() {
    let m = Model::new("sugar");
    let hp = Set::strings(["H1", "H2"]);
    let cp = Set::strings(["C1", "C2"]);
    let st = Set::range(0..2);
    let hcs = &(&hp * &cp) * &st;
    variable!(m, q[(i, j, k) in hcs] >= 0.0);
    constraint!(m, lim[(i, j, k) in hcs], q[i, j, k] <= 10.0);
    objective!(m, Max, sum!(q[i, j, k] for (i, j, k) in hcs));
    assert_eq!(m.num_variables(), 8);
    assert_eq!(m.num_constraints(), 8);
    assert!(m.constraint_id("lim[H1,C2,1]").is_some());
}

#[test]
fn multi_index_sugar_allows_key_reuse() {
    let m = Model::new("reuse");
    let p = Set::strings(["a", "b"]);
    let n = Set::range(0..2);
    let pn = &p * &n;
    variable!(m, s[(pp, nn) in pn] >= 0.0);
    constraint!(m, c[(pp, nn) in pn], s[pp, nn] + s[pp, nn] <= 5.0);
    assert_eq!(m.num_constraints(), 4);
}

#[test]
fn multi_bind_declaration_not_mangled() {
    let m = Model::new("multibind");
    variable!(m, b[i in 0..2, n in 0..3] >= 0.0);
    constraint!(m, lim[i in 0..2, n in 0..3], b[i, n] <= 1.0);
    assert_eq!(m.num_variables(), 6);
    assert_eq!(m.num_constraints(), 6);
}

#[test]
fn computed_constraint_name_in_loop() {
    let m = Model::new("named_loop");
    let labels = ["a", "b", "c"];
    variable!(m, x[i in 0..3] >= 0.0);
    for (i, nm) in labels.iter().enumerate() {
        constraint!(m, name = format!("cap_{nm}"), x[i] <= 1.0);
    }
    constraint!(m, name = "fixed", x[0] >= 0.0);

    assert_eq!(m.num_constraints(), 4);
    assert!(m.constraint_id("cap_a").is_some());
    assert!(m.constraint_id("cap_c").is_some());
    assert!(m.constraint_id("fixed").is_some());
    assert!(m.constraint_id("_c0").is_none());
}

// --- Filtered families

#[test]
fn filtered_constraint_family_keeps_matching_keys() {
    let m = Model::new("ffam");
    variable!(m, x[i in 0..5] >= 0.0);
    constraint!(m, evens[i in 0..5 if i % 2 == 0], x[i] <= 1.0);
    assert_eq!(m.num_constraints(), 3);
    assert!(m.constraint_id("evens[0]").is_some());
    assert!(m.constraint_id("evens[2]").is_some());
    assert!(m.constraint_id("evens[4]").is_some());
    assert!(m.constraint_id("evens[1]").is_none());
    assert!(m.constraint_id("evens[3]").is_none());
}

#[test]
fn filtered_variable_family_only_builds_matching_keys() {
    let m = Model::new("fvar");
    variable!(m, x[i in 0..5 if i % 2 == 0] >= 0.0);
    assert_eq!(m.num_variables(), 3);
    assert_eq!(x.len(), 3);
    objective!(m, Min, x[0] + x[2] + x[4]);
    assert_eq!(m.kind(), ModelKind::LP);
}

#[test]
fn filtered_tuple_family_uses_cross_index_condition() {
    let m = Model::new("ftuple");
    let rows = Set::range(0..3);
    let cols = Set::range(0..3);
    let rc = &rows * &cols;
    variable!(m, y[(i, j) in rc] >= 0.0);
    constraint!(m, diag[(i, j) in rc if i == j], y[i, j] <= 1.0);
    assert_eq!(m.num_constraints(), 3);
    assert!(m.constraint_id("diag[0,0]").is_some());
    assert!(m.constraint_id("diag[1,1]").is_some());
    assert!(m.constraint_id("diag[2,2]").is_some());
    assert!(m.constraint_id("diag[0,1]").is_none());
}

#[test]
fn filtered_family_reads_external_data() {
    let m = Model::new("fdata");
    let unit_of = [0_usize, 0, 1, 1, 2];
    variable!(m, w[i in 0..5] >= 0.0);
    constraint!(m, only_unit1[i in 0..5 if unit_of[i] == 1], w[i] <= 1.0);
    assert_eq!(m.num_constraints(), 2);
    assert!(m.constraint_id("only_unit1[2]").is_some());
    assert!(m.constraint_id("only_unit1[3]").is_some());
    assert!(m.constraint_id("only_unit1[0]").is_none());
}

#[test]
fn filtered_string_family_drops_keys() {
    let m = Model::new("fstr");
    let plants = Set::strings(["a", "skip", "c"]);
    variable!(m, x[p in plants] >= 0.0);
    constraint!(m, keep[p in plants if p != "skip"], x[p] <= 1.0);
    assert_eq!(m.num_constraints(), 2);
    assert!(m.constraint_id("keep[a]").is_some());
    assert!(m.constraint_id("keep[c]").is_some());
    assert!(m.constraint_id("keep[skip]").is_none());
}

#[test]
fn index_sugar_leaves_arrays_untouched() {
    let m = Model::new("arrays");
    let cost = [3.0, 5.0];
    let mat = [[1.0, 0.0], [0.0, 1.0]];
    variable!(m, x[i in 0..2] >= 0.0);
    constraint!(m, c, sum!(cost[i] * x[i] for i in 0..2) <= 100.0);
    objective!(
        m,
        Max,
        sum!(mat[i][j] * x[i] for i in 0..2, j in 0..2) + sum!([3.0, 5.0][i] * x[i] for i in 0..2)
    );
    assert_eq!(m.num_variables(), 2);
    assert_eq!(m.num_constraints(), 1);
}

// --- `set!`: index-set construction as a macro.

#[test]
fn set_macro_plain_range_normalizes() {
    let m = Model::new("set_range");
    set!(items = 0..5);
    assert_eq!(items.len(), 5);
    variable!(m, x[i in items] >= 0.0);
    assert_eq!(x.len(), 5);
    assert_eq!(m.num_variables(), 5);
}

#[test]
fn set_macro_plain_product_borrows_operands() {
    let plants = Set::strings(["seattle", "san-diego"]);
    let markets = Set::strings(["nyc", "chi", "topeka"]);
    set!(routes = plants * markets);
    assert_eq!(routes.len(), 6);
    assert_eq!(plants.len(), 2);
    assert_eq!(markets.len(), 3);
}

#[test]
fn set_macro_plain_product_still_accepts_refs() {
    let plants = Set::strings(["a", "b"]);
    set!(routes = &plants * &plants);
    assert_eq!(routes.len(), 4);
}

#[test]
fn set_macro_comprehension_filters_product() {
    let plants = Set::strings(["seattle", "san-diego"]);
    // Single tuple pattern over a `*` product domain, `if` decoded by-value.
    set!(arcs = (p, q) in &plants * &plants if p != q);
    assert_eq!(arcs.len(), 2);

    let m = Model::new("set_arcs");
    variable!(m, f[(p, q) in arcs] >= 0.0);
    assert_eq!(f.len(), 2);
    assert_eq!(m.num_variables(), 2);
}

#[test]
fn set_macro_comprehension_multi_bind_builds_product() {
    set!(rc = i in 0..2, j in 0..2);
    assert_eq!(rc.len(), 4);

    set!(diag = i in 0..3, j in 0..3 if i == j);
    assert_eq!(diag.len(), 3);

    let m = Model::new("set_diag");
    variable!(m, y[(i, j) in diag] >= 0.0);
    constraint!(m, lim[(i, j) in diag], y[i, j] <= 1.0);
    assert_eq!(m.num_variables(), 3);
    assert_eq!(m.num_constraints(), 3);
}

// - soc_constraint! -----------------------------------------------------------

#[test]
fn soc_named_scalar() {
    let m = Model::new("soc_named");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    let id = soc_constraint!(m, cone, [x, y] <= t);
    assert_eq!(m.soc_constraint_id("cone"), Some(id));
    assert_eq!(m.num_soc_constraints(), 1);
    objective!(m, Min, t);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn soc_anonymous_auto_names() {
    let m = Model::new("soc_anon");
    variable!(m, x);
    variable!(m, t >= 0.0);
    soc_constraint!(m, [x] <= t);
    soc_constraint!(m, [x, t] <= t + 1.0);
    assert!(m.soc_constraint_id("_soc0").is_some());
    assert!(m.soc_constraint_id("_soc1").is_some());
    assert_eq!(m.num_soc_constraints(), 2);
}

#[test]
fn soc_computed_name() {
    let m = Model::new("soc_computed");
    variable!(m, x);
    variable!(m, t >= 0.0);
    let k = 7;
    soc_constraint!(m, name = format!("cone_{k}"), [x] <= t);
    assert!(m.soc_constraint_id("cone_7").is_some());
}

#[test]
fn soc_affine_terms_and_bound() {
    let m = Model::new("soc_affine");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    soc_constraint!(m, cone, [x - y, 2.0 * y + 1.0] <= t + 2.0);
    let socs = m.soc_constraints();
    assert_eq!(socs[0].terms.len(), 2);
}

#[test]
fn soc_family_over_range() {
    let m = Model::new("soc_family");
    let assets = Set::range(0..3);
    variable!(m, u[i in assets]);
    variable!(m, v[i in assets]);
    variable!(m, cap >= 0.0);
    soc_constraint!(m, risk[i in assets], [u[i], v[i]] <= cap);
    assert_eq!(m.num_soc_constraints(), 3);
    assert!(m.soc_constraint_id("risk[0]").is_some());
    assert!(m.soc_constraint_id("risk[2]").is_some());
    objective!(m, Min, cap);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn soc_filtered_family() {
    let m = Model::new("soc_filtered");
    variable!(m, w[i in 0..4]);
    variable!(m, t >= 0.0);
    soc_constraint!(m, even[i in 0..4 if i % 2 == 0], [w[i]] <= t);
    assert_eq!(m.num_soc_constraints(), 2);
    assert!(m.soc_constraint_id("even[0]").is_some());
    assert!(m.soc_constraint_id("even[2]").is_some());
    assert!(m.soc_constraint_id("even[1]").is_none());
}

#[test]
#[should_panic(expected = "non-affine term")]
fn soc_macro_rejects_quadratic_term() {
    let m = Model::new("soc_bad");
    variable!(m, x);
    variable!(m, t >= 0.0);
    soc_constraint!(m, cone, [x * x] <= t);
}
