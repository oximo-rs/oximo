//! Tests for SOC pattern detection (polynomial recognition) and the normalized
//! `SocForm` views backends translate from.

#![expect(clippy::many_single_char_names)]

use oximo_core::prelude::*;

fn detect_first(m: &Model) -> Option<SocForm> {
    let arena = m.arena();
    let vars = m.resolved_variables();
    let model_constraints = m.constraints();
    let constraints = model_constraints.algebraic();
    oximo_expr::extract_quadratic(&arena, constraints[0].lhs).and_then(|q| {
        let form = oximo_core::__detect_soc_from_quadratic(&vars, &constraints[0], &q);
        assert_eq!(
            oximo_core::__is_detected_soc_from_quadratic(&vars, &constraints[0], &q),
            form.is_some(),
        );
        form
    })
}

#[test]
fn accepts_scaled_sum_of_squares() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, 2.0 * x * x + 3.0 * y * y <= 5.0 * t * t);

    let form = detect_first(&m).expect("should detect SOC");
    assert_eq!(form.terms.len(), 2);
    let coef_of = |name: &str| {
        let id = m.variable_id(name).unwrap();
        form.terms.iter().find(|lt| lt.coeffs[0].0 == id).map(|lt| lt.coeffs[0].1).unwrap()
    };
    assert!((coef_of("x") - (2.0f64 / 5.0).sqrt()).abs() < 1e-12);
    assert!((coef_of("y") - (3.0f64 / 5.0).sqrt()).abs() < 1e-12);
    let bound = form.standard_bound().expect("standard head");
    assert_eq!(bound.coeffs, vec![(m.variable_id("t").unwrap(), 1.0)]);
    assert_eq!(bound.constant, 0.0);
}

#[test]
fn rejects_cross_terms() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + x * y + y * y <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_linear_part() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + x <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_nonzero_constant() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + 1.0 <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn detects_ge_standard_cone_like_its_le_mirror() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, 5.0 * t * t >= 2.0 * x * x + 3.0 * y * y);

    let form = detect_first(&m).expect("should detect a `>=` SOC");
    let coef_of = |name: &str| {
        let id = m.variable_id(name).unwrap();
        form.terms.iter().find(|lt| lt.coeffs[0].0 == id).map(|lt| lt.coeffs[0].1).unwrap()
    };
    assert_eq!(form.terms.len(), 2);
    assert!((coef_of("x") - (2.0f64 / 5.0).sqrt()).abs() < 1e-12);
    assert!((coef_of("y") - (3.0f64 / 5.0).sqrt()).abs() < 1e-12);
    let bound = form.standard_bound().expect("standard head");
    assert_eq!(bound.coeffs, vec![(m.variable_id("t").unwrap(), 1.0)]);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn ge_sense_head_is_the_negated_side() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x >= t * t);
    assert!(detect_first(&m).is_none());
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn ge_sense_detects_reversed_head_when_it_is_nonnegative() {
    let m = Model::new("soc");
    variable!(m, x >= 0.0);
    variable!(m, t);
    constraint!(m, c, x * x >= t * t);
    let form = detect_first(&m).expect("`t^2 <= x^2` with `x >= 0`");
    assert_eq!(form.standard_bound().unwrap().coeffs, vec![(m.variable_id("x").unwrap(), 1.0)]);
    assert_eq!(form.terms.len(), 1);
    assert_eq!(form.terms[0].coeffs.to_vec(), vec![(m.variable_id("t").unwrap(), 1.0)]);
}

#[test]
fn ge_rejects_concave_side_of_a_cone() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    constraint!(m, c, x * x + y * y >= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_equality_cone_shape() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, t * t == x * x);
    assert!(detect_first(&m).is_none());
}

#[test]
fn ge_rejects_nonzero_constant() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    constraint!(m, c, t * t >= x * x + 1.0);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_ranged_cone_shape() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    m.add_constraint(
        "band",
        Constraint::new((t * t - x * x).into_function(), Interval { lower: 0.0, upper: 1.0 }),
    );
    assert!(m.constraints().algebraic()[0].as_single().is_none());
    assert!(detect_first(&m).is_none());
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn macro_split_quadratic_range_detects_only_its_cone_half() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    let RangeConstraintHandles::Split { lower, upper } =
        constraint!(m, band, 0.0 <= t * t - x * x <= 1.0)
    else {
        panic!("quadratic macro ranges split");
    };
    let arena = m.arena();
    let vars = m.resolved_variables();
    let rows = m.constraints();
    let detected = |id: ConstraintId| {
        let row = &rows.algebraic()[id.index()];
        let q = oximo_expr::extract_quadratic(&arena, row.lhs).unwrap();
        oximo_core::__detect_soc_from_quadratic(&vars, row, &q)
    };
    let cone = detected(lower.row_id()).expect("lower half is a cone");
    assert_eq!(cone.standard_bound().unwrap().coeffs, vec![(m.variable_id("t").unwrap(), 1.0)]);
    assert!(detected(upper.row_id()).is_none());
    drop((rows, vars, arena));
    assert_eq!(m.kind(), ModelKind::QCP);
}

/// `(variable name, coefficient)` pairs of every component.
fn form_signature(m: &Model, form: &SocForm) -> Vec<Vec<(String, f64)>> {
    let names = m.resolved_variables();
    let mut terms: Vec<_> = form
        .terms
        .iter()
        .map(|t| t.coeffs.iter().map(|&(v, c)| (names[v.index()].name.to_string(), c)).collect())
        .collect::<Vec<Vec<_>>>();
    terms.sort_by(|a, b| a[0].0.cmp(&b[0].0));
    let mut signature: Vec<Vec<_>> = form
        .head
        .iter()
        .map(|h| {
            assert_eq!(h.constant, 0.0);
            h.coeffs.iter().map(|&(v, c)| (names[v.index()].name.to_string(), c)).collect()
        })
        .collect();
    signature.extend(terms);
    signature
}

#[test]
fn le_row_and_ge_mirror_materialize_the_same_form() {
    type Build = fn(&Model, bool);
    let cases: [(&str, Build); 4] = [
        ("standard", |m, ge| {
            variable!(m, x);
            variable!(m, y);
            variable!(m, t >= 0.0);
            if ge {
                constraint!(m, c, 5.0 * t * t >= 2.0 * x * x + 3.0 * y * y);
            } else {
                constraint!(m, c, 2.0 * x * x + 3.0 * y * y <= 5.0 * t * t);
            }
        }),
        ("standard with rhs", |m, ge| {
            variable!(m, x);
            variable!(m, t >= 0.0);
            if ge {
                constraint!(m, c, 4.0 - x * x + t * t >= 4.0);
            } else {
                constraint!(m, c, x * x - t * t + 4.0 <= 4.0);
            }
        }),
        ("rotated", |m, ge| {
            variable!(m, x);
            variable!(m, y);
            variable!(m, u >= 0.0);
            variable!(m, v >= 0.0);
            if ge {
                constraint!(m, c, 6.0 * u * v >= x * x + 4.0 * y * y);
            } else {
                constraint!(m, c, x * x + 4.0 * y * y <= 6.0 * u * v);
            }
        }),
        ("rotated reordered heads", |m, ge| {
            variable!(m, v >= 0.0);
            variable!(m, x);
            variable!(m, u >= 0.0);
            if ge {
                constraint!(m, c, 4.0 * u * v >= x * x);
            } else {
                constraint!(m, c, x * x <= 4.0 * u * v);
            }
        }),
    ];

    for (name, build) in cases {
        let le = Model::new("le");
        build(&le, false);
        let ge = Model::new("ge");
        build(&ge, true);
        let (le_form, ge_form) = (detect_first(&le), detect_first(&ge));
        let (le_form, ge_form) = (le_form.expect(name), ge_form.expect(name));
        assert_eq!(le_form.kind(), ge_form.kind(), "{name}");
        assert_eq!(form_signature(&le, &le_form), form_signature(&ge, &ge_form), "{name}");
        assert_eq!(le.kind(), ge.kind(), "{name}");
    }
}

#[test]
fn rejects_two_negative_squares() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t >= 0.0);
    variable!(m, u >= 0.0);
    constraint!(m, c, x * x - u * u <= t * t);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_all_positive_squares() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, y);
    constraint!(m, c, x * x + y * y <= 4.0);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rejects_unsigned_bound_variable() {
    let m = Model::new("soc");
    variable!(m, x);
    variable!(m, t);
    constraint!(m, c, x * x <= t * t);
    assert!(detect_first(&m).is_none());
}

// Rotated cones

fn rotated_parts(
    form: &SocForm,
) -> (&oximo_expr::LinearTerms<'static>, &oximo_expr::LinearTerms<'static>) {
    match &form.head {
        SocHead::Rotated(u, v) => (u, v),
        SocHead::Standard(_) => panic!("expected rotated head"),
    }
}

#[test]
fn detects_rotated_cone_from_cross_term() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    constraint!(m, c, x * x + 4.0 * y * y <= 6.0 * u * v);

    let form = detect_first(&m).expect("should detect rotated cone");
    let (hu, hv) = rotated_parts(&form);
    let (ui, vi) = (m.variable_id("u").unwrap(), m.variable_id("v").unwrap());
    assert_eq!(hu.coeffs, vec![(ui, 3.0)]);
    assert_eq!(hv.coeffs, vec![(vi, 1.0)]);
    assert_eq!(form.terms.len(), 2);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn detects_ge_rotated_cone_like_its_le_mirror() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    constraint!(m, c, 6.0 * u * v >= x * x + 4.0 * y * y);

    let form = detect_first(&m).expect("should detect a `>=` rotated cone");
    let (hu, hv) = rotated_parts(&form);
    let (ui, vi) = (m.variable_id("u").unwrap(), m.variable_id("v").unwrap());
    assert_eq!(hu.coeffs, vec![(ui, 3.0)]);
    assert_eq!(hv.coeffs, vec![(vi, 1.0)]);
    let mut terms: Vec<_> = form.terms.iter().map(|t| t.coeffs[0]).collect();
    terms.sort_by_key(|&(var, _)| var.0);
    let (xi, yi) = (m.variable_id("x").unwrap(), m.variable_id("y").unwrap());
    assert_eq!(terms, vec![(xi, 1.0), (yi, 2.0)]);
    assert_eq!(m.kind(), ModelKind::SOCP);
}

#[test]
fn ge_rotated_detection_requires_nonnegative_heads() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, u);
    variable!(m, v >= 0.0);
    constraint!(m, c, u * v >= x * x);
    assert!(detect_first(&m).is_none());
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn ge_rotated_rejects_reverse_orientation() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    constraint!(m, c, x * x >= u * v);
    assert!(detect_first(&m).is_none());
}

#[test]
fn ge_rotated_rejects_squared_head() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    constraint!(m, c, u * v >= x * x + u * u);
    assert!(detect_first(&m).is_none());
}

#[test]
fn rotated_detection_requires_nonnegative_heads() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, u);
    variable!(m, v >= 0.0);
    constraint!(m, c, x * x <= u * v);
    assert!(detect_first(&m).is_none());
    assert_eq!(m.kind(), ModelKind::QCP);
}

#[test]
fn rotated_detection_rejects_extra_structure() {
    let build = |f: &dyn Fn(&Model)| {
        let m = Model::new("rsoc");
        f(&m);
        detect_first(&m)
    };
    assert!(
        build(&|m| {
            variable!(m, x);
            variable!(m, u >= 0.0);
            variable!(m, v >= 0.0);
            constraint!(m, c, x * x + u * u <= u * v);
        })
        .is_none()
    );
    assert!(
        build(&|m| {
            variable!(m, x);
            variable!(m, u >= 0.0);
            variable!(m, v >= 0.0);
            constraint!(m, c, x * x + v * v <= u * v);
        })
        .is_none()
    );
    assert!(
        build(&|m| {
            variable!(m, u >= 0.0);
            variable!(m, x);
            variable!(m, v >= 0.0);
            constraint!(m, c, x * x + v * v <= 2.0 * u * v);
        })
        .is_none()
    );
    assert!(
        build(&|m| {
            variable!(m, u >= 0.0);
            variable!(m, x);
            variable!(m, v >= 0.0);
            constraint!(m, c, x * x + u * u <= 2.0 * u * v);
        })
        .is_none()
    );
    assert!(
        build(&|m| {
            variable!(m, u >= 0.0);
            variable!(m, x);
            variable!(m, v >= 0.0);
            variable!(m, y);
            constraint!(m, c, x * x + y * y <= 2.0 * u * v);
        })
        .is_some_and(|form| form.head.is_rotated() && form.terms.len() == 2)
    );
    assert!(
        build(&|m| {
            variable!(m, x);
            variable!(m, u >= 0.0);
            variable!(m, v >= 0.0);
            constraint!(m, c, x * x + u * v <= 0.0);
        })
        .is_none()
    );
    assert!(
        build(&|m| {
            variable!(m, u >= 0.0);
            variable!(m, v >= 0.0);
            constraint!(m, c, (-1.0) * u * v <= 0.0);
        })
        .is_none()
    );
}

#[test]
fn rotated_to_standard_preserves_membership_and_duals() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    constraint!(m, c, x * x <= 2.0 * u * v);
    let form = detect_first(&m).unwrap();
    let standard = form.clone().into_standard();
    assert!(!standard.head.is_rotated());

    let eval = |lt: &oximo_expr::LinearTerms<'static>, pt: &dyn Fn(oximo_expr::VarId) -> f64| {
        lt.constant + lt.coeffs.iter().map(|&(var, c)| c * pt(var)).sum::<f64>()
    };
    let (xi, ui, vi) =
        (m.variable_id("x").unwrap(), m.variable_id("u").unwrap(), m.variable_id("v").unwrap());
    for &(xv, uv, vv) in &[(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (0.5, 0.2, 0.9), (1.0, 0.1, 0.1)] {
        let pt = |var: oximo_expr::VarId| {
            if var == xi {
                xv
            } else if var == ui {
                uv
            } else if var == vi {
                vv
            } else {
                0.0
            }
        };

        let (hu, hv) = rotated_parts(&form);
        let rotated_ok = {
            let w: f64 = form.terms.iter().map(|t| eval(t, &pt).powi(2)).sum();
            w <= 2.0 * eval(hu, &pt) * eval(hv, &pt) + 1e-12
        };
        let t = standard.standard_bound().unwrap();
        let standard_ok = {
            let w: f64 = standard.terms.iter().map(|t| eval(t, &pt).powi(2)).sum();
            w.sqrt() <= eval(t, &pt) + 1e-12
        };
        assert_eq!(rotated_ok, standard_ok, "point {xv},{uv},{vv}");
    }
}

#[test]
fn explicit_rotated_cone_round_trips_through_views() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, y);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    let handle = m.add_rotated_soc_constraint("r", [x, y], u, v);
    assert_eq!(m.kind(), ModelKind::SOCP);

    let rows = m.soc_constraints();
    let row = &rows[handle.soc_id().index()];
    assert_eq!(row.head(), SocHead::Rotated(u.id(), v.id()));
    assert_eq!(row.terms().len(), 2);
    assert_eq!(row.dimension(), 4);

    drop(rows);
    assert_eq!(m.soc_constraint_handle_from_id(handle.soc_id()), None);

    let text = format!("{}", m.display_soc(handle.soc_id()));
    assert_eq!(text, "r: ||x, y||^2 <= 2 (u) (v), u >= 0, v >= 0");
}

#[test]
fn typed_rotated_cone_registers_and_checks_dimension() {
    let m = Model::new("rsoc");
    variable!(m, x);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    let f = VectorAffineFunction::new([u.into_function(), v.into_function(), x.into_function()]);
    let h = m.add_constraint("typed", Constraint::new(f, RotatedSecondOrderCone { dimension: 3 }));
    assert_eq!(m.soc_constraints()[h.soc_id().index()].head(), SocHead::Rotated(u.id(), v.id()));
}

#[test]
fn macro_rotated_forms_register_rotated_cones() {
    let m = Model::new("rsoc");
    variable!(m, x[i in 0..3]);
    variable!(m, u >= 0.0);
    variable!(m, v >= 0.0);
    let named: RotatedSocConstraintHandle =
        soc_constraint!(m, r, [x[0], 2.0 * x[1]] in rotated(u, v + 1.0));
    let auto = soc_constraint!(m, [x[2]] in rotated(u, v));
    soc_constraint!(m, family[i in 0..3], [x[i]] in rotated(u, v));
    soc_constraint!(m, standard, [x[0]] <= u);

    let rows = m.soc_constraints();
    assert_eq!(rows[named.soc_id().index()].dimension(), 4);
    assert!(rows[auto.soc_id().index()].head().is_rotated());

    for i in 0..3 {
        let id = m.soc_constraint_id(&format!("family[{i}]")).unwrap();
        assert!(rows[id.index()].head().is_rotated());
    }
    let standard = m.soc_constraint_id("standard").unwrap();
    assert!(!rows[standard.index()].head().is_rotated());
    assert_eq!(rows[standard.index()].kind(), SocKind::Standard);
    assert_eq!(rows[named.soc_id().index()].kind(), SocKind::Rotated);
    assert_eq!(rows[named.soc_id().index()].terms().len(), 2);
    drop(rows);

    let set_form: SocConstraintHandle = soc_constraint!(m, set_form, [x[0], x[1]] in soc(u + 1.0));
    let rows = m.soc_constraints();
    let row = &rows[set_form.soc_id().index()];
    assert_eq!((row.kind(), row.dimension()), (SocKind::Standard, 3));
    assert_eq!(set_form.soc_kind(), SocKind::Standard);
    assert_eq!(named.soc_kind(), SocKind::Rotated);
    drop(rows);
    assert_eq!(m.rotated_soc_constraint_handle("r"), Some(named));
    assert!(m.soc_constraint_handle("standard").is_some());
    assert!(m.rotated_soc_constraint_handle("standard").is_none());
}

#[test]
fn rotated_detection_orders_heads_by_variable_id() {
    let m = Model::new("rsoc order");
    variable!(m, v >= 0.0);
    variable!(m, x);
    variable!(m, u >= 0.0);
    constraint!(m, c, x * x <= 4.0 * u * v);
    let form = detect_first(&m).unwrap();
    let (first, second) = rotated_parts(&form);
    assert_eq!(first.coeffs, vec![(m.variable_id("v").unwrap(), 2.0)]);
    assert_eq!(second.coeffs, vec![(m.variable_id("u").unwrap(), 1.0)]);
}

#[test]
fn into_standard_merges_unsorted_heads_and_keeps_cancelled_coefficients() {
    use oximo_expr::{LinearTerms, VarId};
    let form = SocForm {
        terms: vec![LinearTerms::owned(vec![(VarId(9), 1.0)], 0.0)],
        head: SocHead::Rotated(
            LinearTerms::owned(vec![(VarId(3), 2.0), (VarId(1), 1.0)], 1.0),
            LinearTerms::owned(vec![(VarId(1), 1.0), (VarId(2), 4.0)], 0.5),
        ),
    };
    let standard = form.into_standard();
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let sum = standard.standard_bound().unwrap();
    assert_eq!(
        sum.coeffs.to_vec(),
        vec![(VarId(1), 2.0 * r), (VarId(2), 4.0 * r), (VarId(3), 2.0 * r)]
    );
    assert!((sum.constant - 1.5 * r).abs() < 1e-15);

    let diff = &standard.terms[0];
    // `x1` cancels in `u - v`, explicit zero.
    assert_eq!(
        diff.coeffs.to_vec(),
        vec![(VarId(1), 0.0), (VarId(2), -4.0 * r), (VarId(3), 2.0 * r)]
    );
    assert_eq!(standard.terms[1].coeffs.to_vec(), vec![(VarId(9), 1.0)]);
}
