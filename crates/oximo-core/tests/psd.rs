#![allow(non_snake_case)]
use oximo_core::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn symmetric_variables_are_shared_and_unconstrained_until_psd_registration() {
    let m = Model::new("matrix");
    symmetric_variable!(m, X[3]);
    assert_eq!(m.num_variables(), 6);
    assert_eq!(X[(0, 2)].var_id(), X[(2, 0)].var_id());
    assert_eq!(m.kind(), ModelKind::LP);
    let cone = psd_constraint!(m, positivity, &X);
    assert_eq!(m.kind(), ModelKind::SDP);
    assert_eq!(m.constraints().positive_semidefinite().len(), 1);
    assert_eq!(m.num_constraints(), 1);
    assert!(m.has_cones());
    assert!(matches!(
        m.constraints().iter().next(),
        Some(ConstraintRef::PositiveSemidefinite { .. })
    ));
    assert!(m.to_string().contains("positivity: symmetric 3x3"));
    m.set_psd_active(cone, false).unwrap();
    assert_eq!(m.kind(), ModelKind::LP);
    assert!(!m.has_active_psd_constraints());
    m.set_psd_active(cone, true).unwrap();
    assert_eq!(m.kind(), ModelKind::SDP);
    variable!(m, _integer, Int);
    assert_eq!(m.kind(), ModelKind::MISDP);
}

#[test]
fn triangle_constructors_and_frobenius_use_ordinary_entries() {
    let a: SymmetricMatrix<f64> =
        SymmetricMatrix::from_upper_triangle(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = SymmetricMatrix::from_lower_triangle(3, [1.0, 2.0, 4.0, 3.0, 5.0, 6.0]);
    assert_eq!(a, b);
    assert_eq!(a.trace(), 10.0);
    assert_eq!(a.frobenius(&a), 136.0);
    assert_eq!((&a + &b).upper_triangle(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    assert_eq!((2.0 * a.clone()).upper_triangle(), (b * 2.0).upper_triangle());
    assert!(a.get(3, 0).is_none());
    assert!(catch_unwind(|| SymmetricMatrix::<f64>::from_upper_triangle(0, [])).is_err());
    assert!(catch_unwind(|| SymmetricMatrix::from_upper_triangle(2, [1.0])).is_err());
    assert!(catch_unwind(|| oximo_core::triangle_len(usize::MAX)).is_err());
}

#[test]
fn affine_matrix_arithmetic_and_parameters_remain_symbolic() {
    let m = Model::new("symbolic");
    symmetric_variable!(m, X[2]);
    variable!(m, t);
    param!(m, p = 2.0);
    let identity = SymmetricMatrix::from_upper_triangle(2, [1.0, 0.0, 1.0]);
    let shifted = &X + t * &identity - p * &identity;
    let cone = psd_constraint!(m, shifted, shifted);
    let first = m.psd_constraints()[cone.index()].matrix[(0, 0)];
    let linear = oximo_expr::extract_linear(&m.arena(), first).unwrap().into_owned();
    assert_eq!(linear.constant, -2.0);
    m.set_param(p, 4.0).unwrap();
    assert_eq!(oximo_expr::extract_linear(&m.arena(), first).unwrap().constant, -4.0);
    let expression = identity.frobenius(&X);
    assert_eq!(oximo_expr::extract_linear(&m.arena(), expression.id()).unwrap().coeffs.len(), 2);
    let function = (&X).into_symmetric_affine_function(&m);
    m.add_constraint(
        "typed",
        Constraint::new(function, PositiveSemidefiniteCone { side_dimension: 2 }),
    );
    m.add_constraint(
        "converted",
        Constraint::new(X.into_function(), PositiveSemidefiniteCone { side_dimension: 2 }),
    );
}

#[test]
fn invalid_registrations_preserve_existing_registries() {
    let m = Model::new("validation");
    variable!(m, x);
    let other = Model::new("other");
    symmetric_variable!(other, foreign[2]);
    assert!(catch_unwind(AssertUnwindSafe(|| m.add_psd_constraint("foreign", &foreign))).is_err());
    assert_eq!(m.num_psd_constraints(), 0);
    let matrix = SymmetricMatrix::from_upper_triangle(1, [x.square().erase()]);
    assert!(catch_unwind(AssertUnwindSafe(|| m.add_psd_constraint("quadratic", matrix))).is_err());
    // Collision at the last entry must not register earlier entries.
    m.__var("taken[1,1]").build();
    let before = m.num_variables();
    assert!(catch_unwind(AssertUnwindSafe(|| m.add_symmetric_variable("taken", 2))).is_err());
    assert_eq!(m.num_variables(), before);
    let matrix = SymmetricMatrix::from_upper_triangle(1, [x]);
    let function = matrix.clone().into_symmetric_affine_function(&m);
    assert!(
        catch_unwind(AssertUnwindSafe(|| Constraint::new(
            function,
            PositiveSemidefiniteCone { side_dimension: 2 }
        )
        .into_ir()))
        .is_err()
    );
    m.add_psd_constraint("same", &matrix);
    assert!(catch_unwind(AssertUnwindSafe(|| m.add_psd_constraint("same", &matrix))).is_err());
    assert_eq!(m.num_psd_constraints(), 1);
}

#[test]
fn indexed_psd_forks_remap_in_deterministic_order_and_roll_back_errors() {
    let m = Model::new("indexed");
    variable!(m, x);
    param!(m, p = 1.0);
    // More than the parallel-registration threshold, with new roots per worker.
    psd_constraint!(m, blocks[i in 0..300], SymmetricMatrix::from_upper_triangle(1, [x + p * f64::from(u32::try_from(i).unwrap())]));
    assert_eq!(m.num_psd_constraints(), 300);
    for (i, cone) in m.psd_constraints().iter().enumerate() {
        assert_eq!(cone.name, format!("blocks[{i}]"));
        assert_eq!(
            oximo_expr::extract_linear(&m.arena(), cone.matrix[(0, 0)]).unwrap().constant,
            f64::from(u32::try_from(i).unwrap())
        );
    }
    m.set_param(p, 2.0).unwrap();
    assert_eq!(
        oximo_expr::extract_linear(&m.arena(), m.psd_constraints()[299].matrix[(0, 0)])
            .unwrap()
            .constant,
        598.0
    );
    let nodes = m.arena().len();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        psd_constraint!(m, bad[i in 0..3], SymmetricMatrix::from_upper_triangle(1, [if i == 2 { x.square().erase() } else { (x + 20.0).erase() }]));
    })).is_err());
    assert_eq!(m.num_psd_constraints(), 300);
    assert_eq!(m.arena().len(), nodes);
    psd_constraint!(m, name = "computed", SymmetricMatrix::from_upper_triangle(1, [x]));
    psd_constraint!(m, SymmetricMatrix::from_upper_triangle(1, [x]));
    assert_eq!(m.num_psd_constraints(), 302);
}

#[test]
fn reformulated_clone_preserves_psd_registry_and_parameters() {
    let m = Model::new("source");
    variable!(m, x);
    param!(m, p = 2.0);
    let handle = psd_constraint!(m, positivity, SymmetricMatrix::from_upper_triangle(1, [x + p]));
    let copy = m.to_reformulated_sos_model(SosReformulationOptions::default()).unwrap();
    let cloned = copy.model();
    assert_ne!(cloned.id(), m.id());
    assert_eq!(cloned.kind(), ModelKind::SDP);
    assert_eq!(cloned.psd_constraint_id("positivity"), Some(handle.id()));
    assert_eq!(
        cloned.psd_constraints()[0].matrix.upper_triangle(),
        m.psd_constraints()[0].matrix.upper_triangle()
    );
    cloned.set_param_id(p.param_id().unwrap(), 7.0);
    let entry = cloned.psd_constraints()[0].matrix[(0, 0)];
    assert_eq!(oximo_expr::extract_linear(&cloned.arena(), entry).unwrap().constant, 7.0);
    assert_eq!(oximo_expr::extract_linear(&m.arena(), entry).unwrap().constant, 2.0);
}
