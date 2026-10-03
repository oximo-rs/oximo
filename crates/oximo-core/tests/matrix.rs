use oximo_core::{SymmetricMatrix, SymmetricMatrixError};

#[test]
fn dense_rows_preserve_values_and_canonical_triangle_order() {
    let rows = [[1, 2, 4], [2, 3, 5], [4, 5, 6]];
    let matrix = SymmetricMatrix::try_from_rows(rows).unwrap();
    assert_eq!(matrix.upper_triangle(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(matrix, SymmetricMatrix::try_from_rows(rows.iter()).unwrap());
    assert_eq!(matrix[(2, 0)], rows[2][0]);

    let owned = vec![
        vec![String::from("a"), String::from("b")],
        vec![String::from("b"), String::from("c")],
    ];
    let borrowed_rows: Vec<&[String]> = owned.iter().map(Vec::as_slice).collect();
    let borrowed = SymmetricMatrix::try_from_rows(borrowed_rows).unwrap();
    assert_eq!(borrowed, SymmetricMatrix::try_from_rows(owned).unwrap());
    assert_eq!(borrowed.to_string(), "[[a, b], [b, c]]");
}

#[test]
fn dense_rows_reject_empty_and_nonsquare_inputs() {
    assert_eq!(
        SymmetricMatrix::try_from_rows(Vec::<Vec<f64>>::new()),
        Err(SymmetricMatrixError::Empty)
    );
    assert_eq!(
        SymmetricMatrix::try_from_rows([Vec::<f64>::new()]),
        Err(SymmetricMatrixError::NonSquare { row: 0, expected: 1, actual: 0 })
    );
    assert_eq!(
        SymmetricMatrix::try_from_rows([[1, 2, 3], [2, 3, 4]]),
        Err(SymmetricMatrixError::NonSquare { row: 0, expected: 2, actual: 3 })
    );
    assert_eq!(
        SymmetricMatrix::try_from_rows([vec![1, 2], vec![2]]),
        Err(SymmetricMatrixError::NonSquare { row: 1, expected: 2, actual: 1 })
    );
}

#[test]
fn dense_rows_require_exact_symmetry_without_averaging() {
    assert_eq!(
        SymmetricMatrix::try_from_rows([[1.0, 2.0], [2.0 + f64::EPSILON * 2.0, 3.0]]),
        Err(SymmetricMatrixError::NotSymmetric { row: 0, column: 1 })
    );
    assert_eq!(
        SymmetricMatrix::try_from_rows([[1.0, f64::NAN], [f64::NAN, 3.0]]),
        Err(SymmetricMatrixError::NotSymmetric { row: 0, column: 1 })
    );
    let indefinite = SymmetricMatrix::try_from_rows([[0, 1], [1, 0]]).unwrap();
    assert_eq!(indefinite.upper_triangle(), &[0, 1, 0]);
}

#[test]
fn numeric_identity_has_unit_diagonal_and_zero_off_diagonals() {
    let identity = SymmetricMatrix::<f64>::identity(3);
    assert_eq!(identity.upper_triangle(), &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(identity.trace(), 3.0);
    assert_eq!(SymmetricMatrix::<i32>::identity(1).upper_triangle(), &[1]);
    assert!(std::panic::catch_unwind(|| SymmetricMatrix::<f64>::identity(0)).is_err());
    assert!(std::panic::catch_unwind(|| SymmetricMatrix::<f64>::identity(usize::MAX)).is_err());
}

#[test]
fn display_shows_full_rows_and_honors_element_precision() {
    let matrix = SymmetricMatrix::try_from_rows([[1.25, -2.5], [-2.5, 3.0]]).unwrap();
    assert_eq!(matrix.to_string(), "[[1.25, -2.5], [-2.5, 3]]");
    assert_eq!(format!("{matrix:.2}"), "[[1.25, -2.50], [-2.50, 3.00]]");
    assert_eq!(SymmetricMatrix::from_upper_triangle(1, [7]).to_string(), "[[7]]");
}
