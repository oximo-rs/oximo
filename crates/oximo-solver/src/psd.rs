//! Adapter coordinate conversions for real symmetric PSD cones.
//! Mathematical modeling always uses ordinary matrix entries.

use oximo_core::{SymmetricMatrix, triangle_len};
use std::f64::consts::SQRT_2;

/// Native triangle order. Used to describe an encoding.
#[derive(Clone, Copy, Debug)]
pub enum PsdTriangleOrder {
    UpperColumn,
    LowerColumn,
}

/// Iterate native coordinates as canonical upper-triangle indices and svec scales.
pub fn svec_coordinates(
    n: usize,
    order: PsdTriangleOrder,
) -> impl ExactSizeIterator<Item = (usize, f64)> + std::iter::FusedIterator {
    SvecCoordinates {
        n,
        order,
        row: 0,
        column: 0,
        index: 0,
        diagonal: 0,
        remaining: triangle_len(n),
    }
}

/// Constant-space iterator mapping native triangle order to stored indices and svec scales.
struct SvecCoordinates {
    n: usize,
    order: PsdTriangleOrder,
    row: usize,
    column: usize,
    index: usize,
    diagonal: usize,
    remaining: usize,
}

impl Iterator for SvecCoordinates {
    type Item = (usize, f64);

    /// Yield the next native coordinate and advance the triangle traversal.
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = (self.index, if self.row == self.column { 1.0 } else { SQRT_2 });
        self.remaining -= 1;
        if self.remaining > 0 {
            match self.order {
                PsdTriangleOrder::UpperColumn => {
                    self.index += 1;
                    self.row += 1;
                    if self.row > self.column {
                        self.column += 1;
                        self.row = 0;
                    }
                }
                PsdTriangleOrder::LowerColumn => {
                    if self.row + 1 == self.n {
                        self.column += 1;
                        self.row = self.column;
                        self.diagonal += self.column + 1;
                        self.index = self.diagonal;
                    } else {
                        self.index += self.row + 1;
                        self.row += 1;
                    }
                }
            }
        }
        Some(item)
    }

    /// Report the exact number of native coordinates remaining.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for SvecCoordinates {}
impl std::iter::FusedIterator for SvecCoordinates {}

/// Pack entries in native triangle order, scaling off-diagonals by sqrt(2).
pub fn pack_svec(matrix: &SymmetricMatrix<f64>, order: PsdTriangleOrder) -> Vec<f64> {
    svec_coordinates(matrix.side_dimension(), order)
        .map(|(k, scale)| matrix.upper_triangle()[k] * scale)
        .collect()
}

/// Restore ordinary matrix entries from a complete, finite svec block.
/// Returns `None` for incomplete or nonfinite native data.
pub fn unpack_svec(
    n: usize,
    values: &[f64],
    order: PsdTriangleOrder,
) -> Option<SymmetricMatrix<f64>> {
    if values.len() != triangle_len(n) || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut entries = vec![0.0; values.len()];
    for ((k, scale), &value) in svec_coordinates(n, order).zip(values) {
        entries[k] = value / scale;
    }
    Some(SymmetricMatrix::from_upper_triangle(n, entries))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Check both triangle orders, remaining lengths, and repeated exhaustion.
    #[test]
    fn coordinate_iteration_preserves_order_and_remaining_length() {
        for n in [1, 2, 3, 8, 32] {
            for order in [PsdTriangleOrder::UpperColumn, PsdTriangleOrder::LowerColumn] {
                let matrix = SymmetricMatrix::from_upper_fn(n, |i, j| (i, j));
                let expected: Vec<_> = match order {
                    PsdTriangleOrder::UpperColumn => {
                        (0..n).flat_map(|j| (0..=j).map(move |i| (i, j))).collect()
                    }
                    PsdTriangleOrder::LowerColumn => {
                        (0..n).flat_map(|j| (j..n).map(move |i| (j, i))).collect()
                    }
                };
                let mut coordinates = svec_coordinates(n, order);
                for (position, &(i, j)) in expected.iter().enumerate() {
                    assert_eq!(coordinates.len(), expected.len() - position);
                    let (index, scale) = coordinates.next().unwrap();
                    assert_eq!(matrix.upper_triangle()[index], (i, j));
                    assert_eq!(scale, if i == j { 1.0 } else { SQRT_2 });
                }
                assert_eq!(coordinates.len(), 0);
                assert!(coordinates.next().is_none());
                assert!(coordinates.next().is_none());
            }
        }
    }

    /// Check packing order, round trips, inner products, and invalid input rejection.
    #[test]
    fn distinct_three_by_three_entries_expose_order_and_preserve_inner_products() {
        let a = SymmetricMatrix::from_upper_triangle(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = SymmetricMatrix::from_upper_triangle(3, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let upper = pack_svec(&a, PsdTriangleOrder::UpperColumn);
        let lower = pack_svec(&a, PsdTriangleOrder::LowerColumn);
        assert_eq!(upper, [1.0, 2.0 * SQRT_2, 3.0, 4.0 * SQRT_2, 5.0 * SQRT_2, 6.0]);
        assert_eq!(lower, [1.0, 2.0 * SQRT_2, 4.0 * SQRT_2, 3.0, 5.0 * SQRT_2, 6.0]);
        for order in [PsdTriangleOrder::UpperColumn, PsdTriangleOrder::LowerColumn] {
            let pa = pack_svec(&a, order);
            let pb = pack_svec(&b, order);
            let restored = unpack_svec(3, &pa, order).unwrap();
            for (&actual, &expected) in restored.upper_triangle().iter().zip(a.upper_triangle()) {
                assert!((actual - expected).abs() < 1e-12);
            }
            let dot: f64 = pa.iter().zip(pb).map(|(a, b)| a * b).sum();
            assert!((dot - a.frobenius(&b)).abs() < 1e-12);
        }
        assert!(unpack_svec(3, &[0.0; 5], PsdTriangleOrder::UpperColumn).is_none());
        assert!(unpack_svec(1, &[f64::NAN], PsdTriangleOrder::UpperColumn).is_none());
        assert!(unpack_svec(1, &[f64::INFINITY], PsdTriangleOrder::LowerColumn).is_none());
    }
}
