//! Explicit matrix convention at the polynomial/derivative boundary.

use rustc_hash::FxHashMap;

use crate::{QuadraticTerms, VarId};

/// Why a Hessian entry cannot cross a checked coefficient boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HessianConversionErrorKind {
    #[error("the entry must be finite")]
    NonFiniteInput,
    #[error("accumulating duplicate entries overflowed")]
    AccumulationOverflow,
    #[error("the diagonal has no exactly representable polynomial coefficient")]
    InexactDiagonal,
    #[error("doubling the polynomial diagonal exceeds the finite f64 range")]
    DiagonalOverflow,
}

/// A failed conversion, including the variable pair and offending input value.
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Hessian conversion at ({row:?}, {column:?}) for input {value:e}: {kind}")]
pub struct HessianConversionError {
    pub row: VarId,
    pub column: VarId,
    pub value: f64,
    pub kind: HessianConversionErrorKind,
}

impl PartialEq for HessianConversionError {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
            && self.column == other.column
            && self.value.to_bits() == other.value.to_bits()
            && self.kind == other.kind
    }
}
impl Eq for HessianConversionError {}

/// Failure to import a Hessian together with linear coefficients and a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum QuadraticImportError {
    #[error(transparent)]
    Coefficients(#[from] crate::QuadraticBuildError),
    #[error(transparent)]
    Hessian(#[from] HessianConversionError),
}

/// Canonical full second derivatives, one entry per unordered variable pair.
///
/// Diagonal entries are actual second derivatives.
/// This is a matrix type; [`QuadraticTerms`] stores literal polynomial coefficients.
#[derive(Clone, Debug)]
pub struct LowerTriangularHessian {
    entries: Vec<(VarId, VarId, f64)>,
}

impl QuadraticTerms {
    /// Convert polynomial coefficients to finite lower-triangle second derivatives.
    /// Diagonal coefficients are doubled, while off-diagonal coefficients are unchanged.
    /// Entries retain canonical column-then-row order.
    ///
    /// ```
    /// use oximo_expr::{QuadraticTerms, VarId};
    /// # fn main() -> Result<(), oximo_expr::HessianConversionError> {
    /// let terms = QuadraticTerms::from_coefficients(
    ///     vec![(VarId(0), VarId(0), 3.0), (VarId(1), VarId(0), 5.0)],
    ///     Vec::new(), 0.0,
    /// ).unwrap();
    ///
    /// let hessian = terms.checked_hessian_entries().collect::<Result<Vec<_>, _>>()?;
    /// assert_eq!(hessian, [(VarId(0), VarId(0), 6.0), (VarId(1), VarId(0), 5.0)]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// An entry returns its variable pair and original polynomial coefficient if
    /// the input is nonfinite or doubling a diagonal overflows.
    pub fn checked_hessian_entries(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<(VarId, VarId, f64), HessianConversionError>> + '_
    {
        self.quadratic().iter().map(|&(row, column, coefficient)| {
            let error = |kind| HessianConversionError { row, column, value: coefficient, kind };
            let value = if row == column { 2.0 * coefficient } else { coefficient };
            if !value.is_finite() {
                let kind = if coefficient.is_finite() {
                    HessianConversionErrorKind::DiagonalOverflow
                } else {
                    HessianConversionErrorKind::NonFiniteInput
                };
                return Err(error(kind));
            }
            Ok((row, column, value))
        })
    }
}

impl LowerTriangularHessian {
    /// Merge duplicates before any diagonal scaling, orient pairs, sort by
    /// column then row, and remove exact zeros.
    ///
    /// # Errors
    /// Rejects nonfinite records or overflowing duplicate sums with entry context.
    pub fn new(entries: Vec<(VarId, VarId, f64)>) -> Result<Self, HessianConversionError> {
        let mut combined = FxHashMap::default();
        for (a, b, value) in entries {
            let (row, column) = if a.0 >= b.0 { (a, b) } else { (b, a) };
            let error = |kind| HessianConversionError { row, column, value, kind };
            if !value.is_finite() {
                return Err(error(HessianConversionErrorKind::NonFiniteInput));
            }
            let current = combined.entry((row, column)).or_insert(0.0);
            let next = *current + value;
            if !next.is_finite() {
                return Err(error(HessianConversionErrorKind::AccumulationOverflow));
            }
            *current = next;
        }

        let mut entries: Vec<_> =
            combined.into_iter().filter(|(_, h)| *h != 0.0).map(|((r, c), h)| (r, c, h)).collect();
        entries.sort_unstable_by_key(|(row, column, _)| (column.0, row.0));

        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[(VarId, VarId, f64)] {
        &self.entries
    }

    pub(crate) fn into_polynomial_coefficients(
        mut self,
    ) -> Result<Vec<(VarId, VarId, f64)>, HessianConversionError> {
        for (row, column, value) in &mut self.entries {
            if row == column {
                let half = 0.5 * *value;
                if (2.0 * half).to_bits() != value.to_bits() {
                    return Err(HessianConversionError {
                        row: *row,
                        column: *column,
                        value: *value,
                        kind: HessianConversionErrorKind::InexactDiagonal,
                    });
                }
                *value = half;
            }
        }

        Ok(self.entries)
    }
}

impl TryFrom<LowerTriangularHessian> for QuadraticTerms {
    type Error = HessianConversionError;

    fn try_from(value: LowerTriangularHessian) -> Result<Self, Self::Error> {
        Ok(Self::from_canonical_parts(value.into_polynomial_coefficients()?, Vec::new(), 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_polynomial_export_preserves_order_and_exact_subnormal_diagonals() {
        let tiny = f64::from_bits(1);
        let terms = QuadraticTerms::from_coefficients(
            vec![
                (VarId(3), VarId(3), -tiny),
                (VarId(0), VarId(2), f64::MAX),
                (VarId(1), VarId(1), -0.5 * f64::MAX),
                (VarId(0), VarId(0), tiny),
            ],
            Vec::new(),
            0.0,
        )
        .unwrap();
        assert_eq!(terms.checked_hessian_entries().len(), 4);

        let entries = terms.checked_hessian_entries().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            entries,
            [
                (VarId(0), VarId(0), 2.0 * tiny),
                (VarId(2), VarId(0), f64::MAX),
                (VarId(1), VarId(1), -f64::MAX),
                (VarId(3), VarId(3), -2.0 * tiny),
            ]
        );

        let imported =
            QuadraticTerms::try_from(LowerTriangularHessian::new(entries).unwrap()).unwrap();
        assert_eq!(imported.quadratic(), terms.quadratic());
    }

    #[test]
    fn checked_polynomial_export_reports_diagonal_overflow_with_original_coefficient() {
        let first_overflow = f64::from_bits((0.5 * f64::MAX).to_bits() + 1);
        for coefficient in [first_overflow, 1e308, f64::MAX] {
            for negate in [false, true] {
                let mut terms = QuadraticTerms::from_coefficients(
                    vec![(VarId(1), VarId(0), 2.0), (VarId(7), VarId(7), coefficient)],
                    Vec::new(),
                    0.0,
                )
                .unwrap();
                if negate {
                    terms.negate();
                }

                let error =
                    terms.checked_hessian_entries().collect::<Result<Vec<_>, _>>().unwrap_err();
                let expected = if negate { -coefficient } else { coefficient };
                assert_eq!(
                    error,
                    HessianConversionError {
                        row: VarId(7),
                        column: VarId(7),
                        value: expected,
                        kind: HessianConversionErrorKind::DiagonalOverflow,
                    }
                );
                assert!(error.to_string().contains("finite f64 range"));
            }
        }
    }

    #[test]
    fn checked_polynomial_export_rejects_nonfinite_diagonal_and_cross_entries() {
        for coefficient in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for column in [VarId(0), VarId(3)] {
                let terms = QuadraticTerms::from_canonical_parts(
                    vec![(VarId(3), column, coefficient)],
                    Vec::new(),
                    0.0,
                );
                assert_eq!(
                    terms.checked_hessian_entries().next().unwrap().unwrap_err(),
                    HessianConversionError {
                        row: VarId(3),
                        column,
                        value: coefficient,
                        kind: HessianConversionErrorKind::NonFiniteInput,
                    }
                );
            }
        }
    }

    #[test]
    fn typed_hessian_conversion_canonicalizes_and_reports_the_offending_entry() {
        let tiny = f64::from_bits(1);
        let matrix = LowerTriangularHessian::new(vec![
            (VarId(0), VarId(2), 2.0),
            (VarId(2), VarId(0), 3.0),
            (VarId(1), VarId(1), tiny),
            (VarId(1), VarId(1), tiny),
        ])
        .unwrap();
        assert_eq!(matrix.entries(), [(VarId(2), VarId(0), 5.0), (VarId(1), VarId(1), 2.0 * tiny)]);

        let terms = QuadraticTerms::try_from(matrix).unwrap();
        assert_eq!(terms.quadratic(), [(VarId(2), VarId(0), 5.0), (VarId(1), VarId(1), tiny)]);

        let error = QuadraticTerms::try_from(
            LowerTriangularHessian::new(vec![(VarId(2), VarId(2), tiny)]).unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            (error.row, error.column, error.value.to_bits()),
            (VarId(2), VarId(2), tiny.to_bits())
        );
        assert_eq!(error.kind, HessianConversionErrorKind::InexactDiagonal);
        assert!(error.to_string().contains("VarId(2)"));

        let error =
            LowerTriangularHessian::new(vec![(VarId(3), VarId(1), f64::INFINITY)]).unwrap_err();
        assert_eq!((error.row, error.column), (VarId(3), VarId(1)));
        assert_eq!(error.kind, HessianConversionErrorKind::NonFiniteInput);

        let error =
            LowerTriangularHessian::new(vec![(VarId(3), VarId(1), f64::MAX); 2]).unwrap_err();
        assert_eq!(error.kind, HessianConversionErrorKind::AccumulationOverflow);
        assert_eq!(error.value, f64::MAX);
    }
}
