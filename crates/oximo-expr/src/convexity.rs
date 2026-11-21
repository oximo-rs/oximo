//! Numerical curvature certificates for quadratic polynomials.

use rustc_hash::FxHashMap;

use crate::{HessianConversionError, QuadraticTerms, VarId};

/// Why the curvature of a quadratic polynomial could not be decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CurvatureError {
    /// A block of coupled variables is larger than the dense check accepts.
    #[error(
        "a block of {dimension} coupled variables exceeds the dense limit of {}",
        QuadraticTerms::MAX_CURVATURE_DIMENSION
    )]
    Dimension { dimension: usize },
    /// A Hessian entry is not finite.
    #[error(transparent)]
    Hessian(#[from] HessianConversionError),
}

impl QuadraticTerms {
    /// Largest block of coupled quadratic variables whose curvature is
    /// checked.
    pub const MAX_CURVATURE_DIMENSION: usize = 512;

    /// Whether the polynomial is convex (positive semidefinite Hessian).
    ///
    /// Necessary conditions on the coefficients are checked exactly
    /// (a negative square coefficient, or a cross term on a variable without a
    /// square term). Other conditions are numerical were each coupled block is
    /// scaled to a unit diagonal and factored, and pivots below a tight tolerance
    /// relative to that diagonal count as zero.
    ///
    /// # Errors
    /// Returns [`CurvatureError::Dimension`] when a block of coupled
    /// variables exceeds [`Self::MAX_CURVATURE_DIMENSION`], and
    /// [`CurvatureError::Hessian`] when doubling a square coefficient
    /// overflows.
    pub fn is_convex(&self) -> Result<bool, CurvatureError> {
        self.semidefinite(1.0)
    }

    /// Whether the polynomial is concave, that is, whether its negation is
    /// convex. See [`Self::is_convex`].
    ///
    /// # Errors
    /// Same as [`Self::is_convex`].
    pub fn is_concave(&self) -> Result<bool, CurvatureError> {
        self.semidefinite(-1.0)
    }

    /// Whether `sign` times the Hessian is positive semidefinite.
    fn semidefinite(&self, sign: f64) -> Result<bool, CurvatureError> {
        let mut index: FxHashMap<VarId, usize> = FxHashMap::default();
        let mut diagonal = Vec::new();
        let mut cross = Vec::new();
        for entry in self.checked_hessian_entries() {
            let (row, column, value) = entry?;
            let mut position = |v: VarId| {
                *index.entry(v).or_insert_with(|| {
                    diagonal.push(0.0);
                    diagonal.len() - 1
                })
            };
            let (i, j) = (position(row), position(column));
            if i == j {
                diagonal[i] += sign * value;
            } else if value != 0.0 {
                cross.push((i, j, sign * value));
            }
        }

        // Exact necessary conditions:
        // - nonnegative diagonal,
        // - no coupling through a zero diagonal entry.
        if diagonal.iter().any(|&d| d < 0.0)
            || cross.iter().any(|&(i, j, _)| diagonal[i] == 0.0 || diagonal[j] == 0.0)
        {
            return Ok(false);
        }

        for (members, entries) in blocks(diagonal.len(), &cross) {
            let n = members.len();
            if n > Self::MAX_CURVATURE_DIMENSION {
                return Err(CurvatureError::Dimension { dimension: n });
            }

            // Scale to a unit diagonal.
            let mut local = FxHashMap::default();
            let mut scale = Vec::with_capacity(n);
            for (k, &i) in members.iter().enumerate() {
                local.insert(i, k);
                scale.push(diagonal[i].sqrt());
            }

            let mut matrix = vec![0.0; n * n];
            for k in 0..n {
                matrix[k * n + k] = 1.0;
            }
            for (i, j, value) in entries {
                let (a, b) = (local[&i], local[&j]);
                let scaled = value / scale[a] / scale[b];
                matrix[a * n + b] += scaled;
                matrix[b * n + a] += scaled;
            }

            if !positive_semidefinite(n, matrix) {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// A cross term `(row, column, value)` in local positions.
type Entry = (usize, usize, f64);

/// Variables of one coupled block and the cross terms among them.
type Block = (Vec<usize>, Vec<Entry>);

/// Group cross terms into blocks of variables connected by cross terms.
/// Variables without cross terms belong to no block.
fn blocks(n: usize, cross: &[Entry]) -> Vec<Block> {
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    let mut parent: Vec<usize> = (0..n).collect();
    for &(i, j, _) in cross {
        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
        parent[a.max(b)] = a.min(b);
    }

    let mut block = vec![usize::MAX; n];
    let mut result: Vec<Block> = Vec::new();
    for &(i, j, value) in cross {
        let r = root(&mut parent, i);
        if block[r] == usize::MAX {
            block[r] = result.len();
            result.push(Default::default());
        }
        result[block[r]].1.push((i, j, value));
    }
    for i in 0..n {
        let r = root(&mut parent, i);
        if block[r] != usize::MAX {
            result[block[r]].0.push(i);
        }
    }

    result
}

/// Symmetric elimination with diagonal pivoting on a dense row-major matrix
/// whose diagonal is at most about one.
fn positive_semidefinite(n: usize, mut a: Vec<f64>) -> bool {
    #[expect(clippy::cast_precision_loss, reason = "dimension is at most 512")]
    let tolerance = 64.0 * f64::EPSILON * n as f64;

    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j])).expect("rows");
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
            for r in 0..n {
                a.swap(r * n + k, r * n + p);
            }
        }

        let pivot = a[k * n + k];
        if pivot.is_nan() || pivot <= tolerance {
            return (k..n).all(|i| (k..n).all(|j| a[i * n + j].abs() <= tolerance));
        }

        for i in k + 1..n {
            let coupling = a[i * n + k];
            if coupling * coupling > pivot * (a[i * n + i] + tolerance) {
                return false;
            }
        }

        for i in k + 1..n {
            let factor = a[i * n + k] / pivot;
            if factor == 0.0 {
                continue;
            }
            let (head, tail) = a.split_at_mut(i * n);
            let pivot_row = &head[k * n + k + 1..k * n + n];
            for (x, &y) in tail[k + 1..n].iter_mut().zip(pivot_row) {
                *x -= factor * y;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(quadratic: Vec<(VarId, VarId, f64)>) -> QuadraticTerms {
        QuadraticTerms::from_coefficients(quadratic, Vec::new(), 0.0).unwrap()
    }

    /// `sum (x_i - x_{i-1})^2` over a chain of `n` variables.
    fn chain(n: u32) -> QuadraticTerms {
        terms(
            (0..n)
                .flat_map(|i| {
                    let square = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
                    let cross = (i > 0).then(|| (VarId(i), VarId(i - 1), -2.0));
                    std::iter::once((VarId(i), VarId(i), square)).chain(cross)
                })
                .collect(),
        )
    }

    #[test]
    fn semidefiniteness_accepts_singular_and_rejects_indefinite_matrices() {
        let psd = |n, a: &[f64]| positive_semidefinite(n, a.to_vec());

        assert!(psd(2, &[1.0, 1.0, 1.0, 1.0]));
        assert!(psd(3, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.25]));
        assert!(psd(2, &[0.0; 4]));
        assert!(!psd(2, &[1.0, 2.0, 2.0, 1.0]));
        assert!(!psd(2, &[0.0, 1.0, 1.0, 0.0]));
        assert!(!psd(2, &[1.0, 0.0, 0.0, -1e-6]));
        assert!(!psd(3, &[1.0, 0.0, 0.0, 0.0, 0.0, 1e-3, 0.0, 1e-3, 0.0]));
        assert!(!psd(2, &[1.0, 1.0 + 1e-10, 1.0 + 1e-10, 1.0]));
    }

    #[test]
    fn curvature_follows_the_sign_of_the_hessian() {
        let (x, y) = (VarId(0), VarId(1));

        let affine = terms(Vec::new());
        assert_eq!((affine.is_convex(), affine.is_concave()), (Ok(true), Ok(true)));

        let squares = terms(vec![(x, x, 1.0), (y, y, 2.0)]);
        assert_eq!((squares.is_convex(), squares.is_concave()), (Ok(true), Ok(false)));

        let saddle = terms(vec![(x, x, 1.0), (y, y, -1.0)]);
        assert_eq!((saddle.is_convex(), saddle.is_concave()), (Ok(false), Ok(false)));

        let bilinear = terms(vec![(y, x, 1.0)]);
        assert_eq!((bilinear.is_convex(), bilinear.is_concave()), (Ok(false), Ok(false)));

        let negated = terms(vec![(x, x, -1.0), (y, x, 2.0), (y, y, -1.0)]);
        assert_eq!((negated.is_convex(), negated.is_concave()), (Ok(false), Ok(true)));
    }

    #[test]
    fn curvature_is_relative_to_each_variable_and_not_the_largest_entry() {
        let (x, y) = (VarId(0), VarId(1));

        assert_eq!(terms(vec![(x, x, 1e20), (y, y, -1.0)]).is_convex(), Ok(false));

        // Det 1e20 - (1e10 + 1)^2 < 0.
        let b = 1e10 + 1.0;
        let coupled = terms(vec![(x, x, 0.5e20), (y, x, b), (y, y, 0.5)]);
        assert_eq!(coupled.is_convex(), Ok(false));

        // (1e10 x + y)^2, convex and singular.
        let square = terms(vec![(x, x, 1e20), (y, x, 2e10), (y, y, 1.0)]);
        assert_eq!(square.is_convex(), Ok(true));
        assert_eq!(terms(vec![(x, x, 1.0), (y, x, 1e-300)]).is_convex(), Ok(false));
    }

    #[test]
    fn curvature_checks_independent_blocks_beyond_the_dense_limit() {
        // sum (x_2i - x_2i+1)^2 over 1000 pairs.
        let pairs = (0..2000).step_by(2).flat_map(|i| {
            let (a, b) = (VarId(i), VarId(i + 1));
            [(a, a, 1.0), (b, a, -2.0), (b, b, 1.0)]
        });
        let separable = terms(pairs.collect());
        assert_eq!(separable.is_convex(), Ok(true));
        assert_eq!(separable.is_concave(), Ok(false));

        let max = u32::try_from(QuadraticTerms::MAX_CURVATURE_DIMENSION).unwrap();
        assert_eq!(chain(max).is_convex(), Ok(true));
    }

    #[test]
    fn curvature_rejects_oversized_blocks() {
        let n = QuadraticTerms::MAX_CURVATURE_DIMENSION + 1;
        assert_eq!(
            chain(u32::try_from(n).unwrap()).is_convex(),
            Err(CurvatureError::Dimension { dimension: n })
        );
    }
}
