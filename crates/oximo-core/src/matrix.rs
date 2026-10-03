//! Symmetric matrices with ordinary (unscaled) entries.
//!
//! [`SymmetricMatrix<T>`] stores the `n(n+1)/2` independent entries of a real
//! symmetric matrix. `(i, j)` and `(j, i)` access the same stored value. For
//! expression matrices, this shares the original scalar variable or expression.
//!
//! # Triangle ordering
//!
//! Constructors require a positive side dimension and an exact entry count.
//! Upper-column order for a 3-by-3 matrix is
//! `[a00, a01, a11, a02, a12, a22]`. Lower-column order is
//! `[a00, a10, a20, a11, a21, a22]`. [`SymmetricMatrix::from_lower_triangle`]
//! converts it to the same canonical storage. [`SymmetricMatrix::from_upper_fn`]
//! calls its rule once per independent entry, with `i <= j`.
//!
//! ```
//! use oximo_core::SymmetricMatrix;
//!
//! let a = SymmetricMatrix::from_upper_triangle(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! let b = SymmetricMatrix::from_lower_triangle(3, [1.0, 2.0, 4.0, 3.0, 5.0, 6.0]);
//! assert_eq!(a, b);
//! assert_eq!(a[(0, 2)], a[(2, 0)]);
//! assert_eq!(a.trace(), 10.0);
//! assert_eq!(a.frobenius(&b), 136.0);
//! ```
//!
//! For dense row data, [`SymmetricMatrix::try_from_rows`] checks that the input
//! is nonempty, square and exactly symmetric before storing its upper triangle.
//! Numeric identity matrices are available through [`SymmetricMatrix::identity`].
//!
//! ```
//! use oximo_core::SymmetricMatrix;
//!
//! let a = SymmetricMatrix::try_from_rows([[2.0, 1.0], [1.0, 2.0]])?;
//! let identity = SymmetricMatrix::<f64>::identity(2);
//! assert_eq!(format!("{a}"), "[[2, 1], [1, 2]]");
//! assert_eq!(format!("{identity:.1}"), "[[1.0, 0.0], [0.0, 1.0]]");
//! # Ok::<(), oximo_core::SymmetricMatrixError>(())
//! ```
//!
//! # Matrix expressions
//!
//! Addition and subtraction are entrywise and require equal dimensions.
//! Scalar multiplication and negation propagate the scalar expression degree.
//! [`SymmetricMatrix::trace`] sums the diagonal.
//! [`SymmetricMatrix::frobenius`] computes `trace(A' * B)`, counting each
//! off-diagonal product twice. These operations work with symbolic entries,
//! so parameter expressions remain rebindable.
//!
//! ```
//! use oximo_core::prelude::*;
//!
//! let model = Model::new("matrix shift");
//! let x = model.add_symmetric_variable("X", 2);
//! variable!(model, t);
//! let identity = SymmetricMatrix::<f64>::identity(2);
//! let shifted = &x - t * &identity;
//! let cone = model.add_psd_constraint("shifted", &shifted);
//! assert_eq!(model.psd_constraints()[cone.index()].matrix.side_dimension(), 2);
//! objective!(model, Min, x.trace());
//! ```
//!
//! See [`crate::psd`] for PSD membership, typed lowering and supported backends.

use std::fmt;
use std::ops::{Add, Index, Mul, Neg, Sub};

/// A symmetric matrix stored as its unscaled upper triangle, column by column.
/// Mirrored indices always refer to the same entry.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetricMatrix<T> {
    side_dimension: usize,
    entries: Vec<T>,
}

/// Invalid dense row input for a symmetric matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SymmetricMatrixError {
    #[error("symmetric matrix must have at least one row")]
    Empty,
    #[error("matrix row {row} has {actual} entries; expected {expected} for a square matrix")]
    NonSquare { row: usize, expected: usize, actual: usize },
    #[error("matrix entries at ({row}, {column}) and ({column}, {row}) differ")]
    NotSymmetric { row: usize, column: usize },
    #[error("symmetric matrix triangle dimension overflow")]
    DimensionOverflow,
}

/// Checked number of independent entries in a nonempty symmetric matrix.
///
/// # Panics
/// Panics for zero dimensions or arithmetic overflow.
pub fn triangle_len(n: usize) -> usize {
    assert!(n > 0, "symmetric matrix dimension must be positive");
    checked_triangle_len(n).expect("triangle dimension overflow")
}

fn checked_triangle_len(n: usize) -> Option<usize> {
    let next = n.checked_add(1)?;
    let (a, b) = if n.is_multiple_of(2) { (n / 2, next) } else { (n, next / 2) };
    a.checked_mul(b)
}

impl<T> SymmetricMatrix<T> {
    /// Construct from dense rows, checking square shape and exact symmetry.
    ///
    /// Accepts owned or borrowed rows, including arrays, vectors and slices.
    /// Mirrored entries are compared with `PartialEq` without a tolerance.
    /// This checks symmetry, not positive semidefiniteness or finite values.
    ///
    /// # Errors
    /// Returns [`SymmetricMatrixError`] for empty, nonsquare or asymmetric input,
    /// or when the independent entry count overflows `usize`. Error coordinates
    /// are zero-based.
    pub fn try_from_rows<R: AsRef<[T]>>(
        rows: impl IntoIterator<Item = R>,
    ) -> Result<Self, SymmetricMatrixError>
    where
        T: Clone + PartialEq,
    {
        let rows: Vec<_> = rows.into_iter().collect();
        let n = rows.len();
        if n == 0 {
            return Err(SymmetricMatrixError::Empty);
        }
        let len = checked_triangle_len(n).ok_or(SymmetricMatrixError::DimensionOverflow)?;
        for (row, entries) in rows.iter().enumerate() {
            let actual = entries.as_ref().len();
            if actual != n {
                return Err(SymmetricMatrixError::NonSquare { row, expected: n, actual });
            }
        }
        let mut entries = Vec::with_capacity(len);
        for (column, mirrored_row) in rows.iter().enumerate() {
            for (row, source_row) in rows.iter().enumerate().take(column + 1) {
                let entry = &source_row.as_ref()[column];
                if row != column && entry != &mirrored_row.as_ref()[row] {
                    return Err(SymmetricMatrixError::NotSymmetric { row, column });
                }
                entries.push(entry.clone());
            }
        }
        Ok(Self { side_dimension: n, entries })
    }

    /// Construct from unscaled upper-triangle entries in column order.
    ///
    /// # Panics
    /// Panics for an invalid dimension or entry count.
    pub fn from_upper_triangle(n: usize, entries: impl IntoIterator<Item = T>) -> Self {
        let len = triangle_len(n);
        let entries: Vec<_> = entries.into_iter().collect();
        assert_eq!(entries.len(), len, "symmetric matrix triangle length mismatch");
        Self { side_dimension: n, entries }
    }

    /// Construct from unscaled lower-triangle entries in column order.
    ///
    /// # Panics
    /// Panics for an invalid dimension or entry count.
    pub fn from_lower_triangle(n: usize, entries: impl IntoIterator<Item = T>) -> Self {
        let len = triangle_len(n);
        let mut entries: Vec<_> = entries.into_iter().map(Some).collect();
        assert_eq!(entries.len(), len, "symmetric matrix triangle length mismatch");
        Self::from_upper_fn(n, |i, j| {
            let start = len - triangle_len_or_zero(n - i);
            entries[start + j - i].take().expect("triangle entry visited once")
        })
    }

    /// Evaluate `f(i, j)` only for `i <= j`, in upper-column order.
    ///
    /// # Panics
    /// Panics for an invalid dimension.
    pub fn from_upper_fn(n: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut entries = Vec::with_capacity(triangle_len(n));
        for j in 0..n {
            for i in 0..=j {
                entries.push(f(i, j));
            }
        }
        Self { side_dimension: n, entries }
    }

    pub const fn side_dimension(&self) -> usize {
        self.side_dimension
    }

    /// Ordinary upper-triangle entries, never solver-scaled coordinates.
    pub fn upper_triangle(&self) -> &[T] {
        &self.entries
    }

    /// Consume the matrix into its ordinary upper triangle.
    pub fn into_upper_triangle(self) -> Vec<T> {
        self.entries
    }

    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> SymmetricMatrix<U> {
        SymmetricMatrix {
            side_dimension: self.side_dimension,
            entries: self.entries.iter().map(f).collect(),
        }
    }

    /// Look up an entry; mirrored indices share storage.
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i >= self.side_dimension || j >= self.side_dimension {
            return None;
        }
        let (i, j) = (i.min(j), i.max(j));
        self.entries.get(triangle_len_or_zero(j) + i)
    }

    /// Sum the diagonal entries.
    pub fn trace(&self) -> T
    where
        T: Copy + Add<Output = T>,
    {
        (1..self.side_dimension).fold(self[(0, 0)], |sum, i| sum + self[(i, i)])
    }

    /// Frobenius inner product `trace(self' * rhs)`, counting off-diagonals twice.
    ///
    /// # Panics
    /// Panics when the dimensions differ.
    pub fn frobenius<R: Copy, O>(&self, rhs: &SymmetricMatrix<R>) -> O
    where
        T: Copy + Mul<R, Output = O>,
        O: Add<Output = O> + Mul<f64, Output = O>,
    {
        assert_eq!(self.side_dimension, rhs.side_dimension, "matrix dimension mismatch");
        let mut sum = self[(0, 0)] * rhs[(0, 0)];
        for j in 1..self.side_dimension {
            for i in 0..=j {
                let term = self[(i, j)] * rhs[(i, j)];
                sum = sum + term * if i == j { 1.0 } else { 2.0 };
            }
        }
        sum
    }
}

impl<T: num_traits::Zero + num_traits::One> SymmetricMatrix<T> {
    /// Construct an identity matrix with ones on the diagonal and zeros elsewhere.
    ///
    /// # Panics
    /// Panics for a zero side dimension or triangle entry count overflow.
    pub fn identity(n: usize) -> Self {
        Self::from_upper_fn(n, |i, j| if i == j { T::one() } else { T::zero() })
    }
}

/// Format full rows, including mirrored entries. Precision and other element
/// formatting options are forwarded to each entry.
impl<T: fmt::Display> fmt::Display for SymmetricMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for row in 0..self.side_dimension {
            if row > 0 {
                f.write_str(", ")?;
            }
            f.write_str("[")?;
            for column in 0..self.side_dimension {
                if column > 0 {
                    f.write_str(", ")?;
                }
                fmt::Display::fmt(&self[(row, column)], f)?;
            }
            f.write_str("]")?;
        }
        f.write_str("]")
    }
}

fn triangle_len_or_zero(n: usize) -> usize {
    if n == 0 { 0 } else { triangle_len(n) }
}

impl<T> Index<(usize, usize)> for SymmetricMatrix<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        self.get(i, j).expect("symmetric matrix index out of bounds")
    }
}

// Modeling macro index sugar borrows each tuple component.
impl<T> Index<(&usize, &usize)> for SymmetricMatrix<T> {
    type Output = T;
    fn index(&self, (i, j): (&usize, &usize)) -> &T {
        &self[(*i, *j)]
    }
}

macro_rules! matrix_binary {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<L: Copy + $trait<R>, R: Copy> $trait<&SymmetricMatrix<R>> for &SymmetricMatrix<L> {
            type Output = SymmetricMatrix<L::Output>;
            fn $method(self, rhs: &SymmetricMatrix<R>) -> Self::Output {
                assert_eq!(self.side_dimension, rhs.side_dimension, "matrix dimension mismatch");
                SymmetricMatrix::from_upper_triangle(self.side_dimension,
                    self.entries.iter().zip(&rhs.entries).map(|(&l, &r)| l $op r))
            }
        }
        impl<L: Copy + $trait<R>, R: Copy> $trait<SymmetricMatrix<R>> for SymmetricMatrix<L> {
            type Output = SymmetricMatrix<L::Output>;
            fn $method(self, rhs: SymmetricMatrix<R>) -> Self::Output { &self $op &rhs }
        }
        impl<L: Copy + $trait<R>, R: Copy> $trait<&SymmetricMatrix<R>> for SymmetricMatrix<L> {
            type Output = SymmetricMatrix<L::Output>;
            fn $method(self, rhs: &SymmetricMatrix<R>) -> Self::Output { &self $op rhs }
        }
        impl<L: Copy + $trait<R>, R: Copy> $trait<SymmetricMatrix<R>> for &SymmetricMatrix<L> {
            type Output = SymmetricMatrix<L::Output>;
            fn $method(self, rhs: SymmetricMatrix<R>) -> Self::Output { self $op &rhs }
        }
    };
}
matrix_binary!(Add, add, +);
matrix_binary!(Sub, sub, -);

mod sealed {
    pub trait Scalar {}
}
/// Scalar types accepted by symmetric matrix multiplication.
pub trait MatrixScalar: Copy + sealed::Scalar {}
macro_rules! scalars {
    ($($t:ty),*) => { $(impl sealed::Scalar for $t {} impl MatrixScalar for $t {})* };
}
scalars!(f64, f32, i32, u32, i64, usize);
impl<D: oximo_expr::Degree> sealed::Scalar for oximo_expr::Expr<'_, D> {}
impl<D: oximo_expr::Degree> MatrixScalar for oximo_expr::Expr<'_, D> {}

impl<T: Copy + Mul<S>, S: MatrixScalar> Mul<S> for &SymmetricMatrix<T> {
    type Output = SymmetricMatrix<T::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        self.map(|&entry| entry * rhs)
    }
}
impl<T: Copy + Mul<S>, S: MatrixScalar> Mul<S> for SymmetricMatrix<T> {
    type Output = SymmetricMatrix<T::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        &self * rhs
    }
}
impl<T: Copy + Neg> Neg for &SymmetricMatrix<T> {
    type Output = SymmetricMatrix<T::Output>;
    fn neg(self) -> Self::Output {
        self.map(|&entry| -entry)
    }
}
impl<T: Copy + Neg> Neg for SymmetricMatrix<T> {
    type Output = SymmetricMatrix<T::Output>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

macro_rules! scalar_left {
    ($($t:ty),*) => { $(
        impl<T: Copy> Mul<&SymmetricMatrix<T>> for $t where $t: Mul<T> {
            type Output = SymmetricMatrix<<$t as Mul<T>>::Output>;
            fn mul(self, rhs: &SymmetricMatrix<T>) -> Self::Output { rhs.map(|&entry| self * entry) }
        }
        impl<T: Copy> Mul<SymmetricMatrix<T>> for $t where $t: Mul<T> {
            type Output = SymmetricMatrix<<$t as Mul<T>>::Output>;
            fn mul(self, rhs: SymmetricMatrix<T>) -> Self::Output { rhs.map(|&entry| self * entry) }
        }
    )* };
}
scalar_left!(f64, f32, i32, u32, i64, usize);
impl<'a, D: oximo_expr::Degree, T: Copy> Mul<&SymmetricMatrix<T>> for oximo_expr::Expr<'a, D>
where
    Self: Mul<T>,
{
    type Output = SymmetricMatrix<<Self as Mul<T>>::Output>;
    fn mul(self, rhs: &SymmetricMatrix<T>) -> Self::Output {
        rhs.map(|&entry| self * entry)
    }
}
impl<'a, D: oximo_expr::Degree, T: Copy> Mul<SymmetricMatrix<T>> for oximo_expr::Expr<'a, D>
where
    Self: Mul<T>,
{
    type Output = SymmetricMatrix<<Self as Mul<T>>::Output>;
    fn mul(self, rhs: SymmetricMatrix<T>) -> Self::Output {
        rhs.map(|&entry| self * entry)
    }
}
