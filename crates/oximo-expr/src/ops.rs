use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::arena::ExprId;
use crate::handle::Expr;
use crate::linear::{add_into, add_n, div_into, mul_into, neg_into, sub_into};

// -----------------------------------------------------------------------------
// Expr <op> Expr
// -----------------------------------------------------------------------------

impl<'a> Add for Expr<'a> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let id = add_into(&mut self.arena.borrow_mut(), self.id, rhs.id);
        Self::new(id, self.arena)
    }
}

impl<'a> Sub for Expr<'a> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let id = sub_into(&mut self.arena.borrow_mut(), self.id, rhs.id);
        Self::new(id, self.arena)
    }
}

impl<'a> Mul for Expr<'a> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let id = mul_into(&mut self.arena.borrow_mut(), self.id, rhs.id);
        Self::new(id, self.arena)
    }
}

impl<'a> Div for Expr<'a> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let id = div_into(&mut self.arena.borrow_mut(), self.id, rhs.id);
        Self::new(id, self.arena)
    }
}

impl<'a> Neg for Expr<'a> {
    type Output = Self;
    fn neg(self) -> Self {
        let id = neg_into(&mut self.arena.borrow_mut(), self.id);
        Self::new(id, self.arena)
    }
}

// -----------------------------------------------------------------------------
// Expr <op> f64 / f64 <op> Expr, and the same for i32 because `2 * x`
// without type annotation is the most common ergonomic case.
// -----------------------------------------------------------------------------

macro_rules! impl_scalar_ops {
    ($scalar:ty, $to_f64:expr) => {
        impl<'a> Add<$scalar> for Expr<'a> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                let id = {
                    let mut a = self.arena.borrow_mut();
                    let rhs_id = a.constant($to_f64(rhs));
                    add_into(&mut a, self.id, rhs_id)
                };
                Self::new(id, self.arena)
            }
        }

        impl<'a> Add<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn add(self, rhs: Expr<'a>) -> Expr<'a> {
                rhs + self
            }
        }

        impl<'a> Sub<$scalar> for Expr<'a> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                let id = {
                    let mut a = self.arena.borrow_mut();
                    let rhs_id = a.constant($to_f64(rhs));
                    sub_into(&mut a, self.id, rhs_id)
                };
                Self::new(id, self.arena)
            }
        }

        impl<'a> Sub<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn sub(self, rhs: Expr<'a>) -> Expr<'a> {
                let id = {
                    let mut a = rhs.arena.borrow_mut();
                    let lhs_id = a.constant($to_f64(self));
                    sub_into(&mut a, lhs_id, rhs.id)
                };
                Expr::new(id, rhs.arena)
            }
        }

        impl<'a> Mul<$scalar> for Expr<'a> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                let id = {
                    let mut a = self.arena.borrow_mut();
                    let rhs_id = a.constant($to_f64(rhs));
                    mul_into(&mut a, self.id, rhs_id)
                };
                Self::new(id, self.arena)
            }
        }

        impl<'a> Mul<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn mul(self, rhs: Expr<'a>) -> Expr<'a> {
                rhs * self
            }
        }

        impl<'a> Div<$scalar> for Expr<'a> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                let id = {
                    let mut a = self.arena.borrow_mut();
                    let rhs_id = a.constant($to_f64(rhs));
                    div_into(&mut a, self.id, rhs_id)
                };
                Self::new(id, self.arena)
            }
        }

        impl<'a> Div<Expr<'a>> for $scalar {
            type Output = Expr<'a>;
            fn div(self, rhs: Expr<'a>) -> Expr<'a> {
                let id = {
                    let mut a = rhs.arena.borrow_mut();
                    let lhs_id = a.constant($to_f64(self));
                    div_into(&mut a, lhs_id, rhs.id)
                };
                Expr::new(id, rhs.arena)
            }
        }
    };
}

impl_scalar_ops!(f64, core::convert::identity);
impl_scalar_ops!(i32, f64::from);

// -----------------------------------------------------------------------------
// std::iter::Sum: the first element of the iterator carries the arena handle,
// so no external zero is required. Collected into a single flat n-ary `Add`.
// -----------------------------------------------------------------------------

impl<'a> std::iter::Sum for Expr<'a> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let items: Vec<Self> = iter.collect();
        let first = *items.first().expect("Expr::sum on empty iterator");
        let ids: Vec<ExprId> = items.iter().map(|e| e.id).collect();
        let id = add_n(&mut first.arena.borrow_mut(), &ids);
        Self::new(id, first.arena)
    }
}

impl<'a, 'b> std::iter::Sum<&'b Expr<'a>> for Expr<'a> {
    fn sum<I: Iterator<Item = &'b Expr<'a>>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

/// Dot product of expressions with scalar coefficients: `sum_{i} c_i * e_i`.
///
/// Both arguments are slices. Pass owned containers by reference:
/// `&vec`, `vec.as_slice()`, or `&array`.
///
/// # Panics
/// Panics if `exprs` and `coeffs` have different lengths, or if `exprs`
/// is empty (the result needs an arena handle).
pub fn dot<'a>(exprs: &[Expr<'a>], coeffs: &[f64]) -> Expr<'a> {
    assert_eq!(
        exprs.len(),
        coeffs.len(),
        "dot: length mismatch (exprs.len() = {}, coeffs.len() = {})",
        exprs.len(),
        coeffs.len(),
    );
    exprs.iter().zip(coeffs).map(|(e, c)| *c * *e).sum()
}
