//! Incremental affine construction and compensated coefficient accumulation.

use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::smallvec;

use crate::arena::{ArenaAccess, ExprId, ExprNode, VarId};
use crate::degree::AddDegree;
use crate::linear::{CoeffAccum, LinearTerms, add_n, as_linear, push_linear};
use crate::{Affine, Degree, Expr, ExprArenaCell};

/// Reusable accumulator for incremental affine construction.
///
/// Unlike repeated binary `+`, adding terms writes no arena nodes.
//// Numeric terms are merged at [`Self::build`] while parameter-dependent
/// terms stay symbolic. Construction is linear in the number of supplied
/// coefficient entries for unshared additive inputs.
/// Shared subexpressions use the usual extraction memoization.
///
/// ```
/// use oximo_expr::{AffineBuilder, Expr, ExprArena, ExprArenaCell, VarId};
/// let arena = ExprArenaCell::new(ExprArena::new());
/// let x = Expr::from_var(&arena, VarId(0));
/// let mut builder = AffineBuilder::new(&arena);
/// builder.add_term(2.0, x).add_constant(3.0);
/// let first = builder.build();
/// builder.add_term(4.0, x);
/// let second = builder.build(); // first is unchanged
/// ```
#[derive(Debug)]
pub struct AffineBuilder<'a> {
    arena: &'a ExprArenaCell,
    terms: Vec<(ExprId, f64)>,
    constant: f64,
    scratch: CoeffAccum,
    children: Vec<ExprId>,
    compensated: Option<Box<CompensatedAccum>>,
}

impl<'a> AffineBuilder<'a> {
    pub fn new(arena: &'a ExprArenaCell) -> Self {
        Self::with_capacity(arena, 0)
    }

    /// Reserve space for `capacity` terms and distinct numeric coefficients.
    pub fn with_capacity(arena: &'a ExprArenaCell, capacity: usize) -> Self {
        Self {
            arena,
            terms: Vec::with_capacity(capacity),
            constant: 0.0,
            scratch: CoeffAccum::with_capacity(capacity),
            children: Vec::new(),
            compensated: None,
        }
    }

    /// Accumulate original numeric terms using Neumaier compensated summation.
    pub fn new_compensated(arena: &'a ExprArenaCell) -> Self {
        Self::with_capacity_compensated(arena, 0)
    }

    /// Reserve a compensated builder for `capacity` terms/distinct variables.
    pub fn with_capacity_compensated(arena: &'a ExprArenaCell, capacity: usize) -> Self {
        let mut builder = Self::new(arena);
        builder.terms.reserve(capacity);
        builder.compensated = Some(Box::new(CompensatedAccum::with_capacity(capacity)));
        builder
    }

    /// Add a weighted constant or affine expression without writing arena nodes.
    /// Dynamic handles must first be checked with [`Expr::try_affine`].
    ///
    /// # Panics
    /// Panics before changing the builder if the expression belongs to another arena.
    pub fn add_term<D>(&mut self, coefficient: f64, expression: Expr<'a, D>) -> &mut Self
    where
        D: Degree + AddDegree<Affine, Output = Affine>,
    {
        assert!(
            std::ptr::eq(self.arena, expression.arena()),
            "expressions belong to different arenas"
        );
        self.terms.push((expression.id(), coefficient));
        self
    }

    pub fn add_constant(&mut self, value: f64) -> &mut Self {
        if let Some(acc) = &mut self.compensated {
            acc.constant.add(value);
        } else {
            self.constant += value;
        }
        self
    }

    /// Discard pending terms while retaining allocated scratch capacity.
    pub fn clear(&mut self) {
        self.terms.clear();
        self.constant = 0.0;
        self.scratch.clear();
        self.children.clear();
        if let Some(acc) = &mut self.compensated {
            acc.clear();
        }
    }

    /// Emit an expression and reset the builder, retaining scratch capacity.
    /// Numeric input emits one `Linear` node (or a constant), while symbolic
	/// input preserves ordered numeric segments under a flat sum.
    #[expect(clippy::float_cmp, reason = "only an exact unit weight can omit multiplication")]
    pub fn build(&mut self) -> Expr<'a, Affine> {
        let id = self.arena.with_mut(|arena| {
            if let Some(acc) = &mut self.compensated {
                for &(id, scalar) in &self.terms {
                    if let ExprNode::Var(var) = arena.get(id) {
                        acc.add(*var, scalar);
                        acc.constant.add(0.0 * scalar);
                    } else if let Some(terms) = as_linear(arena, id, false) {
                        for &(var, coefficient) in terms.coeffs.iter() {
                            acc.add(var, coefficient * scalar);
                        }
                        acc.constant.add(terms.constant * scalar);
                    } else {
                        acc.flush(arena, &mut self.children);
                        let child = if scalar == 1.0 {
                            id
                        } else {
                            let coefficient = arena.constant(scalar);
                            arena.push(ExprNode::Mul(smallvec![coefficient, id]))
                        };
                        self.children.push(child);
                    }
                }
                acc.flush(arena, &mut self.children);
                return match self.children.as_slice() {
                    [] => arena.constant(0.0),
                    [one] => *one,
                    _ => add_n(arena, self.children.iter().copied().collect()),
                };
            }
            let mut constant = self.constant;
            for &(id, scalar) in &self.terms {
                if let ExprNode::Var(var) = arena.get(id) {
                    self.scratch.add(*var, scalar);
                    constant += 0.0 * scalar;
                } else if let Some(terms) = as_linear(arena, id, false) {
                    for &(var, coefficient) in terms.coeffs.iter() {
                        self.scratch.add(var, coefficient * scalar);
                    }
                    constant += terms.constant * scalar;
                } else {
                    flush_segment(arena, &mut self.scratch, &mut constant, &mut self.children);
                    let child = if scalar == 1.0 {
                        id
                    } else {
                        let coefficient = arena.constant(scalar);
                        arena.push(ExprNode::Mul(smallvec![coefficient, id]))
                    };
                    self.children.push(child);
                }
            }
            flush_segment(arena, &mut self.scratch, &mut constant, &mut self.children);
            match self.children.as_slice() {
                [] => arena.constant(constant),
                [one] => *one,
                _ => add_n(arena, self.children.iter().copied().collect()),
            }
        });
        self.clear();
        Expr::from_id(id, self.arena)
    }
}

fn flush_segment(
    arena: &mut (impl ArenaAccess + ?Sized),
    acc: &mut CoeffAccum,
    constant: &mut f64,
    children: &mut Vec<ExprId>,
) {
    if !acc.coeffs.is_empty() || *constant != 0.0 {
        children.push(push_linear(arena, LinearTerms::owned(acc.coeffs.clone(), *constant)));
        acc.clear();
        *constant = 0.0;
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.is_finite() && value.is_finite() && next.is_finite() {
            self.correction += if self.sum.abs() >= value.abs() {
                (self.sum - next) + value
            } else {
                (value - next) + self.sum
            };
        } else {
            // Avoid introducing NaN solely from inf-inf in the correction.
            self.correction = 0.0;
        }
        self.sum = next;
    }

    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Debug)]
struct CompensatedAccum {
    coeffs: Vec<(VarId, CompensatedSum)>,
    slot: FxHashMap<VarId, usize>,
    constant: CompensatedSum,
}

impl CompensatedAccum {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            coeffs: Vec::with_capacity(capacity),
            slot: FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher),
            constant: CompensatedSum::default(),
        }
    }

    fn clear(&mut self) {
        self.coeffs.clear();
        self.slot.clear();
        self.constant = CompensatedSum::default();
    }

    fn add(&mut self, var: VarId, coefficient: f64) {
        let index = *self.slot.entry(var).or_insert_with(|| {
            self.coeffs.push((var, CompensatedSum::default()));
            self.coeffs.len() - 1
        });
        self.coeffs[index].1.add(coefficient);
    }

    fn flush(&mut self, arena: &mut (impl ArenaAccess + ?Sized), children: &mut Vec<ExprId>) {
        let constant = self.constant.value();
        if !self.coeffs.is_empty() || constant != 0.0 {
            let coeffs = self.coeffs.iter().map(|&(var, sum)| (var, sum.value())).collect();
            children.push(push_linear(arena, LinearTerms::owned(coeffs, constant)));
        }
        self.clear();
    }
}
