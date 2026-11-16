//! Incremental affine construction and compensated coefficient accumulation.

use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::{SmallVec, smallvec};

use crate::arena::{ArenaAccess, ExprId, ExprNode, VarId};
use crate::compensated_sum::CompensatedSum;
use crate::degree::AddDegree;
use crate::linear::{CoeffAccum, LinearTerms, add_n, as_linear, push_linear};
use crate::{Affine, Expr, ExprArenaCell};

/// Reusable accumulator for incremental affine construction.
///
/// Unlike repeated binary `+`, adding terms writes no arena nodes.
/// Numeric terms are merged at [`Self::build`] or [`Self::finish`] while parameter-dependent
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
        Self::with_capacities(arena, 0, 0)
    }

    /// Reserve space for `capacity` terms and distinct numeric coefficients.
    /// Use [`Self::with_capacities`] when these counts differ or the number of
    /// numeric coefficients is unknown.
    pub fn with_capacity(arena: &'a ExprArenaCell, capacity: usize) -> Self {
        Self::with_capacities(arena, capacity, capacity)
    }

    /// Reserve terms independently from distinct numeric coefficients.
    pub fn with_capacities(
        arena: &'a ExprArenaCell,
        term_capacity: usize,
        coefficient_capacity: usize,
    ) -> Self {
        Self {
            arena,
            terms: Vec::with_capacity(term_capacity),
            constant: 0.0,
            scratch: CoeffAccum::with_capacity(coefficient_capacity),
            children: Vec::new(),
            compensated: None,
        }
    }

    /// Accumulate original numeric terms using Neumaier compensated summation.
    pub fn new_compensated(arena: &'a ExprArenaCell) -> Self {
        Self::with_capacities_compensated(arena, 0, 0)
    }

    /// Reserve a compensated builder for `capacity` terms/distinct variables.
    /// Use [`Self::with_capacities_compensated`] for independent hints.
    pub fn with_capacity_compensated(arena: &'a ExprArenaCell, capacity: usize) -> Self {
        Self::with_capacities_compensated(arena, capacity, capacity)
    }

    /// Reserve compensated terms independently from distinct numeric coefficients.
    /// See [`Self::with_capacities`] for the meaning of each hint.
    pub fn with_capacities_compensated(
        arena: &'a ExprArenaCell,
        term_capacity: usize,
        coefficient_capacity: usize,
    ) -> Self {
        let mut builder = Self::new(arena);
        builder.terms.reserve(term_capacity);
        builder.compensated = Some(Box::new(CompensatedAccum::with_capacity(coefficient_capacity)));
        builder
    }

    /// Add a weighted constant or affine expression without writing arena nodes.
    /// Dynamic handles must first be checked with [`Expr::as_affine`].
    ///
    /// # Panics
    /// Panics before changing the builder if the expression belongs to another arena.
    pub fn add_term<E>(&mut self, coefficient: f64, expression: E) -> &mut Self
    where
        E: crate::ExprInput<'a>,
        E::Degree: AddDegree<Affine, Output = Affine>,
    {
        let expression = expression.as_expression();
        assert!(
            std::ptr::eq(self.arena, expression.arena()),
            "expressions belong to different arenas"
        );
        self.terms.push((expression.publish_id(), coefficient));
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

    pub(crate) fn reserve_terms(&mut self, additional: usize) {
        self.terms.reserve(additional);
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
    pub fn build(&mut self) -> Expr<'a, Affine> {
        let id = self.emit::<false>();
        self.clear();
        Expr::from_id(id, self.arena)
    }

    /// Emit an expression and consume the builder, reusing its final numeric
    /// coefficient storage where possible. Use [`Self::build`] to reuse the builder.
    pub fn finish(mut self) -> Expr<'a, Affine> {
        let id = self.emit::<true>();
        Expr::from_id(id, self.arena)
    }

    #[expect(clippy::float_cmp, reason = "only an exact unit weight can omit multiplication")]
    fn emit<const CONSUMING: bool>(&mut self) -> ExprId {
        self.arena.with_mut(|arena| {
            if let Some(acc) = &mut self.compensated {
                for &(id, scalar) in &self.terms {
                    if let ExprNode::Var(var) = arena.get(id) {
                        acc.add(*var, scalar);
                        acc.constant.add(0.0 * scalar);
                    } else if let Some(terms) = numeric_terms(arena, id) {
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
                    _ => add_n(arena, self.children.iter().copied().collect(), false).id,
                };
            }
            let mut constant = self.constant;
            for &(id, scalar) in &self.terms {
                if let ExprNode::Var(var) = arena.get(id) {
                    self.scratch.add(*var, scalar);
                    constant += 0.0 * scalar;
                } else if let Some(terms) = numeric_terms(arena, id) {
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
            if CONSUMING {
                let mut coeffs = std::mem::take(&mut self.scratch.coeffs);
                coeffs.shrink_to_fit();
                if self.children.is_empty() {
                    return push_linear(arena, LinearTerms::owned(coeffs, constant));
                }
                if !coeffs.is_empty() || constant != 0.0 {
                    self.children.push(push_linear(arena, LinearTerms::owned(coeffs, constant)));
                }
            } else {
                flush_segment(arena, &mut self.scratch, &mut constant, &mut self.children);
            }
            match self.children.as_slice() {
                [] => arena.constant(constant),
                [one] => *one,
                _ => add_n(arena, self.children.iter().copied().collect(), false).id,
            }
        })
    }
}

/// Extracts numeric affine coefficients and the constant term without resolving parameters.
/// Returns `None` if the expression cannot be represented in that form.
fn numeric_terms<'a>(
    arena: &'a (impl ArenaAccess + ?Sized),
    id: ExprId,
) -> Option<LinearTerms<'a>> {
    match arena.get(id) {
        ExprNode::Const(value) => return Some(LinearTerms::borrowed(&[], *value)),
        ExprNode::Linear { coeffs, constant } => {
            return Some(LinearTerms::borrowed(coeffs, *constant));
        }
        ExprNode::Param(_) => return None,
        _ => {}
    }
    let mut pending = SmallVec::<[ExprId; 16]>::from_slice(&[id]);
    let mut remaining = 64;
    while let Some(next) = pending.pop() {
        if remaining == 0 {
            break;
        }
        remaining -= 1;
        match arena.get(next) {
            ExprNode::Param(_) => return None,
            ExprNode::Add(children) | ExprNode::Mul(children) => {
                if pending.len() + children.len() > pending.inline_size() {
                    break;
                }
                pending.extend_from_slice(children);
            }
            ExprNode::Unary(crate::UnaryOp::Neg, child) => pending.push(*child),
            // Other operations can ignore their inputs (e.g. p^0). Do not
            // infer dependency through them; let the extractor decide.
            _ => {}
        }
    }
    as_linear(arena, id, false)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprArena, extract_linear};

    fn coefficient_capacity(builder: &AffineBuilder<'_>) -> usize {
        if let Some(acc) = &builder.compensated {
            acc.coeffs.capacity()
        } else {
            builder.scratch.coeffs.capacity()
        }
    }

    #[test]
    fn finish_transfers_exact_sized_coefficient_storage() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let mut builder = AffineBuilder::with_capacities(&arena, 16, 16);
        let coefficients = builder.scratch.coeffs.as_ptr();
        for i in 0..16 {
            builder.add_term(2.0, Expr::from_var(&arena, VarId(i)));
        }
        let expression = builder.finish();
        let snapshot = arena.borrow();
        let ExprNode::Linear { coeffs, .. } = snapshot.get(expression.id()) else {
            panic!("numeric finish must produce a Linear node");
        };
        assert_eq!(coeffs.len(), 16);
        assert_eq!(coeffs.as_ptr(), coefficients, "finish must transfer the allocation");
    }

    #[test]
    fn finish_does_not_retain_contribution_sized_coefficient_storage() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let mut builder = AffineBuilder::with_capacities(&arena, 4096, 4096);
        for _ in 0..4096 {
            builder.add_term(2.0, x);
        }
        let expression = builder.finish();
        let snapshot = arena.borrow();
        let ExprNode::Linear { coeffs, .. } = snapshot.get(expression.id()) else {
            panic!("numeric finish must produce a Linear node");
        };
        assert_eq!(coeffs.as_slice(), [(VarId(0), 8192.0)]);
        assert_eq!(coeffs.capacity(), 1);
    }

    #[test]
    fn repeated_contributions_reuse_small_coefficient_reservation() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let y = Expr::from_var(&arena, VarId(1));
        for compensated in [false, true] {
            let mut builder = if compensated {
                AffineBuilder::with_capacities_compensated(&arena, 4096, 2)
            } else {
                AffineBuilder::with_capacities(&arena, 4096, 2)
            };
            assert!(builder.terms.capacity() >= 4096);

            let reserved = coefficient_capacity(&builder);
            assert_eq!(reserved, 2);
            for _ in 0..2048 {
                builder.add_term(2.0, x).add_term(3.0, y);
            }
            let first = builder.build();
            let terms = extract_linear(&arena.borrow(), first.id()).unwrap().into_owned();
            assert_eq!(terms.coeffs.as_ref(), [(VarId(0), 4096.0), (VarId(1), 6144.0)]);
            assert_eq!(coefficient_capacity(&builder), reserved);

            builder.add_term(4.0, y);
            let second = builder.build();
            assert_eq!(
                extract_linear(&arena.borrow(), second.id()).unwrap().coeffs.as_ref(),
                [(VarId(1), 4.0)]
            );
            assert_eq!(coefficient_capacity(&builder), reserved);
        }
    }

    #[test]
    fn symbolic_contributions_leave_numeric_storage_unallocated_until_needed() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let parameter = arena.borrow_mut().new_param(2.0);
        let p = Expr::from_param(&arena, parameter);
        let x = Expr::from_var(&arena, VarId(0));
        for compensated in [false, true] {
            let mut builder = if compensated {
                AffineBuilder::with_capacities_compensated(&arena, 4096, 0)
            } else {
                AffineBuilder::with_capacities(&arena, 4096, 0)
            };
            for _ in 0..4096 {
                builder.add_term(2.0, p);
            }
            let symbolic = builder.build();
            assert_eq!(coefficient_capacity(&builder), 0);
            assert_eq!(builder.scratch.coeffs.capacity(), 0);

            if let Some(acc) = &builder.compensated {
                assert_eq!(acc.slot.capacity(), 0);
            }

            p.set_param_value(3.0).unwrap();
            assert_eq!(
                extract_linear(&arena.borrow(), symbolic.id()).unwrap().constant.to_bits(),
                (4096.0_f64 * 6.0).to_bits()
            );

            builder.add_term(5.0, x);
            let numeric = builder.build();
            assert_eq!(
                extract_linear(&arena.borrow(), numeric.id()).unwrap().coeffs.as_ref(),
                [(VarId(0), 5.0)]
            );
            assert!(coefficient_capacity(&builder) > 0);
        }
    }
}
