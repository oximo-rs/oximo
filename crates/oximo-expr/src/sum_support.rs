//! Implementation details for product fusion in indexed sums.

use std::marker::PhantomData;
use std::ops::Mul;

use crate::degree::AddDegree;
use crate::{Affine, AffineBuilder, Degree, Expr, ExprArenaCell, ExprInput, ExpressionDegree};

/// A term in an indexed sum, defining how it is evaluated and accumulated.
pub trait SumTerm<'a>: Sized {
    type Degree: ExpressionDegree;
    type Buffer;
    type Evaluated: SumTerm<'a, Degree = Self::Degree>;

    /// Keep unfiltered callback conversion/drop timing when loops are flattened.
    fn evaluate(self) -> Self::Evaluated;

    fn buffer(arena: Option<&'a ExprArenaCell>, capacity: usize) -> Self::Buffer;
    fn reserve(_buffer: &mut Self::Buffer, _additional: usize) {}
    fn push(buffer: &mut Self::Buffer, term: Self);
    fn finish(buffer: Self::Buffer) -> Option<Expr<'a, Self::Degree>>;

    fn sum_terms(
        terms: impl Iterator<Item = Self>,
        arena: Option<&'a ExprArenaCell>,
    ) -> Option<Expr<'a, Self::Degree>> {
        let mut collector = SumCollector::with_capacity(arena, terms.size_hint().0);
        for term in terms {
            collector.push(term);
        }
        collector.finish()
    }
}

pub fn evaluate_sum_term<'a, E: SumTerm<'a>>(term: E) -> E::Evaluated {
    term.evaluate()
}

/// Type-directed storage for generated loops, without staging weighted handles.
pub struct SumCollector<'a, E: SumTerm<'a>> {
    buffer: E::Buffer,
    marker: PhantomData<fn(&'a ()) -> E>,
}

impl<'a, E: SumTerm<'a>> std::fmt::Debug for SumCollector<'a, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SumCollector").finish_non_exhaustive()
    }
}

impl<'a, E: SumTerm<'a>> SumCollector<'a, E> {
    pub fn new(arena: Option<&'a ExprArenaCell>) -> Self {
        Self::with_capacity(arena, 0)
    }

    fn with_capacity(arena: Option<&'a ExprArenaCell>, capacity: usize) -> Self {
        Self { buffer: E::buffer(arena, capacity), marker: PhantomData }
    }

    /// Reserve terms using a hint from an already-created domain iterator.
    pub fn reserve(&mut self, additional: usize) {
        E::reserve(&mut self.buffer, additional);
    }

    pub fn push(&mut self, term: E) {
        E::push(&mut self.buffer, term);
    }

    pub fn finish(self) -> Option<Expr<'a, E::Degree>> {
        E::finish(self.buffer)
    }
}

impl<'a, E: ExprInput<'a>> SumTerm<'a> for E {
    type Degree = <E::Degree as Degree>::Computed;
    type Buffer = (Option<&'a ExprArenaCell>, Vec<E>);
    type Evaluated = Expr<'a, E::Degree>;

    fn evaluate(self) -> Self::Evaluated {
        self.into_expression()
    }

    fn buffer(arena: Option<&'a ExprArenaCell>, capacity: usize) -> Self::Buffer {
        (arena, Vec::with_capacity(capacity))
    }

    fn reserve((_, terms): &mut Self::Buffer, additional: usize) {
        terms.reserve(additional);
    }

    fn push((_, terms): &mut Self::Buffer, term: Self) {
        terms.push(term);
    }

    fn finish((arena, terms): Self::Buffer) -> Option<Expr<'a, Self::Degree>> {
        Self::sum_terms(terms.into_iter(), arena)
    }

    fn sum_terms(
        terms: impl Iterator<Item = Self>,
        arena: Option<&'a ExprArenaCell>,
    ) -> Option<Expr<'a, Self::Degree>> {
        match arena {
            Some(arena) => Some(Expr::__sum_terms_in(arena, terms)),
            None => Expr::__sum_terms(terms),
        }
    }
}

/// An explicitly weighted affine input for the modeling sum macro.
/// Construct it with [`weighted_term`].
#[derive(Debug)]
pub struct WeightedTerm<'a, D: Degree> {
    coefficient: f64,
    expression: Expr<'a, D>,
}

/// Select affine coefficient accumulation explicitly.
///
/// ```
/// use oximo_expr::{Affine, ExprInput, WeightedTerm, weighted_term};
/// fn term<'a, E: ExprInput<'a, Degree = Affine>>(input: E) -> WeightedTerm<'a, Affine> {
///     weighted_term(2.0, input)
/// }
/// ```
pub fn weighted_term<'a, E>(coefficient: f64, expression: E) -> WeightedTerm<'a, E::Degree>
where
    E: ExprInput<'a>,
    E::Degree: AddDegree<Affine, Output = Affine>,
{
    WeightedTerm { coefficient, expression: expression.into_expression() }
}

#[derive(Debug)]
pub struct WeightedBuffer<'a> {
    arena: Option<&'a ExprArenaCell>,
    anchored: bool,
    builder: Option<AffineBuilder<'a>>,
    same_arena: bool,
    term_capacity: usize,
}

impl<'a, D> SumTerm<'a> for WeightedTerm<'a, D>
where
    D: Degree + AddDegree<Affine, Output = Affine>,
{
    type Degree = D::Computed;
    type Buffer = WeightedBuffer<'a>;
    type Evaluated = Self;

    fn evaluate(self) -> Self::Evaluated {
        self
    }

    fn buffer(arena: Option<&'a ExprArenaCell>, capacity: usize) -> Self::Buffer {
        WeightedBuffer {
            arena,
            anchored: arena.is_some(),
            builder: None,
            same_arena: true,
            term_capacity: capacity,
        }
    }

    fn reserve(buffer: &mut Self::Buffer, additional: usize) {
        if let Some(builder) = &mut buffer.builder {
            builder.reserve_terms(additional);
        } else {
            buffer.term_capacity = buffer.term_capacity.max(additional);
        }
    }

    fn push(buffer: &mut Self::Buffer, term: Self) {
        let target = *buffer.arena.get_or_insert(term.expression.arena());
        if std::ptr::eq(target, term.expression.arena()) {
            buffer
                .builder
                .get_or_insert_with(|| {
                    // Constant-degree inputs cannot contribute numeric variable
                    // coefficients. We retain the existing estimate otherwise.
                    let coefficients =
                        if crate::degree::may_be_affine::<D>() { buffer.term_capacity } else { 0 };
                    AffineBuilder::with_capacities(target, buffer.term_capacity, coefficients)
                })
                .add_term(term.coefficient, term.expression);
        } else {
            // Do not reject until the remaining callbacks have also finished.
            buffer.same_arena = false;
        }
    }

    fn finish(buffer: Self::Buffer) -> Option<Expr<'a, Self::Degree>> {
        assert!(
            buffer.same_arena,
            "{}",
            if buffer.anchored {
                "sum! terms belong to a different model"
            } else {
                "expressions belong to different arenas"
            }
        );

        let target = buffer.arena?;
        let builder = buffer.builder.unwrap_or_else(|| AffineBuilder::new(target));

        Some(builder.finish().into_degree())
    }
}

/// A type-only receiver for stable autoref dispatch at the macro call site.
#[derive(Debug)]
pub struct ProductTag<L, R>(PhantomData<fn(L, R)>);

pub fn product_tag<L, R>(_: &(L, R)) -> ProductTag<L, R> {
    ProductTag(PhantomData)
}

pub trait SumProduct<L, R> {
    type Output;
    fn sum_product(self, operands: (L, R)) -> Self::Output;
}

impl<L: Mul<R>, R> SumProduct<L, R> for &ProductTag<L, R> {
    type Output = L::Output;

    fn sum_product(self, (left, right): (L, R)) -> Self::Output {
        left * right
    }
}

macro_rules! numeric_products {
    ($scalar:ty) => {
        impl<'a, D> SumProduct<$scalar, Expr<'a, D>> for &&ProductTag<$scalar, Expr<'a, D>>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = WeightedTerm<'a, D>;

            fn sum_product(
                self,
                (coefficient, expression): ($scalar, Expr<'a, D>),
            ) -> Self::Output {
                WeightedTerm { coefficient: f64::from(coefficient), expression }
            }
        }

        impl<'a, D> SumProduct<Expr<'a, D>, $scalar> for &&ProductTag<Expr<'a, D>, $scalar>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = WeightedTerm<'a, D>;

            fn sum_product(
                self,
                (expression, coefficient): (Expr<'a, D>, $scalar),
            ) -> Self::Output {
                WeightedTerm { coefficient: f64::from(coefficient), expression }
            }
        }

        impl<'a, 'b, D> SumProduct<$scalar, &'b Expr<'a, D>>
            for &&ProductTag<$scalar, &'b Expr<'a, D>>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = WeightedTerm<'a, D>;

            fn sum_product(
                self,
                (coefficient, expression): ($scalar, &'b Expr<'a, D>),
            ) -> Self::Output {
                WeightedTerm {
                    coefficient: f64::from(coefficient),
                    expression: expression.into_expression(),
                }
            }
        }

        impl<'a, 'b, D> SumProduct<&'b Expr<'a, D>, $scalar>
            for &&ProductTag<&'b Expr<'a, D>, $scalar>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = WeightedTerm<'a, D>;

            fn sum_product(
                self,
                (expression, coefficient): (&'b Expr<'a, D>, $scalar),
            ) -> Self::Output {
                WeightedTerm {
                    coefficient: f64::from(coefficient),
                    expression: expression.into_expression(),
                }
            }
        }
    };
}

numeric_products!(f64);
numeric_products!(i32);
