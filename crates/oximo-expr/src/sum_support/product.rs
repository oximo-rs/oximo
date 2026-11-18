//! Staged multiplication where only known oximo operations may remain deferred.

use std::marker::PhantomData;
use std::ops::Mul;

use super::{SumTerm, WeightedTerm, sealed};
use crate::degree::{AddDegree, MulDegree};
use crate::{
    Affine, Degree, Expr, ExprArenaCell, ExprInput, ExpressionDegree, QuadraticBuilder,
    QuadraticTerms,
};

/// An ordinary operand or operator result, retaining its original Rust type.
#[derive(Debug)]
pub struct ProductValue<T>(pub T);

impl<'a, T: SumTerm<'a>> sealed::Term<'a> for ProductValue<T> {}

impl<'a, T: SumTerm<'a>> SumTerm<'a> for ProductValue<T> {
    type Degree = T::Degree;
    type Buffer = T::Buffer;
    type Evaluated = ProductValue<T::Evaluated>;

    fn evaluate(self) -> Self::Evaluated {
        ProductValue(self.0.evaluate())
    }

    fn buffer(arena: Option<&'a ExprArenaCell>, capacity: usize) -> Self::Buffer {
        T::buffer(arena, capacity)
    }

    fn reserve(buffer: &mut Self::Buffer, additional: usize) {
        T::reserve(buffer, additional);
    }

    fn push(buffer: &mut Self::Buffer, term: Self) {
        T::push(buffer, term.0);
    }

    fn finish(buffer: Self::Buffer) -> Option<Expr<'a, Self::Degree>> {
        T::finish(buffer)
    }

    fn sum_terms(
        terms: impl Iterator<Item = Self>,
        arena: Option<&'a ExprArenaCell>,
    ) -> Option<Expr<'a, Self::Degree>> {
        T::sum_terms(terms.map(|term| term.0), arena)
    }
}

/// Internal operands normalize to their original type for ordinary multiplication.
pub trait ProductOperand {
    type Original;
    fn materialize(self) -> Self::Original;
}

impl<T> ProductOperand for ProductValue<T> {
    type Original = T;
    fn materialize(self) -> T {
        self.0
    }
}

impl<'a, D: Degree> ProductOperand for WeightedTerm<'a, D> {
    type Original = Expr<'a, D::Computed>;
    fn materialize(self) -> Self::Original {
        self.expression * self.coefficient
    }
}

/// A type-only receiver for stable autoref dispatch at the macro call site.
#[derive(Debug)]
pub struct ProductTag<L, R>(PhantomData<fn(L, R)>);

impl<L, R> ProductTag<L, R> {
    pub fn bind_finish<O, F: FnOnce(R) -> O>(&self, finish: F) -> F {
        finish
    }
}

pub fn product_tag<L, R>(_: &(L, R)) -> ProductTag<L, R> {
    ProductTag(PhantomData)
}

/// Infer the later RHS type without evaluating or capturing its expression.
pub fn product_left_tag<L, R>(_: &L, _: PhantomData<R>) -> ProductTag<L, R> {
    ProductTag(PhantomData)
}

pub fn product_right_type<R>(_: PhantomData<R>, _: &R) {}

pub fn finish_product<R, O>(finish: impl FnOnce(R) -> O, right: R) -> O {
    finish(right)
}

/// Sealed dispatch for deferred products and ordinary multiplication.
pub trait SumProduct<L, R>: sealed::Product<L, R> {
    type Output;
    type PreparedLeft;
    /// Complete an unfused inner operation before evaluating the next operand.
    fn prepare_left(self, left: L) -> Self::PreparedLeft;
    fn sum_product(self, operands: (L, R)) -> Self::Output;
}

impl<L: ProductOperand, R: ProductOperand> sealed::Product<L, R> for &ProductTag<L, R> where
    L::Original: Mul<R::Original>
{
}

impl<L: ProductOperand, R: ProductOperand> SumProduct<L, R> for &ProductTag<L, R>
where
    L::Original: Mul<R::Original>,
{
    type Output = ProductValue<<L::Original as Mul<R::Original>>::Output>;
    type PreparedLeft = ProductValue<L::Original>;

    fn prepare_left(self, left: L) -> Self::PreparedLeft {
        ProductValue(left.materialize())
    }

    fn sum_product(self, (left, right): (L, R)) -> Self::Output {
        ProductValue(left.materialize() * right.materialize())
    }
}

macro_rules! numeric_products {
    ($scalar:ty, $input:ty) => {
        impl<'a, 'b, D> sealed::Product<ProductValue<$scalar>, ProductValue<$input>>
            for &&ProductTag<ProductValue<$scalar>, ProductValue<$input>>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
        }

        impl<'a, 'b, D> SumProduct<ProductValue<$scalar>, ProductValue<$input>>
            for &&ProductTag<ProductValue<$scalar>, ProductValue<$input>>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = WeightedTerm<'a, D>;
            type PreparedLeft = ProductValue<$scalar>;

            fn prepare_left(self, left: ProductValue<$scalar>) -> Self::PreparedLeft {
                left
            }

            fn sum_product(
                self,
                (ProductValue(coefficient), ProductValue(expression)): (
                    ProductValue<$scalar>,
                    ProductValue<$input>,
                ),
            ) -> Self::Output {
                WeightedTerm {
                    coefficient: f64::from(coefficient),
                    expression: expression.into_expression(),
                }
            }
        }

        impl<'a, 'b, D> sealed::Product<ProductValue<$input>, ProductValue<$scalar>>
            for &&ProductTag<ProductValue<$input>, ProductValue<$scalar>>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
        }

        impl<'a, 'b, D> SumProduct<ProductValue<$input>, ProductValue<$scalar>>
            for &&ProductTag<ProductValue<$input>, ProductValue<$scalar>>
        where
            D: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = WeightedTerm<'a, D>;
            type PreparedLeft = ProductValue<$input>;

            fn prepare_left(self, left: ProductValue<$input>) -> Self::PreparedLeft {
                left
            }

            fn sum_product(
                self,
                (ProductValue(expression), ProductValue(coefficient)): (
                    ProductValue<$input>,
                    ProductValue<$scalar>,
                ),
            ) -> Self::Output {
                WeightedTerm {
                    coefficient: f64::from(coefficient),
                    expression: expression.into_expression(),
                }
            }
        }
    };
}

numeric_products!(f64, Expr<'a, D>);
numeric_products!(i32, Expr<'a, D>);
numeric_products!(f64, &'b Expr<'a, D>);
numeric_products!(i32, &'b Expr<'a, D>);

#[derive(Clone, Copy, Debug)]
enum Order {
    Plain,
    WeightLeft,
    WeightRight,
    WeightProduct,
}

/// Two affine factors, with the source grouping retained for symbolic fallback.
#[derive(Debug)]
pub struct QuadraticProduct<'a, D: ExpressionDegree> {
    left: Expr<'a, Affine>,
    right: Expr<'a, Affine>,
    weight: f64,
    order: Order,
    degree: PhantomData<D>,
}

impl<'a, D: ExpressionDegree> ProductOperand for QuadraticProduct<'a, D> {
    type Original = Expr<'a, D>;

    fn materialize(self) -> Self::Original {
        let mut left = self.left;
        let mut right = self.right;
        match self.order {
            Order::WeightLeft => left = left * self.weight,
            Order::WeightRight => right = right * self.weight,
            Order::Plain | Order::WeightProduct => {}
        }
        let product = left * right;
        match self.order {
            Order::WeightProduct => (product * self.weight).into_degree(),
            _ => product.into_degree(),
        }
    }
}

macro_rules! affine_products {
    ($left:ty, $right:ty) => {
        impl<'a, 'b, 'c, L, R> sealed::Product<ProductValue<$left>, ProductValue<$right>>
            for &&ProductTag<ProductValue<$left>, ProductValue<$right>>
        where
            L: Degree + AddDegree<Affine, Output = Affine> + MulDegree<R>,
            R: Degree + AddDegree<Affine, Output = Affine>,
        {
        }

        impl<'a, 'b, 'c, L, R> SumProduct<ProductValue<$left>, ProductValue<$right>>
            for &&ProductTag<ProductValue<$left>, ProductValue<$right>>
        where
            L: Degree + AddDegree<Affine, Output = Affine> + MulDegree<R>,
            R: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = QuadraticProduct<'a, <L as MulDegree<R>>::Output>;
            type PreparedLeft = ProductValue<$left>;

            fn prepare_left(self, left: ProductValue<$left>) -> Self::PreparedLeft {
                left
            }

            fn sum_product(
                self,
                (ProductValue(left), ProductValue(right)): (
                    ProductValue<$left>,
                    ProductValue<$right>,
                ),
            ) -> Self::Output {
                QuadraticProduct {
                    left: left.into_expression().into_degree(),
                    right: right.into_expression().into_degree(),
                    weight: 1.0,
                    order: Order::Plain,
                    degree: PhantomData,
                }
            }
        }
    };
}

affine_products!(Expr<'a, L>, Expr<'a, R>);
affine_products!(&'b Expr<'a, L>, Expr<'a, R>);
affine_products!(Expr<'a, L>, &'c Expr<'a, R>);
affine_products!(&'b Expr<'a, L>, &'c Expr<'a, R>);

macro_rules! weighted_products {
    ($input:ty) => {
        impl<'a, 'b, L, R> sealed::Product<WeightedTerm<'a, L>, ProductValue<$input>>
            for &&ProductTag<WeightedTerm<'a, L>, ProductValue<$input>>
        where
            L: Degree + AddDegree<Affine, Output = Affine>,
            L::Computed: MulDegree<R>,
            R: Degree + AddDegree<Affine, Output = Affine>,
        {
        }

        impl<'a, 'b, L, R> SumProduct<WeightedTerm<'a, L>, ProductValue<$input>>
            for &&ProductTag<WeightedTerm<'a, L>, ProductValue<$input>>
        where
            L: Degree + AddDegree<Affine, Output = Affine>,
            L::Computed: MulDegree<R>,
            R: Degree + AddDegree<Affine, Output = Affine>,
        {
            type Output = QuadraticProduct<'a, <L::Computed as MulDegree<R>>::Output>;
            type PreparedLeft = WeightedTerm<'a, L>;

            fn prepare_left(self, left: WeightedTerm<'a, L>) -> Self::PreparedLeft {
                left
            }

            fn sum_product(
                self,
                (left, ProductValue(right)): (WeightedTerm<'a, L>, ProductValue<$input>),
            ) -> Self::Output {
                QuadraticProduct {
                    left: left.expression.into_degree(),
                    right: right.into_expression().into_degree(),
                    weight: left.coefficient,
                    order: Order::WeightLeft,
                    degree: PhantomData,
                }
            }
        }

        impl<'a, 'b, L, R> sealed::Product<ProductValue<$input>, WeightedTerm<'a, L>>
            for &&ProductTag<ProductValue<$input>, WeightedTerm<'a, L>>
        where
            L: Degree + AddDegree<Affine, Output = Affine>,
            R: Degree + AddDegree<Affine, Output = Affine> + MulDegree<L::Computed>,
        {
        }

        impl<'a, 'b, L, R> SumProduct<ProductValue<$input>, WeightedTerm<'a, L>>
            for &&ProductTag<ProductValue<$input>, WeightedTerm<'a, L>>
        where
            L: Degree + AddDegree<Affine, Output = Affine>,
            R: Degree + AddDegree<Affine, Output = Affine> + MulDegree<L::Computed>,
        {
            type Output = QuadraticProduct<'a, <R as MulDegree<L::Computed>>::Output>;
            type PreparedLeft = ProductValue<$input>;

            fn prepare_left(self, left: ProductValue<$input>) -> Self::PreparedLeft {
                left
            }

            fn sum_product(
                self,
                (ProductValue(left), right): (ProductValue<$input>, WeightedTerm<'a, L>),
            ) -> Self::Output {
                QuadraticProduct {
                    left: left.into_expression().into_degree(),
                    right: right.expression.into_degree(),
                    weight: right.coefficient,
                    order: Order::WeightRight,
                    degree: PhantomData,
                }
            }
        }
    };
}

weighted_products!(Expr<'a, R>);
weighted_products!(&'b Expr<'a, R>);

macro_rules! scale_products {
    ($scalar:ty) => {
        // Two numeric weights around one affine factor keep the existing
        // outer-weight accumulation path and its original inner multiplication.
        impl<'a, D: Degree> sealed::Product<WeightedTerm<'a, D>, ProductValue<$scalar>>
            for &&ProductTag<WeightedTerm<'a, D>, ProductValue<$scalar>>
        {
        }

        impl<'a, D: Degree> SumProduct<WeightedTerm<'a, D>, ProductValue<$scalar>>
            for &&ProductTag<WeightedTerm<'a, D>, ProductValue<$scalar>>
        {
            type Output = WeightedTerm<'a, D::Computed>;
            type PreparedLeft = ProductValue<Expr<'a, D::Computed>>;

            fn prepare_left(self, left: WeightedTerm<'a, D>) -> Self::PreparedLeft {
                ProductValue(left.materialize())
            }
            fn sum_product(
                self,
                (product, ProductValue(weight)): (WeightedTerm<'a, D>, ProductValue<$scalar>),
            ) -> Self::Output {
                WeightedTerm { expression: product.materialize(), coefficient: f64::from(weight) }
            }
        }

        impl<'a, D: Degree> sealed::Product<ProductValue<$scalar>, WeightedTerm<'a, D>>
            for &&ProductTag<ProductValue<$scalar>, WeightedTerm<'a, D>>
        {
        }

        impl<'a, D: Degree> SumProduct<ProductValue<$scalar>, WeightedTerm<'a, D>>
            for &&ProductTag<ProductValue<$scalar>, WeightedTerm<'a, D>>
        {
            type Output = WeightedTerm<'a, D::Computed>;
            type PreparedLeft = ProductValue<$scalar>;

            fn prepare_left(self, left: ProductValue<$scalar>) -> Self::PreparedLeft {
                left
            }
            fn sum_product(
                self,
                (ProductValue(weight), product): (ProductValue<$scalar>, WeightedTerm<'a, D>),
            ) -> Self::Output {
                WeightedTerm { expression: product.materialize(), coefficient: f64::from(weight) }
            }
        }

        impl<'a, D: ExpressionDegree>
            sealed::Product<QuadraticProduct<'a, D>, ProductValue<$scalar>>
            for &&ProductTag<QuadraticProduct<'a, D>, ProductValue<$scalar>>
        {
        }

        impl<'a, D: ExpressionDegree> SumProduct<QuadraticProduct<'a, D>, ProductValue<$scalar>>
            for &&ProductTag<QuadraticProduct<'a, D>, ProductValue<$scalar>>
        {
            type Output = QuadraticProduct<'a, D>;
            type PreparedLeft = QuadraticProduct<'a, D>;

            fn prepare_left(self, left: QuadraticProduct<'a, D>) -> Self::PreparedLeft {
                left
            }
            fn sum_product(
                self,
                (mut product, ProductValue(weight)): (
                    QuadraticProduct<'a, D>,
                    ProductValue<$scalar>,
                ),
            ) -> Self::Output {
                product.weight = f64::from(weight);
                product.order = Order::WeightProduct;
                product
            }
        }

        impl<'a, D: ExpressionDegree>
            sealed::Product<ProductValue<$scalar>, QuadraticProduct<'a, D>>
            for &&ProductTag<ProductValue<$scalar>, QuadraticProduct<'a, D>>
        {
        }

        impl<'a, D: ExpressionDegree> SumProduct<ProductValue<$scalar>, QuadraticProduct<'a, D>>
            for &&ProductTag<ProductValue<$scalar>, QuadraticProduct<'a, D>>
        {
            type Output = QuadraticProduct<'a, D>;
            type PreparedLeft = ProductValue<$scalar>;

            fn prepare_left(self, left: ProductValue<$scalar>) -> Self::PreparedLeft {
                left
            }
            fn sum_product(
                self,
                (ProductValue(weight), mut product): (
                    ProductValue<$scalar>,
                    QuadraticProduct<'a, D>,
                ),
            ) -> Self::Output {
                product.weight = f64::from(weight);
                product.order = Order::WeightProduct;
                product
            }
        }
    };
}

scale_products!(f64);
scale_products!(i32);

#[derive(Debug)]
enum Segment<'a, D: ExpressionDegree> {
    Numeric(QuadraticTerms),
    Symbolic(QuadraticProduct<'a, D>),
}

/// Buffer so numeric terms stream into a builder.
#[derive(Debug)]
pub struct QuadraticBuffer<'a, D: ExpressionDegree> {
    arena: Option<&'a ExprArenaCell>,
    anchored: bool,
    same_arena: bool,
    builder: Option<QuadraticBuilder<'a>>,
    pending: bool,
    segments: Vec<Segment<'a, D>>,
}

impl<'a, D: ExpressionDegree> QuadraticBuffer<'a, D> {
    fn flush(&mut self) {
        if self.pending {
            self.segments.push(Segment::Numeric(self.builder.as_mut().unwrap().build_terms()));
            self.pending = false;
        }
    }
}

impl<'a, D: ExpressionDegree> sealed::Term<'a> for QuadraticProduct<'a, D> {}

impl<'a, D: ExpressionDegree> SumTerm<'a> for QuadraticProduct<'a, D> {
    type Degree = D;
    type Evaluated = Self;
    type Buffer = QuadraticBuffer<'a, D>;

    fn evaluate(self) -> Self {
        self
    }

    fn buffer(arena: Option<&'a ExprArenaCell>, _capacity: usize) -> Self::Buffer {
        QuadraticBuffer {
            arena,
            anchored: arena.is_some(),
            same_arena: true,
            builder: None,
            pending: false,
            segments: Vec::new(),
        }
    }

    fn push(buffer: &mut Self::Buffer, term: Self) {
        let arena = *buffer.arena.get_or_insert(term.left.arena());
        if !std::ptr::eq(arena, term.left.arena()) || !std::ptr::eq(arena, term.right.arena()) {
            buffer.same_arena = false;
            return;
        }

        let builder = buffer.builder.get_or_insert_with(|| QuadraticBuilder::new(arena));
        if builder.add_product(term.weight, &term.left, &term.right).is_ok() {
            buffer.pending = true;
        } else {
            // Builder errors are atomic.
            buffer.flush();
            buffer.segments.push(Segment::Symbolic(term));
        }
    }

    fn finish(mut buffer: Self::Buffer) -> Option<Expr<'a, D>> {
        assert!(
            buffer.same_arena,
            "{}",
            if buffer.anchored {
                "sum! terms belong to a different model"
            } else {
                "expressions belong to different arenas"
            }
        );
        let arena = buffer.arena?;
        if buffer.segments.is_empty() {
            return Some(
                buffer
                    .builder
                    .unwrap_or_else(|| QuadraticBuilder::new(arena))
                    .build()
                    .into_degree(),
            );
        }
        buffer.flush();

        Some(
            Expr::__sum_terms_in(
                arena,
                buffer.segments.into_iter().map(|segment| match segment {
                    Segment::Numeric(terms) => Expr::build(arena, |arena| {
                        crate::quadratic_builder::push_quadratic(arena, terms)
                    }),
                    Segment::Symbolic(term) => term.materialize(),
                }),
            )
            .into_degree(),
        )
    }
}
