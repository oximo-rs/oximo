//! Staged multiplication where only known oximo operations may remain deferred.

use std::marker::PhantomData;
use std::ops::Mul;

use rustc_hash::FxHashMap;

use super::{SumTerm, WeightedTerm, sealed};
use crate::degree::{AddDegree, MulDegree};
use crate::quadratic_builder::MAX_EAGER_COEFFICIENTS;
use crate::{
    Affine, Degree, Expr, ExprArenaCell, ExprId, ExprInput, ExprNode, ExpressionDegree,
    QuadraticBuilder, QuadraticTerms,
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

#[derive(Debug)]
enum Expansion {
    Numeric,
    Factored,
    Oversized,
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

fn coefficient_pairs_fit_budget(left: usize, right: usize) -> bool {
    left.checked_mul(right).is_some_and(|pairs| pairs <= MAX_EAGER_COEFFICIENTS)
}

impl<D: ExpressionDegree> QuadraticProduct<'_, D> {
    fn shared_non_monomial_square(&self, arena: &dyn crate::arena::ArenaAccess) -> bool {
        match (arena.get(self.left.id), arena.get(self.right.id)) {
            (
                ExprNode::Linear { coeffs: left, constant: left_constant },
                ExprNode::Linear { coeffs: right, constant: right_constant },
            ) => {
                (left.len() > 1 || *left_constant != 0.0)
                    && (self.left.id == self.right.id
                        || (left_constant.to_bits() == right_constant.to_bits()
                            && left.len() == right.len()
                            && left
                                .iter()
                                .zip(right)
                                .all(|(&(a, x), &(b, y))| a == b && x.to_bits() == y.to_bits())))
            }
            (ExprNode::Var(_) | ExprNode::Const(_), _) => false,
            _ => self.left.id == self.right.id,
        }
    }

    fn expansion(&self) -> Expansion {
        if self.left.cached_var_id().is_some() && self.right.cached_var_id().is_some() {
            return Expansion::Numeric;
        }

        self.left.arena().with_ref(|arena| {
            let count = |input: &Expr<'_, Affine>| {
                let id = input.id;
                match arena.get(id) {
                    ExprNode::Var(_) => Some(1),
                    ExprNode::Const(_) => Some(0),
                    ExprNode::Linear { coeffs, .. } => Some(coeffs.len()),
                    _ => crate::linear::as_linear(arena, id, false).map(|terms| terms.coeffs.len()),
                }
            };
            match (count(&self.left), count(&self.right)) {
                (Some(left), Some(right)) if !coefficient_pairs_fit_budget(left, right) => {
                    if self.weight.is_finite() { Expansion::Oversized } else { Expansion::Factored }
                }

                // add_product expands (weight * left) * right.
                (Some(_), Some(_))
                    if matches!(self.order, Order::Plain | Order::WeightLeft)
                        && !self.shared_non_monomial_square(arena) =>
                {
                    Expansion::Numeric
                }
                _ => Expansion::Factored,
            }
        })
    }
}

/// Normalize numeric compound roots once for this sum, without copying leaves.
fn normalized_factor<'a>(
    input: Expr<'a, Affine>,
    normalized: &mut FxHashMap<ExprId, ExprId>,
) -> Expr<'a, Affine> {
    let root = input.publish_id();
    let arena = input.arena();
    let id = *normalized.entry(root).or_insert_with(|| {
        arena.with_mut(|arena| {
            if matches!(
                arena.get(root),
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Linear { .. }
            ) {
                return root;
            }
            let Some(terms) = crate::linear::as_linear(arena, root, false) else {
                return root;
            };
            if !terms.constant.is_finite() || terms.coeffs.iter().any(|(_, c)| !c.is_finite()) {
                return root;
            }
            let constant = terms.constant;
            let coeffs = terms.coeffs.into_owned();
            crate::linear::push_linear(arena, crate::LinearTerms::owned(coeffs, constant))
        })
    });

    Expr::from_id(id, arena)
}

impl<'a, D: ExpressionDegree> QuadraticProduct<'a, D> {
    fn materialize_oversized(self, normalized: &mut FxHashMap<ExprId, ExprId>) -> Expr<'a, D> {
        let mut left = normalized_factor(self.left, normalized);
        let mut right = normalized_factor(self.right, normalized);
        match self.order {
            Order::WeightLeft => {
                left = Expr::constant(left.arena(), self.weight).mul_factored(&left);
            }
            Order::WeightRight => {
                right = Expr::constant(right.arena(), self.weight).mul_factored(&right);
            }
            Order::Plain | Order::WeightProduct => {}
        }

        let product = left.mul_factored(right);
        match self.order {
            Order::WeightProduct => {
                product.mul_factored(Expr::constant(product.arena(), self.weight)).into_degree()
            }
            _ => product.into_degree(),
        }
    }
}

impl<'a, D: ExpressionDegree> ProductOperand for QuadraticProduct<'a, D> {
    type Original = Expr<'a, D>;

    fn materialize(self) -> Self::Original {
        let factored = std::ptr::eq(self.left.arena(), self.right.arena())
            && self.left.arena().with_ref(|arena| self.shared_non_monomial_square(arena));
        let mut left = self.left;
        let mut right = self.right;
        match self.order {
            Order::WeightLeft => left = left * self.weight,
            Order::WeightRight => right = right * self.weight,
            Order::Plain | Order::WeightProduct => {}
        }
        let product = if factored { left.mul_factored(&right) } else { left * right };
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
    Oversized(QuadraticProduct<'a, D>),
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

        // Keep compound squares factored and bound ordered coefficient pairs
        // before the builder can allocate expansion scratch or output storage.
        let expansion = term.expansion();
        if matches!(expansion, Expansion::Numeric)
            && buffer
                .builder
                .get_or_insert_with(|| QuadraticBuilder::new(arena))
                .add_product(term.weight, &term.left, &term.right)
                .is_ok()
        {
            buffer.pending = true;
        } else {
            // Builder errors are atomic.
            buffer.flush();
            buffer.segments.push(if matches!(expansion, Expansion::Oversized) {
                Segment::Oversized(term)
            } else {
                Segment::Symbolic(term)
            });
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
        let mut normalized = FxHashMap::default();

        Some(
            Expr::__sum_terms_in(
                arena,
                buffer.segments.into_iter().map(|segment| match segment {
                    Segment::Numeric(terms) => Expr::build(arena, |arena| {
                        crate::quadratic_builder::push_quadratic(arena, terms)
                    }),
                    Segment::Symbolic(term) => term.materialize(),
                    Segment::Oversized(term) => term.materialize_oversized(&mut normalized),
                }),
            )
            .into_degree(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprArena, ExprNode, Quadratic, VarId, evaluate};

    fn factor(arena: &ExprArenaCell, start: u32, count: u32) -> Expr<'_, Affine> {
        let id = arena
            .borrow_mut()
            .linear((start..start + count).map(|i| (VarId(i), 1.0)).collect(), 0.0);
        Expr::new(id, arena).into_affine().unwrap()
    }

    fn product<'a>(
        left: Expr<'a, Affine>,
        right: Expr<'a, Affine>,
    ) -> QuadraticProduct<'a, Quadratic> {
        QuadraticProduct { left, right, weight: 1.0, order: Order::Plain, degree: PhantomData }
    }

    #[test]
    fn oversized_products_bypass_the_builder_before_it_is_created() {
        for (n, m, square) in [
            (8, 5, false),
            (33, 1, false),
            (1, 33, false),
            (128, 128, false),
            (6, 6, true),
            (128, 128, true),
        ] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let left = factor(&arena, 0, n);
            let right = if square { left.clone() } else { factor(&arena, n, m) };
            let before = arena.borrow().len();
            let mut buffer = QuadraticProduct::buffer(Some(&arena), 1);
            QuadraticProduct::push(&mut buffer, product(left, right));
            assert!(
                buffer.builder.is_none(),
                "oversized products must not reserve builder storage"
            );
            assert!(!buffer.pending);
            assert!(matches!(buffer.segments.as_slice(), [Segment::Oversized(_)]));
            assert_eq!(arena.borrow().len(), before);
            let result = QuadraticProduct::finish(buffer).unwrap();
            let snapshot = arena.borrow();
            assert!(matches!(snapshot.get(result.id()), ExprNode::Mul(_)));
            let point = vec![1.0; usize::try_from(n + m).unwrap()];
            assert_eq!(evaluate(&snapshot, result.id(), &&point[..]).unwrap(), f64::from(n * m));
        }
    }

    #[test]
    fn products_at_the_pair_budget_still_accumulate_into_one_node() {
        for (n, m) in [(8, 4), (32, 1), (1, 32), (5, 5)] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let left = factor(&arena, 0, n);
            let right = factor(&arena, n, m);
            let mut buffer = QuadraticProduct::buffer(Some(&arena), 2);
            QuadraticProduct::push(&mut buffer, product(left.clone(), right.clone()));
            QuadraticProduct::push(&mut buffer, product(left, right));
            assert!(buffer.pending);
            assert!(buffer.segments.is_empty());
            let before = arena.borrow().len();
            let result = QuadraticProduct::finish(buffer).unwrap();
            let snapshot = arena.borrow();
            assert_eq!(snapshot.len(), before + 1);
            let ExprNode::Quadratic(terms) = snapshot.get(result.id()) else {
                panic!("within-budget products must use the builder");
            };
            assert_eq!(terms.quadratic().len(), usize::try_from(n * m).unwrap());
            assert!(terms.quadratic().iter().all(|&(_, _, c)| c.to_bits() == 2.0_f64.to_bits()));
        }
    }

    #[test]
    fn pair_budget_counts_compound_affine_inputs() {
        for count in [4, 5] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let a = factor(&arena, 0, 4);
            let b = factor(&arena, 4, 4);
            let root = arena.borrow_mut().push(ExprNode::Add(vec![a.id(), b.id()].into()));
            let left = Expr::new(root, &arena).into_affine().unwrap();
            let right = factor(&arena, 8, count);
            let mut buffer = QuadraticProduct::buffer(Some(&arena), 1);
            QuadraticProduct::push(&mut buffer, product(left, right));
            assert_eq!(buffer.builder.is_some(), count == 4);
            assert_eq!(buffer.pending, count == 4);
        }
    }

    #[test]
    fn oversized_products_preserve_surrounding_numeric_segments() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let small = factor(&arena, 0, 1);
        let left = factor(&arena, 0, 8);
        let right = factor(&arena, 8, 5);
        let mut buffer = QuadraticProduct::buffer(Some(&arena), 3);
        QuadraticProduct::push(&mut buffer, product(small.clone(), small.clone()));
        QuadraticProduct::push(&mut buffer, product(left, right));
        assert!(matches!(buffer.segments.as_slice(), [Segment::Numeric(_), Segment::Oversized(_)]));
        let mut last = product(small.clone(), small);
        last.weight = 3.0;
        QuadraticProduct::push(&mut buffer, last);
        assert!(buffer.pending);
        let result = QuadraticProduct::finish(buffer).unwrap();
        assert_eq!(evaluate(&arena.borrow(), result.id(), &&[1.0; 13][..]).unwrap(), 44.0);
    }

    #[test]
    fn coefficient_pair_budget_rejects_overflow() {
        assert!(coefficient_pairs_fit_budget(MAX_EAGER_COEFFICIENTS, 1));
        assert!(!coefficient_pairs_fit_budget(MAX_EAGER_COEFFICIENTS + 1, 1));
        assert!(!coefficient_pairs_fit_budget(usize::MAX, 2));
        assert!(!coefficient_pairs_fit_budget(2, usize::MAX));
        assert!(!coefficient_pairs_fit_budget(usize::MAX, usize::MAX));
    }
}
