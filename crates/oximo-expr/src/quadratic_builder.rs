//! Validated numeric polynomial construction.

use rustc_hash::FxHashMap;
use thiserror::Error;

use crate::arena::ArenaAccess;
use crate::degree::AddDegree;
use crate::linear::as_linear;
use crate::{
    Affine, Expr, ExprArenaCell, ExprInput, ExprNode, Quadratic, QuadraticTerms, VarId,
    VariableHandle,
};

/// A numeric quadratic could not be constructed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuadraticBuildError {
    #[error("quadratic input belongs to another expression arena")]
    ForeignArena,
    #[error("quadratic input must be affine")]
    NotAffine,
    #[error("quadratic input must be polynomial of degree at most two")]
    NotQuadratic,
    #[error("parameter-dependent coefficients are unsupported")]
    ParameterizedInput,
    #[error("input coefficients must be finite")]
    NonFiniteInput,
    #[error("an intermediate coefficient product overflowed")]
    ProductOverflow,
    #[error("an accumulated coefficient overflowed")]
    AccumulationOverflow,
    #[error("input cannot be expanded into finite quadratic coefficients")]
    ExpansionFailed,
}

/// Reusable numeric quadratic accumulator using ordinary, uncompensated summation.
///
/// `add_product(c, x, y)` expands `(c * x) * y`, including offsets and repeated
/// variables. Parameters are rejected, including at zero weight.
/// Additions write no arena nodes. Regrouping coefficients can change floating-point rounding.
///
/// ```
/// use oximo_expr::{Expr, ExprArena, ExprArenaCell, QuadraticBuilder, VarId};
/// # fn main() -> Result<(), oximo_expr::QuadraticBuildError> {
/// let arena = ExprArenaCell::new(ExprArena::new());
/// let x = Expr::from_var(&arena, VarId(0));
///
/// let mut builder = QuadraticBuilder::new(&arena);
/// builder.add_monomial(3.0, x, x)?; // stores the coefficient 3 for x*x
/// let first = builder.build();
/// builder.add_quadratic(2.0, &first)?; // retains the source's arena identity
/// builder.add_linear(2.0, x)?;
/// let second = builder.build(); // reset after each build
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QuadraticBuilder<'a> {
    arena: &'a ExprArenaCell,
    coefficients: Accumulator,
    scratch: Accumulator,
    square_scratch: Vec<f64>,
}

/// Coefficients are accumulated in polynomial units, including subnormal diagonals.
#[derive(Debug, Default)]
pub(crate) struct Accumulator {
    pub(crate) linear: FxHashMap<VarId, f64>,
    pub(crate) quadratic: FxHashMap<(VarId, VarId), f64>,
    pub(crate) constant: f64,
}

fn checked(value: f64, error: QuadraticBuildError) -> Result<f64, QuadraticBuildError> {
    if value.is_finite() { Ok(value) } else { Err(error) }
}
fn finite(value: f64) -> Result<f64, QuadraticBuildError> {
    checked(value, QuadraticBuildError::NonFiniteInput)
}
fn product(a: f64, b: f64) -> Result<f64, QuadraticBuildError> {
    checked(a * b, QuadraticBuildError::ProductOverflow)
}
fn sum(a: f64, b: f64) -> Result<f64, QuadraticBuildError> {
    checked(a + b, QuadraticBuildError::AccumulationOverflow)
}

/// A cheap uniqueness certificate that does not reorder raw affine inputs.
fn monotonic_variables(coeffs: &[(VarId, f64)]) -> bool {
    coeffs.windows(2).all(|w| w[0].0.0 < w[1].0.0) || coeffs.windows(2).all(|w| w[0].0.0 > w[1].0.0)
}

fn triangular_len(n: usize) -> Option<usize> {
    let next = n.checked_add(1)?;
    if n.is_multiple_of(2) { (n / 2).checked_mul(next) } else { n.checked_mul(next / 2) }
}

fn finish_coefficients<K, T>(
    coefficients: &mut FxHashMap<K, f64>,
    map_entry: impl FnMut((K, f64)) -> T,
) -> Vec<T> {
    if coefficients.len() < coefficients.capacity() / 4 || coefficients.values().any(|&c| c == 0.0)
    {
        coefficients.drain().filter(|(_, c)| *c != 0.0).map(map_entry).collect()
    } else {
        coefficients.drain().map(map_entry).collect()
    }
}

impl Accumulator {
    pub(crate) fn clear(&mut self) {
        self.linear.clear();
        self.quadratic.clear();
        self.constant = 0.0;
    }

    #[inline]
    pub(crate) fn add_linear(&mut self, v: VarId, c: f64) -> Result<(), QuadraticBuildError> {
        finite(c)?;
        if c == 0.0 {
            return Ok(());
        }
        if let Some(current) = self.linear.get_mut(&v) {
            *current = sum(*current, c)?;
        } else {
            self.linear.insert(v, c);
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn add_quadratic(
        &mut self,
        a: VarId,
        b: VarId,
        c: f64,
    ) -> Result<(), QuadraticBuildError> {
        finite(c)?;
        if c == 0.0 {
            return Ok(());
        }
        let key = if a.0 >= b.0 { (a, b) } else { (b, a) };
        if let Some(current) = self.quadratic.get_mut(&key) {
            *current = sum(*current, c)?;
        } else {
            self.quadratic.insert(key, c);
        }

        Ok(())
    }

    /// Check every destination before changing any entry.
    fn merge(&mut self, other: &mut Self) -> Result<(), QuadraticBuildError> {
        let constant = sum(self.constant, other.constant)?;
        let mut new_linear = 0;
        if !self.linear.is_empty() {
            for (&v, &c) in &other.linear {
                let current = self.linear.get(&v);
                sum(current.copied().unwrap_or(0.0), c)?;
                new_linear += usize::from(current.is_none());
            }
        }
        let mut new_quadratic = 0;
        if !self.quadratic.is_empty() {
            for (&pair, &c) in &other.quadratic {
                let current = self.quadratic.get(&pair);
                sum(current.copied().unwrap_or(0.0), c)?;
                new_quadratic += usize::from(current.is_none());
            }
        }

        self.linear.reserve(new_linear);
        self.quadratic.reserve(new_quadratic);
        self.constant = constant;
        if self.linear.is_empty() && !other.linear.is_empty() {
            std::mem::swap(&mut self.linear, &mut other.linear);
        } else {
            for (&v, &c) in &other.linear {
                *self.linear.entry(v).or_default() += c;
            }
        }
        if self.quadratic.is_empty() && !other.quadratic.is_empty() {
            std::mem::swap(&mut self.quadratic, &mut other.quadratic);
        } else {
            for (&pair, &c) in &other.quadratic {
                *self.quadratic.entry(pair).or_default() += c;
            }
        }

        Ok(())
    }

    pub(crate) fn finish(&mut self) -> QuadraticTerms {
        let mut linear = finish_coefficients(&mut self.linear, std::convert::identity);
        let mut quadratic = finish_coefficients(&mut self.quadratic, |((a, b), h)| (a, b, h));
        linear.sort_unstable_by_key(|(v, _)| v.0);
        quadratic.sort_unstable_by_key(|(a, b, _)| (b.0, a.0));

        QuadraticTerms::from_canonical_parts(quadratic, linear, std::mem::take(&mut self.constant))
    }

    fn add_leaf(&mut self, leaf: NumericLeaf<'_>, scale: f64) -> Result<(), QuadraticBuildError> {
        match leaf {
            NumericLeaf::Affine(coeffs, constant) => {
                self.constant = sum(self.constant, product(finite(constant)?, scale)?)?;
                for &(v, c) in coeffs {
                    self.add_linear(v, product(finite(c)?, scale)?)?;
                }
            }
            NumericLeaf::Quadratic(q) => {
                self.constant = sum(self.constant, product(q.constant(), scale)?)?;
                for &(v, c) in q.linear() {
                    self.add_linear(v, product(c, scale)?)?;
                }
                for &(a, b, h) in q.quadratic() {
                    self.add_quadratic(a, b, product(h, scale)?)?;
                }
            }
            NumericLeaf::Var(v) => self.add_linear(v, scale)?,
        }

        Ok(())
    }
}

impl<'a> QuadraticBuilder<'a> {
    pub fn new(arena: &'a ExprArenaCell) -> Self {
        Self::with_capacity(arena, 0, 0)
    }

    /// Reserve distinct linear variables and distinct unordered quadratic pairs.
    pub fn with_capacity(arena: &'a ExprArenaCell, linear: usize, quadratic: usize) -> Self {
        let mut coefficients = Accumulator::default();
        coefficients.linear.reserve(linear);
        coefficients.quadratic.reserve(quadratic);
        Self { arena, coefficients, scratch: Accumulator::default(), square_scratch: Vec::new() }
    }

    /// Add a finite constant.
    ///
    /// # Errors
    /// Rejects nonfinite input or accumulation overflow atomically.
    pub fn add_constant(&mut self, c: f64) -> Result<&mut Self, QuadraticBuildError> {
        self.coefficients.constant = sum(self.coefficients.constant, finite(c)?)?;
        Ok(self)
    }

    /// Add a linear coefficient using a checked variable handle.
    /// Zero weights still validate the variable's arena.
    ///
    /// # Errors
    /// Rejects foreign variables, nonfinite input or accumulation overflow atomically.
    #[inline]
    pub fn add_linear(
        &mut self,
        c: f64,
        variable: VariableHandle<'a>,
    ) -> Result<&mut Self, QuadraticBuildError> {
        finite(c)?;
        if !std::ptr::eq(self.arena, variable.arena()) {
            return Err(QuadraticBuildError::ForeignArena);
        }
        self.add_linear_raw(c, variable.known_var_id())
    }

    /// Add the coefficient of `first * second`, with no diagonal scaling.
    /// Zero weights still validate both variable handles' arenas.
    ///
    /// # Errors
    /// Rejects foreign variables, nonfinite input or accumulation overflow atomically.
    #[inline]
    pub fn add_monomial(
        &mut self,
        c: f64,
        first: VariableHandle<'a>,
        second: VariableHandle<'a>,
    ) -> Result<&mut Self, QuadraticBuildError> {
        finite(c)?;
        if !std::ptr::eq(self.arena, first.arena()) || !std::ptr::eq(self.arena, second.arena()) {
            return Err(QuadraticBuildError::ForeignArena);
        }
        self.add_monomial_raw(c, first.known_var_id(), second.known_var_id())
    }

    /// Add a linear coefficient using a raw, arena-local variable ID.
    /// The caller must ensure the ID belongs to this builder's arena. Raw IDs
    /// carry no model identity, so foreign IDs cannot be detected.
    /// Prefer [`Self::add_linear`] for checked variable handles.
    ///
    /// # Errors
    /// Rejects nonfinite input or accumulation overflow atomically.
    #[inline]
    pub fn add_linear_raw(
        &mut self,
        c: f64,
        variable: VarId,
    ) -> Result<&mut Self, QuadraticBuildError> {
        self.coefficients.add_linear(variable, c)?;
        Ok(self)
    }

    /// Add the coefficient of `first * second` using raw variable IDs, with no
    /// diagonal scaling. The caller must ensure both IDs belong to this builder's
    /// arena.
    ///
    /// Raw IDs carry no model identity, so foreign IDs cannot be detected.
    /// Prefer [`Self::add_monomial`] for checked variable handles.
    ///
    /// # Errors
    /// Rejects nonfinite input or accumulation overflow atomically.
    #[inline]
    pub fn add_monomial_raw(
        &mut self,
        c: f64,
        first: VarId,
        second: VarId,
    ) -> Result<&mut Self, QuadraticBuildError> {
        self.coefficients.add_quadratic(first, second, c)?;
        Ok(self)
    }

    /// Add a weighted numeric affine expression, accepting owned or borrowed handles.
    /// Zero weights still validate the input class and arena.
    ///
    /// # Errors
    /// Rejects foreign, parameter-dependent, non-affine or nonfinite inputs and overflow atomically.
    pub fn add_term<E>(&mut self, c: f64, expression: E) -> Result<&mut Self, QuadraticBuildError>
    where
        E: ExprInput<'a>,
        E::Degree: AddDegree<Affine, Output = Affine>,
    {
        self.add_expr(c, expression)
    }

    /// Runtime-checked counterpart of [`Self::add_term`], accepting dynamic handles.
    /// Input must be numeric affine; use [`Self::add_quadratic`] for quadratic
    /// handles. Zero weights still validate the arena and expression class.
    ///
    /// # Errors
    /// Rejects foreign, parameter-dependent, non-affine or nonfinite inputs and overflow atomically.
    pub fn add_expr<E: ExprInput<'a>>(
        &mut self,
        c: f64,
        expression: E,
    ) -> Result<&mut Self, QuadraticBuildError> {
        finite(c)?;
        let expression = expression.as_expression();
        if !std::ptr::eq(self.arena, expression.arena()) {
            return Err(QuadraticBuildError::ForeignArena);
        }
        if let Some(variable) = expression.cached_var_id() {
            self.coefficients.add_linear(variable, c)?;
            return Ok(self);
        }

        self.arena.with_ref(|arena| {
            if let ExprNode::Var(v) = arena.get(expression.id) {
                return self.coefficients.add_linear(*v, c);
            }
            self.scratch.clear();
            let terms = numeric_affine(arena, expression.id)?;
            self.scratch.constant = product(c, finite(terms.constant)?)?;
            for &(v, a) in terms.coeffs.iter() {
                self.scratch.add_linear(v, product(c, finite(a)?)?)?;
            }
            self.coefficients.merge(&mut self.scratch)
        })?;

        Ok(self)
    }

    /// Add a weighted numeric polynomial of degree at most two from a checked handle.
    /// Accepts owned or borrowed handles, including dynamic expressions.
    /// Parameters are rejected rather than frozen at their current values.
    /// Zero scales still validate the arena, input structure and coefficients.
    /// Copies coefficients without publishing the source or allocating arena nodes.
    ///
    /// # Errors
    /// Rejects foreign, parameter-dependent, nonpolynomial or nonfinite inputs,
    /// unrepresentable expansions, nonfinite scales, and coefficient overflow atomically.
    pub fn add_quadratic<E: ExprInput<'a>>(
        &mut self,
        scale: f64,
        expression: E,
    ) -> Result<&mut Self, QuadraticBuildError> {
        finite(scale)?;
        let expression = expression.as_expression();
        if !std::ptr::eq(self.arena, expression.arena()) {
            return Err(QuadraticBuildError::ForeignArena);
        }
        if let Some(variable) = expression.cached_var_id() {
            self.coefficients.add_linear(variable, scale)?;
            return Ok(self);
        }
        self.arena.with_ref(|arena| {
            self.scratch.clear();
            if let Some(leaf) = numeric_leaf(arena.get(expression.id)) {
                self.scratch.add_leaf(leaf, scale)?;
            } else {
                crate::fold::validate_numeric_inputs(arena, expression.id, false).map_err(
                    |error| match error {
                        crate::fold::NumericInputError::Parameterized => {
                            QuadraticBuildError::ParameterizedInput
                        }
                        crate::fold::NumericInputError::NonFinite => {
                            QuadraticBuildError::NonFiniteInput
                        }
                    },
                )?;
                if crate::classify::classify_access(arena, expression.id)
                    > crate::ExprClass::Quadratic
                {
                    return Err(QuadraticBuildError::NotQuadratic);
                }
                let terms = crate::quadratic::extract_quadratic_access(arena, expression.id)
                    .ok_or(QuadraticBuildError::ExpansionFailed)?;
                self.scratch.add_leaf(NumericLeaf::Quadratic(&terms), scale)?;
            }
            self.coefficients.merge(&mut self.scratch)
        })?;
        Ok(self)
    }

    /// Add existing polynomial, linear and constant coefficients directly.
    /// The caller must ensure all raw variable IDs in `terms` belong to this
    /// builder's arena. Coefficient data carries no model identity, so foreign
    /// IDs cannot be detected.
    /// Use [`Self::add_quadratic`] when the source is an expression handle.
    ///
    /// # Errors
    /// Rejects nonfinite scale, product overflow or accumulation overflow atomically.
    pub fn add_quadratic_raw(
        &mut self,
        scale: f64,
        terms: &QuadraticTerms,
    ) -> Result<&mut Self, QuadraticBuildError> {
        finite(scale)?;
        self.scratch.clear();
        self.scratch.add_leaf(NumericLeaf::Quadratic(terms), scale)?;
        self.coefficients.merge(&mut self.scratch)?;

        Ok(self)
    }

    /// Expand `(c * lhs) * rhs` with a finite intermediate after each multiplication.
    /// Zero weights still validate both input classes and arenas.
    ///
    /// # Errors
    /// Rejects foreign, parameter-dependent, non-affine or nonfinite inputs and overflow atomically.
    #[inline]
    pub fn add_product<L, R>(
        &mut self,
        c: f64,
        lhs: L,
        rhs: R,
    ) -> Result<&mut Self, QuadraticBuildError>
    where
        L: ExprInput<'a>,
        R: ExprInput<'a>,
        L::Degree: AddDegree<Affine, Output = Affine>,
        R::Degree: AddDegree<Affine, Output = Affine>,
    {
        finite(c)?;
        let (lhs, rhs) = (lhs.as_expression(), rhs.as_expression());
        if !std::ptr::eq(self.arena, lhs.arena()) || !std::ptr::eq(self.arena, rhs.arena()) {
            return Err(QuadraticBuildError::ForeignArena);
        }
        if let (Some(first), Some(second)) = (lhs.cached_var_id(), rhs.cached_var_id()) {
            self.coefficients.add_quadratic(first, second, c)?;
            return Ok(self);
        }

        self.arena.with_ref(|arena| {
            if let (ExprNode::Var(a), ExprNode::Var(b)) = (arena.get(lhs.id), arena.get(rhs.id)) {
                return self.coefficients.add_quadratic(*a, *b, c);
            }
            self.scratch.clear();
            let left = numeric_affine(arena, lhs.id)?;
            let right = numeric_affine(arena, rhs.id)?;
            let left_unique = monotonic_variables(&left.coeffs);
            let right_unique =
                if lhs.id == rhs.id { left_unique } else { monotonic_variables(&right.coeffs) };
            if c != 0.0
                && (left.constant != 0.0 || right.constant != 0.0)
                && left_unique
                && right_unique
                && let Some(n) = left.coeffs.len().checked_add(right.coeffs.len())
            {
                self.scratch.linear.reserve(n);
            }
            let offset = product(c, finite(left.constant)?)?;
            self.scratch.constant = product(offset, finite(right.constant)?)?;
            let mut min_scaled = f64::INFINITY;
            let mut min_right = f64::INFINITY;
            for &(v, a) in left.coeffs.iter() {
                let scaled = product(c, finite(a)?)?;
                self.scratch.add_linear(v, product(scaled, finite(right.constant)?)?)?;
                min_scaled = min_scaled.min(scaled.abs());
            }
            for &(v, b) in right.coeffs.iter() {
                let value = finite(b)?;
                self.scratch.add_linear(v, product(offset, value)?)?;
                min_right = min_right.min(value.abs());
            }

            // Uniqueness alone does not imply dense output since products can be
            // zero or underflow.
            let smallest = min_scaled * min_right;
            let dense = left_unique && right_unique && smallest != 0.0 && smallest.is_finite();
            if dense {
                let pairs = if lhs.id == rhs.id {
                    triangular_len(left.coeffs.len())
                } else {
                    left.coeffs.len().checked_mul(right.coeffs.len())
                };

                if let Some(n) = pairs {
                    self.scratch.quadratic.reserve(n);
                }
            }

            if dense
                && lhs.id == rhs.id
                && left.coeffs.len() >= 32
                && let Some(n) = triangular_len(left.coeffs.len())
            {
                self.square_scratch.clear();
                self.square_scratch.resize(n, 0.0);

                // Keep every ordered multiplication and addition
                // Only replace hash lookups by positional indices.
                let mut row = 0;
                for (i, &(_, x)) in left.coeffs.iter().enumerate() {
                    let scaled = product(c, finite(x)?)?;
                    let mut column = 0;
                    for (j, &(_, y)) in right.coeffs.iter().enumerate() {
                        let value = product(scaled, finite(y)?)?;
                        if value != 0.0 {
                            let slot = if j <= i { row + j } else { column + i };
                            self.square_scratch[slot] = sum(self.square_scratch[slot], value)?;
                        }
                        column += j + 1;
                    }
                    row += i + 1;
                }

                let mut slot = 0;
                for (i, &(a, _)) in left.coeffs.iter().enumerate() {
                    for &(b, _) in &left.coeffs[..=i] {
                        self.scratch.add_quadratic(a, b, self.square_scratch[slot])?;
                        slot += 1;
                    }
                }
            } else {
                for &(a, x) in left.coeffs.iter() {
                    let scaled = product(c, finite(x)?)?;
                    for &(b, y) in right.coeffs.iter() {
                        let value = product(scaled, finite(y)?)?;
                        if value != 0.0 {
                            self.scratch.add_quadratic(a, b, value)?;
                        }
                    }
                }
            }
            self.coefficients.merge(&mut self.scratch)
        })?;

        Ok(self)
    }

    /// Discard pending coefficients, retaining allocated capacity.
    pub fn clear(&mut self) -> &mut Self {
        self.coefficients.clear();
        self.scratch.clear();
        self
    }

    /// Emit one fresh node and reset, retaining accumulator capacity.
    pub fn build(&mut self) -> Expr<'a, Quadratic> {
        let terms = self.build_terms();
        Expr::build(self.arena, |arena| push_quadratic(arena, terms))
    }

    /// Finish and reset without allocating any expression node, retaining capacity.
    pub fn build_terms(&mut self) -> QuadraticTerms {
        self.coefficients.finish()
    }
}

fn numeric_affine(
    arena: &(impl ArenaAccess + ?Sized),
    id: crate::ExprId,
) -> Result<crate::LinearTerms<'_>, QuadraticBuildError> {
    as_linear(arena, id, false).ok_or_else(|| {
        if crate::classify::classify_access(arena, id) <= crate::ExprClass::Linear {
            QuadraticBuildError::ParameterizedInput
        } else {
            QuadraticBuildError::NotAffine
        }
    })
}

pub(crate) fn push_quadratic(
    arena: &mut (impl ArenaAccess + ?Sized),
    terms: QuadraticTerms,
) -> crate::ExprId {
    if terms.quadratic.is_empty() {
        let sign = terms.coefficient_sign();
        let mut linear = terms.linear;
        if sign.is_sign_negative() {
            for (_, coefficient) in &mut linear {
                *coefficient = -*coefficient;
            }
        }
        crate::linear::push_linear(arena, crate::LinearTerms::owned(linear, terms.constant))
    } else {
        arena.push(ExprNode::Quadratic(std::sync::Arc::new(terms)))
    }
}

#[derive(Debug, Default)]
pub(crate) struct QuadraticSlots {
    linear: FxHashMap<VarId, usize>,
    quadratic: FxHashMap<(VarId, VarId), usize>,
    linear_initialized: bool,
    quadratic_initialized: bool,
}

#[derive(Clone, Copy)]
pub(crate) enum NumericLeaf<'a> {
    Affine(&'a [(VarId, f64)], f64),
    Quadratic(&'a QuadraticTerms),
    Var(VarId),
}
pub(crate) fn numeric_leaf(node: &ExprNode) -> Option<NumericLeaf<'_>> {
    match node {
        ExprNode::Linear { coeffs, constant } => Some(NumericLeaf::Affine(coeffs, *constant)),
        ExprNode::Quadratic(q) => Some(NumericLeaf::Quadratic(q)),
        ExprNode::Var(v) => Some(NumericLeaf::Var(*v)),
        ExprNode::Const(c) => Some(NumericLeaf::Affine(&[], *c)),
        _ => None,
    }
}

/// Add `scale * right` to `left` in place, merging matching coefficients and
/// removing exact cancellations.
pub(crate) fn merge_unique(
    left: &mut QuadraticTerms,
    slots: &mut QuadraticSlots,
    right: NumericLeaf<'_>,
    scale: f64,
) -> Option<()> {
    let single;
    let (linear, quadratic, offset) = match right {
        NumericLeaf::Affine(coeffs, c) => {
            if !coeffs.windows(2).all(|w| w[0].0.0 < w[1].0.0) {
                let mut normalized = Accumulator::default();
                normalized.add_leaf(right, 1.0).ok()?;
                return merge_unique(
                    left,
                    slots,
                    NumericLeaf::Quadratic(&normalized.finish()),
                    scale,
                );
            }
            (coeffs, &[][..], c)
        }
        NumericLeaf::Quadratic(q) => (q.linear(), q.quadratic(), q.constant()),
        NumericLeaf::Var(v) => {
            single = [(v, 1.0)];
            (&single[..], &[][..], 0.0)
        }
    };

    finite(scale).ok()?;
    if !linear.is_empty() && !slots.linear_initialized {
        slots.linear.extend(left.linear.iter().enumerate().map(|(i, &(v, _))| (v, i)));
        slots.linear_initialized = true;
    }
    if !quadratic.is_empty() && !slots.quadratic_initialized {
        slots
            .quadratic
            .extend(left.quadratic.iter().enumerate().map(|(i, &(a, b, _))| ((a, b), i)));
        slots.quadratic_initialized = true;
    }

    let constant = sum(left.constant, product(finite(offset).ok()?, scale).ok()?).ok()?;
    let coefficient_scale = scale * left.coefficient_sign();
    let mut values_changed = false;
    let mut order_changed = false;
    for &(v, c) in linear {
        let slot = slots.linear.get(&v);
        let current = slot.map_or(0.0, |&i| left.linear[i].1);
        let value = sum(current, product(finite(c).ok()?, coefficient_scale).ok()?).ok()?;
        values_changed |= value.to_bits() != current.to_bits();
        order_changed |= if slot.is_some() { value == 0.0 } else { value != 0.0 };
    }

    for &(a, b, h) in quadratic {
        let slot = slots.quadratic.get(&(a, b));
        let current = slot.map_or(0.0, |&i| left.quadratic[i].2);
        let value = sum(current, product(h, coefficient_scale).ok()?).ok()?;
        values_changed |= value.to_bits() != current.to_bits();
        order_changed |= if slot.is_some() { value == 0.0 } else { value != 0.0 };
    }

    if order_changed {
        left.invalidate_order();
    } else if values_changed {
        left.invalidate_view();
    }

    left.constant = constant;
    for &(v, c) in linear {
        let delta = c * coefficient_scale;
        if let Some(&i) = slots.linear.get(&v) {
            left.linear[i].1 += delta;
            if left.linear[i].1 == 0.0 {
                left.linear.swap_remove(i);
                slots.linear.remove(&v);
                if let Some(&(moved, _)) = left.linear.get(i) {
                    slots.linear.insert(moved, i);
                }
            }
        } else if delta != 0.0 {
            slots.linear.insert(v, left.linear.len());
            left.linear.push((v, delta));
        }
    }

    for &(a, b, h) in quadratic {
        let delta = h * coefficient_scale;
        if let Some(&i) = slots.quadratic.get(&(a, b)) {
            left.quadratic[i].2 += delta;
            if left.quadratic[i].2 == 0.0 {
                left.quadratic.swap_remove(i);
                slots.quadratic.remove(&(a, b));
                if let Some(&(x, y, _)) = left.quadratic.get(i) {
                    slots.quadratic.insert((x, y), i);
                }
            }
        } else if delta != 0.0 {
            slots.quadratic.insert((a, b), left.quadratic.len());
            left.quadratic.push((a, b, delta));
        }
    }

    Some(())
}

pub(crate) fn accumulate_nodes(
    nodes: &mut [ExprNode],
    lhs: usize,
    rhs: usize,
    slots: &mut QuadraticSlots,
    scale: f64,
) -> Option<()> {
    if lhs == rhs {
        let ExprNode::Quadratic(left) = &mut nodes[lhs] else {
            return None;
        };
        return scale_unique(std::sync::Arc::get_mut(left)?, slots, 1.0 + scale);
    }

    let (left, right) = if lhs < rhs {
        let (head, tail) = nodes.split_at_mut(rhs);
        (&mut head[lhs], &tail[0])
    } else {
        let (head, tail) = nodes.split_at_mut(lhs);
        (&mut tail[0], &head[rhs])
    };

    let ExprNode::Quadratic(left) = left else {
        return None;
    };

    merge_unique(std::sync::Arc::get_mut(left)?, slots, numeric_leaf(right)?, scale)
}

#[expect(clippy::float_cmp, reason = "exact negation can flip the stored coefficient sign")]
pub(crate) fn scale_unique(
    left: &mut QuadraticTerms,
    slots: &mut QuadraticSlots,
    scale: f64,
) -> Option<()> {
    finite(scale).ok()?;
    let constant = product(left.constant, scale).ok()?;
    if scale == -1.0 {
        left.negate();
        return Some(());
    }
    for c in left.linear.iter().map(|&(_, c)| c).chain(left.quadratic.iter().map(|&(_, _, h)| h)) {
        product(c, scale).ok()?;
    }

    left.invalidate_view();
    left.constant = constant;
    for (_, c) in &mut left.linear {
        *c *= scale;
    }
    for (_, _, h) in &mut left.quadratic {
        *h *= scale;
    }

    let len = left.linear.len() + left.quadratic.len();
    left.linear.retain(|&(_, c)| c != 0.0);
    left.quadratic.retain(|&(_, _, h)| h != 0.0);
    if len != left.linear.len() + left.quadratic.len() {
        *slots = QuadraticSlots::default();
    }

    Some(())
}

pub(crate) const MAX_EAGER_COEFFICIENTS: usize = 32;
fn leaf_count(node: &ExprNode) -> usize {
    match node {
        ExprNode::Linear { coeffs, .. } => coeffs.len(),
        ExprNode::Quadratic(q) => q.linear.len().saturating_add(q.quadratic.len()),
        ExprNode::Const(_) => 0,
        ExprNode::Var(_) => 1,
        _ => usize::MAX,
    }
}
pub(crate) fn add_small(
    arena: &mut (impl ArenaAccess + ?Sized),
    lhs: crate::ExprId,
    rhs: crate::ExprId,
    scale: f64,
) -> Option<crate::ExprId> {
    let (left, right) = (arena.get(lhs), arena.get(rhs));
    if !matches!(left, ExprNode::Quadratic(_)) && !matches!(right, ExprNode::Quadratic(_)) {
        return None;
    }
    if leaf_count(left).saturating_add(leaf_count(right)) > MAX_EAGER_COEFFICIENTS {
        return None;
    }

    let mut acc = Accumulator::default();
    acc.add_leaf(numeric_leaf(left)?, 1.0).ok()?;
    acc.add_leaf(numeric_leaf(right)?, scale).ok()?;

    Some(push_quadratic(arena, acc.finish()))
}

pub(crate) fn product_small(
    arena: &mut (impl ArenaAccess + ?Sized),
    lhs: crate::ExprId,
    rhs: crate::ExprId,
) -> Option<crate::ExprId> {
    if let Some(id) = scale_product(arena, lhs, rhs) {
        return Some(id);
    }
    if matches!(arena.get(lhs), ExprNode::Const(_)) || matches!(arena.get(rhs), ExprNode::Const(_))
    {
        return None;
    }

    if let (
        ExprNode::Linear { coeffs: left, constant: left_constant },
        ExprNode::Linear { coeffs: right, constant: right_constant },
    ) = (arena.get(lhs), arena.get(rhs))
        && (lhs == rhs
            || (left_constant.to_bits() == right_constant.to_bits()
                && left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(&(a, x), &(b, y))| a == b && x.to_bits() == y.to_bits())))
    {
        // Retain a shared factor for domain checks and stable nonlinear
        // evaluation.
        return Some(arena.push(ExprNode::Mul([lhs, lhs].into_iter().collect())));
    }

    let monomial = |id| match arena.get(id) {
        ExprNode::Var(v) => Some((*v, 1.0)),
        ExprNode::Linear { coeffs, constant } if *constant == 0.0 && coeffs.len() == 1 => {
            Some(coeffs[0])
        }
        _ => None,
    };
    if let (Some((left, a)), Some((right, b))) = (monomial(lhs), monomial(rhs)) {
        let c = product(a, b).ok()?;
        let h = c;
        let (a, b) = if left.0 >= right.0 { (left, right) } else { (right, left) };
        let quadratic = if h == 0.0 { Vec::new() } else { vec![(a, b, h)] };
        return Some(push_quadratic(
            arena,
            QuadraticTerms::from_canonical_parts(quadratic, Vec::new(), 0.0),
        ));
    }

    let count = |id| match arena.get(id) {
        ExprNode::Linear { coeffs, .. } => Some(coeffs.len()),
        ExprNode::Var(_) => Some(1),
        _ => None,
    };
    let (n, m) = (count(lhs)?, count(rhs)?);
    if n == 0 || m == 0 || n.saturating_mul(m) > MAX_EAGER_COEFFICIENTS {
        return None;
    }
    let left = as_linear(arena, lhs, false)?;
    let right = as_linear(arena, rhs, false)?;
    let mut delta = Accumulator {
        constant: product(left.constant, right.constant).ok()?,
        ..Accumulator::default()
    };

    for &(v, a) in left.coeffs.iter() {
        delta.add_linear(v, product(a, right.constant).ok()?).ok()?;
    }
    for &(v, b) in right.coeffs.iter() {
        delta.add_linear(v, product(b, left.constant).ok()?).ok()?;
    }
    for &(a, x) in left.coeffs.iter() {
        for &(b, y) in right.coeffs.iter() {
            delta.add_quadratic(a, b, product(x, y).ok()?).ok()?;
        }
    }

    let result = delta.finish();
    drop(left);
    drop(right);

    Some(push_quadratic(arena, result))
}

fn scale_product(
    arena: &mut (impl ArenaAccess + ?Sized),
    lhs: crate::ExprId,
    rhs: crate::ExprId,
) -> Option<crate::ExprId> {
    let (q, scale) = match (arena.get(lhs), arena.get(rhs)) {
        (ExprNode::Quadratic(q), ExprNode::Const(c))
        | (ExprNode::Const(c), ExprNode::Quadratic(q)) => (q, *c),
        _ => return None,
    };
    finite(scale).ok()?;
    let constant = product(q.constant(), scale).ok()?;
    let mut linear = q.linear().to_vec();
    let mut quadratic = q.quadratic().to_vec();
    for (_, c) in &mut linear {
        *c = product(*c, scale).ok()?;
    }
    for (_, _, h) in &mut quadratic {
        *h = product(*h, scale).ok()?;
    }

    linear.retain(|&(_, c)| c != 0.0);
    quadratic.retain(|&(_, _, h)| h != 0.0);

    Some(push_quadratic(arena, QuadraticTerms::from_canonical_parts(quadratic, linear, constant)))
}
pub(crate) fn sum_numeric(
    arena: &mut (impl ArenaAccess + ?Sized),
    ids: &[crate::ExprId],
) -> Option<crate::ExprId> {
    let mut count = 0usize;
    let mut quadratic = false;
    for &id in ids {
        let node = arena.get(id);
        quadratic |= matches!(node, ExprNode::Quadratic(_));
        count = count.saturating_add(leaf_count(node));
        if count > MAX_EAGER_COEFFICIENTS {
            return None;
        }
    }
    if !quadratic {
        return None;
    }
    let mut acc = Accumulator::default();
    for &id in ids {
        acc.add_leaf(numeric_leaf(arena.get(id))?, 1.0).ok()?;
    }

    Some(push_quadratic(arena, acc.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprArena, evaluate};

    #[test]
    fn finish_preserves_sparse_and_cancelled_support_and_reuses_capacity() {
        for capacity in [16, 4096] {
            for cancelled in [0, 8, 16] {
                let mut acc = Accumulator::default();
                acc.linear.reserve(capacity);
                acc.quadratic.reserve(capacity);
                let capacities = (acc.linear.capacity(), acc.quadratic.capacity());
                for i in (0..16).rev() {
                    let v = VarId(i);
                    let c = if i == 15 { f64::from_bits(1) } else { 1.0 };
                    acc.add_linear(v, c).unwrap();
                    acc.add_quadratic(v, v, c).unwrap();
                    if i < cancelled {
                        acc.add_linear(v, -c).unwrap();
                        acc.add_quadratic(v, v, -c).unwrap();
                    }
                }
                acc.constant = 3.0;
                let terms = acc.finish();
                let expected: Vec<_> = (cancelled..16)
                    .map(|i| (VarId(i), if i == 15 { f64::from_bits(1) } else { 1.0 }))
                    .collect();
                assert_eq!(terms.linear(), expected);
                assert_eq!(
                    terms.quadratic(),
                    expected.iter().map(|&(v, c)| (v, v, c)).collect::<Vec<_>>()
                );
                assert_eq!(terms.constant(), 3.0);
                assert_eq!((acc.linear.capacity(), acc.quadratic.capacity()), capacities);
                assert!(acc.linear.is_empty() && acc.quadratic.is_empty());

                let empty = acc.finish();
                assert!(empty.linear().is_empty() && empty.quadratic().is_empty());
                assert_eq!(empty.constant(), 0.0);

                acc.add_linear(VarId(42), 2.0).unwrap();
                assert_eq!(acc.finish().linear(), [(VarId(42), 2.0)]);
            }
        }
    }

    #[test]
    fn unique_merge_initializes_only_the_needed_index() {
        for quadratic_first in [false, true] {
            let mut left = QuadraticTerms::from_coefficients(
                (0..64).map(|i| (VarId(i), VarId(i), 3.0)).collect(),
                (0..64).map(|i| (VarId(i), 2.0)).collect(),
                2.0,
            )
            .unwrap();
            let mut slots = QuadraticSlots::default();

            merge_unique(&mut left, &mut slots, NumericLeaf::Affine(&[], 3.0), 1.0).unwrap();
            assert!(!slots.linear_initialized && !slots.quadratic_initialized);
            assert_eq!(slots.linear.capacity(), 0);
            assert_eq!(slots.quadratic.capacity(), 0);

            let first = QuadraticTerms::from_coefficients(
                if quadratic_first { vec![(VarId(0), VarId(0), 1.0)] } else { vec![] },
                if quadratic_first { vec![] } else { vec![(VarId(0), 1.0)] },
                0.0,
            )
            .unwrap();
            let leaf = if quadratic_first {
                NumericLeaf::Quadratic(&first)
            } else {
                NumericLeaf::Affine(first.linear(), 0.0)
            };

            merge_unique(&mut left, &mut slots, leaf, 1.0).unwrap();
            assert_eq!(slots.linear_initialized, !quadratic_first);
            assert_eq!(slots.quadratic_initialized, quadratic_first);
            if quadratic_first {
                assert_eq!(slots.linear.capacity(), 0);
            } else {
                assert_eq!(slots.quadratic.capacity(), 0);
            }

            let mixed = QuadraticTerms::from_coefficients(
                vec![(VarId(1), VarId(1), -3.0), (VarId(64), VarId(64), 5.0)],
                vec![(VarId(1), -2.0), (VarId(64), 4.0)],
                0.0,
            )
            .unwrap();
            merge_unique(&mut left, &mut slots, NumericLeaf::Quadratic(&mixed), 1.0).unwrap();
            merge_unique(&mut left, &mut slots, NumericLeaf::Quadratic(&mixed), 1.0).unwrap();
            assert!(slots.linear_initialized && slots.quadratic_initialized);

            let mut expected_linear: Vec<_> = (0..64).map(|i| (VarId(i), 2.0)).collect();
            expected_linear[0].1 += if quadratic_first { 0.0 } else { 1.0 };
            expected_linear[1].1 = -2.0;
            expected_linear.push((VarId(64), 8.0));
            let mut expected_quadratic: Vec<_> =
                (0..64).map(|i| (VarId(i), VarId(i), 3.0)).collect();
            expected_quadratic[0].2 += if quadratic_first { 1.0 } else { 0.0 };
            expected_quadratic[1].2 = -3.0;
            expected_quadratic.push((VarId(64), VarId(64), 10.0));
            assert_eq!(left.linear(), expected_linear);
            assert_eq!(left.quadratic(), expected_quadratic);
            assert_eq!(left.constant().to_bits(), 5.0_f64.to_bits());
        }
    }

    #[test]
    fn scaling_removals_reset_partially_initialized_indices() {
        for quadratic_first in [false, true] {
            let tiny = f64::from_bits(1);
            let mut left = QuadraticTerms::from_coefficients(
                vec![(VarId(0), VarId(0), tiny), (VarId(1), VarId(1), 6.0)],
                vec![(VarId(0), tiny), (VarId(1), 4.0)],
                0.0,
            )
            .unwrap();
            let mut slots = QuadraticSlots::default();
            let quadratic =
                QuadraticTerms::from_coefficients(vec![(VarId(1), VarId(1), 2.0)], vec![], 0.0)
                    .unwrap();
            let first = if quadratic_first {
                NumericLeaf::Quadratic(&quadratic)
            } else {
                NumericLeaf::Var(VarId(1))
            };
            merge_unique(&mut left, &mut slots, first, 1.0).unwrap();
            scale_unique(&mut left, &mut slots, 0.5).unwrap();
            assert!(!slots.linear_initialized && !slots.quadratic_initialized);
            assert_eq!(slots.linear.capacity(), 0);
            assert_eq!(slots.quadratic.capacity(), 0);

            merge_unique(&mut left, &mut slots, NumericLeaf::Var(VarId(1)), 1.0).unwrap();
            assert!(!slots.quadratic_initialized);

            merge_unique(&mut left, &mut slots, NumericLeaf::Quadratic(&quadratic), 1.0).unwrap();
            assert_eq!(left.linear(), [(VarId(1), if quadratic_first { 3.0 } else { 3.5 })]);
            assert_eq!(
                left.quadratic(),
                [(VarId(1), VarId(1), if quadratic_first { 6.0 } else { 5.0 })]
            );
        }
    }

    /// Frozen ordered expansion and merge from before scratch optimizations.
    fn ordered_product_reference(
        destination: &mut Accumulator,
        c: f64,
        left: &crate::LinearTerms<'_>,
        right: &crate::LinearTerms<'_>,
    ) -> Result<(), QuadraticBuildError> {
        finite(c)?;
        let mut scratch = Accumulator::default();
        let offset = product(c, finite(left.constant)?)?;
        scratch.constant = product(offset, finite(right.constant)?)?;

        for &(v, a) in left.coeffs.iter() {
            scratch.add_linear(v, product(product(c, finite(a)?)?, finite(right.constant)?)?)?;
        }
        for &(v, b) in right.coeffs.iter() {
            scratch.add_linear(v, product(offset, finite(b)?)?)?;
        }
        for &(a, x) in left.coeffs.iter() {
            let scaled = product(c, finite(x)?)?;
            for &(b, y) in right.coeffs.iter() {
                scratch.add_quadratic(a, b, product(scaled, finite(y)?)?)?;
            }
        }

        let constant = sum(destination.constant, scratch.constant)?;
        for (&v, &value) in &scratch.linear {
            sum(destination.linear.get(&v).copied().unwrap_or(0.0), value)?;
        }
        for (&pair, &value) in &scratch.quadratic {
            sum(destination.quadratic.get(&pair).copied().unwrap_or(0.0), value)?;
        }

        destination.constant = constant;
        for (&v, &value) in &scratch.linear {
            *destination.linear.entry(v).or_default() += value;
        }
        for (&pair, &value) in &scratch.quadratic {
            *destination.quadratic.entry(pair).or_default() += value;
        }

        Ok(())
    }

    fn assert_accumulator_bits(actual: &Accumulator, expected: &Accumulator) {
        assert_eq!(actual.constant.to_bits(), expected.constant.to_bits());
        assert_eq!(actual.linear.len(), expected.linear.len());
        assert_eq!(actual.quadratic.len(), expected.quadratic.len());
        for (key, value) in &expected.linear {
            assert_eq!(actual.linear.get(key).map(|v| v.to_bits()), Some(value.to_bits()));
        }
        for (key, value) in &expected.quadratic {
            assert_eq!(actual.quadratic.get(key).map(|v| v.to_bits()), Some(value.to_bits()));
        }
    }

    #[test]
    fn product_scratch_matches_original_order_and_atomic_errors() {
        let mut random = 0x1234_5678_9abc_def0_u64;
        let values = [0.0, -0.0, 0.1, 0.3, -0.7, 1.1, f64::from_bits(1), 1e-200, 1e150];
        for n in [8_u32, 32, 33, 64, 256] {
            for layout in 0..4 {
                let coeffs: Vec<_> = (0..n)
                    .map(|i| {
                        random = random.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        let variable = match layout {
                            0 => i,
                            1 => n - i - 1,
                            2 => i % 16,
                            _ => (i * 17) % n,
                        };
                        (VarId(variable), values[(random >> 32) as usize % values.len()])
                    })
                    .collect();

                for (weight, offset) in [(0.1, 0.3), (-0.7, -0.0), (0.0, 1.0), (1e200, 0.0)] {
                    let mut raw = ExprArena::new();
                    let id = raw.linear(coeffs.clone(), offset);
                    let separate = raw.linear(coeffs.clone(), offset);
                    let arena = ExprArenaCell::new(raw);
                    let left = Expr::new(id, &arena).into_affine().unwrap();
                    for right_id in [id, separate] {
                        let right = Expr::new(right_id, &arena).into_affine().unwrap();
                        let terms = crate::LinearTerms::borrowed(&coeffs, offset);
                        let mut builder = QuadraticBuilder::with_capacity(&arena, 16, 64);
                        let mut expected = Accumulator::default();
                        for _ in 0..3 {
                            let reference =
                                ordered_product_reference(&mut expected, weight, &terms, &terms);
                            let actual = builder.add_product(weight, &left, &right).map(|_| ());
                            assert_eq!(actual, reference, "n={n}, layout={layout}, c={weight}");
                            assert_accumulator_bits(&builder.coefficients, &expected);
                        }

                        builder.build_terms();
                        expected.clear();
                        let reference =
                            ordered_product_reference(&mut expected, 0.1, &terms, &terms);
                        assert_eq!(builder.add_product(0.1, &left, &right).map(|_| ()), reference);

                        assert_accumulator_bits(&builder.coefficients, &expected);
                        builder.clear();
                        assert_accumulator_bits(&builder.coefficients, &Accumulator::default());
                    }
                }
            }
        }
    }

    #[test]
    fn square_failures_leave_pending_coefficients_and_allow_reuse() {
        for (weight, large, error) in [
            (2.0, f64::MAX, QuadraticBuildError::ProductOverflow),
            (1.0, 1e200, QuadraticBuildError::ProductOverflow),
            (1.0, 1e154, QuadraticBuildError::AccumulationOverflow),
            (0.0, f64::NAN, QuadraticBuildError::NonFiniteInput),
            (0.0, f64::INFINITY, QuadraticBuildError::NonFiniteInput),
        ] {
            let mut raw = ExprArena::new();
            let bad = raw.linear((0..32).map(|i| (VarId(i), large)).collect(), 0.0);
            let good = raw.linear((0..32).map(|i| (VarId(i), 0.3)).collect(), -0.0);
            let arena = ExprArenaCell::new(raw);
            let bad = Expr::new(bad, &arena).into_affine().unwrap();
            let good = Expr::new(good, &arena).into_affine().unwrap();

            let mut builder = QuadraticBuilder::new(&arena);
            builder.add_constant(7.0).unwrap();
            builder.add_linear_raw(3.0, VarId(0)).unwrap();
            assert_eq!(builder.add_product(weight, &bad, &bad).unwrap_err(), error);

            let mut expected = Accumulator { constant: 7.0, ..Accumulator::default() };
            expected.add_linear(VarId(0), 3.0).unwrap();
            assert_accumulator_bits(&builder.coefficients, &expected);

            let coeffs: Vec<_> = (0..32).map(|i| (VarId(i), 0.3)).collect();
            let terms = crate::LinearTerms::borrowed(&coeffs, -0.0);
            ordered_product_reference(&mut expected, 0.1, &terms, &terms).unwrap();
            builder.add_product(0.1, &good, &good).unwrap();
            assert_accumulator_bits(&builder.coefficients, &expected);
        }
    }

    #[test]
    fn empty_map_transfers_wait_for_all_merge_validation() {
        let mut destination = Accumulator::default();
        destination.quadratic.insert((VarId(0), VarId(0)), f64::MAX);
        let mut scratch = Accumulator::default();
        scratch.linear.insert(VarId(1), 2.0);
        scratch.quadratic.insert((VarId(0), VarId(0)), f64::MAX);
        assert_eq!(destination.merge(&mut scratch), Err(QuadraticBuildError::AccumulationOverflow));
        assert!(destination.linear.is_empty());
        assert_eq!(scratch.linear.len(), 1, "failed merge must not transfer a map");
        assert_eq!(destination.quadratic[&(VarId(0), VarId(0))], f64::MAX);

        destination.clear();
        scratch.constant = -0.0;
        destination.merge(&mut scratch).unwrap();
        assert_eq!(destination.constant.to_bits(), 0.0_f64.to_bits());
        assert_eq!(destination.linear[&VarId(1)], 2.0);
        assert!(scratch.linear.is_empty());
        assert!(scratch.quadratic.is_empty());
    }

    #[test]
    fn square_cross_products_are_not_doubled() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let coeffs: Vec<_> = (0..32).map(|i| (VarId(i), if i == 0 { 0.1 } else { 0.3 })).collect();
        let expression: Expr<'_, Affine> =
            Expr::build(&arena, |a| a.push(ExprNode::Linear { coeffs, constant: 0.0 }));
        let mut builder = QuadraticBuilder::new(&arena);
        builder.add_product(0.1, &expression, &expression).unwrap();
        let first = (0.1_f64 * 0.1) * 0.3;
        let second = (0.1_f64 * 0.3) * 0.1;
        assert_ne!((first + second).to_bits(), (2.0 * first).to_bits());
        assert_eq!(
            builder.coefficients.quadratic[&(VarId(1), VarId(0))].to_bits(),
            (first + second).to_bits()
        );
    }

    #[test]
    fn duplicate_products_do_not_reserve_the_contribution_cartesian_product() {
        let mut raw = ExprArena::new();
        let id = raw.linear((0..256).map(|i| (VarId(i % 16), 1.0)).collect(), 0.0);
        let arena = ExprArenaCell::new(raw);
        let expression = Expr::new(id, &arena).into_affine().unwrap();
        let mut builder = QuadraticBuilder::with_capacity(&arena, 128, 256);
        let reserved_linear = builder.coefficients.linear.capacity();
        builder.add_product(1.0, &expression, &expression).unwrap();
        assert_eq!(builder.coefficients.linear.capacity(), reserved_linear);
        assert_eq!(builder.coefficients.quadratic.len(), 136);
        assert!(builder.coefficients.quadratic.capacity() < 1024);
        assert!(builder.scratch.quadratic.capacity() < 1024);
        assert_eq!(builder.square_scratch, [] as [f64; 0]);
        assert_eq!(triangular_len(0), Some(0));
        assert_eq!(triangular_len(256), Some(32_896));
        assert_eq!(triangular_len(usize::MAX), None);
    }

    #[test]
    fn empty_and_sparse_products_allocate_by_output_support() {
        for (weight, coefficient, first, count) in [
            (0.0, 1.0, 1.0, 0),
            (-0.0, 1.0, 1.0, 0),
            (1.0, 1e-200, 1e-200, 0),
            (1e-200, 1e-200, 1e-200, 0),
            (1.0, 0.0, 0.0, 0),
            (1.0, 1e-200, 1e-160, 1),
            (1.0, 0.0, 2.0, 1),
        ] {
            let coeffs: Vec<_> =
                (0..1024).map(|i| (VarId(i), if i == 0 { first } else { coefficient })).collect();
            let mut raw = ExprArena::new();
            let left = raw.linear(coeffs.clone(), 0.0);
            let separate = raw.linear(coeffs.clone(), 0.0);
            let arena = ExprArenaCell::new(raw);
            let left = Expr::new(left, &arena).into_affine().unwrap();

            for right_id in [left.id, separate] {
                let right = Expr::new(right_id, &arena).into_affine().unwrap();
                let mut builder = QuadraticBuilder::new(&arena);
                builder.add_product(weight, &left, &right).unwrap();
                let terms = crate::LinearTerms::borrowed(&coeffs, 0.0);
                let mut expected = Accumulator::default();
                ordered_product_reference(&mut expected, weight, &terms, &terms).unwrap();
                assert_accumulator_bits(&builder.coefficients, &expected);
                assert_eq!(builder.coefficients.quadratic.len(), count);
                assert_eq!(builder.coefficients.linear.capacity(), 0);
                assert_eq!(builder.scratch.linear.capacity(), 0);
                assert_eq!(builder.square_scratch.capacity(), 0);
                assert_eq!(builder.scratch.quadratic.capacity(), 0);
                if count == 0 {
                    assert_eq!(builder.coefficients.quadratic.capacity(), 0);
                } else {
                    assert!(builder.coefficients.quadratic.capacity() <= 4);
                }
            }
        }
    }

    #[test]
    fn sparse_product_fallback_preserves_validation_and_atomic_errors() {
        for (weight, large, invalid) in [
            (0.0, 1.0, Some(f64::NAN)),
            (0.0, 1.0, Some(f64::INFINITY)),
            (1.0, 1e200, Some(f64::NAN)),
            (1.0, 1e200, None),
            (1.0, 1e154, None),
        ] {
            let mut coeffs: Vec<_> =
                (0..1024).map(|i| (VarId(i), if i < 2 { large } else { 1e-200 })).collect();
            if let Some(value) = invalid {
                coeffs.last_mut().unwrap().1 = value;
            }

            let mut raw = ExprArena::new();
            let id = raw.linear(coeffs.clone(), 0.0);
            let arena = ExprArenaCell::new(raw);
            let expression = Expr::new(id, &arena).into_affine().unwrap();
            let mut builder = QuadraticBuilder::new(&arena);
            builder.add_constant(7.0).unwrap();
            builder.add_linear_raw(3.0, VarId(0)).unwrap();

            let mut expected = Accumulator { constant: 7.0, ..Accumulator::default() };
            expected.add_linear(VarId(0), 3.0).unwrap();
            let terms = crate::LinearTerms::borrowed(&coeffs, 0.0);
            let reference = ordered_product_reference(&mut expected, weight, &terms, &terms);
            assert!(reference.is_err());
            assert_eq!(
                builder.add_product(weight, &expression, &expression).map(|_| ()),
                reference
            );

            assert_accumulator_bits(&builder.coefficients, &expected);
            let good = Expr::from_var(&arena, VarId(0)) + 2.0;
            builder.add_product(0.0, &good, &good).unwrap();
            assert_accumulator_bits(&builder.coefficients, &expected);
        }
    }

    #[test]
    fn square_destination_overflow_is_atomic_after_successful_expansion() {
        let mut raw = ExprArena::new();
        let id = raw
            .linear((0..32).map(|i| (VarId(i), if i == 0 { 1e154 } else { 0.0 })).collect(), 0.0);
        let arena = ExprArenaCell::new(raw);
        let expression = Expr::new(id, &arena).into_affine().unwrap();

        let mut builder = QuadraticBuilder::new(&arena);
        builder.add_monomial_raw(f64::MAX, VarId(0), VarId(0)).unwrap();
        builder.add_constant(9.0).unwrap();
        assert_eq!(
            builder.add_product(1.0, &expression, &expression).unwrap_err(),
            QuadraticBuildError::AccumulationOverflow
        );

        let terms = builder.build_terms();
        assert_eq!(terms.constant(), 9.0);
        assert_eq!(terms.quadratic(), [(VarId(0), VarId(0), f64::MAX)]);

        builder.add_product(0.0, &expression, &expression).unwrap();
        assert_eq!(builder.build_terms().quadratic(), []);
    }

    #[test]
    fn checked_variable_insertions_need_no_arena_reads() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(42));
        let y = Expr::from_var(&arena, VarId(99));
        arena.reset_access_count();
        let dynamic = x.as_dynamic();
        let affine = y.into_affine().unwrap();
        let narrowed = dynamic.as_affine().unwrap();
        let mut builder = QuadraticBuilder::new(&arena);
        builder.add_linear(2.0, x).unwrap();
        builder.add_monomial(3.0, x, y).unwrap();
        builder.add_expr(4.0, &dynamic).unwrap();
        builder.add_term(5.0, &affine).unwrap();
        builder.add_product(6.0, &narrowed, &affine).unwrap();
        assert_eq!(arena.access_count(), 0);
        let terms = builder.build_terms();
        assert_eq!(terms.linear(), [(VarId(42), 6.0), (VarId(99), 5.0)]);
        assert_eq!(terms.quadratic(), [(VarId(99), VarId(42), 9.0)]);
    }

    #[test]
    fn compacted_variables_remain_available_without_arena_reads() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(42));
        let y = Expr::from_var(&arena, VarId(99));
        arena.reset_access_count();
        let compacted_x = x.compact();
        let compacted_y = y.as_dynamic().into_affine().unwrap().compact();
        assert_eq!(compacted_x.id(), x.id());
        assert_eq!(compacted_y.id(), y.id());

        let mut builder = QuadraticBuilder::new(&arena);
        assert_eq!(compacted_x.var_id(), Some(VarId(42)));
        assert_eq!(compacted_y.var_id(), Some(VarId(99)));
        builder.add_term(2.0, &compacted_x).unwrap();
        builder.add_expr(3.0, &compacted_y).unwrap();
        builder.add_product(4.0, &compacted_x, &compacted_y).unwrap();
        assert_eq!(arena.access_count(), 0);

        let terms = builder.build_terms();
        assert_eq!(terms.linear(), [(VarId(42), 2.0), (VarId(99), 3.0)]);
        assert_eq!(terms.quadratic(), [(VarId(99), VarId(42), 4.0)]);
    }

    #[test]
    fn checked_quadratic_copies_do_not_publish_the_source() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let source = x * x;
        let root = source.transient_id();
        let mut builder = QuadraticBuilder::new(&arena);
        arena.reset_access_count();
        builder.add_quadratic(2.0, &source).unwrap();
        assert_eq!(arena.access_count(), 1);

        let copied = builder.build_terms();
        let grown = source + x;
        assert_eq!(grown.id(), root);
        assert_eq!(copied.linear(), []);
        assert_eq!(copied.quadratic(), [(VarId(0), VarId(0), 2.0)]);
    }

    #[test]
    fn checked_quadratic_variable_inputs_need_no_arena_reads() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(42));
        let mut builder = QuadraticBuilder::new(&arena);
        arena.reset_access_count();
        builder.add_quadratic(2.0, x).unwrap();
        builder.add_quadratic(3.0, x.as_dynamic()).unwrap();
        assert_eq!(arena.access_count(), 0);
        assert_eq!(builder.build_terms().linear(), [(VarId(42), 5.0)]);
    }

    #[test]
    fn right_growing_sums_retain_the_larger_unique_destination() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let mut builder = QuadraticBuilder::new(&arena);
        for i in 0..64 {
            builder.add_monomial_raw(1.0, VarId(i), VarId(i)).unwrap();
        }
        let mut q = builder.build();
        let root = q.id;
        for _ in 0..4096 {
            q = x * x + q;
        }
        assert_eq!(q.id, root, "a small fresh left term must not absorb the growing right prefix");

        let q = x + q;
        assert_eq!(
            q.id, root,
            "a borrowed affine left input can also accumulate into the owned right root"
        );

        let alias = q.as_dynamic();
        let changed = x * x + q;
        assert_ne!(changed.id, root, "the larger destination must still respect permanent sealing");
        assert_eq!(
            crate::extract_quadratic(&arena.borrow(), alias.id())
                .unwrap()
                .hessian_entries()
                .collect::<Vec<_>>()[0]
                .2,
            8194.0
        );
    }

    #[test]
    fn consuming_conversions_preserve_ownership_and_borrowed_views_seal() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        for mode in 0..4 {
            let q = x * x;
            let root = q.id;
            let q = match mode {
                0 => q.into_dynamic(),
                1 => {
                    let q: Expr = q.into();
                    q.into_quadratic().unwrap().into_dynamic()
                }
                2 => q.into_nonlinear().into_quadratic().unwrap().into_dynamic(),
                _ => match q.into_classified() {
                    crate::ClassifiedExpr::Quadratic(q) => q.into_dynamic(),
                    _ => panic!("quadratic"),
                },
            };
            assert_eq!((q + x).id, root);
        }

        let q = x * x;
        let root = q.id;
        let alias = q.as_nonlinear();
        assert_ne!((q + x).id, root);
        assert_eq!(evaluate(&arena.borrow(), alias.id(), &&[2.0][..]).unwrap(), 4.0);
    }

    #[test]
    fn weighted_products_apply_the_weight_first_in_every_contribution() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let y = Expr::from_var(&arena, VarId(1));
        let mut builder = QuadraticBuilder::new(&arena);

        // Constant, left-linear, right-linear, and quadratic contributions.
        for (left, right) in [
            (
                Expr::constant(&arena, f64::MAX).as_dynamic(),
                Expr::constant(&arena, 2.0).as_dynamic(),
            ),
            ((f64::MAX * x).as_dynamic(), Expr::constant(&arena, 2.0).as_dynamic()),
            (Expr::constant(&arena, f64::MAX).as_dynamic(), (2.0 * y).as_dynamic()),
            ((f64::MAX * x).as_dynamic(), (2.0 * y).as_dynamic()),
        ] {
            builder
                .add_product(0.5, left.as_affine().unwrap(), right.as_affine().unwrap())
                .unwrap();
            let q = builder.build();
            let terms = crate::extract_quadratic(&arena.borrow(), q.id()).unwrap();
            assert_eq!(
                terms.constant()
                    + terms.linear().iter().map(|&(_, c)| c).sum::<f64>()
                    + terms
                        .hessian_entries()
                        .collect::<Vec<_>>()
                        .iter()
                        .map(|&(_, _, h)| h)
                        .sum::<f64>(),
                f64::MAX
            );
        }

        builder.add_product(1e200, 1e-200 * x, 1e-200 * y).unwrap();
        let q = builder.build();
        let terms = crate::extract_quadratic(&arena.borrow(), q.id()).unwrap();
        assert_eq!(terms.hessian_entries().collect::<Vec<_>>(), [(VarId(1), VarId(0), 1e-200)]);
    }

    #[test]
    fn failures_preserve_pending_coefficients_and_scratch_is_reusable() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let y = Expr::from_var(&arena, VarId(1));
        let mut builder = QuadraticBuilder::new(&arena);
        builder.add_constant(7.0).unwrap().add_linear(3.0, x).unwrap();
        assert_eq!(
            builder.add_product(f64::MAX, 2.0 * x, 0.5 * y).unwrap_err(),
            QuadraticBuildError::ProductOverflow
        );

        builder.add_linear(f64::MAX, y).unwrap();
        assert_eq!(
            builder.add_term(f64::MAX, x + y + 1.0).unwrap_err(),
            QuadraticBuildError::AccumulationOverflow
        );

        let q = builder.build();
        let terms = crate::extract_quadratic(&arena.borrow(), q.id()).unwrap();
        assert_eq!(terms.constant(), 7.0);
        assert_eq!(terms.linear(), [(VarId(0), 3.0), (VarId(1), f64::MAX)]);

        builder.add_constant(9.0).unwrap();
        builder.clear().add_constant(4.0).unwrap();
        let constant = builder.build();
        assert_eq!(
            crate::extract_quadratic(&arena.borrow(), constant.id()).unwrap().constant(),
            4.0
        );

        let zero = builder.build();
        assert_eq!(crate::extract_quadratic(&arena.borrow(), zero.id()).unwrap().constant(), 0.0);
    }

    #[test]
    fn checked_coefficients_normalize_and_preserve_subnormal_hessians() {
        let tiny = f64::from_bits(1);
        let terms = QuadraticTerms::from_coefficients(
            vec![
                (VarId(0), VarId(2), 3.0),
                (VarId(1), VarId(1), tiny),
                (VarId(2), VarId(0), -1.0),
                (VarId(0), VarId(0), 0.0),
            ],
            vec![(VarId(2), 1.0), (VarId(0), 2.0), (VarId(2), -1.0)],
            0.0,
        )
        .unwrap();
        assert!(terms.is_canonical());
        assert_eq!(terms.quadratic(), [(VarId(2), VarId(0), 2.0), (VarId(1), VarId(1), tiny)]);
        assert_eq!(terms.linear(), [(VarId(0), 2.0)]);

        for bad in [f64::NAN, f64::INFINITY] {
            assert!(
                QuadraticTerms::from_coefficients(vec![(VarId(0), VarId(0), bad)], Vec::new(), 0.0)
                    .is_err()
            );
        }
        assert_eq!(
            QuadraticTerms::from_coefficients(
                vec![(VarId(0), VarId(1), f64::MAX); 2],
                Vec::new(),
                0.0
            )
            .unwrap_err(),
            QuadraticBuildError::AccumulationOverflow
        );

        let arena = ExprArenaCell::new(ExprArena::new());
        let raw = arena.borrow_mut().push(ExprNode::Quadratic(std::sync::Arc::new(terms)));
        let summed: Expr = std::iter::once(Expr::new(raw, &arena)).sum();
        assert_eq!(summed.id, raw);

        let q = crate::extract_quadratic(&arena.borrow(), summed.id()).unwrap();
        assert_eq!(q.quadratic()[1].2.to_bits(), 1);

        let doubled: Expr =
            [Expr::new(raw, &arena), Expr::constant(&arena, 1.0).as_dynamic()].into_iter().sum();
        assert_eq!(
            crate::extract_quadratic(&arena.borrow(), doubled.id()).unwrap().quadratic()[1]
                .2
                .to_bits(),
            1
        );

        let scaled = Expr::new(raw, &arena) * 2.0;
        assert_eq!(
            crate::extract_quadratic(&arena.borrow(), scaled.id()).unwrap().quadratic()[1]
                .2
                .to_bits(),
            2
        );
    }

    #[test]
    fn raw_linear_inputs_are_normalized_in_both_addition_orders() {
        for reverse in [false, true] {
            for shared in [false, true] {
                let arena = ExprArenaCell::new(ExprArena::new());
                let x = Expr::from_var(&arena, VarId(0));
                let raw = arena
                    .borrow_mut()
                    .linear(vec![(VarId(2), 0.0), (VarId(0), 1.0), (VarId(0), 2.0)], 0.0);
                let q = x * x;
                let q = if shared { q.clone() } else { q };
                let sum =
                    if reverse { q + Expr::new(raw, &arena) } else { Expr::new(raw, &arena) + q };
                let snapshot = arena.borrow();
                let ExprNode::Quadratic(q) = snapshot.get(sum.id()) else {
                    panic!("direct quadratic")
                };
                assert!(q.is_canonical());
                assert_eq!(q.linear(), [(VarId(0), 3.0)]);
            }
        }
    }

    #[test]
    fn fresh_powers_sums_and_scaling_keep_unique_roots() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let two = Expr::constant(&arena, 2.0);
        for mode in 0..6 {
            let before = arena.with_ref(|a| a.len());
            let q = match mode {
                0 => x.square().into_dynamic(),
                1 => x.powi(2),
                2 => x.powf(2.0),
                3 => std::iter::once(x.powi(2)).sum(),
                4 => Expr::__sum_terms(std::iter::once(x.powi(2))).unwrap(),
                _ => x.pow(&two),
            };
            assert_eq!(arena.with_ref(|a| a.len()), before + 1);

            let root = q.id;
            let count = arena.with_ref(|a| a.len());
            let q = -(q * 2.0);
            let q = q + x;
            assert_eq!(q.id, root);
            assert_eq!(arena.with_ref(|a| a.len()), count);
        }
        let q: Expr = [x.powi(2), x.powi(2)].into_iter().sum();
        let root = q.id;
        assert_eq!((q + x).id, root);

        let original = x.powi(2);
        let id = original.id;
        let reused = original.powi(1);
        assert!(!reused.owned, "a borrowed identity power must not recover ownership");

        let _ = reused + x;
        assert_eq!(evaluate(&arena.borrow(), id, &&[3.0][..]).unwrap(), 9.0);
    }

    #[test]
    fn eager_sums_obey_the_coefficient_budget() {
        for count in [31, 32, 33, 4096] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let sum: Expr = (0..count).map(|_| x.powi(2)).sum();
            let snapshot = arena.borrow();
            assert_eq!(
                matches!(snapshot.get(sum.id()), ExprNode::Quadratic(_)),
                count <= MAX_EAGER_COEFFICIENTS
            );
        }
    }

    #[test]
    fn published_views_are_canonical_after_unordered_growth_cancellation_and_scaling() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars: Vec<_> = (0..64).map(|i| Expr::from_var(&arena, VarId(i))).collect();
        let mut q = vars[0] * vars[0];
        let root = q.id;
        for i in (1..64).rev() {
            q = q + vars[i] * vars[(i + 1) % 64] + vars[i];
        }
        q = q - vars[20] * vars[21] - vars[20];
        let q = -(q * 2.0);
        assert_eq!(q.id, root);

        let snapshot = arena.borrow();
        let ExprNode::Quadratic(terms) = snapshot.get(q.id()) else { panic!("compact quadratic") };
        assert!(terms.is_canonical());
        assert!(!terms.linear().iter().any(|&(v, _)| v == VarId(20)));
        assert!(
            !terms
                .hessian_entries()
                .collect::<Vec<_>>()
                .iter()
                .any(|&(a, b, _)| (a, b) == (VarId(21), VarId(20)))
        );
        assert!(std::ptr::eq(terms.linear().as_ptr(), terms.linear().as_ptr()));
    }

    #[test]
    fn failed_unique_scaling_does_not_change_the_original() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let q = (f64::MAX / 2.0) * x * x;
        let root = q.id;
        let result = q * 4.0;
        assert_ne!(result.id, root);

        let snapshot = arena.borrow();
        let ExprNode::Quadratic(before) = snapshot.get(root) else { panic!("compact quadratic") };
        assert_eq!(before.quadratic(), [(VarId(0), VarId(0), f64::MAX / 2.0)]);
    }

    #[test]
    fn builder_and_unique_accumulation_work_inside_a_fork() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![arena.__with_fork(snapshot.clone(), || {
            let mut builder = QuadraticBuilder::new(&arena);
            builder.add_product(2.0, x, x).unwrap();
            let q = builder.build();
            let id = q.id;
            let q = -(q + x) * 2.0;
            assert_eq!(q.id, id);
            q.id()
        })];
        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let root = remaps[0].apply(forks[0].value);
        drop(batch);
        assert_eq!(evaluate(&arena.borrow(), root, &&[3.0][..]).unwrap(), -42.0);
    }

    #[test]
    fn checked_factored_quadratic_ingestion_works_inside_a_fork() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let mut forks = vec![arena.__with_fork(snapshot.clone(), || {
            let source = (x + 1.0).square();
            let mut builder = QuadraticBuilder::new(&arena);
            builder.add_quadratic(2.0, &source).unwrap();
            builder.add_quadratic(1.0, x).unwrap();
            builder.build().into_id()
        })];
        drop(snapshot);
        let remaps = batch.merge(&mut forks);
        let root = remaps[0].apply(forks[0].value);
        drop(batch);
        assert_eq!(evaluate(&arena.borrow(), root, &&[3.0][..]).unwrap(), 35.0);
    }

    #[test]
    fn scalar_offsets_keep_large_unique_quadratics_compact() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let mut builder = QuadraticBuilder::new(&arena);
        for i in 0..64 {
            let x = Expr::from_var(&arena, VarId(i));
            builder.add_product(1.0, x, x).unwrap();
        }
        let q = builder.build();
        let original = q.id;
        let q = q + 3.0 - 1;
        assert_eq!(q.id, original);

        let snapshot = arena.borrow();
        let ExprNode::Quadratic(terms) = snapshot.get(q.id) else { panic!("compact quadratic") };
        assert_eq!(terms.constant, 2.0);
        assert_eq!(terms.quadratic.len(), 64);
    }

    #[test]
    fn unique_sums_reuse_the_root_and_sharing_seals_permanently() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let y = Expr::from_var(&arena, VarId(1));
        let mut q = x * x;
        let original = q.id;
        for _ in 0..4096 {
            q = q + x * y;
        }
        assert_eq!(q.id, original, "unique accumulation must not copy the prefix");

        {
            let _alias = q.clone();
        }
        q = q + x * y;
        assert_ne!(q.id, original, "dropping an alias must not recover uniqueness");

        let snapshot = arena.borrow();
        assert_eq!(evaluate(&snapshot, original, &&[2.0, 3.0][..]).unwrap(), 24580.0);
        assert_eq!(evaluate(&snapshot, q.id, &&[2.0, 3.0][..]).unwrap(), 24586.0);
    }

    #[test]
    fn published_ids_snapshots_and_parent_insertion_protect_coefficients() {
        for mode in 0..4 {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let mut q = x * x;
            let original = q.id;
            match mode {
                0 => {
                    let _ = q.publish_id();
                }
                1 => {
                    let _ = arena.borrow();
                }
                2 => {
                    let _ = q.sin();
                }
                _ => {
                    let _ = Expr::new(original, &arena);
                }
            }
            q = q + x * x;
            assert_ne!(q.id, original);

            let snapshot = arena.borrow();
            assert_eq!(evaluate(&snapshot, original, &&[3.0][..]).unwrap(), 9.0);
            assert_eq!(evaluate(&snapshot, q.id, &&[3.0][..]).unwrap(), 18.0);
        }
    }

    #[test]
    fn speculative_overflow_keeps_the_previous_polynomial_intact() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let q = (f64::MAX * 0.75) * x * x;
        let original = q.id;
        let result = q + (f64::MAX / 2.0) * x * x;
        let snapshot = arena.borrow();
        let ExprNode::Quadratic(q) = snapshot.get(original) else { panic!("compact quadratic") };
        assert_eq!(q.quadratic[0].2, f64::MAX * 0.75);
        assert_ne!(result.id, original);
    }

    #[test]
    fn raw_duplicate_delta_is_validated_before_mutation() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let q = x * x;
        let original = q.id;
        let raw = arena.with_mut(|a| {
            a.push(ExprNode::Linear {
                coeffs: vec![(VarId(0), f64::MAX), (VarId(0), f64::MAX)],
                constant: 0.0,
            })
        });
        let result = q + Expr::new(raw, &arena);
        assert_ne!(result.id, original);

        let snapshot = arena.borrow();
        let ExprNode::Quadratic(before) = snapshot.get(original) else {
            panic!("compact quadratic")
        };
        assert_eq!(before.linear.as_slice(), []);
        assert_eq!(before.quadratic[0].2, 1.0);
    }
}
