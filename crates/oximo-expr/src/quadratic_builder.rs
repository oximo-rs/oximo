//! Validated numeric polynomial construction.

use rustc_hash::FxHashMap;
use thiserror::Error;

use crate::arena::ArenaAccess;
use crate::degree::AddDegree;
use crate::linear::as_linear;
use crate::{Affine, Expr, ExprArenaCell, ExprInput, ExprNode, Quadratic, QuadraticTerms, VarId};

/// A numeric quadratic could not be constructed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuadraticBuildError {
    #[error("quadratic input belongs to another expression arena")]
    ForeignArena,
    #[error("quadratic input must be affine")]
    NotAffine,
    #[error("parameter-dependent coefficients are unsupported")]
    ParameterizedInput,
    #[error("input coefficients must be finite")]
    NonFiniteInput,
    #[error("an intermediate coefficient product overflowed")]
    ProductOverflow,
    #[error("an accumulated coefficient overflowed")]
    AccumulationOverflow,
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
/// builder.add_product(3.0, x, x)?; // stores the coefficient 3 for x*x
/// let first = builder.build();
/// builder.add_linear(2.0, VarId(0))?;
/// let second = builder.build(); // reset after each build
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QuadraticBuilder<'a> {
    arena: &'a ExprArenaCell,
    coefficients: Accumulator,
    scratch: Accumulator,
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

    /// Check every destination before changing any entry. Keep scratch capacity.
    fn merge(&mut self, other: &Self) -> Result<(), QuadraticBuildError> {
        let constant = sum(self.constant, other.constant)?;
        for (&v, &c) in &other.linear {
            sum(self.linear.get(&v).copied().unwrap_or(0.0), c)?;
        }
        for (&pair, &c) in &other.quadratic {
            sum(self.quadratic.get(&pair).copied().unwrap_or(0.0), c)?;
        }

        self.constant = constant;
        for (&v, &c) in &other.linear {
            *self.linear.entry(v).or_default() += c;
        }
        for (&pair, &c) in &other.quadratic {
            *self.quadratic.entry(pair).or_default() += c;
        }

        Ok(())
    }

    pub(crate) fn finish(&mut self) -> QuadraticTerms {
        let mut linear: Vec<_> = self.linear.drain().filter(|(_, c)| *c != 0.0).collect();
        let mut quadratic: Vec<_> = self
            .quadratic
            .drain()
            .filter(|(_, c)| *c != 0.0)
            .map(|((a, b), h)| (a, b, h))
            .collect();
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
        Self { arena, coefficients, scratch: Accumulator::default() }
    }

    /// Add a finite constant.
    ///
    /// # Errors
    /// Rejects nonfinite input or accumulation overflow atomically.
    pub fn add_constant(&mut self, c: f64) -> Result<&mut Self, QuadraticBuildError> {
        self.coefficients.constant = sum(self.coefficients.constant, finite(c)?)?;
        Ok(self)
    }

    /// Add a linear coefficient using an arena-local variable ID.
    ///
    /// # Errors
    /// Rejects nonfinite input or accumulation overflow atomically.
    pub fn add_linear(
        &mut self,
        c: f64,
        variable: VarId,
    ) -> Result<&mut Self, QuadraticBuildError> {
        self.coefficients.add_linear(variable, c)?;
        Ok(self)
    }

    /// Add the coefficient of `first * second`, with no diagonal scaling.
    /// Variable IDs must belong to this builder's arena.
    ///
    /// # Errors
    /// Rejects nonfinite input or accumulation overflow atomically.
    pub fn add_monomial(
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
    /// Input must be numeric affine; use [`Self::add_quadratic`] for coefficient
    /// quadratics. Zero weights still validate the arena and expression class.
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
            self.coefficients.merge(&self.scratch)
        })?;

        Ok(self)
    }

    /// Add existing polynomial, linear and constant coefficients directly.
    /// Variable IDs are local to this arena.
    ///
    /// # Errors
    /// Rejects nonfinite scale, product overflow or accumulation overflow atomically.
    pub fn add_quadratic(
        &mut self,
        scale: f64,
        terms: &QuadraticTerms,
    ) -> Result<&mut Self, QuadraticBuildError> {
        finite(scale)?;
        self.scratch.clear();
        self.scratch.add_leaf(NumericLeaf::Quadratic(terms), scale)?;
        self.coefficients.merge(&self.scratch)?;

        Ok(self)
    }

    /// Expand `(c * lhs) * rhs` with a finite intermediate after each multiplication.
    /// Zero weights still validate both input classes and arenas.
    ///
    /// # Errors
    /// Rejects foreign, parameter-dependent, non-affine or nonfinite inputs and overflow atomically.
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

        self.arena.with_ref(|arena| {
            if let (ExprNode::Var(a), ExprNode::Var(b)) = (arena.get(lhs.id), arena.get(rhs.id)) {
                return self.coefficients.add_quadratic(*a, *b, c);
            }
            self.scratch.clear();
            let left = numeric_affine(arena, lhs.id)?;
            let right = numeric_affine(arena, rhs.id)?;
            let offset = product(c, finite(left.constant)?)?;
            self.scratch.constant = product(offset, finite(right.constant)?)?;
            for &(v, a) in left.coeffs.iter() {
                self.scratch
                    .add_linear(v, product(product(c, finite(a)?)?, finite(right.constant)?)?)?;
            }
            for &(v, b) in right.coeffs.iter() {
                self.scratch.add_linear(v, product(offset, finite(b)?)?)?;
            }
            for &(a, x) in left.coeffs.iter() {
                let scaled = product(c, finite(x)?)?;
                for &(b, y) in right.coeffs.iter() {
                    self.scratch.add_quadratic(a, b, product(scaled, finite(y)?)?)?;
                }
            }
            self.coefficients.merge(&self.scratch)
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
        crate::linear::push_linear(arena, crate::LinearTerms::owned(terms.linear, terms.constant))
    } else {
        arena.push(ExprNode::Quadratic(std::sync::Arc::new(terms)))
    }
}

#[derive(Debug, Default)]
pub(crate) struct QuadraticSlots {
    linear: FxHashMap<VarId, usize>,
    quadratic: FxHashMap<(VarId, VarId), usize>,
    initialized: bool,
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
    if !slots.initialized {
        slots.linear.extend(left.linear.iter().enumerate().map(|(i, &(v, _))| (v, i)));
        slots
            .quadratic
            .extend(left.quadratic.iter().enumerate().map(|(i, &(a, b, _))| ((a, b), i)));
        slots.initialized = true;
    }

    let constant = sum(left.constant, product(finite(offset).ok()?, scale).ok()?).ok()?;
    let mut values_changed = false;
    let mut order_changed = false;
    for &(v, c) in linear {
        let slot = slots.linear.get(&v);
        let current = slot.map_or(0.0, |&i| left.linear[i].1);
        let value = sum(current, product(finite(c).ok()?, scale).ok()?).ok()?;
        values_changed |= value.to_bits() != current.to_bits();
        order_changed |= if slot.is_some() { value == 0.0 } else { value != 0.0 };
    }

    for &(a, b, h) in quadratic {
        let slot = slots.quadratic.get(&(a, b));
        let current = slot.map_or(0.0, |&i| left.quadratic[i].2);
        let value = sum(current, product(h, scale).ok()?).ok()?;
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
        let delta = c * scale;
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
        let delta = h * scale;
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

pub(crate) fn scale_unique(
    left: &mut QuadraticTerms,
    slots: &mut QuadraticSlots,
    scale: f64,
) -> Option<()> {
    finite(scale).ok()?;
    let constant = product(left.constant, scale).ok()?;
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
    fn right_growing_sums_retain_the_larger_unique_destination() {
        let arena = ExprArenaCell::new(ExprArena::new());
        let x = Expr::from_var(&arena, VarId(0));
        let mut builder = QuadraticBuilder::new(&arena);
        for i in 0..64 {
            builder.add_monomial(1.0, VarId(i), VarId(i)).unwrap();
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
            (Expr::constant(&arena, f64::MAX).erase(), Expr::constant(&arena, 2.0).erase()),
            ((f64::MAX * x).erase(), Expr::constant(&arena, 2.0).erase()),
            (Expr::constant(&arena, f64::MAX).erase(), (2.0 * y).erase()),
            ((f64::MAX * x).erase(), (2.0 * y).erase()),
        ] {
            builder
                .add_product(0.5, left.try_affine().unwrap(), right.try_affine().unwrap())
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
        builder.add_constant(7.0).unwrap().add_linear(3.0, VarId(0)).unwrap();
        assert_eq!(
            builder.add_product(f64::MAX, 2.0 * x, 0.5 * y).unwrap_err(),
            QuadraticBuildError::ProductOverflow
        );

        builder.add_linear(f64::MAX, VarId(1)).unwrap();
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
            [Expr::new(raw, &arena), Expr::constant(&arena, 1.0).erase()].into_iter().sum();
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
                0 => {
                    let q = x.square();
                    Expr::from_construction(q.id, q.owned, &arena)
                }
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

        let alias = q.clone();
        drop(alias);
        q = q + x * y;
        assert_ne!(q.id, original, "dropping an alias must not recover uniqueness");

        let snapshot = arena.borrow();
        assert_eq!(evaluate(&snapshot, original, &&[2.0, 3.0][..]).unwrap(), 24580.0);
        assert_eq!(evaluate(&snapshot, q.id, &&[2.0, 3.0][..]).unwrap(), 24586.0);
    }

    #[test]
    fn raw_ids_snapshots_and_parent_insertion_protect_coefficients() {
        for mode in 0..4 {
            let arena = ExprArenaCell::new(ExprArena::new());
            let x = Expr::from_var(&arena, VarId(0));
            let mut q = x * x;
            let original = q.id;
            match mode {
                0 => {
                    let _ = q.id();
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
        assert!(before.linear.is_empty());
        assert_eq!(before.quadratic[0].2, 1.0);
    }
}
