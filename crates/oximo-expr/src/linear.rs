use std::borrow::Cow;

use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::{SmallVec, smallvec};

use crate::arena::{ArenaAccess, Children, ExprArena, ExprId, ExprNode, UnaryOp, VarId};

const MAX_EAGER_COEFFICIENTS: usize = 32;
const MAX_EAGER_NODES: usize = 64;

/// Stack-backed merging avoids a hash table and per-variable vectors for the
/// small numeric expressions that dominate scalar model construction.
struct SmallLinear {
    coeffs: SmallVec<[(VarId, f64); MAX_EAGER_COEFFICIENTS]>,
    constant: f64,
}

impl SmallLinear {
    fn new() -> Self {
        Self { coeffs: SmallVec::new(), constant: 0.0 }
    }

    fn add(&mut self, var: VarId, coefficient: f64) -> bool {
        if let Some((_, value)) = self.coeffs.iter_mut().find(|(v, _)| *v == var) {
            *value += coefficient;
        } else {
            if self.coeffs.len() == MAX_EAGER_COEFFICIENTS {
                return false;
            }
            self.coeffs.push((var, coefficient));
        }
        true
    }

    fn terminal(&mut self, node: &ExprNode) -> bool {
        match node {
            ExprNode::Const(c) => self.constant += c,
            ExprNode::Var(v) => {
                if !self.add(*v, 1.0) {
                    return false;
                }
                self.constant += 0.0;
            }
            ExprNode::Linear { coeffs, constant } => {
                for &(var, coefficient) in coeffs {
                    if !self.add(var, coefficient) {
                        return false;
                    }
                }
                self.constant += constant;
            }
            _ => return false,
        }
        true
    }

    fn operand(&mut self, arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> bool {
        if let ExprNode::Add(children) = arena.get(id) {
            // We use wide flat numeric prefix because it is cheap if its
            // merged support is small.
            let mut segment = Self::new();
            for &child in children {
                if !segment.terminal(arena.get(child)) {
                    return false;
                }
            }
            self.constant += segment.constant;
            for (var, coefficient) in segment.coeffs {
                if !self.add(var, coefficient) {
                    return false;
                }
            }
            true
        } else {
            let ok = self.terminal(arena.get(id));
            // Direct leaves have not gone through a sum's initial +0.
            self.constant = match arena.get(id) {
                ExprNode::Const(c) | ExprNode::Linear { constant: c, .. } => *c,
                _ => self.constant,
            };
            ok
        }
    }

    fn push(self, arena: &mut (impl ArenaAccess + ?Sized)) -> ExprId {
        push_linear(arena, LinearTerms::owned(self.coeffs.into_vec(), self.constant))
    }
}

/// Coefficients of a linear expression: `sum(coeff * var) + constant`.
///
/// The coefficient storage borrows directly from an [`ExprNode::Linear`] when
/// possible and owns coefficients that must be synthesized while walking an
/// expression tree. Call [`LinearTerms::into_owned`] before storing extracted
/// terms beyond the lifetime of their [`ExprArena`].
#[derive(Clone, Debug, Default)]
pub struct LinearTerms<'a> {
    pub coeffs: Cow<'a, [(VarId, f64)]>,
    pub constant: f64,
}

impl<'a> LinearTerms<'a> {
    /// Construct linear terms with owned coefficient storage.
    pub fn owned(coeffs: Vec<(VarId, f64)>, constant: f64) -> Self {
        Self { coeffs: Cow::Owned(coeffs), constant }
    }

    /// Construct linear terms borrowing coefficient storage.
    pub fn borrowed(coeffs: &'a [(VarId, f64)], constant: f64) -> Self {
        Self { coeffs: Cow::Borrowed(coeffs), constant }
    }

    /// Detach these terms from their borrowed source.
    ///
    /// This does not allocate when the coefficient storage is already owned.
    pub fn into_owned(self) -> LinearTerms<'static> {
        LinearTerms { coeffs: Cow::Owned(self.coeffs.into_owned()), constant: self.constant }
    }
}

/// Accumulator that merges duplicate `(VarId, coeff)` terms while
/// preserving the order each variable is first seen.
#[derive(Debug)]
pub(crate) struct CoeffAccum {
    pub(crate) coeffs: Vec<(VarId, f64)>,
    slot: FxHashMap<VarId, usize>,
}

impl CoeffAccum {
    pub(crate) fn clear(&mut self) {
        self.coeffs.clear();
        self.slot.clear();
    }
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            coeffs: Vec::with_capacity(n),
            slot: FxHashMap::with_capacity_and_hasher(n, FxBuildHasher),
        }
    }

    /// Add `c` to `v`'s running coefficient, appending `v` the first time it is
    /// seen.
    pub(crate) fn add(&mut self, v: VarId, c: f64) {
        match self.slot.entry(v) {
            std::collections::hash_map::Entry::Occupied(slot) => self.coeffs[*slot.get()].1 += c,
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(self.coeffs.len());
                self.coeffs.push((v, c));
            }
        }
    }

    pub(crate) fn extend_from_slice(&mut self, terms: &[(VarId, f64)]) {
        for &(v, c) in terms {
            self.add(v, c);
        }
    }

    pub(crate) fn into_coeffs(self) -> Vec<(VarId, f64)> {
        self.coeffs
    }
}

/// Try to interpret `id` as a linear expression. Returns `None` for any
/// nonlinear node (Mul of two non-constants, Pow, transcendentals, ...).
///
/// When `resolve_params` is set, an [`ExprNode::Param`] folds to its current
/// arena value and counts as a constant.
#[cfg(test)]
fn recursive_linear<'a, A: ArenaAccess + ?Sized>(
    arena: &'a A,
    id: ExprId,
    resolve_params: bool,
) -> Option<LinearTerms<'a>> {
    match arena.get(id) {
        ExprNode::Const(c) => Some(LinearTerms::borrowed(&[], *c)),
        ExprNode::Param(p) if resolve_params => {
            Some(LinearTerms::borrowed(&[], arena.param_value(*p)))
        }
        ExprNode::Var(v) => {
            Some(LinearTerms { coeffs: Cow::Owned(vec![(*v, 1.0)]), constant: 0.0 })
        }
        ExprNode::Linear { coeffs, constant } => Some(LinearTerms::borrowed(coeffs, *constant)),
        ExprNode::Unary(UnaryOp::Neg, inner) => {
            let inner = *inner;
            recursive_linear(arena, inner, resolve_params).map(|t| {
                let mut coeffs = t.coeffs.into_owned();
                for (_, c) in &mut coeffs {
                    *c = -*c;
                }
                LinearTerms { coeffs: Cow::Owned(coeffs), constant: -t.constant }
            })
        }
        ExprNode::Add(children) => {
            let mut acc = CoeffAccum::with_capacity(children.len() * 4);
            let mut constant = 0.0;
            for &child in children {
                let t = recursive_linear(arena, child, resolve_params)?;
                acc.extend_from_slice(&t.coeffs);
                constant += t.constant;
            }
            Some(LinearTerms::owned(acc.into_coeffs(), constant))
        }
        ExprNode::Mul(children) => {
            let mut product = LinearTerms::borrowed(&[], 1.0);
            for &child in children {
                product =
                    multiply_linear(product, recursive_linear(arena, child, resolve_params)?)?;
            }
            Some(product)
        }
        ExprNode::Pow(base, exp) => {
            let ExprNode::Const(exponent) = *arena.get(*exp) else { return None };
            if !exponent.is_finite()
                || (exponent - exponent.round()).abs() >= f64::EPSILON
                || exponent < 0.0
            {
                return None;
            }
            if exponent.round() == 0.0 {
                return Some(LinearTerms::borrowed(&[], 1.0));
            }
            let terms = recursive_linear(arena, *base, resolve_params)?;
            affine_power(terms, exponent.round())
        }
        _ => None,
    }
}

struct LinearFolder<'a, 'b, A: ?Sized> {
    arena: &'a A,
    resolve_params: bool,
    root: ExprId,
    // Decide lazily so flat sums keep their allocation-free traversal setup.
    flatten_sums: &'b std::cell::Cell<Option<bool>>,
    sum_magnitude: &'b std::cell::Cell<f64>,
}

// Keep the wide-sum loop independent of the larger deferred fold state.
fn flat_scaled_var<A: ArenaAccess + ?Sized>(
    arena: &A,
    children: &[ExprId],
    resolve_params: bool,
) -> Option<(VarId, f64)> {
    let [left, right] = children else { return None };
    let scalar = |id| match arena.get(id) {
        ExprNode::Const(value) => Some(*value),
        ExprNode::Param(param) if resolve_params => Some(arena.param_value(*param)),
        _ => None,
    };
    match (arena.get(*left), arena.get(*right)) {
        (_, ExprNode::Var(var)) => Some((*var, 1.0 * scalar(*left)?)),
        (ExprNode::Var(var), _) => Some((*var, 1.0 * scalar(*right)?)),
        _ => None,
    }
}

#[inline(never)]
fn flat_linear<'a, A: ArenaAccess + ?Sized>(
    arena: &'a A,
    children: &[ExprId],
    resolve_params: bool,
) -> Option<LinearTerms<'a>> {
    let terminal = |id| match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Linear { .. } => true,
        ExprNode::Param(_) => resolve_params,
        ExprNode::Mul(children) => flat_scaled_var(arena, children, resolve_params).is_some(),
        _ => false,
    };
    if children.first().is_some_and(|&id| !terminal(id))
        || children.last().is_some_and(|&id| !terminal(id))
    {
        return None;
    }
    let mut acc = CoeffAccum::with_capacity(children.len());
    let mut constant = 0.0;
    for &child in children {
        match arena.get(child) {
            ExprNode::Var(var) => {
                acc.add(*var, 1.0);
                constant += 0.0;
            }
            ExprNode::Const(value) => constant += value,
            ExprNode::Param(param) if resolve_params => constant += arena.param_value(*param),
            ExprNode::Linear { coeffs, constant: value } => {
                if let [(var, coefficient)] = coeffs.as_slice() {
                    acc.add(*var, *coefficient);
                } else {
                    acc.extend_from_slice(coeffs);
                }
                constant += value;
            }
            ExprNode::Mul(children) => {
                let (var, scalar) = flat_scaled_var(arena, children, resolve_params)?;
                acc.add(var, 1.0 * scalar);
                constant += 0.0 * scalar;
            }
            _ => return None,
        }
    }
    Some(LinearTerms::owned(acc.into_coeffs(), constant))
}

enum LinearState<'a> {
    Sum { rest: &'a [ExprId], pending: crate::fold::SumFrames<'a>, acc: CoeffAccum, constant: f64 },
    Scale { child: Option<ExprId>, value: Option<LinearTerms<'a>>, scalar: f64, negate: bool },
    Product { rest: &'a [ExprId], value: Option<LinearTerms<'a>> },
    Power { child: Option<ExprId>, value: Option<LinearTerms<'a>>, exponent: f64 },
}

#[expect(clippy::float_cmp, reason = "the exponent has been validated and rounded to an integer")]
fn affine_power(terms: LinearTerms<'_>, exponent: f64) -> Option<LinearTerms<'_>> {
    if exponent == 1.0 {
        Some(terms)
    } else if terms.coeffs.is_empty() {
        Some(LinearTerms::borrowed(&[], terms.constant.powf(exponent)))
    } else {
        None
    }
}

/// Multiply affine terms only when one side has no decision variables.
/// Parameter-dependent constants are resolved by the caller at extraction time.
fn multiply_linear<'a>(left: LinearTerms<'a>, right: LinearTerms<'a>) -> Option<LinearTerms<'a>> {
    let (mut terms, scalar) = if left.coeffs.is_empty() {
        (right, left.constant)
    } else if right.coeffs.is_empty() {
        (left, right.constant)
    } else {
        return None;
    };
    if !terms.coeffs.is_empty() {
        for (_, coefficient) in terms.coeffs.to_mut() {
            *coefficient *= scalar;
        }
    }
    terms.constant *= scalar;
    Some(terms)
}

impl<'a, A: ArenaAccess + ?Sized> crate::fold::Folder for LinearFolder<'a, '_, A> {
    type Value = LinearTerms<'a>;
    type State = LinearState<'a>;

    fn start(&self, id: ExprId) -> std::ops::ControlFlow<Option<Self::Value>, Self::State> {
        use std::ops::ControlFlow::{Break, Continue};
        Break(match self.arena.get(id) {
            ExprNode::Const(c) => Some(LinearTerms::borrowed(&[], *c)),
            ExprNode::Param(p) if self.resolve_params => {
                Some(LinearTerms::borrowed(&[], self.arena.param_value(*p)))
            }
            ExprNode::Var(v) => Some(LinearTerms::owned(vec![(*v, 1.0)], 0.0)),
            ExprNode::Linear { coeffs, constant } => Some(LinearTerms::borrowed(coeffs, *constant)),
            ExprNode::Add(children) => {
                return Continue(LinearState::Sum {
                    rest: children,
                    pending: None,
                    acc: CoeffAccum::with_capacity(children.len()),
                    constant: 0.0,
                });
            }
            ExprNode::Unary(UnaryOp::Neg, child) => {
                return Continue(LinearState::Scale {
                    child: Some(*child),
                    value: None,
                    scalar: 1.0,
                    negate: true,
                });
            }
            ExprNode::Mul(children) => {
                let mut scalar = 1.0;
                let mut linear = None;
                for &child in children {
                    match self.arena.get(child) {
                        ExprNode::Const(c) => scalar *= c,
                        ExprNode::Param(p) if self.resolve_params => {
                            scalar *= self.arena.param_value(*p);
                        }
                        _ if linear.is_none() => linear = Some(child),
                        node => {
                            let has_variables = |node: &ExprNode| match node {
                                ExprNode::Var(_) => true,
                                ExprNode::Linear { coeffs, .. } => !coeffs.is_empty(),
                                _ => false,
                            };
                            if has_variables(node)
                                && has_variables(self.arena.get(linear.expect("first factor")))
                            {
                                return Break(None);
                            }
                            return Continue(LinearState::Product {
                                rest: children,
                                value: Some(LinearTerms::borrowed(&[], 1.0)),
                            });
                        }
                    }
                }
                if linear.is_some() {
                    return Continue(LinearState::Scale {
                        child: linear,
                        value: None,
                        scalar,
                        negate: false,
                    });
                }
                Some(LinearTerms::owned(Vec::new(), scalar))
            }
            ExprNode::Pow(base, exp) => {
                let ExprNode::Const(exponent) = *self.arena.get(*exp) else { return Break(None) };
                if !exponent.is_finite()
                    || (exponent - exponent.round()).abs() >= f64::EPSILON
                    || exponent < 0.0
                {
                    return Break(None);
                }
                if exponent.round() == 0.0 {
                    return Break(Some(LinearTerms::borrowed(&[], 1.0)));
                }
                return Continue(LinearState::Power {
                    child: Some(*base),
                    value: None,
                    exponent: exponent.round(),
                });
            }
            _ => None,
        })
    }

    fn next(&self, state: &mut Self::State) -> std::ops::ControlFlow<Option<Self::Value>, ExprId> {
        use std::ops::ControlFlow::{Break, Continue};
        match state {
            LinearState::Power { child, value, exponent } => {
                if let Some(child) = child.take() {
                    return Continue(child);
                }
                Break(value.take().and_then(|terms| affine_power(terms, *exponent)))
            }
            LinearState::Sum { rest, pending, acc, constant } => {
                loop {
                    let Some((&child, tail)) = rest.split_first() else {
                        if let Some((next, previous_constant)) =
                            pending.as_mut().and_then(|frames| frames.pop())
                        {
                            *constant += previous_constant;
                            *rest = next;
                            continue;
                        }
                        break;
                    };
                    *rest = tail;
                    // Accumulate terminals directly: a wide sum needs no per-variable vectors.
                    match self.arena.get(child) {
                        ExprNode::Var(v) => {
                            acc.add(*v, 1.0);
                            *constant += 0.0;
                            crate::fold::record_sum_magnitude(self.sum_magnitude, 1.0);
                        }
                        ExprNode::Const(c) => {
                            *constant += c;
                            crate::fold::record_sum_magnitude(self.sum_magnitude, *c);
                        }
                        ExprNode::Param(p) if self.resolve_params => {
                            let value = self.arena.param_value(*p);
                            *constant += value;
                            crate::fold::record_sum_magnitude(self.sum_magnitude, value);
                        }
                        ExprNode::Linear { coeffs, constant: c } => {
                            acc.extend_from_slice(coeffs);
                            *constant += c;
                            crate::fold::record_sum_magnitude(
                                self.sum_magnitude,
                                c.abs() + coeffs.iter().map(|(_, c)| c.abs()).sum::<f64>(),
                            );
                        }
                        _ => return Continue(child),
                    }
                }
                Break(Some(LinearTerms::owned(std::mem::take(&mut acc.coeffs), *constant)))
            }
            LinearState::Scale { child, value, scalar, negate } => {
                if let Some(child) = child.take() {
                    return Continue(child);
                }
                Break(value.take().map(|t| {
                    let mut coeffs = t.coeffs.into_owned();
                    for (_, c) in &mut coeffs {
                        *c = if *negate { -*c } else { *c * *scalar };
                    }
                    LinearTerms::owned(
                        coeffs,
                        if *negate { -t.constant } else { t.constant * *scalar },
                    )
                }))
            }
            LinearState::Product { rest, value } => {
                if value.is_some()
                    && let Some((&child, tail)) = rest.split_first()
                {
                    *rest = tail;
                    return Continue(child);
                }
                Break(value.take())
            }
        }
    }

    fn accept(&self, state: &mut Self::State, value: Self::Value) {
        match state {
            LinearState::Sum { acc, constant, .. } => {
                acc.extend_from_slice(&value.coeffs);
                *constant += value.constant;
                crate::fold::record_sum_magnitude(
                    self.sum_magnitude,
                    value.constant.abs() + value.coeffs.iter().map(|(_, c)| c.abs()).sum::<f64>(),
                );
            }
            LinearState::Scale { value: slot, .. } | LinearState::Power { value: slot, .. } => {
                *slot = Some(value);
            }
            LinearState::Product { value: slot, .. } => {
                *slot = slot.take().and_then(|left| multiply_linear(left, value));
            }
        }
    }

    fn inline(&self, state: &mut Self::State, child: ExprId) -> bool {
        if let LinearState::Sum { rest, pending, constant, .. } = state
            && let ExprNode::Add(children) = self.arena.get(child)
        {
            let flatten = self.flatten_sums.get().unwrap_or_else(|| {
                let flatten = !bounded_affine(self.arena, &[self.root], self.resolve_params);
                self.flatten_sums.set(Some(flatten));
                flatten
            });
            if !flatten {
                return false;
            }
            pending.get_or_insert_with(Box::default).push((*rest, *constant));
            *constant = 0.0;
            *rest = children;
            return true;
        }
        false
    }
}

/// Keep operator construction bounded even when an operand is a large sum.
#[inline]
fn small_affine(arena: &(impl ArenaAccess + ?Sized), ids: &[ExprId]) -> bool {
    // Most eager operands are already terminals, we avoid traversal scratch and
    // the full node match for these common scalar modeling operations.
    let mut coefficients = 0;
    for &id in ids {
        coefficients += match arena.get(id) {
            ExprNode::Const(_) => 0,
            ExprNode::Var(_) => 1,
            ExprNode::Linear { coeffs, .. } => coeffs.len(),
            _ => return bounded_affine(arena, ids, false),
        };
        if coefficients > MAX_EAGER_COEFFICIENTS {
            return false;
        }
    }
    true
}

pub(crate) fn bounded_affine(
    arena: &(impl ArenaAccess + ?Sized),
    ids: &[ExprId],
    resolve_params: bool,
) -> bool {
    let mut stack = SmallVec::<[ExprId; MAX_EAGER_NODES]>::from_slice(ids);
    let mut nodes = 0;
    let mut coefficients = 0;
    while let Some(id) = stack.pop() {
        nodes += 1;
        if nodes > MAX_EAGER_NODES {
            return false;
        }
        match arena.get(id) {
            ExprNode::Const(_) => {}
            ExprNode::Param(_) if resolve_params => {}
            ExprNode::Var(_) => coefficients += 1,
            ExprNode::Linear { coeffs, .. } => coefficients += coeffs.len(),
            ExprNode::Add(children) | ExprNode::Mul(children) => {
                if children.len() > MAX_EAGER_NODES - nodes
                    || stack.len() + children.len() > MAX_EAGER_NODES
                {
                    return false;
                }
                stack.extend_from_slice(children);
            }
            ExprNode::Unary(UnaryOp::Neg, child) => stack.push(*child),
            ExprNode::Pow(base, exp) => stack.extend_from_slice(&[*base, *exp]),
            _ => return false,
        }
        if coefficients > MAX_EAGER_COEFFICIENTS {
            return false;
        }
    }
    true
}

pub(crate) fn as_linear<'a, A: ArenaAccess + ?Sized>(
    arena: &'a A,
    mut id: ExprId,
    resolve_params: bool,
) -> Option<LinearTerms<'a>> {
    let mut negations = 0;
    while let ExprNode::Unary(UnaryOp::Neg, child) = arena.get(id) {
        id = *child;
        negations += 1;
    }
    if negations == 0
        && let ExprNode::Add(children) = arena.get(id)
        && let Some(terms) = flat_linear(arena, children, resolve_params)
    {
        return Some(terms);
    }
    let flatten_sums = std::cell::Cell::new(None);
    let sum_magnitude = std::cell::Cell::new(0.0);
    let mut value = crate::fold::fold(
        arena,
        id,
        LinearFolder {
            arena,
            resolve_params,
            root: id,
            flatten_sums: &flatten_sums,
            sum_magnitude: &sum_magnitude,
        },
    )?;
    if flatten_sums.get() == Some(true)
        && (!sum_magnitude.get().is_finite()
            || !value.constant.is_finite()
            || value.coeffs.iter().any(|(_, c)| !c.is_finite()))
        && crate::fold::finite_inputs(arena, id, resolve_params)
    {
        // Preserve original grouping on either side of an overflow.
        value = crate::fold::fold(
            arena,
            id,
            LinearFolder {
                arena,
                resolve_params,
                root: id,
                flatten_sums: &std::cell::Cell::new(Some(false)),
                sum_magnitude: &std::cell::Cell::new(f64::INFINITY),
            },
        )?;
    }
    if negations != 0 {
        let coeffs = value.coeffs.to_mut();
        for (_, c) in coeffs.iter_mut() {
            for _ in 0..negations {
                *c = -*c;
            }
        }
        for _ in 0..negations {
            value.constant = -value.constant;
        }
    }
    Some(value)
}

/// Materialize linear terms into a fresh `Linear` node in the arena.
pub(crate) fn push_linear(arena: &mut (impl ArenaAccess + ?Sized), t: LinearTerms<'_>) -> ExprId {
    let mut coeffs = t.coeffs.into_owned();
    coeffs.retain(|(_, c)| *c != 0.0);
    if coeffs.is_empty() {
        return arena.constant(t.constant);
    }
    arena.push(ExprNode::Linear { coeffs, constant: t.constant })
}

/// Build `lhs + rhs`, eagerly merging small numeric affine operands.
/// Large or symbolic operands retain an immutable binary `Add` node.
pub(crate) fn add_into(
    arena: &mut (impl ArenaAccess + ?Sized),
    lhs: ExprId,
    rhs: ExprId,
) -> ExprId {
    let mut left = SmallLinear::new();
    if left.operand(arena, lhs) {
        let merged = if matches!(arena.get(rhs), ExprNode::Add(_)) {
            let mut right = SmallLinear::new();
            if right.operand(arena, rhs) {
                left.constant += right.constant;
                right.coeffs.into_iter().all(|(var, coefficient)| left.add(var, coefficient))
            } else {
                false
            }
        } else {
            left.terminal(arena.get(rhs))
        };
        if merged {
            return left.push(arena);
        }
    }
    if small_affine(arena, &[lhs, rhs])
        && let (Some(lt), Some(rt)) = (as_linear(arena, lhs, false), as_linear(arena, rhs, false))
    {
        let constant = lt.constant + rt.constant;
        let mut acc = CoeffAccum::with_capacity(lt.coeffs.len() + rt.coeffs.len());
        acc.extend_from_slice(&lt.coeffs);
        acc.extend_from_slice(&rt.coeffs);
        let coeffs = acc.into_coeffs();
        drop(lt);
        drop(rt);
        return push_linear(arena, LinearTerms { coeffs: Cow::Owned(coeffs), constant });
    }
    arena.push(ExprNode::Add(smallvec![lhs, rhs]))
}

/// Build a flat n-ary sum of `ids` as a single `Add` node.
/// `as_linear`/`split_linear` collapse the resulting `Add`
/// in one pass at extraction, so the linear fast-path is preserved.
///
/// # Panics
/// Panics if `ids` is empty (callers supply at least one term).
pub(crate) fn add_n(arena: &mut (impl ArenaAccess + ?Sized), ids: Children) -> ExprId {
    match ids.as_slice() {
        [] => panic!("add_n on an empty term list"),
        [one] => *one,
        _ => arena.push(ExprNode::Add(ids)),
    }
}

/// Build `lhs - rhs`. Same linear fast-path as `add_into`.
pub(crate) fn sub_into(
    arena: &mut (impl ArenaAccess + ?Sized),
    lhs: ExprId,
    rhs: ExprId,
) -> ExprId {
    let neg = neg_into(arena, rhs);
    add_into(arena, lhs, neg)
}

/// Build `lhs * rhs`. A constant times a small numeric affine operand uses
/// the eager fast path. Large or symbolic operands retain a binary `Mul`.
pub(crate) fn mul_into(
    arena: &mut (impl ArenaAccess + ?Sized),
    lhs: ExprId,
    rhs: ExprId,
) -> ExprId {
    if let ExprNode::Const(c) = *arena.get(lhs)
        && small_affine(arena, &[rhs])
        && let Some(t) = as_linear(arena, rhs, false)
    {
        let constant = t.constant * c;
        let mut coeffs = t.coeffs.into_owned();
        for (_, co) in &mut coeffs {
            *co *= c;
        }
        return push_linear(arena, LinearTerms { coeffs: Cow::Owned(coeffs), constant });
    }
    if let ExprNode::Const(c) = *arena.get(rhs)
        && small_affine(arena, &[lhs])
        && let Some(t) = as_linear(arena, lhs, false)
    {
        let constant = t.constant * c;
        let mut coeffs = t.coeffs.into_owned();
        for (_, co) in &mut coeffs {
            *co *= c;
        }
        return push_linear(arena, LinearTerms { coeffs: Cow::Owned(coeffs), constant });
    }
    // Normalize at the boundary from affine arithmetic to a product of
    // expressions.
    if !scalar_leaf(arena.get(lhs))
        && !scalar_leaf(arena.get(rhs))
        && !small_scalar(arena, lhs)
        && !small_scalar(arena, rhs)
    {
        let left = compact_numeric_into(arena, lhs);
        let right = if lhs == rhs { left } else { compact_numeric_into(arena, rhs) };
        if left != lhs || right != rhs {
            if let (Some(lt), Some(rt)) =
                (as_linear(arena, left, false), as_linear(arena, right, false))
                && let Some(terms) = multiply_linear(lt, rt)
            {
                let terms = terms.into_owned();
                return push_linear(arena, terms);
            }
            return mul_into(arena, left, right);
        }
    }
    arena.push(ExprNode::Mul(smallvec![lhs, rhs]))
}

fn small_scalar(arena: &(impl ArenaAccess + ?Sized), id: ExprId) -> bool {
    match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Param(_) => true,
        ExprNode::Var(_) => false,
        ExprNode::Linear { coeffs, .. } => coeffs.is_empty(),
        _ => crate::classify::is_constant_access(arena, id),
    }
}

fn scalar_leaf(node: &ExprNode) -> bool {
    matches!(node, ExprNode::Const(_) | ExprNode::Param(_))
}

/// Compact only numeric affine operands, keeping parameters symbolic.
pub(crate) fn compact_numeric_into(arena: &mut (impl ArenaAccess + ?Sized), id: ExprId) -> ExprId {
    if matches!(
        arena.get(id),
        ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
    ) {
        return id;
    }
    // A cached structural degree of a growing product makes this rejection
    // bounded.
    if crate::classify::classify_access(arena, id) != crate::ExprClass::Linear {
        return id;
    }
    let parameter_child = match arena.get(id) {
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            children.iter().any(|child| matches!(arena.get(*child), ExprNode::Param(_)))
        }
        ExprNode::Unary(UnaryOp::Neg, child) => matches!(arena.get(*child), ExprNode::Param(_)),
        _ => false,
    };
    // A direct parameter makes numeric extraction certain to fail.
    if parameter_child && let Some(replacement) = compact_parameter_parent(arena, id) {
        return replacement;
    }
    if !parameter_child && let Some(terms) = as_linear(arena, id, false) {
        let terms = terms.into_owned();
        return push_linear(arena, terms);
    }
    compact_symbolic_numeric_children(arena, id)
}

/// When parameters and numeric children under one parent, extract each numeric
/// child once, without scanning its descendants to build parameter metadata first.
fn compact_parameter_parent(arena: &mut (impl ArenaAccess + ?Sized), id: ExprId) -> Option<ExprId> {
    let mut node = arena.get(id).clone();
    let mut children = FxHashMap::default();
    let mut extract = |child: ExprId| -> Option<()> {
        if matches!(
            arena.get(child),
            ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
        ) {
            return Some(());
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = children.entry(child) {
            entry.insert(as_linear(arena, child, false)?.into_owned());
        }
        Some(())
    };
    match &node {
        ExprNode::Add(ids) | ExprNode::Mul(ids) => {
            for &child in ids {
                extract(child)?;
            }
        }
        ExprNode::Unary(UnaryOp::Neg, child) => extract(*child)?,
        _ => return None,
    }
    if children.is_empty() {
        return Some(id);
    }
    let replacements: FxHashMap<_, _> =
        children.into_iter().map(|(child, terms)| (child, push_linear(arena, terms))).collect();
    let rewrite = |child: &mut ExprId| {
        if let Some(&replacement) = replacements.get(child) {
            *child = replacement;
        }
    };
    match &mut node {
        ExprNode::Add(ids) | ExprNode::Mul(ids) => ids.iter_mut().for_each(rewrite),
        ExprNode::Unary(UnaryOp::Neg, child) => rewrite(child),
        _ => unreachable!("only affine arithmetic reaches this point"),
    }
    Some(arena.push(node))
}

/// Rewrite maximal numeric regions under a symbolic affine root. Parameters
/// are never evaluated, and shared regions are compacted at most once. Numeric
/// prefixes stay deferred until a symbolic parent needs their normalized form.
fn compact_symbolic_numeric_children(
    arena: &mut (impl ArenaAccess + ?Sized),
    root: ExprId,
) -> ExprId {
    let mut numeric = FxHashMap::<ExprId, bool>::default();
    let mut pending = vec![(root, false)];
    while let Some((id, finish)) = pending.pop() {
        if numeric.contains_key(&id) {
            continue;
        }
        if !finish {
            pending.push((id, true));
            match arena.get(id) {
                ExprNode::Add(children) | ExprNode::Mul(children) => {
                    pending.extend(children.iter().rev().map(|&child| (child, false)));
                }
                ExprNode::Unary(UnaryOp::Neg, child) => pending.push((*child, false)),
                ExprNode::Pow(base, exponent) => {
                    pending.extend([(*exponent, false), (*base, false)]);
                }
                _ => {}
            }
            continue;
        }
        let value = match arena.get(id) {
            ExprNode::Add(children) | ExprNode::Mul(children) => {
                children.iter().all(|child| numeric[child])
            }
            ExprNode::Unary(UnaryOp::Neg, child) => numeric[child],
            ExprNode::Pow(base, exponent) => numeric[base] && numeric[exponent],
            ExprNode::Param(_) => false,
            _ => true,
        };
        numeric.insert(id, value);
    }

    // Only rewrite the symbolic spine.
    // Maximal numeric regions are extracted directly.
    let mut rewritten = FxHashMap::<ExprId, ExprId>::default();
    pending.push((root, false));
    while let Some((id, finish)) = pending.pop() {
        if rewritten.contains_key(&id) {
            continue;
        }
        if matches!(
            arena.get(id),
            ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
        ) {
            rewritten.insert(id, id);
            continue;
        }
        if numeric[&id] {
            let terms = as_linear(arena, id, false).map(LinearTerms::into_owned);
            let replacement = terms.map_or(id, |terms| push_linear(arena, terms));
            rewritten.insert(id, replacement);
            continue;
        }
        if !finish {
            pending.push((id, true));
            match arena.get(id) {
                ExprNode::Add(children) | ExprNode::Mul(children) => {
                    pending.extend(children.iter().rev().map(|&child| (child, false)));
                }
                ExprNode::Unary(UnaryOp::Neg, child) => pending.push((*child, false)),
                ExprNode::Pow(base, exponent) => {
                    pending.extend([(*exponent, false), (*base, false)]);
                }
                _ => unreachable!("only affine arithmetic reaches this point"),
            }
            continue;
        }
        let mut node = arena.get(id).clone();
        let mut changed = false;
        let mut rewrite_child = |child: &mut ExprId| {
            let replacement = rewritten[child];
            changed |= replacement != *child;
            *child = replacement;
        };
        match &mut node {
            ExprNode::Add(children) | ExprNode::Mul(children) => {
                for child in children {
                    rewrite_child(child);
                }
            }
            ExprNode::Unary(UnaryOp::Neg, child) => rewrite_child(child),
            ExprNode::Pow(base, exponent) => {
                rewrite_child(base);
                rewrite_child(exponent);
            }
            _ => unreachable!("only affine arithmetic reaches this point"),
        }
        let replacement = if changed { arena.push(node) } else { id };
        rewritten.insert(id, replacement);
    }
    rewritten[&root]
}

pub(crate) fn pow_into(
    arena: &mut (impl ArenaAccess + ?Sized),
    base: ExprId,
    exponent: ExprId,
) -> ExprId {
    let base = if matches!(arena.get(exponent), ExprNode::Const(e) if e.is_finite() && *e >= 2.0 && (*e - e.round()).abs() < f64::EPSILON)
    {
        compact_numeric_into(arena, base)
    } else {
        base
    };
    arena.push(ExprNode::Pow(base, exponent))
}

/// Build `num / den`. If `den` is a nonzero constant `c`, fold to `num * (1/c)`
/// so a constant-denominator division stays on the linear fast-path. Otherwise
/// produce a `Div` node (always nonlinear, even when the numerator is linear).
pub(crate) fn div_into(
    arena: &mut (impl ArenaAccess + ?Sized),
    num: ExprId,
    den: ExprId,
) -> ExprId {
    if let ExprNode::Const(c) = *arena.get(den)
        && c != 0.0
    {
        if small_affine(arena, &[num])
            && let Some(t) = as_linear(arena, num, false)
        {
            let inv = 1.0 / c;
            let constant = t.constant * inv;
            let mut coeffs = t.coeffs.into_owned();
            for (_, co) in &mut coeffs {
                *co *= inv;
            }
            return push_linear(arena, LinearTerms { coeffs: Cow::Owned(coeffs), constant });
        }
        let inv = arena.push(ExprNode::Const(1.0 / c));
        return mul_into(arena, num, inv);
    }
    arena.push(ExprNode::Div(num, den))
}

/// Build `-rhs`, eagerly merging small numeric affine operands.
pub(crate) fn neg_into(arena: &mut (impl ArenaAccess + ?Sized), rhs: ExprId) -> ExprId {
    if small_affine(arena, &[rhs])
        && let Some(t) = as_linear(arena, rhs, false)
    {
        let constant = -t.constant;
        let mut coeffs = t.coeffs.into_owned();
        for (_, c) in &mut coeffs {
            *c = -*c;
        }
        return push_linear(arena, LinearTerms { coeffs: Cow::Owned(coeffs), constant });
    }
    arena.push(ExprNode::Unary(UnaryOp::Neg, rhs))
}

/// Extract the linear terms of `id`, if any. Used by solver backends to extract
/// LP coefficients without walking the tree themselves.
///
/// The result borrows coefficients for a direct [`ExprNode::Linear`] and owns
/// coefficients synthesized from a larger expression tree. Its lifetime is
/// tied to `arena` in either case; call [`LinearTerms::into_owned`] to retain a
/// snapshot independently of the arena.
///
/// Parameters are folded to their current arena values, so the returned terms
/// reflect the latest [`ExprArena::set_param_value`] binding.
///
/// [`ExprArena::set_param_value`]: crate::ExprArena::set_param_value
pub fn extract_linear<'a>(arena: &'a ExprArena, id: ExprId) -> Option<LinearTerms<'a>> {
    as_linear(arena, id, true)
}

/// A nonlinear residual summand: the existing arena node `id`, taken with a
/// leading negation when `neg` is set. Carrying the sign as a flag lets
/// [`split_linear`] run without a mutable arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SignedExpr {
    pub id: ExprId,
    pub neg: bool,
}

// Classify additive regions once, rather than retrying extraction at every
// prefix of a mixed sum.
fn additive_affinity(arena: &ExprArena, root: ExprId) -> FxHashMap<ExprId, bool> {
    let mut affine = FxHashMap::default();
    let mut pending = vec![(root, false)];
    while let Some((id, finish)) = pending.pop() {
        if affine.contains_key(&id) {
            continue;
        }
        let value = match arena.get(id) {
            ExprNode::Add(children) if finish => children.iter().all(|child| affine[child]),
            ExprNode::Unary(UnaryOp::Neg, child) if finish => affine[child],
            ExprNode::Add(children) => {
                pending.push((id, true));
                pending.extend(children.iter().rev().map(|&child| (child, false)));
                continue;
            }
            ExprNode::Unary(UnaryOp::Neg, child) => {
                pending.push((id, true));
                pending.push((*child, false));
                continue;
            }
            ExprNode::Const(_)
            | ExprNode::Var(_)
            | ExprNode::Param(_)
            | ExprNode::Linear { .. } => true,
            _ => as_linear(arena, id, true).is_some(),
        };
        affine.insert(id, value);
    }
    affine
}

/// Split an expression into its linear part and a nonlinear residual. The
/// returned `(LinearTerms, Vec<SignedExpr>)` satisfies
///
/// ```text
/// value(id) == sum_i coef_i * var_i + constant + sum_j (-1)^neg_j value(id_j)
/// ```
///
/// The residual is empty when the whole expression is linear and otherwise
/// lists the remaining nonlinear summands (each a pre-existing arena node,
/// optionally negated). `LinearTerms` may have empty `coeffs` and
/// `constant == 0.0` when the whole expression is purely nonlinear.
///
/// As with [`extract_linear`], call [`LinearTerms::into_owned`] before retaining
/// the linear terms independently of `arena`.
///
/// # Panics
/// Panics if `id` or a reachable child is not a valid ID in `arena`.
pub fn split_linear<'a>(arena: &'a ExprArena, id: ExprId) -> (LinearTerms<'a>, Vec<SignedExpr>) {
    if let Some(lt) = as_linear(arena, id, true) {
        return (lt, Vec::new());
    }
    // Non-additive nonlinear roots are already a single residual.
    if !matches!(arena.get(id), ExprNode::Add(_) | ExprNode::Unary(UnaryOp::Neg, _)) {
        return (LinearTerms::borrowed(&[], 0.0), vec![SignedExpr { id, neg: false }]);
    }
    let mut lin = CoeffAccum::with_capacity(0);
    let mut constant = 0.0;
    let mut residual: Vec<SignedExpr> = Vec::new();
    let affine = additive_affinity(arena, id);
    let mut extracted = FxHashMap::default();
    let mut sign_stack: smallvec::SmallVec<[(ExprId, f64); 8]> = smallvec![(id, 1.0)];
    while let Some((cur, sign)) = sign_stack.pop() {
        if affine[&cur] {
            // Consume a maximal affine subtree as one term.
            // Extraction's fold memoizes its shared descendants.
            let t = extracted.entry(cur).or_insert_with(|| {
                as_linear(arena, cur, true).expect("classified additive affine region")
            });
            for &(v, c) in t.coeffs.iter() {
                lin.add(v, c * sign);
            }
            constant += t.constant * sign;
            continue;
        }
        match arena.get(cur) {
            ExprNode::Add(children) => {
                for c in children.iter().copied() {
                    sign_stack.push((c, sign));
                }
            }
            ExprNode::Unary(UnaryOp::Neg, inner) => sign_stack.push((*inner, -sign)),
            _ => residual.push(SignedExpr { id: cur, neg: sign < 0.0 }),
        }
    }
    let mut coeffs = lin.into_coeffs();
    coeffs.retain(|(_, c)| *c != 0.0);
    (LinearTerms { coeffs: Cow::Owned(coeffs), constant }, residual)
}

/// Render the first nonlinear summand of `id` as a short infix string, resolving
/// each [`VarId`] to a display name via `resolve`. Returns `None` when `id` is
/// fully affine (no nonlinear residual).
pub fn describe_nonlinear_term(
    arena: &ExprArena,
    id: ExprId,
    resolve: &impl Fn(VarId) -> String,
) -> Option<String> {
    use crate::render::{PREC_ADD, PREC_UNARY, render_node};
    let (_, residual) = split_linear(arena, id);
    residual.first().map(|s| {
        if s.neg {
            format!("-{}", render_node(arena, s.id, resolve, PREC_UNARY))
        } else {
            render_node(arena, s.id, resolve, PREC_ADD)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growing_nonlinear_products_and_powers_use_bounded_work_per_step() {
        use crate::arena::ParamId;
        struct CountedArena {
            arena: ExprArena,
            reads: std::cell::Cell<usize>,
        }
        impl ArenaAccess for CountedArena {
            fn get(&self, id: ExprId) -> &ExprNode {
                self.reads.set(self.reads.get() + 1);
                self.arena.get(id)
            }
            fn param_value(&self, id: ParamId) -> f64 {
                self.arena.param_value(id)
            }
            fn push(&mut self, node: ExprNode) -> ExprId {
                self.arena.push(node)
            }
            fn cached_degree(&self, id: ExprId) -> Option<crate::classify::Degree> {
                self.arena.cached_degree(id)
            }
            fn cache_degree(&self, id: ExprId, degree: crate::classify::Degree) {
                self.arena.cache_degree(id, degree);
            }
        }
        for powers in [false, true] {
            let mut arena =
                CountedArena { arena: ExprArena::new(), reads: std::cell::Cell::new(0) };
            let x = arena.var(VarId(0));
            let exponent = arena.constant(2.0);
            let mut root = x;
            for _ in 0..4_000 {
                root = if powers {
                    pow_into(&mut arena, root, exponent)
                } else {
                    mul_into(&mut arena, root, x)
                };
            }
            assert!(
                arena.reads.get() < 400_000,
                "quadratic traversal: {} reads, powers={powers}",
                arena.reads.get()
            );
            assert_eq!(crate::classify::classify_access(&arena, root), crate::ExprClass::Nonlinear);
        }
    }

    #[test]
    fn small_eager_merges_preserve_ieee_grouping_with_raw_duplicate_coefficients() {
        let mut arena = ExprArena::new();
        let v = VarId(0);
        let mut ids = vec![arena.var(v)];
        for value in [-0.0, 0.0, 1.0, 1e16, -1e16, f64::INFINITY, f64::NAN] {
            ids.push(arena.constant(value));
            ids.push(arena.linear(vec![(v, value)], value));
        }
        ids.push(arena.linear(vec![(v, 1.0), (v, 1e16), (v, -1e16)], -0.0));
        ids.push(arena.push(ExprNode::Add(ids[..4].iter().copied().collect())));
        let bits = |terms: LinearTerms<'_>| {
            (
                terms.coeffs.iter().map(|&(v, c)| (v, c.to_bits())).collect::<Vec<_>>(),
                terms.constant.to_bits(),
            )
        };
        for &lhs in &ids {
            for &rhs in &ids {
                let expected = {
                    let lt = recursive_linear(&arena, lhs, false).unwrap();
                    let rt = recursive_linear(&arena, rhs, false).unwrap();
                    let mut acc = CoeffAccum::with_capacity(lt.coeffs.len() + rt.coeffs.len());
                    acc.extend_from_slice(&lt.coeffs);
                    acc.extend_from_slice(&rt.coeffs);
                    let mut coeffs = acc.into_coeffs();
                    coeffs.retain(|(_, c)| *c != 0.0);
                    bits(LinearTerms::owned(coeffs, lt.constant + rt.constant))
                };
                let result = add_into(&mut arena, lhs, rhs);
                assert_eq!(
                    bits(as_linear(&arena, result, false).unwrap()),
                    expected,
                    "{lhs:?} + {rhs:?}"
                );
            }
        }
    }

    #[test]
    fn flat_weighted_sum_preserves_ieee_arithmetic_and_parameter_rebinding() {
        let mut arena = ExprArena::new();
        let var = arena.var(VarId(0));
        let param = arena.new_param(0.0);
        let p = arena.param(param);
        for scalar in [-0.0, 0.0, 1.0, 1e308, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            arena.set_param_value(param, scalar);
            let constant = arena.constant(scalar);
            for factor in [constant, p] {
                let left = arena.push(ExprNode::Mul(smallvec![factor, var]));
                let right = arena.push(ExprNode::Mul(smallvec![var, factor]));
                let sum = arena.push(ExprNode::Add(smallvec![left, right, var]));
                for resolve in [false, true] {
                    let bits = |t: LinearTerms<'_>| {
                        (
                            t.constant.to_bits(),
                            t.coeffs.iter().map(|(v, c)| (*v, c.to_bits())).collect::<Vec<_>>(),
                        )
                    };
                    assert_eq!(
                        as_linear(&arena, sum, resolve).map(bits),
                        recursive_linear(&arena, sum, resolve).map(bits)
                    );
                }
            }
        }
        let term = arena.push(ExprNode::Mul(smallvec![p, var]));
        let sum = arena.push(ExprNode::Add(smallvec![term, term]));
        for value in [2.0, 3.0] {
            arena.set_param_value(param, value);
            assert_eq!(extract_linear(&arena, sum).unwrap().coeffs, vec![(VarId(0), 2.0 * value)]);
        }
    }

    #[test]
    fn iterative_linear_matches_recursive_arithmetic_and_order() {
        let (arena, ids) = crate::fold::test_arena();
        for id in ids {
            for resolve in [false, true] {
                let expected = recursive_linear(&arena, id, resolve);
                let actual = as_linear(&arena, id, resolve);
                let bits = |t: LinearTerms<'_>| {
                    (
                        t.constant.to_bits(),
                        t.coeffs.iter().map(|(v, c)| (*v, c.to_bits())).collect::<Vec<_>>(),
                    )
                };
                assert_eq!(actual.map(bits), expected.map(bits), "node {id:?}");
            }
        }
    }
    use crate::arena::{ExprArena, ExprNode, VarId};

    #[test]
    fn direct_linear_extraction_borrows_arena_coefficients() {
        let mut arena = ExprArena::new();
        let id = arena.push(ExprNode::Linear {
            coeffs: vec![(VarId(0), 2.0), (VarId(1), -3.0)],
            constant: 4.0,
        });
        let expected = match arena.get(id) {
            ExprNode::Linear { coeffs, .. } => coeffs.as_slice(),
            _ => unreachable!(),
        };

        let terms = extract_linear(&arena, id).expect("linear");
        let Cow::Borrowed(actual) = terms.coeffs else {
            panic!("direct Linear extraction allocated coefficient storage");
        };
        assert!(std::ptr::eq(actual, expected));
        assert!((terms.constant - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn merged_tree_extraction_owns_coefficients() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![x, y, x]));

        let terms = extract_linear(&arena, sum).expect("linear");
        let Cow::Owned(coeffs) = terms.coeffs else {
            panic!("merged expression did not own synthesized coefficients");
        };
        assert_eq!(coeffs, vec![(VarId(0), 2.0), (VarId(1), 1.0)]);
    }

    #[test]
    fn into_owned_detaches_borrowed_terms_from_arena() {
        let mut arena = ExprArena::new();
        let id = arena.push(ExprNode::Linear { coeffs: vec![(VarId(0), 2.0)], constant: 1.0 });

        let terms: LinearTerms<'static> = extract_linear(&arena, id).unwrap().into_owned();
        arena.push(ExprNode::Const(5.0));

        assert!(matches!(terms.coeffs, Cow::Owned(_)));
        assert_eq!(terms.coeffs.as_ref(), &[(VarId(0), 2.0)]);
        assert!((terms.constant - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn split_linear_preserves_borrowing_only_for_direct_linear_nodes() {
        let mut arena = ExprArena::new();
        let linear = arena.push(ExprNode::Linear { coeffs: vec![(VarId(0), 2.0)], constant: 1.0 });
        let (direct, residual) = split_linear(&arena, linear);
        assert!(matches!(direct.coeffs, Cow::Borrowed(_)));
        assert_eq!(residual, Vec::<SignedExpr>::new());

        let y = arena.push(ExprNode::Var(VarId(1)));
        let nonlinear = arena.push(ExprNode::Unary(UnaryOp::Sin, y));
        let mixed = arena.push(ExprNode::Add(smallvec::smallvec![linear, nonlinear]));
        let (split, residual) = split_linear(&arena, mixed);
        assert!(matches!(split.coeffs, Cow::Owned(_)));
        assert_eq!(split.coeffs.as_ref(), &[(VarId(0), 2.0)]);
        assert!((split.constant - 1.0).abs() < f64::EPSILON);
        assert_eq!(residual, vec![SignedExpr { id: nonlinear, neg: false }]);
    }

    #[test]
    fn param_times_var_stays_symbolic_until_extracted() {
        // Build `price * x` through the operator helper. The parameter must NOT
        // be folded into a Linear node at build time (so it stays re-bindable)
        let mut arena = ExprArena::new();
        let pid = arena.new_param(3.0);
        let price = arena.param(pid);
        let xnode = arena.push(ExprNode::Var(VarId(0)));
        let prod = mul_into(&mut arena, price, xnode);
        assert!(matches!(arena.get(prod), ExprNode::Mul(_)));

        let terms = extract_linear(&arena, prod).expect("linear");
        assert_eq!(&*terms.coeffs, &[(VarId(0), 3.0)][..]);
        assert!(terms.constant.abs() < f64::EPSILON);
    }

    #[test]
    fn rebinding_param_updates_extracted_coeff() {
        let mut arena = ExprArena::new();
        let pid = arena.new_param(3.0);
        let price = arena.param(pid);
        let xnode = arena.push(ExprNode::Var(VarId(0)));
        let prod = mul_into(&mut arena, price, xnode);

        arena.set_param_value(pid, 10.0);
        let terms = extract_linear(&arena, prod).expect("linear");
        assert_eq!(&*terms.coeffs, &[(VarId(0), 10.0)][..]);
    }

    #[test]
    fn param_plus_var_resolves_constant() {
        let mut arena = ExprArena::new();
        let pid = arena.new_param(5.0);
        let price = arena.param(pid);
        let xnode = arena.push(ExprNode::Var(VarId(0)));
        let sum = add_into(&mut arena, price, xnode);
        let terms = extract_linear(&arena, sum).expect("linear");
        assert_eq!(&*terms.coeffs, &[(VarId(0), 1.0)][..]);
        assert!((terms.constant - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn add_extraction_is_first_seen_ordered_and_merges() {
        // `z + x + y + x`: coefficients come out in first-seen order [z, x, y]
        // and the repeated `x` is merged to coeff 2.
        let mut arena = ExprArena::new();
        let z = arena.push(ExprNode::Var(VarId(2)));
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![z, x, y, x]));

        let terms = extract_linear(&arena, sum).expect("linear");
        assert_eq!(&*terms.coeffs, &[(VarId(2), 1.0), (VarId(0), 2.0), (VarId(1), 1.0)][..]);
        assert!(terms.constant.abs() < f64::EPSILON);
        assert_eq!(&*extract_linear(&arena, sum).unwrap().coeffs, &*terms.coeffs);
    }

    #[test]
    fn wide_sum_merges_repeated_vars_in_order() {
        let mut arena = ExprArena::new();
        let n = 50u32;
        let mut ids = Vec::new();
        for _ in 0..3 {
            for v in 0..n {
                ids.push(arena.push(ExprNode::Var(VarId(v))));
            }
        }
        let sum = arena.push(ExprNode::Add(ids.into_iter().collect()));
        let terms = extract_linear(&arena, sum).expect("linear");
        let expected: Vec<(VarId, f64)> = (0..n).map(|v| (VarId(v), 3.0)).collect();
        assert_eq!(&*terms.coeffs, expected.as_slice());
    }

    fn names(v: VarId) -> String {
        match v.0 {
            0 => "x".to_string(),
            1 => "y".to_string(),
            n => format!("v{n}"),
        }
    }

    #[test]
    fn describe_renders_the_first_nonlinear_summand() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));

        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![x, y]));
        assert_eq!(describe_nonlinear_term(&arena, prod, &names).as_deref(), Some("x * y"));

        let two = arena.constant(2.0);
        let pow = arena.push(ExprNode::Pow(x, two));
        assert_eq!(describe_nonlinear_term(&arena, pow, &names).as_deref(), Some("x^2"));

        let s = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
        assert_eq!(describe_nonlinear_term(&arena, s, &names).as_deref(), Some("sin(x)"));

        let div = arena.push(ExprNode::Div(x, y));
        assert_eq!(describe_nonlinear_term(&arena, div, &names).as_deref(), Some("x / y"));
    }

    #[test]
    fn describe_isolates_the_nonlinear_part_of_a_mixed_expression() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let y = arena.push(ExprNode::Var(VarId(1)));
        let z = arena.push(ExprNode::Var(VarId(2)));
        let two = arena.constant(2.0);
        let two_z = arena.push(ExprNode::Mul(smallvec::smallvec![two, z]));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![x, y]));
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![two_z, prod]));
        assert_eq!(describe_nonlinear_term(&arena, sum, &names).as_deref(), Some("x * y"));
    }

    #[test]
    fn describe_returns_none_for_affine() {
        let mut arena = ExprArena::new();
        let x = arena.push(ExprNode::Var(VarId(0)));
        let three = arena.constant(3.0);
        let sum = arena.push(ExprNode::Add(smallvec::smallvec![x, three]));
        assert_eq!(describe_nonlinear_term(&arena, sum, &names), None);
    }

    #[test]
    fn describe_falls_back_to_index_for_unknown_var() {
        let mut arena = ExprArena::new();
        let a = arena.push(ExprNode::Var(VarId(7)));
        let b = arena.push(ExprNode::Var(VarId(8)));
        let prod = arena.push(ExprNode::Mul(smallvec::smallvec![a, b]));
        assert_eq!(describe_nonlinear_term(&arena, prod, &names).as_deref(), Some("v7 * v8"));
    }
}
