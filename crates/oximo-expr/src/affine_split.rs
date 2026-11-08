//! Affine extraction from additive polynomial and nonlinear expressions.

use std::cmp::Ordering;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::smallvec;

use crate::compensated_sum::CompensatedSum;
use crate::{ExprArena, ExprId, ExprNode, LinearTerms, SignedExpr, UnaryOp, VarId, extract_linear};

/// A signed summand remaining after [`split_affine`] extracts affine terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AffineResidual {
    /// The complete source expression.
    Full(SignedExpr),
    /// Only the source [`ExprNode::Quadratic`]'s monomials.
    QuadraticPart(SignedExpr),
}

impl AffineResidual {
    /// The original signed source node.
    ///
    /// For [`Self::QuadraticPart`], evaluating this node also includes
    /// the affine terms already extracted by the split.
    pub fn source(self) -> SignedExpr {
        match self {
            Self::Full(source) | Self::QuadraticPart(source) => source,
        }
    }
}

/// Split an expression into affine terms and signed nonlinear summands.
///
/// Returned residual IDs are published to protect them from in-place updates.
/// Constants and repeated coefficients across additive summands use Neumaier
/// compensated summation. Parameters are resolved at their current arena values.
/// Nonfinite results use ordinary IEEE arithmetic.
///
/// # Panics
/// Panics if `root` or a reachable child is not a valid ID in `arena`.
pub fn split_affine(arena: &ExprArena, root: ExprId) -> (LinearTerms<'_>, Vec<AffineResidual>) {
    if let ExprNode::Quadratic(terms) = arena.get(root) {
        let residual = if terms.quadratic().is_empty() {
            Vec::new()
        } else {
            arena.seal(root);
            vec![AffineResidual::QuadraticPart(SignedExpr { id: root, neg: false })]
        };
        return (LinearTerms::borrowed(terms.linear(), terms.constant()), residual);
    }

    if !matches!(arena.get(root), ExprNode::Add(_) | ExprNode::Unary(UnaryOp::Neg, _)) {
        return extract_linear(arena, root).map_or_else(
            || {
                (
                    LinearTerms::borrowed(&[], 0.0),
                    vec![AffineResidual::Full(SignedExpr { id: root, neg: false })],
                )
            },
            |terms| (terms, Vec::new()),
        );
    }

    if let ExprNode::Add(children) = arena.node(root)
        && let Some(linear) = variable_sum(arena, children)
    {
        return (linear, Vec::new());
    }
    if let Some(split) = stream_additive(arena, root) {
        return split;
    }

    let regions = AdditiveRegions::new(arena, root);
    let linear = regions.accumulate(arena, root);
    let mut residual = Vec::new();
    let mut pending: smallvec::SmallVec<[(ExprId, bool); 8]> = smallvec![(root, false)];
    while let Some((id, neg)) = pending.pop() {
        if regions.affine[&id] {
            continue;
        }

        match arena.get(id) {
            ExprNode::Add(children) => {
                pending.extend(children.iter().map(|&child| (child, neg)));
            }
            ExprNode::Unary(UnaryOp::Neg, child) => pending.push((*child, !neg)),
            ExprNode::Quadratic(_) => {
                arena.seal(id);
                residual.push(AffineResidual::QuadraticPart(SignedExpr { id, neg }));
            }
            _ => residual.push(AffineResidual::Full(SignedExpr { id, neg })),
        }
    }

    (linear, residual)
}

fn variable_sum(arena: &ExprArena, children: &[ExprId]) -> Option<LinearTerms<'static>> {
    if !matches!(arena.node(*children.first()?), ExprNode::Var(_)) {
        return None;
    }

    let mut coefficients = Coefficients::with_capacity(children.len());
    for &child in children {
        let ExprNode::Var(variable) = arena.node(child) else { return None };
        coefficients.add(*variable, 1.0);
    }

    Some(LinearTerms::owned(coefficients.finish(), 0.0))
}

/// Stream trees directly.
fn stream_additive(
    arena: &ExprArena,
    root: ExprId,
) -> Option<(LinearTerms<'static>, Vec<AffineResidual>)> {
    let capacity = match arena.get(root) {
        ExprNode::Add(children) => children.len(),
        _ => 0,
    };

    let mut coefficients = Coefficients::with_capacity(capacity);
    let mut constant = CompensatedSum::default();
    let mut residual = Vec::new();
    let mut extracted = FxHashMap::default();
    let mut seen: Option<FxHashSet<ExprId>> = None;
    let roots = [root];
    let mut pending: smallvec::SmallVec<[(&[ExprId], bool); 8]> = smallvec![(&roots[..], false)];
    while let Some((children, neg)) = pending.last_mut() {
        let Some((&id, rest)) = children.split_last() else {
            pending.pop();
            continue;
        };

        let neg = *neg;
        *children = rest;
        if rest.is_empty() {
            pending.pop();
        }

        let sign = if neg { -1.0 } else { 1.0 };
        let terms = match arena.get(id) {
            ExprNode::Add(children) => {
                if seen.is_none() {
                    let mut composite = children.iter().copied().filter(|&child| {
                        matches!(
                            arena.node(child),
                            ExprNode::Add(_) | ExprNode::Unary(UnaryOp::Neg, _)
                        )
                    });
                    if let (Some(first), Some(second)) = (composite.next(), composite.next()) {
                        if first == second {
                            return None;
                        }
                        seen = Some(FxHashSet::default());
                    }
                }

                if let Some(seen) = &mut seen
                    && !seen.insert(id)
                {
                    return None;
                }

                pending.push((children, neg));
                continue;
            }
            ExprNode::Unary(UnaryOp::Neg, child) => {
                if let Some(seen) = &mut seen
                    && !seen.insert(id)
                {
                    return None;
                }
                pending.push((std::slice::from_ref(child), !neg));
                continue;
            }
            ExprNode::Var(variable) => {
                coefficients.add(*variable, sign);
                continue;
            }
            ExprNode::Const(value) => LinearTerms::borrowed(&[], *value),
            ExprNode::Param(parameter) => LinearTerms::borrowed(&[], arena.param_value(*parameter)),
            ExprNode::Linear { coeffs, constant } => LinearTerms::borrowed(coeffs, *constant),
            ExprNode::Quadratic(terms) => {
                if !terms.quadratic().is_empty() {
                    arena.seal(id);
                    residual.push(AffineResidual::QuadraticPart(SignedExpr { id, neg }));
                }
                LinearTerms::borrowed(terms.linear(), terms.constant())
            }
            _ => {
                let Some(terms) = extracted.entry(id).or_insert_with(|| extract_linear(arena, id))
                else {
                    residual.push(AffineResidual::Full(SignedExpr { id, neg }));
                    continue;
                };
                coefficients.extend(&terms.coeffs, sign);
                constant.add(sign * terms.constant);
                continue;
            }
        };

        coefficients.extend(&terms.coeffs, sign);
        constant.add(sign * terms.constant);
    }

    Some((LinearTerms::owned(coefficients.finish(), constant.value()), residual))
}

struct Coefficients {
    entries: Vec<(VarId, CompensatedSum)>,
    order: Option<Ordering>,
    slots: FxHashMap<VarId, usize>,
    dense: Option<DenseCoefficients>,
}

struct DenseCoefficients {
    base: usize,
    values: Vec<f64>,
    corrections: Vec<f64>,
    capacity: usize,
}

impl DenseCoefficients {
    #[inline]
    fn add(&mut self, index: usize, value: f64) {
        let sum = &mut self.values[index];
        if *sum == 0.0 {
            *sum = value;
            if index < self.corrections.len() && !value.is_finite() {
                self.corrections[index] = 0.0;
            }
            return;
        }

        let mut correction = 0.0;
        CompensatedSum::add_parts(sum, &mut correction, value);
        if correction != 0.0 {
            self.corrections.resize(self.values.len(), 0.0);
            self.corrections[index] += correction;
        } else if index < self.corrections.len() && !sum.is_finite() {
            self.corrections[index] = 0.0;
        }
    }

    fn into_coefficients(mut self) -> Vec<(VarId, f64)> {
        for (sum, correction) in self.values.iter_mut().zip(self.corrections) {
            *sum += correction;
        }

        let mut coefficients = Vec::with_capacity(self.capacity);
        for (index, coefficient) in self.values.into_iter().enumerate() {
            if coefficient != 0.0 {
                let variable =
                    VarId(u32::try_from(self.base + index).expect("observed variable ID"));
                coefficients.push((variable, coefficient));
            }
        }

        coefficients
    }

    fn into_entries(self) -> Vec<(VarId, CompensatedSum)> {
        let mut entries = Vec::with_capacity(self.capacity);
        for (index, sum) in self.values.into_iter().enumerate() {
            let correction = self.corrections.get(index).copied().unwrap_or(0.0);
            if sum != 0.0 || correction != 0.0 {
                let variable =
                    VarId(u32::try_from(self.base + index).expect("observed variable ID"));
                entries.push((variable, CompensatedSum::from_parts(sum, correction)));
            }
        }

        entries
    }
}

impl Coefficients {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            order: Some(Ordering::Equal),
            slots: FxHashMap::default(),
            dense: None,
        }
    }

    #[inline]
    fn add(&mut self, variable: VarId, value: f64) {
        if let Some(order) = &mut self.order {
            if let Some((last, coefficient)) = self.entries.last_mut() {
                let direction = last.index().cmp(&variable.index());
                if direction == Ordering::Equal {
                    coefficient.add(value);
                    return;
                }
                if *order == Ordering::Equal {
                    *order = direction;
                }
                if *order != direction {
                    self.index(variable);
                    self.order = None;
                }
            }

            if self.order.is_some() {
                self.entries.push((variable, CompensatedSum::new(value)));
                return;
            }
        }

        if let Some(dense) = &mut self.dense
            && let Some(index) =
                variable.index().checked_sub(dense.base).filter(|&i| i < dense.values.len())
        {
            dense.add(index, value);
            return;
        }

        if let Some(dense) = self.dense.take() {
            self.entries = dense.into_entries();
            self.index_sparse();
        }

        match self.slots.entry(variable) {
            std::collections::hash_map::Entry::Occupied(slot) => {
                self.entries[*slot.get()].1.add(value);
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(self.entries.len());
                self.entries.push((variable, CompensatedSum::new(value)));
            }
        }
    }

    fn index(&mut self, variable: VarId) {
        let (minimum, maximum) = self
            .entries
            .iter()
            .fold((variable.index(), variable.index()), |(minimum, maximum), (var, _)| {
                (minimum.min(var.index()), maximum.max(var.index()))
            });

        // Align a bounded dense range around clustered IDs.
        let limit = self.entries.capacity().saturating_mul(4).clamp(128, 131_072);
        let mut size = self.entries.capacity().clamp(16, 131_072).next_power_of_two();
        while size <= limit {
            let base = minimum & !(size - 1);

            if maximum - base < size {
                let mut dense = DenseCoefficients {
                    base,
                    values: vec![0.0; size],
                    corrections: Vec::new(),
                    capacity: self.entries.capacity().min(size),
                };
                for (variable, sum) in std::mem::take(&mut self.entries) {
                    let index = variable.index() - base;
                    let (sum, correction) = sum.parts();
                    dense.values[index] = sum;
                    if correction != 0.0 {
                        dense.corrections.resize(size, 0.0);
                        dense.corrections[index] = correction;
                    }
                }
                self.dense = Some(dense);
                return;
            }

            size *= 2;
        }

        self.index_sparse();
    }

    fn index_sparse(&mut self) {
        self.slots.reserve(self.entries.capacity());
        self.slots.extend(self.entries.iter().enumerate().map(|(i, (var, _))| (*var, i)));
    }

    fn extend(&mut self, terms: &[(VarId, f64)], sign: f64) {
        if self.order == Some(Ordering::Less) {
            for &(variable, coefficient) in terms {
                self.add(variable, sign * coefficient);
            }
        } else {
            for &(variable, coefficient) in terms.iter().rev() {
                self.add(variable, sign * coefficient);
            }
        }
    }

    fn finish(self) -> Vec<(VarId, f64)> {
        if let Some(dense) = self.dense {
            return dense.into_coefficients();
        }
        let mut coefficients: Vec<_> = self
            .entries
            .into_iter()
            .map(|(variable, coefficient)| (variable, coefficient.value()))
            .filter(|&(_, coefficient)| coefficient != 0.0)
            .collect();
        match self.order {
            Some(Ordering::Greater) => coefficients.reverse(),
            None => coefficients.sort_unstable_by_key(|&(variable, _)| variable.index()),
            _ => {}
        }
        coefficients
    }
}

/// Additive DAG nodes in child-before-parent order.
struct AdditiveRegions<'a> {
    order: Vec<ExprId>,
    affine: FxHashMap<ExprId, bool>,
    extracted: FxHashMap<ExprId, Option<LinearTerms<'a>>>,
}

impl<'a> AdditiveRegions<'a> {
    fn new(arena: &'a ExprArena, root: ExprId) -> Self {
        let mut regions = Self {
            order: Vec::new(),
            affine: FxHashMap::default(),
            extracted: FxHashMap::default(),
        };
        let mut pending = vec![(root, false)];
        while let Some((id, finish)) = pending.pop() {
            if regions.affine.contains_key(&id) {
                continue;
            }

            let affine = match arena.get(id) {
                ExprNode::Add(children) if finish => {
                    children.iter().all(|child| regions.affine[child])
                }
                ExprNode::Unary(UnaryOp::Neg, child) if finish => regions.affine[child],
                ExprNode::Add(children) => {
                    pending.push((id, true));
                    pending.extend(children.iter().map(|&child| (child, false)));
                    continue;
                }
                ExprNode::Unary(UnaryOp::Neg, child) => {
                    pending.extend([(id, true), (*child, false)]);
                    continue;
                }
                ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Var(_)
                | ExprNode::Linear { .. } => true,
                ExprNode::Quadratic(terms) => terms.quadratic().is_empty(),
                _ => {
                    let terms = extract_linear(arena, id);
                    let affine = terms.is_some();
                    regions.extracted.insert(id, terms);
                    affine
                }
            };
            regions.affine.insert(id, affine);
            regions.order.push(id);
        }

        regions
    }

    fn accumulate(&self, arena: &ExprArena, root: ExprId) -> LinearTerms<'static> {
        // Propagate signed occurrence counts through the DAG.
        let mut weights: FxHashMap<ExprId, CompensatedSum> = FxHashMap::default();
        weights.insert(root, CompensatedSum::new(1.0));

        let mut coefficients = Coefficients::with_capacity(0);
        let mut constant = CompensatedSum::default();
        for &id in self.order.iter().rev() {
            let weight = weights[&id].value();
            let terms = match arena.get(id) {
                ExprNode::Add(children) => {
                    for &child in children {
                        weights.entry(child).or_default().add(weight);
                    }
                    continue;
                }
                ExprNode::Unary(UnaryOp::Neg, child) => {
                    weights.entry(*child).or_default().add(-weight);
                    continue;
                }
                ExprNode::Var(variable) => {
                    coefficients.add(*variable, weight);
                    continue;
                }
                ExprNode::Const(value) => LinearTerms::borrowed(&[], *value),
                ExprNode::Param(parameter) => {
                    LinearTerms::borrowed(&[], arena.param_value(*parameter))
                }
                ExprNode::Linear { coeffs, constant } => LinearTerms::borrowed(coeffs, *constant),
                ExprNode::Quadratic(terms) => {
                    LinearTerms::borrowed(terms.linear(), terms.constant())
                }
                _ => {
                    let Some(terms) = &self.extracted[&id] else { continue };
                    accumulate_terms(&mut coefficients, &mut constant, terms, weight);
                    continue;
                }
            };

            accumulate_terms(&mut coefficients, &mut constant, &terms, weight);
        }

        LinearTerms::owned(coefficients.finish(), constant.value())
    }
}

fn accumulate_terms(
    coefficients: &mut Coefficients,
    constant: &mut CompensatedSum,
    terms: &LinearTerms<'_>,
    weight: f64,
) {
    coefficients.extend(&terms.coeffs, weight);
    constant.add(weight * terms.constant);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{QuadraticTerms, evaluate};

    use super::*;

    #[test]
    fn dense_and_sparse_coefficients_preserve_corrections_when_partial_sums_cancel() {
        for sparse in [false, true] {
            let mut arena = ExprArena::new();
            let mut terms = vec![(10, 1e16), (8, 1.0), (9, 1.0), (10, 1.0)];
            if sparse {
                terms.push((u32::MAX, 1.0));
            }
            terms.extend([(10, -1e16), (10, 2.0)]);
            let children = terms
                .into_iter()
                .rev()
                .map(|(variable, coefficient)| {
                    arena.linear(vec![(VarId(variable), coefficient)], 0.0)
                })
                .collect();

            let root = arena.push(ExprNode::Add(children));
            let (linear, residual) = split_affine(&arena, root);
            let mut expected = vec![(VarId(8), 1.0), (VarId(9), 1.0), (VarId(10), 3.0)];
            if sparse {
                expected.push((VarId(u32::MAX), 1.0));
            }
            assert_eq!(linear.coeffs.as_ref(), expected);
            assert_eq!(residual, []);
        }
    }

    #[test]
    fn unweighted_variable_sums_merge_duplicates_and_preserve_large_ids() {
        let mut arena = ExprArena::new();
        let variables = [u32::MAX, 3, 1, 3, u32::MAX, 2, 1];
        let children = variables.map(|variable| arena.var(VarId(variable))).into_iter().collect();
        let root = arena.push(ExprNode::Add(children));
        let (linear, residual) = split_affine(&arena, root);
        assert_eq!(
            linear.coeffs.as_ref(),
            [(VarId(1), 2.0), (VarId(2), 1.0), (VarId(3), 2.0), (VarId(u32::MAX), 2.0)]
        );
        assert_eq!(linear.constant, 0.0);
        assert_eq!(residual, []);
    }

    #[test]
    fn shared_affine_leaf_extraction_observes_parameter_rebinding() {
        let mut arena = ExprArena::new();
        let variable = VarId(0);
        let affine = arena.linear(vec![(variable, 2.0)], 1.0);
        let parameter_id = arena.new_param(1.5);
        let parameter = arena.param(parameter_id);
        let scaled = arena.push(ExprNode::Mul([parameter, affine].into_iter().collect()));
        let root = arena.push(ExprNode::Add((0..1024).map(|_| scaled).collect()));
        for value in [1.5, 2.5] {
            arena.set_param_value(parameter_id, value).unwrap();
            let (linear, residual) = split_affine(&arena, root);
            assert_eq!(linear.coeffs.as_ref(), [(variable, value * 2048.0)]);
            assert_eq!(linear.constant, value * 1024.0);
            assert_eq!(residual, []);
        }
    }

    #[test]
    fn clustered_high_ids_and_dense_to_sparse_transitions_keep_sorted_coefficients() {
        for variables in [
            [1_000_010, 1_000_008, 1_000_009, 1_000_031, 1_000_008, 1_000_031],
            [10, 8, 9, u32::MAX, 8, u32::MAX],
        ] {
            let mut arena = ExprArena::new();
            let children = variables
                .iter()
                .rev()
                .map(|&var| arena.linear(vec![(VarId(var), 1.0)], 0.0))
                .collect();
            let root = arena.push(ExprNode::Add(children));
            let (linear, residual) = split_affine(&arena, root);
            let expected = [
                (VarId(variables[1]), 2.0),
                (VarId(variables[2]), 1.0),
                (VarId(variables[0]), 1.0),
                (VarId(variables[3]), 2.0),
            ];
            assert_eq!(linear.coeffs.as_ref(), expected);
            assert_eq!(residual, []);
        }
    }

    #[test]
    fn ordered_and_shuffled_coefficients_keep_compensated_duplicate_sums() {
        let mut arena = ExprArena::new();
        let mut terms: Vec<_> = (0..16)
            .flat_map(|variable| {
                [1e16, 1.0, -1e16].map(|coefficient| (VarId(variable), coefficient))
            })
            .collect();
        for order in 0..3 {
            match order {
                1 => terms.reverse(),
                2 => {
                    let count = terms.len();
                    for index in 0..count {
                        terms.swap(index, (index * 19 + 7) % count);
                    }
                }
                _ => {}
            }
            let children = terms
                .iter()
                .map(|&(variable, coefficient)| arena.linear(vec![(variable, coefficient)], 0.0))
                .collect();
            let root = arena.push(ExprNode::Add(children));
            let (linear, residual) = split_affine(&arena, root);
            let expected: Vec<_> = (0..16).map(|variable| (VarId(variable), 1.0)).collect();
            assert_eq!(linear.coeffs.as_ref(), expected);
            assert_eq!(residual, []);
        }
    }

    #[test]
    fn polynomial_affine_terms_preserve_small_values_through_cancellation() {
        let mut arena = ExprArena::new();
        let variable = VarId(0);
        for values in [[1e16, 1.0, -1e16], [-1e16, 1e16, 1.0], [1.0, -1e16, 1e16]] {
            let children = values
                .into_iter()
                .map(|value| {
                    arena.push(ExprNode::Quadratic(Arc::new(
                        QuadraticTerms::from_coefficients(
                            vec![(variable, variable, 1.0)],
                            vec![(variable, value)],
                            value,
                        )
                        .unwrap(),
                    )))
                })
                .collect();

            let root = arena.push(ExprNode::Add(children));
            let negated = arena.push(ExprNode::Unary(UnaryOp::Neg, root));
            for (root, expected) in [(root, 1.0), (negated, -1.0)] {
                let (linear, residual) = split_affine(&arena, root);
                assert_eq!(linear.constant, expected);
                assert_eq!(linear.coeffs.as_ref(), &[(variable, expected)]);
                assert_eq!(residual.len(), 3);
            }
        }
    }

    #[test]
    fn additive_affine_subregions_are_not_rounded_before_polynomial_splitting() {
        let mut arena = ExprArena::new();
        let variable = VarId(0);
        let constants = [1e16, 1.0, -1e16].map(|value| arena.constant(value));
        let affine = arena.push(ExprNode::Add(constants.into_iter().collect()));
        let (linear, residual) = split_affine(&arena, affine);
        assert_eq!(linear.constant, 1.0);
        assert_eq!(residual, []);

        let quadratic = arena.push(ExprNode::Quadratic(Arc::new(
            QuadraticTerms::from_coefficients(vec![(variable, variable, 1.0)], Vec::new(), 2.0)
                .unwrap(),
        )));
        let root = arena.push(ExprNode::Add([affine, quadratic].into_iter().collect()));
        let (linear, residual) = split_affine(&arena, root);
        assert_eq!(linear.constant, 3.0);
        assert_eq!(residual.len(), 1);
    }

    #[test]
    fn shared_affine_dags_do_not_expand_into_every_additive_path() {
        let mut arena = ExprArena::new();
        let variable = VarId(0);
        let mut root = arena.linear(vec![(variable, 2.0)], 1.0);
        for _ in 0..40 {
            root = arena.push(ExprNode::Add([root, root].into_iter().collect()));
        }
        let (linear, residual) = split_affine(&arena, root);
        assert_eq!(linear.constant, 2.0_f64.powi(40));
        assert_eq!(linear.coeffs.as_ref(), &[(variable, 2.0_f64.powi(41))]);
        assert_eq!(residual, []);
    }

    #[test]
    fn direct_polynomials_borrow_coefficients_and_nonfinite_constants_keep_ieee_values() {
        let mut arena = ExprArena::new();
        let variable = VarId(0);
        let quadratic = arena.push(ExprNode::Quadratic(Arc::new(
            QuadraticTerms::from_coefficients(
                vec![(variable, variable, 1.0)],
                vec![(variable, 2.0)],
                3.0,
            )
            .unwrap(),
        )));
        let (linear, _) = split_affine(&arena, quadratic);
        assert!(matches!(linear.coeffs, std::borrow::Cow::Borrowed(_)));

        let infinity = arena.constant(f64::INFINITY);
        let root = arena.push(ExprNode::Add([infinity, quadratic].into_iter().collect()));
        assert_eq!(split_affine(&arena, root).0.constant, f64::INFINITY);

        let negative = arena.push(ExprNode::Unary(UnaryOp::Neg, infinity));
        let invalid = arena.push(ExprNode::Add([infinity, negative].into_iter().collect()));
        assert!(split_affine(&arena, invalid).0.constant.is_nan());
    }

    #[test]
    fn split_affine_preserves_values_for_direct_nested_and_negated_quadratics() {
        let mut arena = ExprArena::new();
        let [x, y, z] = [VarId(0), VarId(1), VarId(2)];
        let quadratic = arena.push(ExprNode::Quadratic(Arc::new(
            QuadraticTerms::from_coefficients(
                vec![(x, y, 2.0), (y, y, 3.0)],
                vec![(x, 5.0), (z, -4.0)],
                7.0,
            )
            .unwrap(),
        )));
        let affine = arena.linear(vec![(z, 11.0)], -2.0);
        let nested = arena.push(ExprNode::Add([affine, quadratic].into_iter().collect()));
        let negated = arena.push(ExprNode::Unary(UnaryOp::Neg, quadratic));
        let nonlinear = arena.push(ExprNode::Unary(UnaryOp::Sin, quadratic));
        let mixed = arena.push(ExprNode::Add([affine, quadratic, nonlinear, negated].into()));
        let negated_mixed = arena.push(ExprNode::Unary(UnaryOp::Neg, mixed));

        for root in [quadratic, nested, negated, nonlinear, mixed, negated_mixed] {
            let (linear, residual) = split_affine(&arena, root);
            for values in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 3.0, -4.0]] {
                let linear_value = linear.constant
                    + linear
                        .coeffs
                        .iter()
                        .map(|&(variable, coefficient)| coefficient * values[variable.index()])
                        .sum::<f64>();
                let residual_value = residual
                    .iter()
                    .map(|term| {
                        let source = term.source();
                        let value = match term {
                            AffineResidual::Full(_) => {
                                evaluate(&arena, source.id, &&values[..]).unwrap()
                            }
                            AffineResidual::QuadraticPart(_) => {
                                let ExprNode::Quadratic(terms) = arena.get(source.id) else {
                                    panic!("quadratic residual")
                                };
                                terms
                                    .quadratic()
                                    .iter()
                                    .map(|&(a, b, coefficient)| {
                                        coefficient * values[a.index()] * values[b.index()]
                                    })
                                    .sum()
                            }
                        };
                        if source.neg { -value } else { value }
                    })
                    .sum::<f64>();
                let expected = evaluate(&arena, root, &&values[..]).unwrap();
                assert!(
                    (linear_value + residual_value - expected).abs() < 1e-10,
                    "root {root:?} at {values:?}: linear {linear_value} + residual {residual_value} != {expected}"
                );
            }
        }
    }
}
