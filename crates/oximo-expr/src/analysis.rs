//! Structural expression facts shared by preparation classification and requirements.

#[cfg(test)]
use crate::arena::ArenaAccess;
use crate::classify::Degree;
use crate::{ExprArena, ExprId, ExprNode, UnaryOp};

/// Expression vocabulary consumed by an adapter before constant folding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionOperator {
    Unary(UnaryOp),
    Add,
    Multiply,
    Divide,
    ConstantPower,
    VariablePower,
    Atan2,
    Min,
    Max,
}

impl ExpressionOperator {
    #[doc(hidden)]
    pub const fn bit(self) -> u64 {
        let index = match self {
            Self::Unary(op) => match op {
                UnaryOp::Neg => 0,
                UnaryOp::Abs => 1,
                UnaryOp::Sqrt => 2,
                UnaryOp::Cbrt => 3,
                UnaryOp::Exp => 4,
                UnaryOp::Exp2 => 5,
                UnaryOp::Expm1 => 6,
                UnaryOp::Log => 7,
                UnaryOp::Log2 => 8,
                UnaryOp::Log10 => 9,
                UnaryOp::Log1p => 10,
                UnaryOp::Sin => 11,
                UnaryOp::Cos => 12,
                UnaryOp::Tan => 13,
                UnaryOp::Asin => 14,
                UnaryOp::Acos => 15,
                UnaryOp::Atan => 16,
                UnaryOp::Sinh => 17,
                UnaryOp::Cosh => 18,
                UnaryOp::Tanh => 19,
                UnaryOp::Asinh => 20,
                UnaryOp::Acosh => 21,
                UnaryOp::Atanh => 22,
            },
            Self::Add => 23,
            Self::Multiply => 24,
            Self::Divide => 25,
            Self::ConstantPower => 26,
            Self::VariablePower => 27,
            Self::Atan2 => 28,
            Self::Min => 29,
            Self::Max => 30,
        };
        1_u64 << index
    }

    /// Complete public expression vocabulary.
    pub const ALL: &'static [Self] = &[
        Self::Unary(UnaryOp::Neg),
        Self::Unary(UnaryOp::Abs),
        Self::Unary(UnaryOp::Sqrt),
        Self::Unary(UnaryOp::Cbrt),
        Self::Unary(UnaryOp::Exp),
        Self::Unary(UnaryOp::Exp2),
        Self::Unary(UnaryOp::Expm1),
        Self::Unary(UnaryOp::Log),
        Self::Unary(UnaryOp::Log2),
        Self::Unary(UnaryOp::Log10),
        Self::Unary(UnaryOp::Log1p),
        Self::Unary(UnaryOp::Sin),
        Self::Unary(UnaryOp::Cos),
        Self::Unary(UnaryOp::Tan),
        Self::Unary(UnaryOp::Asin),
        Self::Unary(UnaryOp::Acos),
        Self::Unary(UnaryOp::Atan),
        Self::Unary(UnaryOp::Sinh),
        Self::Unary(UnaryOp::Cosh),
        Self::Unary(UnaryOp::Tanh),
        Self::Unary(UnaryOp::Asinh),
        Self::Unary(UnaryOp::Acosh),
        Self::Unary(UnaryOp::Atanh),
        Self::Add,
        Self::Multiply,
        Self::Divide,
        Self::ConstantPower,
        Self::VariablePower,
        Self::Atan2,
        Self::Min,
        Self::Max,
    ];
}

impl std::fmt::Display for ExpressionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unary(op) => write!(f, "{op}"),
            Self::Add => f.write_str("add"),
            Self::Multiply => f.write_str("multiply"),
            Self::Divide => f.write_str("divide"),
            Self::ConstantPower => f.write_str("constant-exponent pow"),
            Self::VariablePower => f.write_str("variable-exponent pow"),
            Self::Atan2 => f.write_str("atan2"),
            Self::Min => f.write_str("min"),
            Self::Max => f.write_str("max"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct ExpressionSummary {
    pub(crate) degree: Degree,
    pub operators: u64,
    pub depends_on_variable: bool,
    pub depends_on_parameter: bool,
    pub nonsmooth: bool,
}

/// At most 8 MiB of indexed scratch.
#[derive(Debug, Default)]
struct SummaryScratch {
    dense: Option<Vec<Option<ExpressionSummary>>>,
    sparse: rustc_hash::FxHashMap<ExprId, ExpressionSummary>,
    initialized: bool,
    len: usize,
}

impl SummaryScratch {
    fn prepare(&mut self, arena_len: usize) {
        let limit = (8 * 1024 * 1024) / std::mem::size_of::<Option<ExpressionSummary>>();
        if let Some(dense) = &mut self.dense {
            if arena_len <= limit {
                dense.resize(arena_len, None);
            } else {
                for (id, summary) in
                    self.dense.take().expect("dense storage").into_iter().enumerate()
                {
                    if let Some(summary) = summary {
                        self.sparse
                            .insert(ExprId(u32::try_from(id).expect("arena index")), summary);
                    }
                }
            }
        }
        if !self.initialized {
            self.initialized = true;
            if arena_len <= (8 * 1024 * 1024) / std::mem::size_of::<Option<ExpressionSummary>>() {
                self.dense = Some(vec![None; arena_len]);
            }
        }
    }

    fn get(&self, id: ExprId) -> Option<&ExpressionSummary> {
        self.dense.as_ref().map_or_else(
            || self.sparse.get(&id),
            |dense| dense.get(id.index()).and_then(Option::as_ref),
        )
    }

    fn insert(&mut self, id: ExprId, summary: ExpressionSummary) {
        let previous = if let Some(dense) = &mut self.dense {
            dense[id.index()].replace(summary)
        } else {
            self.sparse.insert(id, summary)
        };
        self.len += usize::from(previous.is_none());
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }
}

fn children(node: &ExprNode, mut visit: impl FnMut(ExprId)) {
    match node {
        ExprNode::Unary(_, child) => visit(*child),
        ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
            visit(*a);
            visit(*b);
        }
        ExprNode::Add(items)
        | ExprNode::Mul(items)
        | ExprNode::Min(items)
        | ExprNode::Max(items) => {
            for &child in items {
                visit(child);
            }
        }
        ExprNode::Const(_)
        | ExprNode::Param(_)
        | ExprNode::Var(_)
        | ExprNode::Linear { .. }
        | ExprNode::Quadratic(_) => {}
    }
}

/// Lazy structural analysis with bounded indexed storage and sparse fallback.
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct ExpressionAnalysis {
    summaries: SummaryScratch,
    stack: Vec<(ExprId, bool)>,
}

impl ExpressionAnalysis {
    /// Number of retained compound-node summaries.
    pub fn cached_nodes(&self) -> usize {
        self.summaries.len
    }

    pub(crate) fn leaf(arena: &ExprArena, id: ExprId) -> Option<ExpressionSummary> {
        Self::leaf_node(arena.get(id))
    }

    fn leaf_node(node: &ExprNode) -> Option<ExpressionSummary> {
        match node {
            ExprNode::Const(_) => Some(ExpressionSummary::default()),
            ExprNode::Param(_) => Some(ExpressionSummary {
                depends_on_parameter: true,
                ..ExpressionSummary::default()
            }),
            ExprNode::Var(_) => Some(ExpressionSummary {
                degree: Degree::One,
                depends_on_variable: true,
                ..ExpressionSummary::default()
            }),
            ExprNode::Linear { coeffs, .. } => Some(ExpressionSummary {
                degree: if coeffs.is_empty() { Degree::Zero } else { Degree::One },
                depends_on_variable: !coeffs.is_empty(),
                ..ExpressionSummary::default()
            }),
            ExprNode::Quadratic(q) => Some(ExpressionSummary {
                degree: if q.quadratic.is_empty() {
                    if q.linear.is_empty() { Degree::Zero } else { Degree::One }
                } else {
                    Degree::Two
                },
                depends_on_variable: !q.quadratic.is_empty() || !q.linear.is_empty(),
                ..ExpressionSummary::default()
            }),
            _ => None,
        }
    }

    /// Look up a fact for this analysis's immutable arena snapshot.
    pub fn known(&self, arena: &ExprArena, id: ExprId) -> Option<ExpressionSummary> {
        arena.get(id);
        self.known_node(arena, id)
    }

    fn known_node(&self, arena: &ExprArena, id: ExprId) -> Option<ExpressionSummary> {
        self.summaries.get(id).copied().or_else(|| Self::leaf_node(arena.node(id)))
    }

    /// Compute all structural facts together, retaining shared descendants.
    /// Use an analysis with one immutable arena, or append-only versions of it.
    ///
    /// # Panics
    ///
    /// Panics for an invalid expression ID or a graph containing invalid children.
    pub fn summarize(&mut self, arena: &ExprArena, root: ExprId) -> ExpressionSummary {
        arena.get(root);
        if let Some(summary) = self.known_node(arena, root) {
            return summary;
        }
        self.summaries.prepare(arena.len());

        self.stack.push((root, false));
        while let Some((id, visited)) = self.stack.pop() {
            if self.known_node(arena, id).is_some() {
                continue;
            }
            let node = arena.node(id);
            if !visited {
                self.stack.push((id, true));
                children(node, |child| {
                    if self.known_node(arena, child).is_none() {
                        self.stack.push((child, false));
                    }
                });
                continue;
            }

            #[cfg(test)]
            arena.record_classification_visit();
            let mut summary = ExpressionSummary::default();
            children(node, |child| {
                let child = self.known_node(arena, child).expect("postorder child summary");
                summary.degree = if matches!(node, ExprNode::Mul(_)) {
                    summary.degree.mul(child.degree)
                } else {
                    summary.degree.add(child.degree)
                };
                summary.operators |= child.operators;
                summary.depends_on_variable |= child.depends_on_variable;
                summary.depends_on_parameter |= child.depends_on_parameter;
                summary.nonsmooth |= child.nonsmooth;
            });
            let operator = match node {
                ExprNode::Unary(op, _) => ExpressionOperator::Unary(*op),
                ExprNode::Add(_) => ExpressionOperator::Add,
                ExprNode::Mul(_) => ExpressionOperator::Multiply,
                ExprNode::Div(_, _) => ExpressionOperator::Divide,
                ExprNode::Pow(_, exponent) => {
                    if self
                        .known_node(arena, *exponent)
                        .expect("exponent summary")
                        .depends_on_variable
                    {
                        ExpressionOperator::VariablePower
                    } else {
                        ExpressionOperator::ConstantPower
                    }
                }
                ExprNode::Atan2(_, _) => ExpressionOperator::Atan2,
                ExprNode::Min(_) => ExpressionOperator::Min,
                ExprNode::Max(_) => ExpressionOperator::Max,
                _ => unreachable!("leaves are handled without traversal"),
            };

            summary.degree = match node {
                ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Unary(UnaryOp::Neg, _) => {
                    summary.degree
                }
                ExprNode::Pow(base, exponent) => match arena.node(*exponent) {
                    ExprNode::Const(e)
                        if e.is_finite() && (*e - e.round()).abs() < f64::EPSILON && *e >= 0.0 =>
                    {
                        let n = match e.round() {
                            v if v < 0.5 => 0,
                            v if v < 1.5 => 1,
                            v if v < 2.5 => 2,
                            _ => 3,
                        };
                        self.known_node(arena, *base).expect("base summary").degree.pow(n)
                    }
                    _ => Degree::Higher,
                },
                _ => Degree::Higher,
            };
            summary.operators |= operator.bit();
            let nonsmooth = matches!(operator, ExpressionOperator::Min | ExpressionOperator::Max)
                || matches!(operator, ExpressionOperator::Unary(op) if op.is_nonsmooth());
            summary.nonsmooth |= nonsmooth && summary.depends_on_variable;
            self.summaries.insert(id, summary);
        }

        *self.summaries.get(root).expect("root summary")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprClass, VarId, classify};
    use smallvec::smallvec;

    #[test]
    fn combined_degrees_match_existing_symbolic_rules_with_dense_and_sparse_scratch() {
        let (arena, roots) = crate::fold::test_arena();
        let mut indexed = ExpressionAnalysis::default();
        let mut sparse = ExpressionAnalysis::default();
        sparse.summaries.prepare(usize::MAX);
        for root in roots {
            let summary = indexed.summarize(&arena, root);
            assert_eq!(summary, sparse.summarize(&arena, root));

            let class = match summary.degree {
                Degree::Zero | Degree::One => ExprClass::Linear,
                Degree::Two => ExprClass::Quadratic,
                Degree::Higher => ExprClass::Nonlinear,
            };
            assert_eq!(class, classify(&arena, root), "root {root:?}");
            assert_eq!(
                summary.degree == Degree::Zero,
                crate::classify::is_constant_access(&arena, root)
            );
        }
        assert!(indexed.summaries.dense.is_some());
        assert!(sparse.summaries.dense.is_none());
        assert_eq!(indexed.summaries.len(), sparse.summaries.len());
        assert!(
            indexed.summaries.dense.as_ref().unwrap().len()
                * std::mem::size_of::<Option<ExpressionSummary>>()
                <= 8 * 1024 * 1024
        );
    }

    #[test]
    fn classification_and_requirements_cache_their_walks_independently() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let terms = (0..150).map(|_| arena.push(ExprNode::Mul(smallvec![x, x]))).collect();
        let root = arena.push(ExprNode::Add(terms));
        arena.__enable_preparation_classifications();
        arena.enable_classification_counters();
        assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        assert!(arena.__cached_expression_summary(root).is_none());

        let before = arena.classification_stats();
        let summary = arena.__preparation_expression_summary(root).unwrap();
        assert_eq!(
            summary.operators,
            ExpressionOperator::Add.bit() | ExpressionOperator::Multiply.bit()
        );
        assert!(summary.depends_on_variable);
        assert!(!summary.depends_on_parameter);
        let after = arena.classification_stats();
        assert!(after.visited_nodes > before.visited_nodes);
        assert_eq!(arena.__preparation_expression_summary(root), Some(summary));
        assert_eq!(classify(&arena, root), ExprClass::Quadratic);
        assert_eq!(arena.classification_stats().visited_nodes, after.visited_nodes);
    }

    #[test]
    fn early_nonlinear_classification_does_not_walk_remaining_branches() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let sin = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
        let terms = (0..10_000).map(|_| arena.push(ExprNode::Mul(smallvec![x, x]))).collect();
        let wide = arena.push(ExprNode::Add(terms));
        let root = arena.push(ExprNode::Add(smallvec![sin, wide]));
        arena.__enable_preparation_classifications();
        arena.enable_classification_counters();
        assert_eq!(classify(&arena, root), ExprClass::Nonlinear);
        assert!(arena.classification_stats().visited_nodes <= 4);
        assert!(arena.__cached_expression_summary(root).is_none());
    }

    #[test]
    fn summaries_follow_append_and_mutation_without_leaking_into_clones() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let root = arena.push(ExprNode::Mul(smallvec![x, x]));
        arena.__enable_preparation_classifications();
        assert_eq!(arena.__preparation_expression_summary(root).unwrap().degree, Degree::Two);
        assert!(arena.clone().__cached_expression_summary(root).is_none());

        let appended = arena.push(ExprNode::Add(smallvec![root, root]));
        assert_eq!(arena.__preparation_expression_summary(appended).unwrap().degree, Degree::Two);

        *arena.get_mut(x) = ExprNode::Const(2.0);
        assert!(arena.__cached_expression_summary(root).is_none());

        arena.__enable_preparation_classifications();
        assert_eq!(arena.__preparation_expression_summary(appended).unwrap().degree, Degree::Zero);
    }

    #[test]
    fn zero_parameters_and_cancelled_values_keep_symbolic_operators_and_degree() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let p = arena.new_param(0.0);
        let parameter = arena.param(p);
        let term = arena.push(ExprNode::Mul(smallvec![parameter, x, x]));
        let neg = arena.push(ExprNode::Unary(UnaryOp::Neg, term));
        let root = arena.push(ExprNode::Add(smallvec![term, neg]));
        arena.__enable_preparation_classifications();
        let expected = arena.__preparation_expression_summary(root).unwrap();
        assert_eq!(expected.degree, Degree::Two);
        assert!(expected.depends_on_parameter);

        for value in [1.0, -1.0, 0.0] {
            arena.set_param_value(p, value).unwrap();
            assert_eq!(arena.__preparation_expression_summary(root), Some(expected));
        }
    }

    #[test]
    fn preparation_analysis_is_stack_safe_and_shared_by_concurrent_readers() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let mut root = arena.push(ExprNode::Mul(smallvec![x, x]));
        for _ in 0..10_000 {
            root = arena.push(ExprNode::Unary(UnaryOp::Neg, root));
        }
        arena.__enable_preparation_classifications();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let arena = &arena;
                scope.spawn(move || {
                    assert_eq!(classify(arena, root), ExprClass::Quadratic);

                    let summary = arena.__preparation_expression_summary(root).unwrap();
                    assert_eq!(summary.degree, Degree::Two);
                    assert_eq!(
                        summary.operators,
                        ExpressionOperator::Multiply.bit()
                            | ExpressionOperator::Unary(UnaryOp::Neg).bit()
                    );
                });
            }
        });
    }
}
