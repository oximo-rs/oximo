use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    EvalContext, EvalError, ExprArena, ExprId, ExprNode, ModelId, ParameterSnapshot,
    WithParameters, evaluate,
};

/// An immutable, owned copy of just the nodes reachable from one expression.
///
/// ```
/// use oximo_expr::{Expr, ExprArena, ExprArenaCell, ParameterSnapshot, VarId};
/// let cell = ExprArenaCell::new(ExprArena::new());
/// let x = Expr::from_var(&cell, VarId(0));
/// let captured = (x * x).capture();
/// drop(cell);
/// let values: &[f64] = &[3.0];
/// assert_eq!(captured.evaluate(&values, &ParameterSnapshot::default()).unwrap(), 9.0);
/// ```
#[derive(Clone, Debug)]
pub struct CapturedExpr {
    model_id: ModelId,
    arena: ExprArena,
    root: ExprId,
}

impl CapturedExpr {
    pub(crate) fn new(model_id: ModelId, arena: &ExprArena, root: ExprId) -> Self {
        // Post-order retains child order and gives every child a lower ID than
        // its parent, including when the source graph contains forward references.
        let mut nodes = Vec::new();
        let mut seen = FxHashSet::default();
        seen.insert(root);
        let mut stack = vec![(root, 0)];
        while let Some((id, next)) = stack.last_mut() {
            let current = *id;
            if let Some(child) = crate::eval::eval_child(arena, current, *next) {
                *next += 1;
                if seen.insert(child) {
                    stack.push((child, 0));
                }
            } else {
                nodes.push((current, arena.get(current).clone()));
                stack.pop();
            }
        }
        let remap: FxHashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(index, (id, _))| {
                (*id, ExprId(u32::try_from(index).expect("captured expressions fit source IDs")))
            })
            .collect();
        let root = remap[&root];
        let mut arena = ExprArena::with_capacity(nodes.len());
        for (_, mut node) in nodes {
            match &mut node {
                ExprNode::Add(children)
                | ExprNode::Mul(children)
                | ExprNode::Min(children)
                | ExprNode::Max(children) => {
                    for child in children {
                        *child = remap[child];
                    }
                }
                ExprNode::Unary(_, child) => *child = remap[child],
                ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                    *a = remap[a];
                    *b = remap[b];
                }
                ExprNode::Const(_)
                | ExprNode::Var(_)
                | ExprNode::Param(_)
                | ExprNode::Linear { .. } => {}
            }
            arena.push(node);
        }
        Self { model_id, arena, root }
    }

    /// Model that owned the captured expression.
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Evaluate with caller-supplied variables and saved parameter values.
    ///
    /// This is a raw-ID evaluation API.
    /// Callers must ensure the context and parameter IDs belong to [`Self::model_id`],
    /// or have an explicit compatible mapping.
    ///
    /// # Errors
    /// Returns an error for a missing variable or parameter.
    pub fn evaluate<C: EvalContext>(
        &self,
        context: &C,
        parameters: &ParameterSnapshot,
    ) -> Result<f64, EvalError> {
        evaluate(&self.arena, self.root, &WithParameters::new(context, parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, ExprArenaCell, UnaryOp, VarId};
    use smallvec::smallvec;

    #[test]
    fn capture_matches_raw_evaluation_for_shared_graphs_and_all_node_shapes() {
        let (mut arena, mut roots) = crate::fold::test_arena();
        let a = arena.var(VarId(0));
        let b = arena.constant(2.0);

        for node in [
            ExprNode::Div(a, b),
            ExprNode::Atan2(a, b),
            ExprNode::Min(smallvec![a, b, a]),
            ExprNode::Max(smallvec![b, a, b]),
        ] {
            roots.push(arena.push(node));
        }

        let cell = ExprArenaCell::new(arena);
        let values: &[f64] = &[0.25, -0.5, 1.0, 2.0, 0.0];
        let parameters = ParameterSnapshot::default();
        for root in roots {
            let expression = Expr::new(root, &cell);
            let captured = expression.capture();
            let expected = evaluate(&cell.borrow(), root, &values).unwrap();
            let actual = captured.evaluate(&values, &parameters).unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(captured.model_id(), expression.model_id());

            for index in 0..captured.arena.len() {
                let parent = ExprId(u32::try_from(index).unwrap());
                let mut next = 0;
                while let Some(child) = crate::eval::eval_child(&captured.arena, parent, next) {
                    assert!(child.index() < index);
                    next += 1;
                }
            }
        }
    }

    #[test]
    fn captured_linear_nodes_preserve_variable_ids_order_and_missing_values() {
        let mut arena = ExprArena::new();
        let root = arena.push(ExprNode::Linear {
            coeffs: vec![(VarId(2), 3.0), (VarId(0), -2.0)],
            constant: -1.0,
        });

        let captured = CapturedExpr::new(ModelId::UNASSIGNED, &arena, root);
        let values: &[f64] = &[4.0, 99.0, 5.0];
        assert_eq!(captured.arena.len(), 1);
        assert_eq!(captured.evaluate(&values, &ParameterSnapshot::default()).unwrap(), 6.0);

        let missing: &[f64] = &[4.0];
        assert!(matches!(
            captured.evaluate(&missing, &ParameterSnapshot::default()),
            Err(EvalError::UnboundVar(VarId(2)))
        ));
    }

    #[test]
    fn capture_only_keeps_reachable_nodes_and_does_not_share_the_source_allocation() {
        let mut arena = ExprArena::new();
        let variable = arena.var(VarId(0));
        let root = arena.push(ExprNode::Add(smallvec![variable, variable]));
        for _ in 0..10_000 {
            arena.constant(99.0);
        }

        let pointer = arena.nodes().as_ptr();
        let captured = CapturedExpr::new(ModelId::UNASSIGNED, &arena, root);
        assert_eq!(captured.arena.len(), 2);

        *arena.get_mut(root) = ExprNode::Const(99.0);
        assert_eq!(arena.nodes().as_ptr(), pointer);

        drop(arena);
        let values: &[f64] = &[3.0];
        assert_eq!(captured.evaluate(&values, &ParameterSnapshot::default()).unwrap(), 6.0);
    }

    #[test]
    fn captured_parameters_require_explicit_saved_values_and_missing_variables_stay_missing() {
        let mut arena = ExprArena::new();
        let parameter = arena.new_param(2.0);
        let parameters = arena.parameter_snapshot();
        let parameter = arena.param(parameter);
        let variable = arena.var(VarId(0));
        let root = arena.push(ExprNode::Mul(smallvec![parameter, variable]));
        let cell = ExprArenaCell::new(arena);
        let captured = Expr::new(root, &cell).capture();
        let values: &[f64] = &[3.0];
        assert_eq!(captured.evaluate(&values, &parameters).unwrap(), 6.0);
        assert!(matches!(
            captured.evaluate(&values, &ParameterSnapshot::default()),
            Err(EvalError::UnboundParam(_))
        ));

        let missing: &[f64] = &[];
        assert!(matches!(
            captured.evaluate(&missing, &parameters),
            Err(EvalError::UnboundVar(VarId(0)))
        ));
    }

    #[test]
    fn capture_and_evaluation_are_stack_safe_and_reorder_source_forward_references() {
        let mut arena = ExprArena::new();
        let root = arena.constant(0.0);
        let mut last = arena.var(VarId(0));
        for _ in 0..30_000 {
            last = arena.push(ExprNode::Unary(UnaryOp::Neg, last));
        }
        *arena.get_mut(root) = ExprNode::Unary(UnaryOp::Neg, last);
        let cell = ExprArenaCell::new(arena);
        let captured = Expr::new(root, &cell).capture();
        assert_eq!(captured.root.index(), captured.arena.len() - 1);

        let values: &[f64] = &[3.0];
        assert_eq!(captured.evaluate(&values, &ParameterSnapshot::default()).unwrap(), -3.0);
    }
}
