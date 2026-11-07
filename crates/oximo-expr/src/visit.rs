use crate::arena::{ExprArena, ExprId, ExprNode};

/// Pre-order visitor over an arena. Backends implement this to translate the
/// expression tree into solver-specific representations without copying.
pub trait Visitor {
    fn visit(&mut self, arena: &ExprArena, id: ExprId, node: &ExprNode);
}

/// Walk the subtree rooted at `id` in pre-order.
///
/// Implemented with an explicit stack so arbitrarily deep expressions are
/// safe.
pub fn walk<V: Visitor>(arena: &ExprArena, id: ExprId, visitor: &mut V) {
    let mut stack = vec![id];
    while let Some(cur) = stack.pop() {
        let node = arena.get(cur);
        visitor.visit(arena, cur, node);
        match node {
            ExprNode::Add(children)
            | ExprNode::Mul(children)
            | ExprNode::Min(children)
            | ExprNode::Max(children) => {
                for &child in children.iter().rev() {
                    stack.push(child);
                }
            }
            ExprNode::Unary(_, inner) => {
                stack.push(*inner);
            }
            ExprNode::Pow(base, exp) | ExprNode::Atan2(base, exp) => {
                stack.push(*exp);
                stack.push(*base);
            }
            ExprNode::Div(num, den) => {
                stack.push(*den);
                stack.push(*num);
            }
            ExprNode::Const(_)
            | ExprNode::Var(_)
            | ExprNode::Param(_)
            | ExprNode::Linear { .. }
            | ExprNode::Quadratic(_) => {}
        }
    }
}

/// Walk the distinct nodes reachable from `id` in pre-order, visiting each
/// shared subexpression exactly once.
pub fn walk_shared<V: Visitor>(arena: &ExprArena, id: ExprId, visitor: &mut V) {
    let mut visited = rustc_hash::FxHashSet::default();
    let mut stack = vec![id];
    while let Some(cur) = stack.pop() {
        if !visited.insert(cur) {
            continue;
        }
        let node = arena.get(cur);
        visitor.visit(arena, cur, node);
        match node {
            ExprNode::Add(children)
            | ExprNode::Mul(children)
            | ExprNode::Min(children)
            | ExprNode::Max(children) => {
                for &child in children.iter().rev() {
                    stack.push(child);
                }
            }
            ExprNode::Unary(_, inner) => {
                stack.push(*inner);
            }
            ExprNode::Pow(base, exp) | ExprNode::Atan2(base, exp) => {
                stack.push(*exp);
                stack.push(*base);
            }
            ExprNode::Div(num, den) => {
                stack.push(*den);
                stack.push(*num);
            }
            ExprNode::Const(_)
            | ExprNode::Var(_)
            | ExprNode::Param(_)
            | ExprNode::Linear { .. }
            | ExprNode::Quadratic(_) => {}
        }
    }
}
