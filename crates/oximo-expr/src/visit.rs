use crate::arena::{ExprArena, ExprId, ExprNode};

/// Pre-order visitor over an arena. Backends implement this to translate the
/// expression tree into solver-specific representations without copying.
pub trait Visitor {
    fn visit(&mut self, arena: &ExprArena, id: ExprId, node: &ExprNode);
}

/// Walk the subtree rooted at `id` in pre-order.
pub fn walk<V: Visitor>(arena: &ExprArena, id: ExprId, visitor: &mut V) {
    let node = arena.get(id);
    visitor.visit(arena, id, node);
    match node {
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            let kids: Vec<ExprId> = children.iter().copied().collect();
            for child in kids {
                walk(arena, child, visitor);
            }
        }
        ExprNode::Neg(inner)
        | ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Log(inner)
        | ExprNode::Abs(inner) => {
            let inner = *inner;
            walk(arena, inner, visitor);
        }
        ExprNode::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;
            walk(arena, base, visitor);
            walk(arena, exp, visitor);
        }
        ExprNode::Div(num, den) => {
            let num = *num;
            let den = *den;
            walk(arena, num, visitor);
            walk(arena, den, visitor);
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {}
    }
}
