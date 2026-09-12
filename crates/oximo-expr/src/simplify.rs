use crate::arena::{ExprArena, ExprId, ExprNode};

/// Apply local algebraic simplifications to the subtree rooted at `id`,
/// returning a (possibly fresh) `ExprId` that is observationally equivalent.
///
/// Current rules: constant folding for unary nodes, `Pow`, `Div`, `Atan2`, and
/// all-constant `Min`/`Max` nodes. The linear fast-path is already canonical,
/// so we leave `Linear` and n-ary `Add`/`Mul` alone.
///
/// TODO: Extend this once we add a CSE pass.
pub fn simplify(arena: &mut ExprArena, id: ExprId) -> ExprId {
    let folded = match arena.get(id).clone() {
        ExprNode::Unary(op, inner) => match arena.get(inner) {
            ExprNode::Const(c) => Some(ExprNode::Const(op.apply(*c))),
            _ => None,
        },
        ExprNode::Pow(base, exp) => match (arena.get(base), arena.get(exp)) {
            (ExprNode::Const(b), ExprNode::Const(e)) => Some(ExprNode::Const(b.powf(*e))),
            _ => None,
        },
        ExprNode::Div(num, den) => match (arena.get(num), arena.get(den)) {
            (ExprNode::Const(n), ExprNode::Const(d)) => Some(ExprNode::Const(n / d)),
            _ => None,
        },
        ExprNode::Atan2(y, x) => match (arena.get(y), arena.get(x)) {
            (ExprNode::Const(y), ExprNode::Const(x)) => Some(ExprNode::Const(y.atan2(*x))),
            _ => None,
        },
        ExprNode::Min(children) | ExprNode::Max(children) => {
            let is_min = matches!(arena.get(id), ExprNode::Min(_));
            let mut values = children.iter().map(|child| match arena.get(*child) {
                ExprNode::Const(value) => Some(*value),
                _ => None,
            });
            values.next().flatten().and_then(|first| {
                values
                    .try_fold(first, |acc, value| {
                        value.map(|v| if is_min { acc.min(v) } else { acc.max(v) })
                    })
                    .map(ExprNode::Const)
            })
        }
        _ => None,
    };
    match folded {
        Some(node) => arena.push(node),
        None => id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ExprArena, ExprNode, UnaryOp};

    #[test]
    fn folds_abs_of_const() {
        let mut a = ExprArena::new();
        let c = a.push(ExprNode::Const(-5.0));
        let abs = a.push(ExprNode::Unary(UnaryOp::Abs, c));
        let folded = simplify(&mut a, abs);
        assert!(matches!(a.get(folded), ExprNode::Const(v) if (*v - 5.0).abs() < f64::EPSILON));
    }
}
