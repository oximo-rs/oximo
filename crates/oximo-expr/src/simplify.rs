use crate::arena::{ExprArena, ExprId, ExprNode};

/// Apply local algebraic simplifications to the subtree rooted at `id`,
/// returning a (possibly fresh) `ExprId` that is observationally equivalent.
///
/// Current rules: constant folding for unary nodes and `Pow`. The linear
/// fast-path is already canonical, so we leave `Linear` and n-ary `Add`/`Mul`
/// alone.
///
/// TODO: Extend this once we add a CSE pass.
pub fn simplify(arena: &mut ExprArena, id: ExprId) -> ExprId {
    let folded = match arena.get(id).clone() {
        ExprNode::Neg(inner) => match arena.get(inner) {
            ExprNode::Const(c) => Some(ExprNode::Const(-*c)),
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
        ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Log(inner)
        | ExprNode::Abs(inner) => {
            let node = arena.get(id).clone();
            if let ExprNode::Const(c) = arena.get(inner) {
                Some(ExprNode::Const(match node {
                    ExprNode::Sin(_) => c.sin(),
                    ExprNode::Cos(_) => c.cos(),
                    ExprNode::Exp(_) => c.exp(),
                    ExprNode::Log(_) => c.ln(),
                    ExprNode::Abs(_) => c.abs(),
                    _ => unreachable!(),
                }))
            } else {
                None
            }
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
    use crate::arena::{ExprArena, ExprNode};

    #[test]
    fn folds_abs_of_const() {
        let mut a = ExprArena::new();
        let c = a.push(ExprNode::Const(-5.0));
        let abs = a.push(ExprNode::Abs(c));
        let folded = simplify(&mut a, abs);
        assert!(matches!(a.get(folded), ExprNode::Const(v) if (*v - 5.0).abs() < f64::EPSILON));
    }
}
