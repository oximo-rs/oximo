use crate::arena::{ExprArena, ExprId, ExprNode};

/// Highest-degree polynomial class an expression belongs to, ignoring constant
/// folding. Used by backends to pick between linear, quadratic, and general
/// nonlinear translation paths.
///
/// Variants are ordered by increasing degree, so `max` of two classes yields the
/// dominating one (e.g. a model with a quadratic objective and a nonlinear
/// constraint is `Nonlinear`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExprClass {
    Linear,
    Quadratic,
    Nonlinear,
}

/// Polynomial-degree bucket. `Higher` is a saturating sentinel for "anything
/// above quadratic". Both polynomial degree > 2 and transcendentals collapse
/// into it, since neither fits a QP solver's quadratic API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Degree {
    Zero,
    One,
    Two,
    Higher,
}

impl Degree {
    /// `+` on a sum: take the maximum, saturating at `Higher`.
    fn add(self, other: Degree) -> Degree {
        self.max(other)
    }

    /// `*` on a product: add ordinal degrees, saturating at `Higher`.
    fn mul(self, other: Degree) -> Degree {
        match (self, other) {
            (Degree::Higher, _) | (_, Degree::Higher) => Degree::Higher,
            (Degree::Zero, x) | (x, Degree::Zero) => x,
            (Degree::One, Degree::One) => Degree::Two,
            _ => Degree::Higher,
        }
    }

    /// `^n` on a power: multiply by `n`, saturating at `Higher`.
    fn pow(self, n: u32) -> Degree {
        match (self, n) {
            (_, 0) | (Degree::Zero, _) => Degree::Zero,
            (d, 1) => d,
            (Degree::One, 2) => Degree::Two,
            _ => Degree::Higher,
        }
    }
}

fn degree(arena: &ExprArena, id: ExprId) -> Degree {
    match arena.get(id) {
        ExprNode::Const(_) => Degree::Zero,
        ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => Degree::One,
        ExprNode::Neg(inner) => degree(arena, *inner),
        ExprNode::Add(children) => {
            let mut d = Degree::Zero;
            for c in children {
                d = d.add(degree(arena, *c));
                if d == Degree::Higher {
                    return d;
                }
            }
            d
        }
        ExprNode::Mul(children) => {
            let mut d = Degree::Zero;
            for c in children {
                d = d.mul(degree(arena, *c));
                if d == Degree::Higher {
                    return d;
                }
            }
            d
        }
        ExprNode::Pow(base, exp) => {
            let ExprNode::Const(e) = arena.get(*exp) else { return Degree::Higher };
            if (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
                return Degree::Higher;
            }
            // Bucket the exponent into the only values `Degree::pow` treats
            // distinctly.
            let n = match e.round() {
                v if v < 0.5 => 0,
                v if v < 1.5 => 1,
                v if v < 2.5 => 2,
                _ => 3,
            };
            degree(arena, *base).pow(n)
        }
        // Transcendentals are always > quadratic. Division is too: `div_into`
        // folds the only degree-preserving case (constant denominator) before a
        // `Div` node is created, so any other `Div` has a non-constant
        // denominator.
        ExprNode::Div(_, _)
        | ExprNode::Sin(_)
        | ExprNode::Cos(_)
        | ExprNode::Exp(_)
        | ExprNode::Log(_)
        | ExprNode::Abs(_) => Degree::Higher,
    }
}

/// Classify an expression as Linear, Quadratic (polynomial degree <= 2 with at
/// least one degree-2 term), or Nonlinear (transcendentals, non-integer powers,
/// or polynomial degree > 2).
pub fn classify(arena: &ExprArena, id: ExprId) -> ExprClass {
    match degree(arena, id) {
        Degree::Zero | Degree::One => ExprClass::Linear,
        Degree::Two => ExprClass::Quadratic,
        Degree::Higher => ExprClass::Nonlinear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ExprArena, ExprNode, VarId};
    use smallvec::smallvec;

    fn var(arena: &mut ExprArena, i: u32) -> ExprId {
        arena.push(ExprNode::Var(VarId(i)))
    }

    #[test]
    fn linear_var_sum() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let sum = a.push(ExprNode::Add(smallvec![x, y]));
        assert_eq!(classify(&a, sum), ExprClass::Linear);
    }

    #[test]
    fn quadratic_mul_two_vars() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let xy = a.push(ExprNode::Mul(smallvec![x, y]));
        assert_eq!(classify(&a, xy), ExprClass::Quadratic);
    }

    #[test]
    fn quadratic_pow_two() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(x, two));
        assert_eq!(classify(&a, sq), ExprClass::Quadratic);
    }

    #[test]
    fn nonlinear_pow_three() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let three = a.push(ExprNode::Const(3.0));
        let cube = a.push(ExprNode::Pow(x, three));
        assert_eq!(classify(&a, cube), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_div() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let q = a.push(ExprNode::Div(x, y));
        assert_eq!(classify(&a, q), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_sin() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let s = a.push(ExprNode::Sin(x));
        assert_eq!(classify(&a, s), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_abs() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let s = a.push(ExprNode::Abs(x));
        assert_eq!(classify(&a, s), ExprClass::Nonlinear);
    }

    #[test]
    fn nonlinear_triple_mul() {
        let mut arena = ExprArena::new();
        let x = var(&mut arena, 0);
        let y = var(&mut arena, 1);
        let z = var(&mut arena, 2);
        let prod = arena.push(ExprNode::Mul(smallvec![x, y, z]));
        assert_eq!(classify(&arena, prod), ExprClass::Nonlinear);
    }

    #[test]
    fn linear_promoted_by_const_mul() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let c = a.push(ExprNode::Const(3.0));
        let m = a.push(ExprNode::Mul(smallvec![c, x]));
        assert_eq!(classify(&a, m), ExprClass::Linear);
    }
}
