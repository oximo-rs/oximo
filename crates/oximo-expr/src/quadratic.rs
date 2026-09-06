use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::arena::{ExprArena, ExprId, ExprNode, VarId};

/// Quadratic decomposition of an expression: its Hessian, gradient-linear
/// part, and constant.
///
/// For a degree-`<= 2` polynomial `f(x)`, this holds the exact Taylor data
///
/// ```text
/// f(x) = constant + sum_i linear_i * x_i + 0.5 * x' Q x
/// ```
///
/// where `Q` is the (constant) Hessian. Returned by
/// [`extract_quadratic`].
#[derive(Clone, Debug, Default)]
pub struct QuadraticTerms {
    /// Lower-triangle Hessian entries `(row, col, h)` with `row >= col`, where
    /// `h = partial^2 f / partial x_row partial x_col`. Diagonal entries are the full second
    /// derivative, so `a * x^2` yields `(x, x, 2a)`. This matches the
    /// `0.5 * x' Q x` convention used by QP solvers.
    pub hessian: Vec<(VarId, VarId, f64)>,
    /// Linear coefficients `(var, coeff)`, the gradient of `f` at the origin.
    pub linear: Vec<(VarId, f64)>,
    /// The constant term `f(0)`.
    pub constant: f64,
}

/// Internal accumulator while walking the expression. `quad` keys are ordered
/// `(min, max)` variable pairs and hold the polynomial coefficient of
/// `x_i * x_j` (i.e. the coefficient of `x_i^2` on the diagonal), not yet the
/// doubled Hessian value.
#[derive(Clone, Default)]
struct Poly {
    quad: FxHashMap<(VarId, VarId), f64>,
    linear: FxHashMap<VarId, f64>,
    constant: f64,
}

impl Poly {
    fn constant(c: f64) -> Self {
        Self { constant: c, ..Self::default() }
    }

    fn var(v: VarId) -> Self {
        let mut linear = FxHashMap::with_capacity_and_hasher(1, FxBuildHasher);
        linear.insert(v, 1.0);
        Self { linear, ..Self::default() }
    }

    fn is_constant(&self) -> bool {
        self.quad.is_empty() && self.linear.is_empty()
    }

    fn is_linear(&self) -> bool {
        self.quad.is_empty()
    }

    fn scale(mut self, s: f64) -> Self {
        self.constant *= s;
        for c in self.linear.values_mut() {
            *c *= s;
        }
        for c in self.quad.values_mut() {
            *c *= s;
        }
        self
    }

    fn neg(self) -> Self {
        self.scale(-1.0)
    }

    fn add_assign(&mut self, other: Poly) {
        self.constant += other.constant;
        for (v, c) in other.linear {
            *self.linear.entry(v).or_insert(0.0) += c;
        }
        for (k, c) in other.quad {
            *self.quad.entry(k).or_insert(0.0) += c;
        }
    }
}

/// Ordered `(min, max)` variable pair, used as the canonical quad key.
fn pair(a: VarId, b: VarId) -> (VarId, VarId) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

/// Multiply two linear polynomials, producing the degree-2 product. Both
/// operands must be linear (`quad` empty), the caller guarantees this.
fn mul_linear(a: &Poly, b: &Poly) -> Poly {
    let mut out = Poly::constant(a.constant * b.constant);
    // a.constant * b.linear + b.constant * a.linear
    for (v, c) in &b.linear {
        *out.linear.entry(*v).or_insert(0.0) += a.constant * c;
    }
    for (v, c) in &a.linear {
        *out.linear.entry(*v).or_insert(0.0) += b.constant * c;
    }
    // a.linear[i] * b.linear[j] -> quad term x_i x_j
    for (vi, ci) in &a.linear {
        for (vj, cj) in &b.linear {
            *out.quad.entry(pair(*vi, *vj)).or_insert(0.0) += ci * cj;
        }
    }
    out
}

/// Recursively interpret `id` as a polynomial of degree `<= 2`. Returns `None`
/// for anything of higher degree, transcendentals, or division. Parameters fold
/// to their live arena value (a degree-0 constant).
#[cfg(test)]
fn recursive_poly(arena: &ExprArena, id: ExprId) -> Option<Poly> {
    match arena.get(id) {
        ExprNode::Const(c) => Some(Poly::constant(*c)),
        ExprNode::Var(v) => Some(Poly::var(*v)),
        ExprNode::Linear { coeffs, constant } => {
            let mut linear: FxHashMap<VarId, f64> =
                FxHashMap::with_capacity_and_hasher(coeffs.len(), FxBuildHasher);
            for (v, c) in coeffs {
                *linear.entry(*v).or_insert(0.0) += *c;
            }
            Some(Poly { quad: FxHashMap::default(), linear, constant: *constant })
        }
        ExprNode::Neg(inner) => recursive_poly(arena, *inner).map(Poly::neg),
        ExprNode::Add(children) => {
            let mut acc = Poly::default();
            for child in children {
                acc.add_assign(recursive_poly(arena, *child)?);
            }
            Some(acc)
        }
        ExprNode::Mul(children) => {
            let mut acc = Poly::constant(1.0);
            for child in children {
                let p = recursive_poly(arena, *child)?;
                acc = if acc.is_constant() {
                    p.scale(acc.constant)
                } else if p.is_constant() {
                    acc.scale(p.constant)
                } else if acc.is_linear() && p.is_linear() {
                    mul_linear(&acc, &p)
                } else {
                    return None;
                };
            }
            Some(acc)
        }
        ExprNode::Pow(base, exp) => {
            let ExprNode::Const(e) = arena.get(*exp) else { return None };
            if (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
                return None;
            }
            match e.round() {
                n if n < 0.5 => Some(Poly::constant(1.0)),
                n if n < 1.5 => recursive_poly(arena, *base),
                n if n < 2.5 => {
                    let p = recursive_poly(arena, *base)?;
                    if !p.is_linear() {
                        return None;
                    }
                    Some(mul_linear(&p, &p))
                }
                _ => None,
            }
        }
        ExprNode::Param(p) => Some(Poly::constant(arena.param_value(*p))),
        ExprNode::Div(_, _)
        | ExprNode::Sin(_)
        | ExprNode::Cos(_)
        | ExprNode::Exp(_)
        | ExprNode::Log(_)
        | ExprNode::Abs(_) => None,
    }
}

struct PolyFolder<'a>(&'a ExprArena);
enum PolyOp {
    Add,
    Mul,
    Neg,
    Identity,
    Square,
}
struct PolyState<'a> {
    rest: &'a [ExprId],
    value: Option<Poly>,
    op: PolyOp,
}

impl<'a> crate::fold::Folder for PolyFolder<'a> {
    type Value = Poly;
    type State = PolyState<'a>;

    fn start(&self, id: ExprId) -> std::ops::ControlFlow<Option<Poly>, Self::State> {
        use std::ops::ControlFlow::{Break, Continue};
        let (rest, op, value) = match self.0.get(id) {
            ExprNode::Const(c) => return Break(Some(Poly::constant(*c))),
            ExprNode::Param(p) => return Break(Some(Poly::constant(self.0.param_value(*p)))),
            ExprNode::Var(v) => return Break(Some(Poly::var(*v))),
            ExprNode::Linear { coeffs, constant } => {
                let mut linear = FxHashMap::with_capacity_and_hasher(coeffs.len(), FxBuildHasher);
                for &(v, c) in coeffs {
                    *linear.entry(v).or_insert(0.0) += c;
                }
                return Break(Some(Poly { linear, constant: *constant, ..Poly::default() }));
            }
            ExprNode::Add(c) => (c.as_slice(), PolyOp::Add, Poly::default()),
            ExprNode::Mul(c) => (c.as_slice(), PolyOp::Mul, Poly::constant(1.0)),
            ExprNode::Neg(c) => (std::slice::from_ref(c), PolyOp::Neg, Poly::default()),
            ExprNode::Pow(base, exp) => {
                let ExprNode::Const(e) = self.0.get(*exp) else { return Break(None) };
                if (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
                    return Break(None);
                }
                let op = match e.round() {
                    n if n < 0.5 => return Break(Some(Poly::constant(1.0))),
                    n if n < 1.5 => PolyOp::Identity,
                    n if n < 2.5 => PolyOp::Square,
                    _ => return Break(None),
                };
                (std::slice::from_ref(base), op, Poly::default())
            }
            _ => return Break(None),
        };
        Continue(PolyState { rest, op, value: Some(value) })
    }

    fn next(&self, state: &mut Self::State) -> std::ops::ControlFlow<Option<Poly>, ExprId> {
        use std::ops::ControlFlow::{Break, Continue};
        if state.value.is_none() {
            return Break(None);
        }
        while let Some((&child, tail)) = state.rest.split_first() {
            state.rest = tail;
            if matches!(state.op, PolyOp::Add) {
                let acc = state.value.as_mut().expect("checked above");
                // Avoid creating a one-entry hash table for every variable in a sum.
                match self.0.get(child) {
                    ExprNode::Var(v) => {
                        acc.constant += 0.0;
                        *acc.linear.entry(*v).or_insert(0.0) += 1.0;
                        continue;
                    }
                    ExprNode::Const(c) => {
                        acc.constant += c;
                        continue;
                    }
                    ExprNode::Param(p) => {
                        acc.constant += self.0.param_value(*p);
                        continue;
                    }
                    _ => {}
                }
            }
            return Continue(child);
        }
        Break(state.value.take())
    }

    fn accept(&self, state: &mut Self::State, p: Poly) {
        let mut acc =
            state.value.take().expect("failed products terminate before requesting children");
        state.value = match state.op {
            PolyOp::Add => {
                acc.add_assign(p);
                Some(acc)
            }
            PolyOp::Mul if acc.is_constant() => Some(p.scale(acc.constant)),
            PolyOp::Mul if p.is_constant() => Some(acc.scale(p.constant)),
            PolyOp::Mul if acc.is_linear() && p.is_linear() => Some(mul_linear(&acc, &p)),
            PolyOp::Neg => Some(p.neg()),
            PolyOp::Identity => Some(p),
            PolyOp::Square if p.is_linear() => Some(mul_linear(&p, &p)),
            PolyOp::Mul | PolyOp::Square => None,
        };
    }
}

fn as_poly(arena: &ExprArena, mut id: ExprId) -> Option<Poly> {
    let mut negations = 0;
    while let ExprNode::Neg(child) = arena.get(id) {
        id = *child;
        negations += 1;
    }
    let mut value = crate::fold::fold(arena, id, PolyFolder(arena))?;
    for _ in 0..negations {
        value = value.neg();
    }
    Some(value)
}

/// Snapshot the quadratic structure of `id`, if it is a polynomial of degree
/// `<= 2`. Returns the Hessian (lower triangle), the linear
/// coefficients, and the constant (see [`QuadraticTerms`]).
///
/// `None` is returned for any expression `classify` would call
/// `Nonlinear` (degree `> 2`, transcendentals, non-integer/negative powers,
/// division). Parameters are folded to their current arena values, so a
/// polynomial whose coefficients are parameters is still extracted.
///
/// A purely linear (or constant) expression yields an empty `hessian`.
pub fn extract_quadratic(arena: &ExprArena, id: ExprId) -> Option<QuadraticTerms> {
    let poly = as_poly(arena, id)?;

    let mut hessian: Vec<(VarId, VarId, f64)> = Vec::with_capacity(poly.quad.len());
    for ((lo, hi), c) in poly.quad {
        if c == 0.0 {
            continue;
        }
        if lo == hi {
            // Diagonal: partial^2 (c x^2)/partial x^2 = 2c.
            hessian.push((lo, lo, 2.0 * c));
        } else {
            // Off-diagonal: store in the lower triangle (row = larger index).
            hessian.push((hi, lo, c));
        }
    }

    let mut linear: Vec<(VarId, f64)> =
        poly.linear.into_iter().filter(|(_, c)| *c != 0.0).collect();
    linear.sort_unstable_by_key(|(v, _)| v.0);
    hessian.sort_unstable_by_key(|(r, c, _)| (c.0, r.0));

    Some(QuadraticTerms { hessian, linear, constant: poly.constant })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterative_poly_matches_recursive_arithmetic() {
        let (arena, ids) = crate::fold::test_arena();
        for id in ids {
            let bits = |p: Poly| {
                let mut linear: Vec<_> =
                    p.linear.into_iter().map(|(v, c)| (v, c.to_bits())).collect();
                let mut quad: Vec<_> = p.quad.into_iter().map(|(v, c)| (v, c.to_bits())).collect();
                linear.sort_unstable_by_key(|(v, _)| v.0);
                quad.sort_unstable_by_key(|((a, b), _)| (a.0, b.0));
                (p.constant.to_bits(), linear, quad)
            };
            assert_eq!(
                as_poly(&arena, id).map(bits),
                recursive_poly(&arena, id).map(bits),
                "node {id:?}"
            );
        }
    }
    use crate::arena::{ExprArena, ExprNode, VarId};
    use smallvec::smallvec;

    fn var(arena: &mut ExprArena, i: u32) -> ExprId {
        arena.push(ExprNode::Var(VarId(i)))
    }

    fn v(i: u32) -> VarId {
        VarId(i)
    }

    #[test]
    fn square_doubles_diagonal() {
        // x0^2 -> Hessian (0,0,2), no linear, constant 0.
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(x, two));
        let q = extract_quadratic(&a, sq).unwrap();
        assert_eq!(q.hessian, vec![(v(0), v(0), 2.0)]);
        assert!(q.linear.is_empty());
        assert!(q.constant.abs() < f64::EPSILON);
    }

    #[test]
    fn bilinear_off_diagonal() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let xy = a.push(ExprNode::Mul(smallvec![x, y]));
        let q = extract_quadratic(&a, xy).unwrap();
        assert_eq!(q.hessian, vec![(v(1), v(0), 1.0)]);
        assert!(q.linear.is_empty());
    }

    #[test]
    fn cvxopt_objective_recovers_hessian() {
        // 2*x0^2 + x0*x1 + x1^2 + x0 + x1 -> Q = [[4,1],[1,2]], c = [1,1].
        let mut a = ExprArena::new();
        let x0 = var(&mut a, 0);
        let x1 = var(&mut a, 1);
        let two = a.push(ExprNode::Const(2.0));
        let x0sq = a.push(ExprNode::Pow(x0, two));
        let term0 = a.push(ExprNode::Mul(smallvec![two, x0sq]));
        let x0x1 = a.push(ExprNode::Mul(smallvec![x0, x1]));
        let two_b = a.push(ExprNode::Const(2.0));
        let x1sq = a.push(ExprNode::Pow(x1, two_b));
        let sum = a.push(ExprNode::Add(smallvec![term0, x0x1, x1sq, x0, x1]));
        let q = extract_quadratic(&a, sum).unwrap();
        assert_eq!(q.hessian, vec![(v(0), v(0), 4.0), (v(1), v(0), 1.0), (v(1), v(1), 2.0)]);
        assert_eq!(q.linear, vec![(v(0), 1.0), (v(1), 1.0)]);
        assert!(q.constant.abs() < f64::EPSILON);
    }

    #[test]
    fn square_of_sum_cross_term() {
        // (x0 + x1)^2 = x0^2 + 2 x0 x1 + x1^2 -> Q = [[2,2],[2,2]].
        let mut a = ExprArena::new();
        let x0 = var(&mut a, 0);
        let x1 = var(&mut a, 1);
        let sum = a.push(ExprNode::Add(smallvec![x0, x1]));
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(sum, two));
        let q = extract_quadratic(&a, sq).unwrap();
        assert_eq!(q.hessian, vec![(v(0), v(0), 2.0), (v(1), v(0), 2.0), (v(1), v(1), 2.0)]);
    }

    #[test]
    fn linear_only_has_empty_hessian() {
        // 3*x0 + 5 -> empty hessian, linear [(0,3)], constant 5.
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let three = a.push(ExprNode::Const(3.0));
        let mul = a.push(ExprNode::Mul(smallvec![three, x]));
        let five = a.push(ExprNode::Const(5.0));
        let expr = a.push(ExprNode::Add(smallvec![mul, five]));
        let q = extract_quadratic(&a, expr).unwrap();
        assert!(q.hessian.is_empty());
        assert_eq!(q.linear, vec![(v(0), 3.0)]);
        assert!((q.constant - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn constant_only() {
        let mut a = ExprArena::new();
        let c = a.push(ExprNode::Const(7.0));
        let q = extract_quadratic(&a, c).unwrap();
        assert!(q.hessian.is_empty());
        assert!(q.linear.is_empty());
        assert!((q.constant - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn negation_flips_signs() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(x, two));
        let inner = a.push(ExprNode::Add(smallvec![sq, x]));
        let neg = a.push(ExprNode::Neg(inner));
        let q = extract_quadratic(&a, neg).unwrap();
        assert_eq!(q.hessian, vec![(v(0), v(0), -2.0)]);
        assert_eq!(q.linear, vec![(v(0), -1.0)]);
    }

    #[test]
    fn cubic_is_none() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let three = a.push(ExprNode::Const(3.0));
        let cube = a.push(ExprNode::Pow(x, three));
        assert!(extract_quadratic(&a, cube).is_none());
    }

    #[test]
    fn triple_product_is_none() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let z = var(&mut a, 2);
        let prod = a.push(ExprNode::Mul(smallvec![x, y, z]));
        assert!(extract_quadratic(&a, prod).is_none());
    }

    #[test]
    fn transcendental_is_none() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let s = a.push(ExprNode::Sin(x));
        assert!(extract_quadratic(&a, s).is_none());
    }

    #[test]
    fn const_times_square_scales() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let y = var(&mut a, 1);
        let xy = a.push(ExprNode::Mul(smallvec![x, y]));
        let three = a.push(ExprNode::Const(3.0));
        let scaled = a.push(ExprNode::Mul(smallvec![three, xy]));
        let q = extract_quadratic(&a, scaled).unwrap();
        assert_eq!(q.hessian, vec![(v(1), v(0), 3.0)]);
    }
}
