use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::arena::{ExprArena, ExprId, ExprNode, UnaryOp, VarId};

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
    fn add_monomial(&mut self, term: Monomial) {
        match term.degree {
            0 => self.constant += term.coefficient,
            1 => {
                self.constant += 0.0;
                *self.linear.entry(term.variables[0]).or_insert(0.0) += term.coefficient;
            }
            2 => {
                self.constant += 0.0;
                *self.quad.entry(pair(term.variables[0], term.variables[1])).or_insert(0.0) +=
                    term.coefficient;

                // Preserve the general product's zero linear entries for a
                // surrounding nonfinite scale.
                self.linear.entry(term.variables[0]).or_insert(0.0);
                self.linear.entry(term.variables[1]).or_insert(0.0);
            }
            _ => unreachable!("bounded monomial degree"),
        }
    }

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

/// Expand a square with one hash insertion per unordered pair.
fn square_linear(p: &Poly) -> Poly {
    let n = p.linear.len();
    let mut out = Poly {
        constant: p.constant * p.constant,
        linear: FxHashMap::with_capacity_and_hasher(n, FxBuildHasher),
        quad: FxHashMap::with_capacity_and_hasher(n * (n + 1) / 2, FxBuildHasher),
    };

    let terms: Vec<_> = p.linear.iter().map(|(&v, &c)| (v, c)).collect();
    for (i, &(vi, ci)) in terms.iter().enumerate() {
        let linear = p.constant * ci;
        out.linear.insert(vi, (0.0 + linear) + linear);
        out.quad.insert((vi, vi), 0.0 + ci * ci);
        for &(vj, cj) in &terms[i + 1..] {
            let product = ci * cj;
            out.quad.insert(pair(vi, vj), (0.0 + product) + product);
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
        ExprNode::Unary(UnaryOp::Neg, inner) => recursive_poly(arena, *inner).map(Poly::neg),
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
            if !e.is_finite() || (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
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
        | ExprNode::Atan2(_, _)
        | ExprNode::Min(_)
        | ExprNode::Max(_)
        | ExprNode::Unary(_, _) => None,
    }
}

struct PolyFolder<'a> {
    arena: &'a ExprArena,
    root: ExprId,
    flatten_sums: &'a std::cell::Cell<Option<bool>>,
    sum_magnitude: &'a std::cell::Cell<f64>,
}

/// A bounded probe for a single term.
#[derive(Clone, Copy)]
struct Monomial {
    variables: [VarId; 2],
    degree: usize,
    coefficient: f64,
}

const MONOMIAL_VISITS: usize = 64;

fn monomial(arena: &ExprArena, id: ExprId) -> Option<Monomial> {
    let mut remaining = MONOMIAL_VISITS;
    monomial_with_budget(arena, id, 0, &mut remaining)
}

#[expect(clippy::float_cmp, reason = "only exact squares and zero constant terms qualify")]
fn monomial_with_budget(
    arena: &ExprArena,
    id: ExprId,
    depth: usize,
    remaining: &mut usize,
) -> Option<Monomial> {
    if depth == 8 {
        return None;
    }

    // Depth alone cannot bound repeated visits to shared, wide products.
    *remaining = remaining.checked_sub(1)?;
    let constant = |coefficient| Monomial { variables: [VarId(0); 2], degree: 0, coefficient };
    let term = match arena.get(id) {
        ExprNode::Const(c) => constant(*c),
        ExprNode::Param(p) => constant(arena.param_value(*p)),
        ExprNode::Var(v) => Monomial { variables: [*v, *v], degree: 1, coefficient: 1.0 },
        ExprNode::Linear { coeffs, constant: c } if coeffs.is_empty() => constant(*c),
        ExprNode::Linear { coeffs, constant: c } if coeffs.len() == 1 && *c == 0.0 => {
            Monomial { variables: [coeffs[0].0; 2], degree: 1, coefficient: 0.0 + coeffs[0].1 }
        }
        ExprNode::Unary(UnaryOp::Neg, child) => {
            let mut term = monomial_with_budget(arena, *child, depth + 1, remaining)?;
            term.coefficient *= -1.0;
            term
        }
        ExprNode::Mul(children) => {
            let mut acc = constant(1.0);
            for &child in children {
                let term = monomial_with_budget(arena, child, depth + 1, remaining)?;
                if acc.degree + term.degree > 2 {
                    return None;
                }

                let coefficient = if acc.degree == 0 || term.degree == 0 {
                    acc.coefficient * term.coefficient
                } else {
                    0.0 + acc.coefficient * term.coefficient
                };

                for &v in &term.variables[..term.degree] {
                    acc.variables[acc.degree] = v;
                    acc.degree += 1;
                }
                acc.coefficient = coefficient;
                if !coefficient.is_finite() {
                    return None;
                }
            }

            acc
        }
        ExprNode::Pow(base, exponent) if matches!(arena.get(*exponent), ExprNode::Const(e) if *e == 2.0) =>
        {
            let mut term = monomial_with_budget(arena, *base, depth + 1, remaining)?;
            if term.degree > 1 {
                return None;
            }

            term.variables[1] = term.variables[0];
            term.degree *= 2;
            term.coefficient = 0.0 + term.coefficient * term.coefficient;
            term
        }
        _ => return None,
    };

    term.coefficient.is_finite().then_some(term)
}

fn flat_monomials(arena: &ExprArena, children: &[ExprId]) -> Option<Poly> {
    if !children.first().into_iter().chain(children.last()).any(|&id| {
        matches!(
            arena.get(id),
            ExprNode::Mul(_) | ExprNode::Pow(_, _) | ExprNode::Unary(UnaryOp::Neg, _)
        )
    }) {
        return None;
    }

    monomial(arena, *children.first()?)?;
    monomial(arena, *children.last()?)?;
    let mut acc = Poly {
        quad: FxHashMap::with_capacity_and_hasher(children.len(), FxBuildHasher),
        ..Poly::default()
    };
    for &child in children {
        let term = monomial(arena, child)?;
        acc.add_monomial(term);
    }

    Some(acc)
}

#[inline(never)]
fn flat_poly(arena: &ExprArena, children: &[ExprId]) -> Option<Poly> {
    let terminal = |id| match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) => true,
        ExprNode::Linear { coeffs, .. } => coeffs.len() <= 1,
        _ => false,
    };
    if children.first().is_some_and(|&id| !terminal(id))
        || children.last().is_some_and(|&id| !terminal(id))
    {
        return None;
    }
    let mut acc = Poly::default();
    for &child in children {
        match arena.get(child) {
            ExprNode::Var(var) => {
                *acc.linear.entry(*var).or_insert(0.0) += 1.0;
                acc.constant += 0.0;
            }
            ExprNode::Const(value) => acc.constant += value,
            ExprNode::Param(param) => acc.constant += arena.param_value(*param),
            ExprNode::Linear { coeffs, constant } if coeffs.len() <= 1 => {
                for &(var, coefficient) in coeffs {
                    // Poly's ordinary leaf path first adds the coefficient
                    // to +0; preserve that IEEE sign behavior here too.
                    *acc.linear.entry(var).or_insert(0.0) += 0.0 + coefficient;
                }
                acc.constant += constant;
            }
            _ => return None,
        }
    }
    Some(acc)
}
enum PolyOp {
    Add,
    Mul,
    Neg,
    Identity,
    Square,
}
struct PolyState<'a> {
    rest: &'a [ExprId],
    pending: crate::fold::SumFrames<'a>,
    value: Option<Poly>,
    op: PolyOp,
}

impl<'a> crate::fold::Folder for PolyFolder<'a> {
    type Value = Poly;
    type State = PolyState<'a>;

    fn start(&self, id: ExprId) -> std::ops::ControlFlow<Option<Poly>, Self::State> {
        use std::ops::ControlFlow::{Break, Continue};
        let (rest, op, value) = match self.arena.get(id) {
            ExprNode::Const(c) => return Break(Some(Poly::constant(*c))),
            ExprNode::Param(p) => return Break(Some(Poly::constant(self.arena.param_value(*p)))),
            ExprNode::Var(v) => return Break(Some(Poly::var(*v))),
            ExprNode::Linear { coeffs, constant } => {
                let mut linear = FxHashMap::with_capacity_and_hasher(coeffs.len(), FxBuildHasher);
                for &(v, c) in coeffs {
                    *linear.entry(v).or_insert(0.0) += c;
                }
                return Break(Some(Poly { linear, constant: *constant, ..Poly::default() }));
            }
            ExprNode::Add(c) => {
                if let Some(poly) = flat_monomials(self.arena, c) {
                    return Break(Some(poly));
                }
                (c.as_slice(), PolyOp::Add, Poly::default())
            }
            ExprNode::Mul(c) if c.len() == 2 && c[0] == c[1] => {
                (&c[..1], PolyOp::Square, Poly::default())
            }
            ExprNode::Mul(c) => (c.as_slice(), PolyOp::Mul, Poly::constant(1.0)),
            ExprNode::Unary(UnaryOp::Neg, c) => {
                (std::slice::from_ref(c), PolyOp::Neg, Poly::default())
            }
            ExprNode::Pow(base, exp) => {
                let ExprNode::Const(e) = self.arena.get(*exp) else { return Break(None) };
                if !e.is_finite() || (*e - e.round()).abs() >= f64::EPSILON || *e < 0.0 {
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
        Continue(PolyState { rest, pending: None, op, value: Some(value) })
    }

    fn next(&self, state: &mut Self::State) -> std::ops::ControlFlow<Option<Poly>, ExprId> {
        use std::ops::ControlFlow::{Break, Continue};
        if state.value.is_none() {
            return Break(None);
        }
        loop {
            let Some((&child, tail)) = state.rest.split_first() else {
                if let Some((next, previous_constant)) =
                    state.pending.as_mut().and_then(|frames| frames.pop())
                {
                    let acc = state.value.as_mut().expect("checked above");
                    acc.constant += previous_constant;
                    state.rest = next;
                    continue;
                }
                break;
            };
            state.rest = tail;
            if matches!(state.op, PolyOp::Add) {
                let acc = state.value.as_mut().expect("checked above");
                // Avoid creating a one-entry hash table for every variable in a sum.
                match self.arena.get(child) {
                    ExprNode::Var(v) => {
                        acc.constant += 0.0;
                        *acc.linear.entry(*v).or_insert(0.0) += 1.0;
                        crate::fold::record_sum_magnitude(self.sum_magnitude, 1.0);
                        continue;
                    }
                    ExprNode::Const(c) => {
                        acc.constant += c;
                        crate::fold::record_sum_magnitude(self.sum_magnitude, *c);
                        continue;
                    }
                    ExprNode::Param(p) => {
                        let value = self.arena.param_value(*p);
                        acc.constant += value;
                        crate::fold::record_sum_magnitude(self.sum_magnitude, value);
                        continue;
                    }
                    _ => {}
                }

                // Nested sums may be inlined by the DAG folder before start()
                // sees them, so we accumulate their monomials directly too.
                if let Some(term) = monomial(self.arena, child) {
                    if term.degree == 2 && acc.quad.is_empty() {
                        acc.quad.reserve(tail.len() + 1);
                    }
                    crate::fold::record_sum_magnitude(self.sum_magnitude, term.coefficient);
                    acc.add_monomial(term);
                    continue;
                }
            }

            return Continue(child);
        }

        Break(state.value.take())
    }

    fn accept(&self, state: &mut Self::State, p: Poly) {
        if matches!(state.op, PolyOp::Add) {
            crate::fold::record_sum_magnitude(
                self.sum_magnitude,
                p.constant.abs()
                    + p.linear.values().chain(p.quad.values()).map(|c| c.abs()).sum::<f64>(),
            );
            state.value.as_mut().expect("successful sum").add_assign(p);
            return;
        }
        let acc = state.value.take().expect("failed products terminate before requesting children");
        state.value = match state.op {
            PolyOp::Add => unreachable!("handled above"),
            PolyOp::Mul if acc.is_constant() => Some(p.scale(acc.constant)),
            PolyOp::Mul if p.is_constant() => Some(acc.scale(p.constant)),
            PolyOp::Mul if acc.is_linear() && p.is_linear() => Some(mul_linear(&acc, &p)),
            PolyOp::Neg => Some(p.neg()),
            PolyOp::Identity => Some(p),
            PolyOp::Square if p.is_linear() => Some(square_linear(&p)),
            PolyOp::Mul | PolyOp::Square => None,
        };
    }

    fn inline(&self, state: &mut Self::State, child: ExprId) -> bool {
        if matches!(state.op, PolyOp::Add)
            && let ExprNode::Add(children) = self.arena.get(child)
        {
            let flatten = self.flatten_sums.get().unwrap_or_else(|| {
                let flatten = !crate::linear::bounded_affine(self.arena, &[self.root], true);
                self.flatten_sums.set(Some(flatten));
                flatten
            });
            if !flatten {
                return false;
            }
            let acc = state.value.as_mut().expect("successful sum");
            state.pending.get_or_insert_with(Box::default).push((state.rest, acc.constant));
            acc.constant = 0.0;
            state.rest = children;
            return true;
        }
        false
    }
}

fn as_poly(arena: &ExprArena, mut id: ExprId) -> Option<Poly> {
    let mut negations = 0;
    while let ExprNode::Unary(UnaryOp::Neg, child) = arena.get(id) {
        id = *child;
        negations += 1;
    }
    if negations == 0
        && let ExprNode::Add(children) = arena.get(id)
        && let Some(poly) = flat_poly(arena, children)
    {
        return Some(poly);
    }
    let flatten_sums = std::cell::Cell::new(None);
    let sum_magnitude = std::cell::Cell::new(0.0);
    let mut value = crate::fold::fold(
        arena,
        id,
        PolyFolder { arena, root: id, flatten_sums: &flatten_sums, sum_magnitude: &sum_magnitude },
    )?;
    if flatten_sums.get() == Some(true)
        && (!sum_magnitude.get().is_finite()
            || !value.constant.is_finite()
            || value.linear.values().chain(value.quad.values()).any(|c| !c.is_finite()))
        && crate::fold::finite_inputs(arena, id, true)
    {
        value = crate::fold::fold(
            arena,
            id,
            PolyFolder {
                arena,
                root: id,
                flatten_sums: &std::cell::Cell::new(Some(false)),
                sum_magnitude: &std::cell::Cell::new(f64::INFINITY),
            },
        )?;
    }
    if negations != 0 {
        for coefficient in value.linear.values_mut().chain(value.quad.values_mut()) {
            for _ in 0..negations {
                *coefficient *= -1.0;
            }
        }
        for _ in 0..negations {
            value.constant *= -1.0;
        }
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
    dense_expanded_quadratic(arena, id).or_else(|| as_poly(arena, id).map(canonicalize))
}

fn canonicalize(poly: Poly) -> QuadraticTerms {
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

    QuadraticTerms { hessian, linear, constant: poly.constant }
}

/// Extract canonical quadratic coefficients from a wide sum of quadratic
/// monomials, optionally combined with a leading affine node.
///
/// Accumulate duplicate terms in a dense triangular buffer for variable IDs
/// `0..64`, remove exact zeros, and emit the lower-triangular Hessian, linear
/// coefficients, and constant in canonical order without hashing or sorting.
///
/// Return `None` when the expression shape, variable IDs, or finite accumulation
/// requirements do not fit this path, allowing `extract_quadratic` to fall back
/// to the general polynomial extractor.
fn dense_expanded_quadratic(arena: &ExprArena, root: ExprId) -> Option<QuadraticTerms> {
    const VARIABLES: usize = 64;
    const TRIANGLE: usize = VARIABLES * (VARIABLES + 1) / 2;
    const MIN_TERMS: usize = 128;

    let ExprNode::Add(children) = arena.get(root) else { return None };
    let (affine, terms) = if children.len() >= MIN_TERMS {
        if let ExprNode::Linear { coeffs, constant } = arena.get(children[0]) {
            (Some((coeffs, *constant)), &children[1..])
        } else {
            (None, children.as_slice())
        }
    } else if let [linear, quadratic] = children.as_slice()
        && let ExprNode::Linear { coeffs, constant } = arena.get(*linear)
        && let ExprNode::Add(terms) = arena.get(*quadratic)
        && terms.len() >= MIN_TERMS
    {
        (Some((coeffs, *constant)), terms.as_slice())
    } else {
        return None;
    };

    // Reject common large-variable sums before allocating dense scratch.
    for &endpoint in [terms.first()?, terms.last()?] {
        let term = monomial(arena, endpoint)?;
        if term.degree != 2 || term.variables.iter().any(|v| v.index() >= VARIABLES) {
            return None;
        }
    }
    let mut linear = [0.0_f64; VARIABLES];
    let mut constant = 0.0;

    if let Some((coeffs, offset)) = affine {
        if !offset.is_finite() {
            return None;
        }
        constant += offset;
        for &(v, c) in coeffs {
            let value = linear.get_mut(v.index())?;
            *value += c;
            if !value.is_finite() {
                return None;
            }
        }
    }

    let mut quadratic = vec![0.0_f64; TRIANGLE];
    let mut magnitude = 0.0_f64;
    for &id in terms {
        let term = monomial(arena, id)?;
        if term.degree != 2 {
            return None;
        }
        let (lo, hi) = pair(term.variables[0], term.variables[1]);
        let (lo, hi) = (lo.index(), hi.index());
        if hi >= VARIABLES {
            return None;
        }

        // Column-major triangular order.
        let index = lo * (2 * VARIABLES - lo + 1) / 2 + hi - lo;
        quadratic[index] += term.coefficient;
        magnitude += term.coefficient.abs();
        if !quadratic[index].is_finite() || !magnitude.is_finite() {
            return None;
        }
    }

    let mut hessian = Vec::with_capacity(terms.len().min(TRIANGLE));
    let mut index = 0;
    for col in 0..VARIABLES {
        for row in col..VARIABLES {
            let coefficient = quadratic[index];
            index += 1;
            if coefficient != 0.0 {
                hessian.push((
                    VarId(u32::try_from(row).expect("bounded dense row")),
                    VarId(u32::try_from(col).expect("bounded dense column")),
                    if row == col { 2.0 * coefficient } else { coefficient },
                ));
            }
        }
    }

    let linear = linear
        .into_iter()
        .enumerate()
        .filter(|&(_, coefficient)| coefficient != 0.0)
        .map(|(v, coefficient)| {
            (VarId(u32::try_from(v).expect("bounded dense variable")), coefficient)
        })
        .collect();

    Some(QuadraticTerms { hessian, linear, constant })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_terms(actual: &QuadraticTerms, expected: &QuadraticTerms) {
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        assert!(same(actual.constant, expected.constant), "constant");
        assert_eq!(actual.linear.len(), expected.linear.len());
        assert_eq!(actual.hessian.len(), expected.hessian.len());
        for (&(v, a), &(w, b)) in actual.linear.iter().zip(&expected.linear) {
            assert_eq!(v, w);
            assert!(same(a, b), "linear {v:?}: {a} != {b}");
        }
        for (&(r, c, a), &(s, d, b)) in actual.hessian.iter().zip(&expected.hessian) {
            assert_eq!((r, c), (s, d));
            assert!(same(a, b), "Hessian ({r:?}, {c:?}): {a} != {b}");
        }
    }

    #[test]
    fn dense_expanded_sums_match_general_extraction_across_parameters_and_grouping() {
        let mut arena = ExprArena::new();
        let vars: Vec<_> = (0..50).map(|v| arena.var(VarId(v))).collect();
        let parameter = arena.new_param(1.0);
        let p = arena.param(parameter);
        let mut terms = smallvec::SmallVec::new();
        for i in 0..650 {
            let coefficient = arena.push(ExprNode::Const(if i % 3 == 0 { -1.5 } else { 0.75 }));
            let scaled = arena.push(ExprNode::Mul(smallvec::smallvec![coefficient, p]));
            terms.push(arena.push(ExprNode::Mul(smallvec::smallvec![
                scaled,
                vars[i % 50],
                vars[(i * 7) % 50],
            ])));
        }

        let affine = arena.push(ExprNode::Linear {
            coeffs: vec![(VarId(0), 1e16), (VarId(0), -1e16), (VarId(0), 1.0), (VarId(49), -3.0)],
            constant: -0.0,
        });
        let sum = arena.push(ExprNode::Add(terms.clone()));
        let nested = arena.push(ExprNode::Add(smallvec::smallvec![affine, sum]));
        terms.insert(0, affine);
        let flat = arena.push(ExprNode::Add(terms));
        for value in [1.0, 0.0, -0.0, -1.0, 1e-300, 1e300] {
            arena.set_param_value(parameter, value).unwrap();
            for root in [sum, nested, flat] {
                let expected = canonicalize(as_poly(&arena, root).unwrap());
                let actual = extract_quadratic(&arena, root).unwrap();
                assert_same_terms(&actual, &expected);
                assert!(dense_expanded_quadratic(&arena, root).is_some());
            }
        }

        let product = arena.push(ExprNode::Mul(smallvec::smallvec![vars[49], vars[3]]));
        let zero = arena.push(ExprNode::Const(0.0));
        let padding = arena.push(ExprNode::Mul(smallvec::smallvec![zero, vars[0], vars[0]]));
        let mut terms = vec![padding; 650];
        terms[0] = product;
        let root = arena.push(ExprNode::Add(terms.into()));
        let actual = dense_expanded_quadratic(&arena, root).unwrap();
        assert_eq!(actual.hessian, vec![(VarId(49), VarId(3), 1.0)]);
    }

    #[test]
    fn dense_expanded_sums_fall_back_for_sparse_ids_overflow_and_nonlinear_terms() {
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let far = arena.var(VarId(1_000_000));
        let product = arena.push(ExprNode::Mul(smallvec::smallvec![x, x]));
        let sparse = arena.push(ExprNode::Mul(smallvec::smallvec![x, far]));
        let huge = arena.push(ExprNode::Const(1e308));
        let overflow = arena.push(ExprNode::Mul(smallvec::smallvec![huge, x, x]));
        let nonlinear = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
        let infinity = arena.push(ExprNode::Const(f64::INFINITY));
        let nan = arena.push(ExprNode::Const(f64::NAN));
        let infinite = arena.push(ExprNode::Mul(smallvec::smallvec![infinity, x, x]));
        let not_a_number = arena.push(ExprNode::Mul(smallvec::smallvec![nan, x, x]));

        for middle in [sparse, overflow, nonlinear, infinite, not_a_number] {
            let mut terms: smallvec::SmallVec<_> = vec![product; 130].into();
            terms[64] = middle;
            if middle == overflow {
                terms[65] = overflow;
            }

            let root = arena.push(ExprNode::Add(terms));
            assert!(dense_expanded_quadratic(&arena, root).is_none());

            let expected = as_poly(&arena, root).map(canonicalize);
            let actual = extract_quadratic(&arena, root);
            match (actual, expected) {
                (Some(a), Some(b)) => assert_same_terms(&a, &b),
                (None, None) => {}
                _ => panic!("fallback changed polynomial recognition"),
            }
        }
    }

    #[test]
    fn dense_expanded_sums_preserve_randomized_coefficient_bits_and_triangle_boundaries() {
        let mut seed = 0x6412_9e37_u64;
        for _ in 0..64 {
            let mut arena = ExprArena::new();
            let vars: Vec<_> = (0..64).map(|v| arena.var(VarId(v))).collect();
            let mut terms = smallvec::SmallVec::new();
            for _ in 0..256 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let i = usize::try_from((seed >> 32) % 64).unwrap();
                let j = usize::try_from((seed >> 40) % 64).unwrap();
                let value = [0.0, -0.0, 1.0, -1.0, 1e-300, 1e300]
                    [usize::try_from((seed >> 48) % 6).unwrap()];
                let c = arena.push(ExprNode::Const(value));
                terms.push(arena.push(ExprNode::Mul(smallvec::smallvec![c, vars[i], vars[j]])));
            }
            let root = arena.push(ExprNode::Add(terms));
            let fast = dense_expanded_quadratic(&arena, root).expect("bounded dense sum");
            assert_same_terms(&fast, &canonicalize(as_poly(&arena, root).unwrap()));
        }
    }

    fn assert_same_poly(actual: &Poly, expected: &Poly) {
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        assert!(same(actual.constant, expected.constant), "constant");
        assert_eq!(actual.linear.len(), expected.linear.len(), "linear keys");
        assert_eq!(actual.quad.len(), expected.quad.len(), "quadratic keys");

        for (v, &c) in &expected.linear {
            assert!(same(actual.linear[v], c), "linear {v:?}: {} != {c}", actual.linear[v]);
        }

        for (v, &c) in &expected.quad {
            assert!(same(actual.quad[v], c), "quadratic {v:?}: {} != {c}", actual.quad[v]);
        }
    }

    #[test]
    fn monomial_sums_preserve_grouping_parameters_and_ieee_fallbacks() {
        use smallvec::smallvec;
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let y = arena.var(VarId(1));
        let p = arena.new_param(1.5);
        let parameter = arena.param(p);
        let mut terms = smallvec::SmallVec::new();
        for c in [0.0, -0.0, 1.5, -3.0, 1e-300, 1e300] {
            let coefficient = arena.push(ExprNode::Const(c));
            let cx = arena.push(ExprNode::Mul(smallvec![coefficient, x]));
            terms.push(arena.push(ExprNode::Mul(smallvec![cx, y])));
            let compact =
                arena.push(ExprNode::Linear { coeffs: vec![(VarId(0), c)], constant: 0.0 });
            terms.push(arena.push(ExprNode::Mul(smallvec![compact, y])));
        }

        terms.push(arena.push(ExprNode::Mul(smallvec![parameter, x, x])));
        let sum = arena.push(ExprNode::Add(terms));
        assert!(
            flat_monomials(
                &arena,
                match arena.get(sum) {
                    ExprNode::Add(c) => c,
                    _ => unreachable!(),
                }
            )
            .is_some()
        );

        for c in [2.0, 0.0, -1.0] {
            arena.set_param_value(p, c).unwrap();
            assert_same_poly(&as_poly(&arena, sum).unwrap(), &recursive_poly(&arena, sum).unwrap());
        }
        arena.set_param_value(p, 1.5).unwrap();
        let infinity = arena.push(ExprNode::Const(f64::INFINITY));
        let scaled = arena.push(ExprNode::Mul(smallvec![sum, infinity]));
        assert_same_poly(
            &as_poly(&arena, scaled).unwrap(),
            &recursive_poly(&arena, scaled).unwrap(),
        );

        for c in [f64::INFINITY, f64::NAN] {
            let coefficient = arena.push(ExprNode::Const(c));
            let term = arena.push(ExprNode::Mul(smallvec![coefficient, x, y]));
            let mixed = arena.push(ExprNode::Add(smallvec![term, sum]));
            assert_same_poly(
                &as_poly(&arena, mixed).unwrap(),
                &recursive_poly(&arena, mixed).unwrap(),
            );
        }
        let nonlinear = arena.push(ExprNode::Unary(UnaryOp::Sin, y));
        let mixed = arena.push(ExprNode::Add(smallvec![sum, nonlinear]));
        assert!(as_poly(&arena, mixed).is_none());
    }

    #[test]
    fn affine_squares_match_ordered_products_at_float_extremes() {
        for constant in [0.0, -0.0, 1.5, 1e308, f64::INFINITY, f64::NAN] {
            let mut p = Poly::constant(constant);
            for (i, coefficient) in
                [0.0, -0.0, 1e-308, -1e-154, 1.0, 1e154, 1e308].into_iter().enumerate()
            {
                p.linear.insert(VarId(u32::try_from(i).unwrap()), coefficient);
            }
            assert_same_poly(&square_linear(&p), &mul_linear(&p, &p));
        }
    }

    #[test]
    fn deep_monomials_use_the_stack_safe_fallback() {
        use smallvec::smallvec;
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let one = arena.push(ExprNode::Const(1.0));
        let mut term = arena.push(ExprNode::Mul(smallvec![x, x]));
        for _ in 0..10_000 {
            term = arena.push(ExprNode::Mul(smallvec![one, term]));
        }

        let root = arena.push(ExprNode::Add(smallvec![term, x]));
        assert!(monomial(&arena, term).is_none());

        let result = extract_quadratic(&arena, root).unwrap();
        assert_eq!(result.hessian, vec![(VarId(0), VarId(0), 2.0)]);
        assert_eq!(result.linear, vec![(VarId(0), 1.0)]);
    }

    #[test]
    fn shared_monomial_probes_fall_back_after_a_bounded_number_of_visits() {
        for fanout in [2, 4, 8, 32] {
            let mut arena = ExprArena::new();
            let parameter = arena.new_param(1.0);
            let mut term = arena.param(parameter);
            for _ in 0..7 {
                term = arena.push(ExprNode::Mul(vec![term; fanout].into()));
            }

            let x = arena.var(VarId(0));
            let children = [term, x];
            let root = arena.push(ExprNode::Add(children.as_slice().into()));
            let mut remaining = MONOMIAL_VISITS;
            assert!(monomial_with_budget(&arena, term, 0, &mut remaining).is_none());
            assert_eq!(remaining, 0, "the visit budget must stop the probe");
            assert!(flat_monomials(&arena, &children).is_none());

            for value in [-1.0, 0.0, 1.0] {
                arena.set_param_value(parameter, value).unwrap();
                let result = extract_quadratic(&arena, root).unwrap();
                assert_eq!(result.hessian, Vec::<(VarId, VarId, f64)>::new());
                assert_eq!(result.linear, vec![(VarId(0), 1.0)]);
                assert_eq!(result.constant, value.abs());
            }
        }
    }

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
        assert_eq!(q.linear, Vec::<(VarId, f64)>::new());
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
        assert_eq!(q.linear, Vec::<(VarId, f64)>::new());
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
        assert_eq!(q.hessian, Vec::<(VarId, VarId, f64)>::new());
        assert_eq!(q.linear, vec![(v(0), 3.0)]);
        assert!((q.constant - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn constant_only() {
        let mut a = ExprArena::new();
        let c = a.push(ExprNode::Const(7.0));
        let q = extract_quadratic(&a, c).unwrap();
        assert_eq!(q.hessian, Vec::<(VarId, VarId, f64)>::new());
        assert_eq!(q.linear, Vec::<(VarId, f64)>::new());
        assert!((q.constant - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn negation_flips_signs() {
        let mut a = ExprArena::new();
        let x = var(&mut a, 0);
        let two = a.push(ExprNode::Const(2.0));
        let sq = a.push(ExprNode::Pow(x, two));
        let inner = a.push(ExprNode::Add(smallvec![sq, x]));
        let neg = a.push(ExprNode::Unary(UnaryOp::Neg, inner));
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
        let s = a.push(ExprNode::Unary(UnaryOp::Sin, x));
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
