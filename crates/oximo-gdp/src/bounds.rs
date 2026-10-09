//! Domain validation and outward-rounded bounds over global and conditional boxes.

use oximo_core::gdp::BooleanId;
use oximo_core::{AlgebraicConstraint, Domain, Variable};
use oximo_expr::{ExprArena, ExprId, ExprNode, LinearTerms, UnaryOp, extract_linear};
use rustc_hash::{FxHashMap, FxHashSet};
use std::f64::consts::{FRAC_PI_2, PI};
use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug)]
pub(crate) enum BoundError {
    Domain(&'static str),
    InvalidNumber,
}

impl Bounds {
    const UNKNOWN: Self = Self { lower: f64::NEG_INFINITY, upper: f64::INFINITY };
    fn point(x: f64) -> Self {
        Self { lower: x, upper: x }
    }

    fn new(lower: f64, upper: f64) -> Self {
        if lower.is_nan() || upper.is_nan() { Self::UNKNOWN } else { Self { lower, upper } }
    }

    fn widened(lower: f64, upper: f64) -> Self {
        Self::new(down(lower), up(upper))
    }

    fn contains_zero(self) -> bool {
        self.lower <= 0.0 && self.upper >= 0.0
    }

    #[expect(clippy::float_cmp, reason = "equal finite endpoints cancel exactly")]
    fn add(self, rhs: Self) -> Self {
        if self.lower == 0.0 && self.upper == 0.0 {
            return rhs;
        }

        if rhs.lower == 0.0 && rhs.upper == 0.0 {
            return self;
        }

        // Zero operands and cancellation of equal finite endpoints are exact.
        // Widening those zeros would create a negative square-root domain.
        let endpoint = |a: f64, b: f64, round: fn(f64) -> f64| {
            if a == 0.0 {
                b
            } else if b == 0.0 {
                a
            } else if a.is_finite() && a == -b {
                0.0
            } else {
                round(a + b)
            }
        };
        Self::new(endpoint(self.lower, rhs.lower, down), endpoint(self.upper, rhs.upper, up))
    }

    fn divide_scalar(self, divisor: f64) -> Self {
        let endpoint = |value: f64, round: fn(f64) -> f64| {
            // Only an exactly zero numerator can bypass outward rounding.
            if value == 0.0 { 0.0 } else { round(value / divisor) }
        };

        if divisor > 0.0 {
            Self::new(endpoint(self.lower, down), endpoint(self.upper, up))
        } else {
            Self::new(endpoint(self.upper, down), endpoint(self.lower, up))
        }
    }

    fn mul(self, rhs: Self) -> Self {
        if (self.lower == 0.0 && self.upper == 0.0) || (rhs.lower == 0.0 && rhs.upper == 0.0) {
            return Self::point(0.0);
        }

        let product = |a: f64, b: f64| if a == 0.0 || b == 0.0 { 0.0 } else { a * b };
        let vals = [
            product(self.lower, rhs.lower),
            product(self.lower, rhs.upper),
            product(self.upper, rhs.lower),
            product(self.upper, rhs.upper),
        ];
        let mut bounds = Self::widened(
            vals.into_iter().fold(f64::INFINITY, f64::min),
            vals.into_iter().fold(f64::NEG_INFINITY, f64::max),
        );

        if (self.lower >= 0.0 && rhs.lower >= 0.0) || (self.upper <= 0.0 && rhs.upper <= 0.0) {
            bounds.lower = bounds.lower.max(0.0);
        }

        if (self.lower >= 0.0 && rhs.upper <= 0.0) || (self.upper <= 0.0 && rhs.lower >= 0.0) {
            bounds.upper = bounds.upper.min(0.0);
        }
        bounds
    }

    fn neg(self) -> Self {
        Self::new(-self.upper, -self.lower)
    }
}

fn down(x: f64) -> f64 {
    if x.is_finite() { x.next_down() } else { x }
}

fn up(x: f64) -> f64 {
    if x.is_finite() { x.next_up() } else { x }
}

// Transcendental functions use a numerical guard in addition to directed
// endpoint widening. The std library does not promise correctly rounded
// transcendental results. These are some numerical bounds.
fn libm_bounds(a: f64, b: f64) -> Bounds {
    let margin = |x: f64| 64.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE);
    Bounds::new(
        if a.is_finite() { down(a - margin(a)) } else { a },
        if b.is_finite() { up(b + margin(b)) } else { b },
    )
}

fn require(condition: bool, op: &'static str) -> Result<(), BoundError> {
    if condition { Ok(()) } else { Err(BoundError::Domain(op)) }
}

#[expect(clippy::float_cmp, reason = "only exactly equal endpoints represent a constant exponent")]
fn pow(base: Bounds, exponent: Bounds) -> Result<Bounds, BoundError> {
    if exponent.lower == exponent.upper {
        let p = exponent.lower;

        if p == 0.0 {
            return Ok(Bounds::point(1.0));
        }

        if p.fract() == 0.0 {
            if p < 0.0 {
                require(!base.contains_zero(), "negative power")?;
            }

            let a = base.lower.powf(p);
            let b = base.upper.powf(p);
            let lower = if p > 0.0 && p.rem_euclid(2.0) == 0.0 && base.contains_zero() {
                0.0
            } else {
                a.min(b)
            };

            let mut result = libm_bounds(lower, a.max(b));

            if p.rem_euclid(2.0) == 0.0 {
                result.lower = result.lower.max(0.0);
            }

            return Ok(result);
        }
        require(if p > 0.0 { base.lower >= 0.0 } else { base.lower > 0.0 }, "fractional power")?;
        let a = base.lower.powf(p);
        let b = base.upper.powf(p);
        let mut result = libm_bounds(a.min(b), a.max(b));
        result.lower = result.lower.max(0.0);
        return Ok(result);
    }
    require(base.lower > 0.0, "variable exponent power")?;
    unary(UnaryOp::Exp, unary(UnaryOp::Log, base)?.mul(exponent))
}

fn unary(op: UnaryOp, b: Bounds) -> Result<Bounds, BoundError> {
    use UnaryOp as U;
    let (lo, hi) = (b.lower, b.upper);
    let mut result = match op {
        U::Neg => return Ok(b.neg()),
        U::Abs => {
            let min = if b.contains_zero() { 0.0 } else { lo.abs().min(hi.abs()) };
            return Ok(Bounds::new(min, lo.abs().max(hi.abs())));
        }
        U::Sqrt => {
            require(lo >= 0.0, "sqrt")?;
            libm_bounds(lo.sqrt(), hi.sqrt())
        }
        U::Cbrt => libm_bounds(lo.cbrt(), hi.cbrt()),
        U::Exp => libm_bounds(lo.exp(), hi.exp()),
        U::Exp2 => libm_bounds(lo.exp2(), hi.exp2()),
        U::Expm1 => libm_bounds(lo.exp_m1(), hi.exp_m1()),
        U::Log | U::Log2 | U::Log10 => {
            require(lo > 0.0, op.name())?;
            libm_bounds(op.apply(lo), op.apply(hi))
        }
        U::Log1p => {
            require(lo > -1.0, "log1p")?;
            libm_bounds(lo.ln_1p(), hi.ln_1p())
        }

        // A full-period enclosure avoids relying on rounded critical-point tests.
        U::Sin | U::Cos => return Ok(Bounds::new(-1.0, 1.0)),
        U::Tan => {
            require(lo.is_finite() && hi.is_finite() && lo.abs().max(hi.abs()) < 1e12, "tan")?;
            let left = down(down(lo - FRAC_PI_2) / PI).ceil();
            let right = up(up(hi - FRAC_PI_2) / PI).floor();
            require(left > right, "tan pole")?;
            libm_bounds(lo.tan(), hi.tan())
        }
        U::Asin => {
            require(lo >= -1.0 && hi <= 1.0, "asin")?;
            libm_bounds(lo.asin(), hi.asin())
        }
        U::Acos => {
            require(lo >= -1.0 && hi <= 1.0, "acos")?;
            libm_bounds(hi.acos(), lo.acos())
        }
        U::Atan => libm_bounds(lo.atan(), hi.atan()),
        U::Sinh => libm_bounds(lo.sinh(), hi.sinh()),
        U::Cosh => libm_bounds(
            if b.contains_zero() { 1.0 } else { lo.cosh().min(hi.cosh()) },
            lo.cosh().max(hi.cosh()),
        ),
        U::Tanh => libm_bounds(lo.tanh(), hi.tanh()),
        U::Asinh => libm_bounds(lo.asinh(), hi.asinh()),
        U::Acosh => {
            require(lo >= 1.0, "acosh")?;
            libm_bounds(lo.acosh(), hi.acosh())
        }
        U::Atanh => {
            require(lo > -1.0 && hi < 1.0, "atanh")?;
            libm_bounds(lo.atanh(), hi.atanh())
        }
    };

    match op {
        U::Sqrt | U::Exp | U::Exp2 | U::Acosh => result.lower = result.lower.max(0.0),
        U::Cosh => result.lower = result.lower.max(1.0),
        U::Tanh => {
            result.lower = result.lower.max(-1.0);
            result.upper = result.upper.min(1.0);
        }
        U::Expm1 => result.lower = result.lower.max(-1.0),
        _ => {}
    }

    Ok(result)
}

enum AffineTerms<'a> {
    Direct(LinearTerms<'a>),
    Shared(Arc<LinearTerms<'static>>),
}

impl<'a> Deref for AffineTerms<'a> {
    type Target = LinearTerms<'a>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Direct(terms) => terms,
            Self::Shared(terms) => terms,
        }
    }
}

// Limit work on global rows.
fn global_affine_terms<'a>(
    arena: &'a ExprArena,
    root: ExprId,
    budget: &mut usize,
    stack: &mut Vec<ExprId>,
) -> Option<LinearTerms<'a>> {
    stack.clear();
    stack.push(root);
    let mut nodes = 0;
    let mut eligible = true;

    while let Some(id) = stack.pop() {
        nodes += 1;

        if nodes > 64 || nodes > *budget {
            eligible = false;
            break;
        }

        match arena.get(id) {
            ExprNode::Add(children) | ExprNode::Mul(children) if children.len() <= 32 => {
                stack.extend(children);
            }
            ExprNode::Unary(UnaryOp::Neg, child) => stack.push(*child),
            ExprNode::Linear { coeffs, .. } if coeffs.len() <= 32 => {}
            ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) => {}
            _ => {
                eligible = false;
                break;
            }
        }
    }
    *budget = budget.saturating_sub(nodes);

    if !eligible {
        return None;
    }

    let mut terms = extract_linear(arena, root)?;

    if !terms.constant.is_finite()
        || terms.coeffs.len() > 32
        || terms.coeffs.iter().any(|(_, c)| !c.is_finite())
    {
        return None;
    }
    *budget = budget.saturating_sub(terms.coeffs.len());
    let coefficients = terms.coeffs.to_mut();
    let mut length = 0;

    for i in 0..coefficients.len() {
        let (variable, coefficient) = coefficients[i];

        if let Some((_, combined)) = coefficients[..length].iter_mut().find(|(v, _)| *v == variable)
        {
            *combined += coefficient;

            if !combined.is_finite() {
                return None;
            }
        } else {
            coefficients[length] = (variable, coefficient);
            length += 1;
        }
    }
    coefficients.truncate(length);
    coefficients.retain(|(_, c)| *c != 0.0);
    (!terms.coeffs.is_empty()).then_some(terms)
}

/// A single domain-validated expression at the current parameter snapshot.
/// Affine coefficients may be materialized only together with their bounds.
pub(crate) struct ExpressionAnalysis {
    pub bounds: Bounds,
    pub affine: Option<LinearTerms<'static>>,
}

pub(crate) struct Bounder<'a> {
    arena: &'a ExprArena,
    variables: &'a [Variable],
    cache: IntervalCache,
    // Forward intervals stay separate from final affine refinements so a
    // previously queried root does not change subsequent domain validation.
    completed: FxHashMap<ExprId, Bounds>,
    stack: Vec<(ExprId, bool)>,
    affine_seen: FxHashSet<ExprId>,
    affine_cache: FxHashMap<ExprId, Option<Arc<LinearTerms<'static>>>>,
    affine_cache_coefficients: usize,
    affine_coefficients: Vec<(oximo_expr::VarId, f64)>,
    affine_slots: FxHashMap<oximo_expr::VarId, usize>,
    tightened: FxHashMap<oximo_expr::VarId, Bounds>,
    supporting_roots: FxHashSet<ExprId>,
    scope: Option<(&'a BoundScopes, Option<BooleanId>)>,
}

/// Reusable propagation storage contains owned coefficients.
#[derive(Default)]
pub(crate) struct PropagationScratch {
    rows: Vec<(ExprId, f64, f64, LinearTerms<'static>)>,
    supporting_roots: FxHashSet<ExprId>,
    stack: Vec<ExprId>,
}

/// Sparse bound changes for each ancestor context.
pub(crate) struct BoundScopes {
    global: FxHashMap<oximo_expr::VarId, Bounds>,
    scopes: Vec<Option<BoundScope>>,
}

struct BoundScope {
    parent: Option<BooleanId>,
    bounds: FxHashMap<oximo_expr::VarId, Bounds>,
}

impl BoundScopes {
    pub fn new(global: &Bounder<'_>, booleans: usize) -> Self {
        Self {
            global: global.tightened.clone(),
            scopes: std::iter::repeat_with(|| None).take(booleans).collect(),
        }
    }

    pub fn contains(&self, id: BooleanId) -> bool {
        self.scopes[id.index()].is_some()
    }

    pub fn insert(
        &mut self,
        id: BooleanId,
        parent: Option<BooleanId>,
        bounds: FxHashMap<oximo_expr::VarId, Bounds>,
    ) {
        self.scopes[id.index()] = Some(BoundScope { parent, bounds });
    }

    fn variable(&self, mut scope: Option<BooleanId>, id: oximo_expr::VarId) -> Option<Bounds> {
        while let Some(indicator) = scope {
            let context = self.scopes[indicator.index()].as_ref().expect("ancestor context");

            if let Some(&bounds) = context.bounds.get(&id) {
                return Some(bounds);
            }
            scope = context.parent;
        }

        self.global.get(&id).copied()
    }
}

enum IntervalCache {
    Dense(Vec<Option<Bounds>>),
    Sparse { entries: FxHashMap<ExprId, Bounds>, arena_len: usize },
}

impl IntervalCache {
    fn clear(&mut self) {
        match self {
            Self::Dense(entries) => entries.fill(None),
            Self::Sparse { entries, .. } => entries.clear(),
        }
    }

    fn new(arena_len: usize) -> Self {
        if arena_len <= 4096 {
            Self::Dense(vec![None; arena_len])
        } else {
            Self::Sparse { entries: FxHashMap::default(), arena_len }
        }
    }

    fn get(&self, id: ExprId) -> Option<Bounds> {
        match self {
            Self::Dense(entries) => entries[id.index()],
            Self::Sparse { entries, .. } => entries.get(&id).copied(),
        }
    }

    fn insert(&mut self, id: ExprId, bounds: Bounds) {
        match self {
            Self::Dense(entries) => entries[id.index()] = Some(bounds),
            Self::Sparse { entries, arena_len } => {
                entries.insert(id, bounds);

                // Dense indexing wins once a substantial part of the arena is
                // reachable.
                if entries.len() > *arena_len / 8 {
                    let mut dense = vec![None; *arena_len];

                    for (&id, &bounds) in entries.iter() {
                        dense[id.index()] = Some(bounds);
                    }
                    *self = Self::Dense(dense);
                }
            }
        }
    }
}

impl<'a> Bounder<'a> {
    pub fn new(arena: &'a ExprArena, variables: &'a [Variable]) -> Self {
        Self::with_scope(arena, variables, None)
    }

    fn with_scope(
        arena: &'a ExprArena,
        variables: &'a [Variable],
        scope: Option<(&'a BoundScopes, Option<BooleanId>)>,
    ) -> Self {
        Self {
            arena,
            variables,
            cache: if scope.is_some() {
                IntervalCache::Sparse { entries: FxHashMap::default(), arena_len: arena.len() }
            } else {
                IntervalCache::new(arena.len())
            },
            completed: FxHashMap::default(),
            stack: Vec::new(),
            affine_seen: FxHashSet::default(),
            affine_cache: FxHashMap::default(),
            affine_cache_coefficients: 0,
            affine_coefficients: Vec::new(),
            affine_slots: FxHashMap::default(),
            tightened: FxHashMap::default(),
            supporting_roots: FxHashSet::default(),
            scope,
        }
    }

    pub fn arena(&self) -> &'a ExprArena {
        self.arena
    }

    pub fn scoped<'b>(&self, scopes: &'b BoundScopes, parent: Option<BooleanId>) -> Bounder<'b>
    where
        'a: 'b,
    {
        Bounder::with_scope(self.arena, self.variables, Some((scopes, parent)))
    }

    pub fn set_scope(&mut self, parent: BooleanId) {
        let (scopes, previous) = self.scope.expect("scoped bounder");

        if previous != Some(parent) {
            self.scope = Some((scopes, Some(parent)));
            self.cache.clear();
            self.completed.clear();
        }
    }

    pub fn assume_true(&mut self, binary: oximo_expr::VarId) {
        self.tightened.insert(binary, Bounds::point(1.0));
    }

    pub fn scoped_reusing<'b>(
        &self,
        scopes: &'b BoundScopes,
        parent: Option<BooleanId>,
        scratch: &mut PropagationScratch,
    ) -> Bounder<'b>
    where
        'a: 'b,
    {
        scratch.rows.clear();
        scratch.supporting_roots.clear();
        let mut local = self.scoped(scopes, parent);
        local.supporting_roots = std::mem::take(&mut scratch.supporting_roots);
        local
    }

    pub fn into_context(
        self,
        scratch: &mut PropagationScratch,
    ) -> FxHashMap<oximo_expr::VarId, Bounds> {
        scratch.supporting_roots = self.supporting_roots;
        self.tightened
    }

    fn variable(&self, id: oximo_expr::VarId) -> Result<Bounds, BoundError> {
        if let Some(&bounds) = self.tightened.get(&id) {
            return Ok(bounds);
        }

        if let Some((scopes, parent)) = self.scope
            && let Some(bounds) = scopes.variable(parent, id)
        {
            return Ok(bounds);
        }

        let v = self.variables.get(id.index()).ok_or(BoundError::InvalidNumber)?;
        let (lo, hi) = match v.domain {
            Domain::SemiContinuous { threshold } | Domain::SemiInteger { threshold } => {
                (threshold.min(0.0), v.ub.max(0.0))
            }
            _ => (v.lb, v.ub),
        };

        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(BoundError::InvalidNumber);
        }

        Ok(Bounds::new(lo, hi))
    }

    /// A bounded affine propagation pass over active unconditional rows.
    /// Only rows connected to a pending GDP source are used. All supporting
    /// expressions must be locked together with the original source.
    pub fn tighten_global(
        &mut self,
        constraints: &[AlgebraicConstraint],
        relevant: &FxHashSet<oximo_expr::VarId>,
    ) {
        self.tighten_rows(
            constraints.iter().filter(|row| row.active).map(|row| (row.lhs, row.lower, row.upper)),
            Some(relevant),
        );
    }

    pub fn tighten_rows(
        &mut self,
        constraints: impl IntoIterator<Item = (ExprId, f64, f64)>,
        relevant: Option<&FxHashSet<oximo_expr::VarId>>,
    ) {
        let mut scratch = PropagationScratch::default();
        self.tighten_rows_reusing(constraints, relevant, &mut scratch);
    }

    pub fn tighten_rows_reusing(
        &mut self,
        constraints: impl IntoIterator<Item = (ExprId, f64, f64)>,
        relevant: Option<&FxHashSet<oximo_expr::VarId>>,
        scratch: &mut PropagationScratch,
    ) {
        let mut budget = 50_000_usize;
        scratch.rows.clear();

        for (lhs, lower, upper) in constraints {
            if !lower.is_finite() && !upper.is_finite() {
                continue;
            }

            if budget == 0 {
                break;
            }

            if let Some(terms) =
                global_affine_terms(self.arena, lhs, &mut budget, &mut scratch.stack)
            {
                scratch.rows.push((lhs, lower, upper, terms.into_owned()));
            }
        }

        let all_rows = relevant.is_none();
        let mut relevant = relevant.cloned().unwrap_or_default();

        for _ in 0..4 {
            let mut changed = false;

            for &(lhs, lower_bound, upper_bound, ref terms) in &scratch.rows {
                if !all_rows && !terms.coeffs.iter().any(|(v, _)| relevant.contains(v)) {
                    continue;
                }

                if !all_rows {
                    let previous = relevant.len();
                    relevant.extend(terms.coeffs.iter().map(|(v, _)| *v));
                    changed |= relevant.len() != previous;
                }

                for &(variable, coefficient) in terms.coeffs.iter() {
                    let mut rest = Bounds::point(terms.constant);
                    let mut valid = true;

                    for &(other, c) in terms.coeffs.iter() {
                        if other != variable {
                            let Ok(bounds) = self.variable(other) else {
                                valid = false;
                                break;
                            };
                            rest = rest.add(bounds.mul(Bounds::point(c)));
                        }
                    }

                    if !valid {
                        continue;
                    }

                    let numerator = Bounds::new(lower_bound, upper_bound).add(rest.neg());
                    let candidate = numerator.divide_scalar(coefficient);
                    let Ok(current) = self.variable(variable) else { continue };
                    let lower = current.lower.max(candidate.lower);
                    let upper = current.upper.min(candidate.upper);

                    // Contradictory global rows remain the solver's responsibility.
                    if lower <= upper && (lower > current.lower || upper < current.upper) {
                        self.tightened.insert(variable, Bounds::new(lower, upper));
                        self.supporting_roots.insert(lhs);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }
        scratch.rows.clear();
    }

    pub fn supporting_roots(&self) -> &FxHashSet<ExprId> {
        &self.supporting_roots
    }

    #[cfg(test)]
    fn bound(&mut self, root: ExprId) -> Result<Bounds, BoundError> {
        self.analyze(root).map(|analysis| analysis.bounds)
    }

    pub fn analyze(&mut self, root: ExprId) -> Result<ExpressionAnalysis, BoundError> {
        if let Some(&bounds) = self.completed.get(&root) {
            let affine = self.affine_terms(root).map(|terms| terms.deref().clone().into_owned());
            return Ok(ExpressionAnalysis { bounds, affine });
        }

        self.stack.clear();
        self.stack.push((root, false));

        while let Some((id, visited)) = self.stack.pop() {
            if self.cache.get(id).is_some() {
                continue;
            }

            let node = self.arena.get(id);

            if !visited {
                self.stack.push((id, true));

                match node {
                    ExprNode::Add(cs)
                    | ExprNode::Mul(cs)
                    | ExprNode::Min(cs)
                    | ExprNode::Max(cs) => self.stack.extend(cs.iter().map(|&c| (c, false))),
                    ExprNode::Unary(_, c) => self.stack.push((*c, false)),
                    ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                        self.stack.push((*a, false));
                        self.stack.push((*b, false));
                    }
                    _ => {}
                }
                continue;
            }

            let child = |id: ExprId| self.cache.get(id).expect("children visited");
            let bounds = match node {
                ExprNode::Const(x) => {
                    if !x.is_finite() {
                        return Err(BoundError::InvalidNumber);
                    }
                    Bounds::point(*x)
                }
                ExprNode::Param(p) => {
                    let x = self.arena.param_value(*p);

                    if !x.is_finite() {
                        return Err(BoundError::InvalidNumber);
                    }
                    Bounds::point(x)
                }
                ExprNode::Var(v) => self.variable(*v)?,
                ExprNode::Linear { coeffs, constant } => {
                    if !constant.is_finite() {
                        return Err(BoundError::InvalidNumber);
                    }

                    let mut b = Bounds::point(*constant);

                    for &(v, c) in coeffs {
                        if !c.is_finite() {
                            return Err(BoundError::InvalidNumber);
                        }

                        if c != 0.0 {
                            b = b.add(self.variable(v)?.mul(Bounds::point(c)));
                        }
                    }
                    b
                }
                ExprNode::Add(cs) => cs.iter().fold(Bounds::point(0.0), |b, &c| b.add(child(c))),
                ExprNode::Mul(cs) if cs.len() == 2 && cs[0] == cs[1] => {
                    pow(child(cs[0]), Bounds::point(2.0))?
                }
                ExprNode::Mul(cs) => cs.iter().fold(Bounds::point(1.0), |b, &c| b.mul(child(c))),
                ExprNode::Unary(op, c) => unary(*op, child(*c))?,
                ExprNode::Pow(a, b) => {
                    let exponent = extract_linear(self.arena, *b)
                        .filter(|terms| terms.coeffs.is_empty() && terms.constant.is_finite())
                        .map_or_else(|| child(*b), |terms| Bounds::point(terms.constant));
                    pow(child(*a), exponent)?
                }
                ExprNode::Div(a, b) => {
                    let denominator = child(*b);
                    require(!denominator.contains_zero(), "division")?;
                    child(*a).mul(Bounds::widened(1.0 / denominator.upper, 1.0 / denominator.lower))
                }
                ExprNode::Atan2(a, b) => {
                    require(
                        !(child(*a).contains_zero() && child(*b).contains_zero()),
                        "atan2 origin",
                    )?;
                    Bounds::new(down(-PI), up(PI))
                }
                ExprNode::Min(cs) => Bounds::new(
                    cs.iter().map(|&c| child(c).lower).fold(f64::INFINITY, f64::min),
                    cs.iter().map(|&c| child(c).upper).fold(f64::INFINITY, f64::min),
                ),
                ExprNode::Max(cs) => Bounds::new(
                    cs.iter().map(|&c| child(c).lower).fold(f64::NEG_INFINITY, f64::max),
                    cs.iter().map(|&c| child(c).upper).fold(f64::NEG_INFINITY, f64::max),
                ),
            };

            self.cache.insert(id, bounds);
        }

        let affine = self.affine_analysis(root)?;
        let bounds = affine
            .as_ref()
            .map_or_else(|| self.cache.get(root).expect("root visited"), |(bounds, _)| *bounds);
        self.completed.insert(root, bounds);

        Ok(ExpressionAnalysis { bounds, affine: affine.map(|(_, terms)| terms) })
    }

    fn affine_terms(&mut self, id: ExprId) -> Option<AffineTerms<'a>> {
        if matches!(
            self.arena.get(id),
            ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
        ) {
            return extract_linear(self.arena, id).map(AffineTerms::Direct);
        }

        if let Some(terms) = self.affine_cache.get(&id) {
            return terms.as_ref().map(|terms| AffineTerms::Shared(Arc::clone(terms)));
        }

        let repeated = !self.affine_seen.insert(id);
        let terms = extract_linear(self.arena, id);
        let coefficients = terms.as_ref().map_or(0, |terms| terms.coeffs.len());

        // Admit only reused subexpressions, with a bounded coefficient budget.
        if repeated
            && self.affine_cache.len() < 128
            && self.affine_cache_coefficients + coefficients <= 4096
        {
            let shared = terms.map(|terms| Arc::new(terms.into_owned()));
            self.affine_cache_coefficients += coefficients;
            self.affine_cache.insert(id, shared.clone());
            shared.map(AffineTerms::Shared)
        } else {
            terms.map(AffineTerms::Direct)
        }
    }

    fn affine_analysis(
        &mut self,
        root: ExprId,
    ) -> Result<Option<(Bounds, LinearTerms<'static>)>, BoundError> {
        if let ExprNode::Add(children) = self.arena.get(root) {
            self.affine_coefficients.clear();
            self.affine_slots.clear();
            let mut constant = 0.0;

            for &child in children {
                let Some(terms) = self.affine_terms(child) else { return Ok(None) };
                constant += terms.constant;

                for &(variable, coefficient) in terms.coeffs.iter() {
                    if let Some(&slot) = self.affine_slots.get(&variable) {
                        self.affine_coefficients[slot].1 += coefficient;
                    } else {
                        self.affine_slots.insert(variable, self.affine_coefficients.len());
                        self.affine_coefficients.push((variable, coefficient));
                    }
                }
            }

            let bounds = self.affine_box(constant, &self.affine_coefficients)?;
            return Ok(Some((
                bounds,
                LinearTerms::owned(self.affine_coefficients.clone(), constant),
            )));
        }

        let Some(terms) = self.affine_terms(root) else { return Ok(None) };
        let bounds = self.affine_box(terms.constant, &terms.coeffs)?;

        Ok(Some((bounds, terms.deref().clone().into_owned())))
    }

    pub fn affine_box(
        &self,
        constant: f64,
        coefficients: &[(oximo_expr::VarId, f64)],
    ) -> Result<Bounds, BoundError> {
        if !constant.is_finite() {
            return Err(BoundError::InvalidNumber);
        }

        let mut bounds = Bounds::point(constant);

        for &(variable, coefficient) in coefficients {
            if !coefficient.is_finite() {
                return Err(BoundError::InvalidNumber);
            }

            if coefficient != 0.0 {
                bounds = bounds.add(self.variable(variable)?.mul(Bounds::point(coefficient)));
            }
        }

        Ok(bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_expr::evaluate;
    fn encloses(bounds: Bounds, value: f64) {
        assert!(bounds.lower <= value && value <= bounds.upper, "{value} outside {bounds:?}");
    }

    #[test]
    fn exact_zero_endpoints_do_not_mask_rounded_underflow() {
        let sum = Bounds::new(0.0, 1.0).add(Bounds::new(0.0, 1.0));
        assert_eq!(sum.lower.to_bits(), 0.0_f64.to_bits());
        let cancelled = Bounds::point(1.0).add(Bounds::point(-1.0));
        assert_eq!(cancelled.lower.to_bits(), 0.0_f64.to_bits());
        assert_eq!(cancelled.upper.to_bits(), 0.0_f64.to_bits());

        for divisor in [-2.0, 2.0] {
            let zero = Bounds::point(0.0).divide_scalar(divisor);
            assert_eq!(zero.lower.to_bits(), 0.0_f64.to_bits());
            assert_eq!(zero.upper.to_bits(), 0.0_f64.to_bits());
            let rounded = Bounds::point(f64::from_bits(1)).divide_scalar(divisor);
            assert!(rounded.lower < 0.0 && rounded.upper > 0.0);
        }
    }

    #[test]
    fn unary_vocabulary_encloses_interior_and_endpoint_values() {
        use UnaryOp as U;
        let cases = [
            (U::Neg, -2.0, 3.0),
            (U::Abs, -2.0, 3.0),
            (U::Sqrt, 0.0, 3.0),
            (U::Cbrt, -2.0, 3.0),
            (U::Exp, -2.0, 3.0),
            (U::Exp2, -2.0, 3.0),
            (U::Expm1, -2.0, 3.0),
            (U::Log, 0.1, 3.0),
            (U::Log2, 0.1, 3.0),
            (U::Log10, 0.1, 3.0),
            (U::Log1p, -0.9, 3.0),
            (U::Sin, -4.0, 5.0),
            (U::Cos, -4.0, 5.0),
            (U::Tan, -1.0, 1.0),
            (U::Asin, -1.0, 1.0),
            (U::Acos, -1.0, 1.0),
            (U::Atan, -4.0, 5.0),
            (U::Sinh, -2.0, 3.0),
            (U::Cosh, -2.0, 3.0),
            (U::Tanh, -2.0, 3.0),
            (U::Asinh, -2.0, 3.0),
            (U::Acosh, 1.0, 3.0),
            (U::Atanh, -0.9, 0.9),
        ];

        for (op, lo, hi) in cases {
            let bounds = unary(op, Bounds::new(lo, hi)).unwrap();

            for i in 0..=100 {
                let x = lo + (hi - lo) * f64::from(i) / 100.0;

                // Protect the endpoint against rounding outside a closed domain.
                encloses(bounds, op.apply(x.clamp(lo, hi)));
            }
        }
    }

    #[test]
    fn poles_and_partial_domains_are_rejected() {
        use UnaryOp as U;

        for (op, lo, hi) in [
            (U::Sqrt, -1.0, 1.0),
            (U::Log, 0.0, 1.0),
            (U::Log1p, -1.0, 1.0),
            (U::Asin, -2.0, 0.0),
            (U::Acos, 0.0, 2.0),
            (U::Acosh, 0.0, 2.0),
            (U::Atanh, -1.0, 0.5),
            (U::Tan, 1.0, 2.0),
            (U::Tan, -2.0, -1.0),
        ] {
            assert!(matches!(unary(op, Bounds::new(lo, hi)), Err(BoundError::Domain(_))));
        }
        assert!(pow(Bounds::new(-1.0, 1.0), Bounds::point(-2.0)).is_err());
        assert!(pow(Bounds::new(-1.0, 1.0), Bounds::point(0.5)).is_err());
        assert!(pow(Bounds::new(-1.0, 1.0), Bounds::new(1.0, 2.0)).is_err());
    }

    #[test]
    fn powers_cover_negative_bases_zero_and_variable_exponents() {
        for (lo, hi, p) in [
            (-3.0, 2.0, 2.0),
            (-3.0, 2.0, 3.0),
            (-3.0, -1.0, -2.0),
            (-3.0, -1.0, -3.0),
            (0.0, 3.0, 0.5),
            (0.1, 3.0, -0.5),
        ] {
            let b = pow(Bounds::new(lo, hi), Bounds::point(p)).unwrap();

            for i in 0..=100 {
                encloses(b, (lo + (hi - lo) * f64::from(i) / 100.0).powf(p));
            }
        }

        let b = pow(Bounds::new(0.5, 3.0), Bounds::new(-1.0, 2.0)).unwrap();

        for a in [0.5_f64, 1.0, 3.0] {
            for p in [-1.0, 0.0, 2.0] {
                encloses(b, a.powf(p));
            }
        }
    }

    #[test]
    fn compound_dag_enclosures_and_shared_square() {
        let m = oximo_core::Model::new("bounds");
        let x = m.__var("x").bounds(-3.0, 2.0).build();
        let y = m.__var("y").bounds(1.0, 4.0).build();
        let expressions = [
            (x.square() + y).erase(),
            (x / y).erase(),
            x.atan2(y).erase(),
            x.pow(y),
            x.min(y).erase(),
            x.max(y).erase(),
        ];
        let arena = m.arena();
        let variables = m.variables();
        let mut bounder = Bounder::new(&arena, &variables);

        for (i, e) in expressions.iter().enumerate() {
            let b = bounder.bound(e.id);

            if i == 3 {
                assert!(b.is_err());
                continue;
            }

            let b = b.unwrap();

            for a in [-3.0, -1.0, 0.0, 2.0] {
                for c in [1.0, 2.0, 4.0] {
                    encloses(b, evaluate(&arena, e.id, &&[a, c][..]).unwrap());
                }
            }
        }
    }

    #[test]
    fn deep_expression_traversal_is_iterative() {
        let m = oximo_core::Model::new("deep");
        let x = m.__var("x").bounds(-1.0, 2.0).build();
        let mut arena = m.__sum_context().borrow_mut();
        let mut id = x.id;

        for _ in 0..20_000 {
            id = arena.push(ExprNode::Unary(UnaryOp::Neg, id));
        }
        drop(arena);
        let arena = m.arena();
        let variables = m.variables();
        let bounds = Bounder::new(&arena, &variables).bound(id).unwrap();
        encloses(bounds, -1.0);
        encloses(bounds, 2.0);
    }

    #[test]
    fn cached_affine_refinements_do_not_change_domain_validation() {
        let m = oximo_core::Model::new("cache domains");
        let x = m.__var("x").bounds(-1.0, 1.0).build();
        let p = m.__param("p", 1.0);
        let cancelled = p * x - p * x + 1.0;
        let log = cancelled.ln();
        let arena = m.arena();
        let variables = m.variables();
        let mut bounder = Bounder::new(&arena, &variables);
        let first = bounder.bound(cancelled.id).unwrap();
        let second = bounder.bound(cancelled.id).unwrap();
        assert_eq!(first.lower.to_bits(), second.lower.to_bits());
        assert_eq!(first.upper.to_bits(), second.upper.to_bits());
        assert!(matches!(bounder.bound(log.id), Err(BoundError::Domain("log"))));
        assert!(matches!(
            Bounder::new(&arena, &variables).bound(log.id),
            Err(BoundError::Domain("log"))
        ));
    }

    #[test]
    fn reused_affine_subexpressions_keep_each_rows_parameter_value() {
        let m = oximo_core::Model::new("shared affine");
        let x = m.__var("x").bounds(-1.0, 1.0).build();
        let p = m.__param("p", 2.0);
        let q = m.__param("q", 5.0);
        let mut arena = m.__sum_context().borrow_mut();
        let mut shared = x.id;

        for _ in 0..1000 {
            shared = arena.push(ExprNode::Unary(UnaryOp::Neg, shared));
        }

        let roots = [p.id, q.id, p.id]
            .map(|parameter| arena.push(ExprNode::Add([shared, parameter].into_iter().collect())));
        drop(arena);
        let arena = m.arena();
        let variables = m.variables();
        let mut bounder = Bounder::new(&arena, &variables);

        for (root, value) in roots.into_iter().zip([2.0, 5.0, 2.0]) {
            let analysis = bounder.analyze(root).unwrap();
            let bounds = analysis.bounds;
            encloses(bounds, value - 1.0);
            encloses(bounds, value + 1.0);
            assert!((bounds.lower - (value - 1.0)).abs() < 1e-12);
            assert!((bounds.upper - (value + 1.0)).abs() < 1e-12);
            let terms = analysis.affine.unwrap();
            assert!((terms.constant - value).abs() < f64::EPSILON);
            assert_eq!(terms.coeffs.len(), 1);
            assert!((terms.coeffs[0].1 - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn interval_cache_keeps_sparse_storage_and_promotes_without_losing_bounds() {
        let mut cache = IntervalCache::new(16_000);

        for id in 0..32_u32 {
            cache.insert(ExprId(id), Bounds::point(f64::from(id)));
        }
        assert!(matches!(cache, IntervalCache::Sparse { .. }));
        assert!(cache.get(ExprId(15_999)).is_none());

        for id in 32..2001_u32 {
            cache.insert(ExprId(id), Bounds::point(f64::from(id)));
        }
        assert!(matches!(cache, IntervalCache::Dense(_)));

        for id in 0..2001_u32 {
            let bounds = cache.get(ExprId(id)).unwrap();
            assert!((bounds.lower - f64::from(id)).abs() < f64::EPSILON);
            assert!((bounds.upper - f64::from(id)).abs() < f64::EPSILON);
        }
        assert!(cache.get(ExprId(15_999)).is_none());
    }

    #[test]
    fn global_propagation_skips_inactive_rows_and_encloses_feasible_points() {
        use oximo_core::prelude::*;
        let model = Model::new("propagation enclosure");
        variable!(model, -10.0 <= x <= 10.0);
        variable!(model, -10.0 <= y <= 10.0);
        constraint!(model, 1.0 <= x + 2.0 * y <= 5.0);
        constraint!(model, -2.0 <= y <= 3.0);
        constraint!(model, inactive, x <= -100.0);
        let mut rows = model.constraints().algebraic().to_vec();
        rows.last_mut().unwrap().active = false;
        let arena = model.arena();
        let variables = model.variables();
        let mut bounder = Bounder::new(&arena, &variables);
        bounder.tighten_global(&rows, &FxHashSet::from_iter([x.var_id().unwrap()]));
        let xb = bounder.bound(x.id).unwrap();
        let yb = bounder.bound(y.id).unwrap();
        assert!(xb.lower > -10.0 && xb.upper < 10.0);

        for xv in -20..=20 {
            for yv in -20..=20 {
                let (xv, yv) = (f64::from(xv) / 2.0, f64::from(yv) / 2.0);

                if (-2.0..=3.0).contains(&yv) && (1.0..=5.0).contains(&(xv + 2.0 * yv)) {
                    encloses(xb, xv);
                    encloses(yb, yv);
                }
            }
        }
        assert!(!bounder.supporting_roots().contains(&rows.last().unwrap().lhs));
    }
}
