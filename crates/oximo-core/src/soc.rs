use oximo_expr::{ExprArena, ExprId, LinearTerms, QuadraticTerms, VarId, extract_quadratic};
use smol_str::SmolStr;

use crate::constraint::{AlgebraicConstraint, Sense};
use crate::var::ResolvedVariable as Variable;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SocConstraintId(pub u32);

/// Typed reference to a vector-affine function in a second-order cone.
pub type SocConstraintHandle =
    crate::ConstraintHandle<crate::semantic::VectorAffineData, crate::SecondOrderCone>;

/// Typed reference to a vector-affine function in a rotated second-order cone.
pub type RotatedSocConstraintHandle =
    crate::ConstraintHandle<crate::semantic::VectorAffineData, crate::RotatedSecondOrderCone>;

impl SocConstraintId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The kind of an explicit second-order cone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SocKind {
    /// `||w||_2 <= t`, components `[t, w...]`.
    Standard,
    /// `||w||_2^2 <= 2 u v` with `u, v >= 0`, components `[u, v, w...]`.
    Rotated,
}

impl SocKind {
    /// Number of head components: one (`t`) or two (`u, v`).
    #[inline]
    pub const fn head_len(self) -> usize {
        match self {
            Self::Standard => 1,
            Self::Rotated => 2,
        }
    }

    /// Smallest valid cone dimension.
    #[inline]
    pub const fn min_dimension(self) -> usize {
        self.head_len() + 1
    }
}

/// The scalar head of a second-order cone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SocHead<T> {
    Standard(T),
    Rotated(T, T),
}

impl<T> SocHead<T> {
    /// The cone kind this head denotes.
    #[inline]
    pub const fn kind(&self) -> SocKind {
        match self {
            Self::Standard(_) => SocKind::Standard,
            Self::Rotated(..) => SocKind::Rotated,
        }
    }

    /// Whether this is the rotated cone.
    #[inline]
    pub const fn is_rotated(&self) -> bool {
        matches!(self, Self::Rotated(..))
    }

    /// Number of head components: one or two.
    #[inline]
    pub const fn component_count(&self) -> usize {
        self.kind().head_len()
    }

    /// Head components in declared order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let (first, second) = match self {
            Self::Standard(t) => (t, None),
            Self::Rotated(u, v) => (u, Some(v)),
        };
        std::iter::once(first).chain(second)
    }

    /// Borrow every head component.
    pub const fn as_ref(&self) -> SocHead<&T> {
        match self {
            Self::Standard(t) => SocHead::Standard(t),
            Self::Rotated(u, v) => SocHead::Rotated(u, v),
        }
    }

    /// Map every head component.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SocHead<U> {
        match self {
            Self::Standard(t) => SocHead::Standard(f(t)),
            Self::Rotated(u, v) => SocHead::Rotated(f(u), f(v)),
        }
    }

    /// Map every head component, stopping at the first error.
    ///
    /// # Errors
    /// Returns the first error produced by `f`.
    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<SocHead<U>, E> {
        Ok(match self {
            Self::Standard(t) => SocHead::Standard(f(t)?),
            Self::Rotated(u, v) => SocHead::Rotated(f(u)?, f(v)?),
        })
    }
}

impl<T: Copy> SocHead<T> {
    /// Read a `kind` head from the front of `[head..., terms...]` components,
    /// or `None` when `components` is shorter than the head.
    pub fn from_prefix(kind: SocKind, components: &[T]) -> Option<Self> {
        match (kind, components) {
            (SocKind::Standard, [t, ..]) => Some(Self::Standard(*t)),
            (SocKind::Rotated, [u, v, ..]) => Some(Self::Rotated(*u, *v)),
            _ => None,
        }
    }
}

/// An explicit second-order cone constraint: `||w||_2 <= t`
/// ([`SocKind::Standard`]) or `||w||_2^2 <= 2 u v`, `u, v >= 0`
/// ([`SocKind::Rotated`]).
#[derive(Clone, Debug)]
pub struct SocConstraint {
    pub name: SmolStr,
    pub active: bool,
    components: crate::SharedSlice<ExprId>,
    kind: SocKind,
}

impl SocConstraint {
    /// Store `[head..., terms...]` components.
    ///
    /// # Panics
    /// Panics when `components` has no term after the head.
    pub(crate) fn new(
        name: SmolStr,
        components: crate::SharedSlice<ExprId>,
        kind: SocKind,
    ) -> Self {
        assert_soc_dimension(kind, components.len());
        Self { name, active: true, components, kind }
    }

    /// The cone kind.
    #[inline]
    pub const fn kind(&self) -> SocKind {
        self.kind
    }

    /// The head roots.
    ///
    /// # Panics
    /// Never panics since registration checks that the head fits.
    #[inline]
    pub fn head(&self) -> SocHead<ExprId> {
        SocHead::from_prefix(self.kind, &self.components).expect("validated SOC dimension")
    }

    /// The norm terms `w`.
    #[inline]
    pub fn terms(&self) -> &[ExprId] {
        &self.components[self.kind.head_len()..]
    }

    /// Every affine root in declared order.
    #[inline]
    pub fn components(&self) -> &[ExprId] {
        &self.components
    }

    /// The shared component buffer (for semantic function views).
    #[inline]
    pub(crate) const fn shared_components(&self) -> &crate::SharedSlice<ExprId> {
        &self.components
    }

    /// Replace every component, keeping the kind.
    pub(crate) fn set_components(&mut self, components: crate::SharedSlice<ExprId>) {
        assert_soc_dimension(self.kind, components.len());
        self.components = components;
    }

    /// Total number of cone components (head plus terms).
    #[inline]
    pub fn dimension(&self) -> usize {
        self.components.len()
    }
}

/// Panic unless a `kind` cone of `dimension` components has at least one term.
pub(crate) fn assert_soc_dimension(kind: SocKind, dimension: usize) {
    match kind {
        SocKind::Standard => assert!(
            dimension >= 2,
            "SOC needs at least two components (the bound and one term), got {dimension}"
        ),
        SocKind::Rotated => assert!(
            dimension >= 3,
            "rotated SOC needs at least three components (two heads and one term), got {dimension}"
        ),
    }
}

/// Normalized second-order cone data, one [`LinearTerms`] per affine member.
/// Standard: `|| A x + a ||_2 <= b'x + beta`.
/// Rotated: `|| A x + a ||_2^2 <= 2 (u'x + u0)(v'x + v0)`
/// with both heads nonnegative.
#[derive(Clone, Debug)]
pub struct SocForm {
    pub terms: Vec<LinearTerms<'static>>,
    pub head: SocHead<LinearTerms<'static>>,
}

impl SocForm {
    /// The cone kind.
    #[inline]
    pub const fn kind(&self) -> SocKind {
        self.head.kind()
    }

    /// Number of cone components (head plus terms).
    pub fn dimension(&self) -> usize {
        self.head.component_count() + self.terms.len()
    }

    /// Every affine component in declared order.
    pub fn components(&self) -> impl Iterator<Item = &LinearTerms<'static>> {
        self.head.iter().chain(self.terms.iter())
    }

    /// The bound `t` of a standard cone, or `None` for a rotated one.
    pub fn standard_bound(&self) -> Option<&LinearTerms<'static>> {
        match &self.head {
            SocHead::Standard(t) => Some(t),
            SocHead::Rotated(..) => None,
        }
    }

    /// Exact orthogonal rewrite of a rotated cone as a standard one:
    /// `||((u-v)/sqrt2, w)|| <= (u+v)/sqrt2`.
    /// Standard forms are returned unchanged.
    /// The result's component order is `[t', s', w...]`.
    ///
    /// Coefficients that cancel exactly are kept as explicit zeros.
    #[must_use]
    pub fn into_standard(self) -> Self {
        let SocHead::Rotated(u, v) = &self.head else {
            return self;
        };
        let (sum, diff) = rotate_heads(u, v);
        let mut terms = Vec::with_capacity(self.terms.len() + 1);
        terms.push(diff);
        terms.extend(self.terms);

        Self { terms, head: SocHead::Standard(sum) }
    }
}

/// Rewrite a rotated head `(u, v)` as `((u+v)/sqrt2, (u-v)/sqrt2)`, the first
/// two components of [`SocForm::into_standard`].
///
/// Both results share one sparsity pattern, the union of the variables of `u`
/// and `v` (exact cancellations are kept as explicit zeros).
pub fn rotate_heads(
    u: &LinearTerms<'_>,
    v: &LinearTerms<'_>,
) -> (LinearTerms<'static>, LinearTerms<'static>) {
    (combine_heads(u, v, 1.0), combine_heads(u, v, -1.0))
}

/// `(first + sign*second)/sqrt2` by a two-pointer merge of the variables.
fn combine_heads(
    first: &LinearTerms<'_>,
    second: &LinearTerms<'_>,
    sign: f64,
) -> LinearTerms<'static> {
    fn sorted(coeffs: &[(VarId, f64)]) -> std::borrow::Cow<'_, [(VarId, f64)]> {
        if coeffs.is_sorted_by_key(|&(var, _)| var.0) {
            std::borrow::Cow::Borrowed(coeffs)
        } else {
            let mut owned = coeffs.to_vec();
            owned.sort_unstable_by_key(|&(var, _)| var.0);

            std::borrow::Cow::Owned(owned)
        }
    }

    let scale = std::f64::consts::FRAC_1_SQRT_2;
    let (left, right) = (sorted(&first.coeffs), sorted(&second.coeffs));
    let mut merged: Vec<(VarId, f64)> = Vec::with_capacity(left.len() + right.len());
    let mut left = left.iter().copied().peekable();
    let mut right = right.iter().map(|&(var, coefficient)| (var, sign * coefficient)).peekable();
    loop {
        let (var, coefficient) = match (left.peek(), right.peek()) {
            (None, None) => break,
            (Some(&(lv, lc)), Some(&(rv, rc))) if lv == rv => {
                left.next();
                right.next();
                (lv, lc + rc)
            }
            (Some(&(lv, _)), Some(&(rv, _))) if lv.0 > rv.0 => right.next().expect("peeked"),
            (Some(_), _) => left.next().expect("peeked"),
            (None, Some(_)) => right.next().expect("peeked"),
        };
        merged.push((var, coefficient * scale));
    }

    LinearTerms::owned(merged, (first.constant + sign * second.constant) * scale)
}

/// Structural shape of a recognized cone row.
/// `sum p_i x_i^2 <= n t^2` for a standard head `t`, or
/// `sum p_i x_i^2 <= c y z` for a rotated head `(y, z)`.
struct SocShape {
    head: SocHead<VarId>,
    /// `n` for a standard head, `c` for a rotated one.
    scale: f64,
}

/// Extract and validate the quadratic form shared by SOC recognition
/// and model-kind inference.
fn soc_quadratic(
    vars: &[Variable],
    c: &AlgebraicConstraint,
    q: &QuadraticTerms,
) -> Option<SocShape> {
    let shape = soc_quadratic_candidate(c, q)?;
    let nonnegative = shape.head.iter().all(|v| vars[v.index()].lb >= 0.0);
    nonnegative.then_some(shape)
}

/// One pass over the canonical entries, which are sorted by `(col, row)` with
/// `row >= col`.
fn soc_quadratic_candidate(c: &AlgebraicConstraint, q: &QuadraticTerms) -> Option<SocShape> {
    let (sense, rhs) = c.as_single()?;
    if sense != Sense::Le {
        return None;
    }
    if !q.linear().is_empty() || q.constant() - rhs != 0.0 {
        return None;
    }

    let mut positives = 0;
    let mut negative: Option<(VarId, f64)> = None;
    let mut cross: Option<(VarId, VarId, f64)> = None;

    // Whether the current column group started with a diagonal entry.
    let mut group: Option<(VarId, bool)> = None;
    for &(row, col, h) in q.quadratic() {
        if group.is_none_or(|(current, _)| current != col) {
            group = Some((col, row == col));
        }
        if row != col {
            // One neg. cross term `-c y z` forms the rotated head, and
            // `y = col` may not carry a squared term.
            if h >= 0.0 || cross.is_some() || group.is_some_and(|(_, diagonal)| diagonal) {
                return None;
            }
            cross = Some((col, row, -h));
            continue;
        }

        // `z` may not carry a squared term either.
        if cross.is_some_and(|(_, z, _)| z == row) {
            return None;
        }
        if h > 0.0 {
            positives += 1;
        } else if h < 0.0 {
            if negative.is_some() {
                return None;
            }
            negative = Some((row, -h));
        }
    }

    if positives == 0 {
        return None;
    }

    match (negative, cross) {
        (Some((t, n)), None) => Some(SocShape { head: SocHead::Standard(t), scale: n }),
        (None, Some((y, z, coef))) => Some(SocShape { head: SocHead::Rotated(y, z), scale: coef }),
        _ => None,
    }
}

/// Whether an algebraic constraint has the supported detected-SOC shape.
///
/// This does not materialize a [`SocForm`]. Model-kind
/// inference only needs this predicate and can avoid allocating one
/// `LinearTerms` coefficient vector per cone member.
pub(crate) fn is_detected_soc(
    arena: &ExprArena,
    vars: &[Variable],
    c: &AlgebraicConstraint,
) -> bool {
    if !matches!(c.as_single(), Some((Sense::Le, _))) {
        return false;
    }
    extract_quadratic(arena, c.lhs).is_some_and(|q| __is_detected_soc_from_quadratic(vars, c, &q))
}

/// Test an already extracted quadratic for the supported detected-SOC shape.
#[doc(hidden)]
pub fn __is_detected_soc_from_quadratic(
    vars: &[Variable],
    c: &AlgebraicConstraint,
    q: &QuadraticTerms,
) -> bool {
    soc_quadratic(vars, c, q).is_some()
}

/// Head variables of a structurally eligible SOC row, ignoring their bounds:
/// `t` for `sum p_i x_i^2 <= n t^2`,
/// `(y, z)` for `sum p_i x_i^2 <= c y z`.
/// The row is a cone exactly when every head variable has `lb >= 0`.
#[doc(hidden)]
pub fn __soc_head_variables(
    row: &AlgebraicConstraint,
    terms: &QuadraticTerms,
) -> Option<SocHead<VarId>> {
    soc_quadratic_candidate(row, terms).map(|shape| shape.head)
}

// TODO: Here we are deliberately conservative and purely structural

/// Recognize an algebraic quadratic constraint as second-order-cone shaped.
///
/// A constraint is recognized iff:
///
/// - it is single-sided `lhs <= rhs` (no ranges, `>=`, or equalities),
/// - `lhs - rhs` is a pure quadratic form: no linear terms, no constant,
/// - it has at least one positive diagonal entry `p_i` and either
///   - exactly one negative diagonal entry `-n` (on the bound variable `t`)
///     and no cross terms, or
///   - exactly one negative cross term `-c y z` and no negative diagonal
///     entry, with `y` and `z` carrying no squared term,
/// - every head variable (`t`, or `y` and `z`) has lower bound `>= 0`.
///
/// That is `sum_i p_i x_i^2 <= n t^2`, equivalent to
/// `|| sqrt(p_i/n) x_i ||_2 <= t`, or `sum_i p_i x_i^2 <= c y z`, the rotated
/// cone `||sqrt(p_i) x_i||^2 <= 2 (c/2 y) z` with `y` the head variable with
/// the smaller ID. Other cross-term (Cholesky-factorized) quadratic forms are
/// not detected.
///
/// Backend hook for recognizing a row that has already been decomposed.
/// `q` must describe `c.lhs` at the current parameter values.
#[doc(hidden)]
pub fn __detect_soc_from_quadratic(
    vars: &[Variable],
    c: &AlgebraicConstraint,
    q: &QuadraticTerms,
) -> Option<SocForm> {
    let SocShape { head, scale } = soc_quadratic(vars, c, q)?;

    // Standard rows normalize by `n` (`||sqrt(p/n) x|| <= t`);
    // rotated rows use `w_i = sqrt(p_i) x_i` with `u = (c/2) y`, `v = z`.
    let divisor = if head.is_rotated() { 1.0 } else { scale };

    // Head variables carry no squared term.
    let terms = q
        .quadratic()
        .iter()
        .filter(|&&(row, col, h)| row == col && h > 0.0)
        .map(|&(row, _, h)| LinearTerms {
            coeffs: vec![(row, (h / divisor).sqrt())].into(),
            constant: 0.0,
        })
        .collect();
    let single = |var: VarId, coefficient: f64| LinearTerms {
        coeffs: vec![(var, coefficient)].into(),
        constant: 0.0,
    };
    let head = match head {
        SocHead::Standard(t) => SocHead::Standard(single(t, 1.0)),
        SocHead::Rotated(y, z) => SocHead::Rotated(single(y, scale / 2.0), single(z, 1.0)),
    };
    Some(SocForm { terms, head })
}
