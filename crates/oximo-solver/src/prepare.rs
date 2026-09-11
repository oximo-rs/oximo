//! Solver-independent model views.

use std::cell::Ref;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use oximo_core::{
    Constraint, Model, ModelConstraints, ModelKind, Objective, ObjectiveSense, Sense,
    SocConstraint, SocForm, Variable,
};
use oximo_expr::{
    ExprArena, ExprId, ExprNode, LinearTerms, QuadraticTerms, extract_linear, extract_quadratic,
};
use rustc_hash::FxHashMap;

use crate::SolverError;

/// Affine coefficients borrowed from the arena or shared with the preparation cache.
#[derive(Clone, Debug)]
pub enum AffineTerms<'a> {
    /// Direct terms, were the inner `Cow` borrows affine nodes or owns uncached folds.
    Borrowed(LinearTerms<'a>),
    Shared(Arc<LinearTerms<'static>>),
}

impl<'a> Deref for AffineTerms<'a> {
    type Target = LinearTerms<'a>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(t) => t,
            Self::Shared(t) => t,
        }
    }
}

impl AffineTerms<'_> {
    #[inline]
    pub fn into_owned(self) -> LinearTerms<'static> {
        match self {
            Self::Borrowed(t) => t.into_owned(),
            Self::Shared(t) => Arc::unwrap_or_clone(t),
        }
    }
}

/// A one-use decomposition, or a shared decomposition after observed reuse.
#[derive(Clone, Debug)]
pub enum Extracted<T> {
    Owned(T),
    Shared(Arc<T>),
}

/// A direct affine node or an owned/shared degree-at-most-two decomposition.
#[derive(Clone, Debug)]
pub enum PolynomialTerms<'a> {
    Affine(LinearTerms<'a>),
    Quadratic(Extracted<QuadraticTerms>),
}

impl<T> Deref for Extracted<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        match self {
            Self::Owned(value) => value,
            Self::Shared(value) => value,
        }
    }
}

/// One solve's immutable model metadata and lazy expression decompositions.
///
/// Original IDs, expressions, bounds and activity flags are retained.
/// Native ordering, capability checks and generated entities belong to
/// the adapter. Create a fresh preparation after model or parameter changes.
/// Resident solver handles should retain their native state, not this borrowed view.
#[derive(Debug)]
pub struct LoweringContext<'a> {
    expressions: PreparedExpressions,
    variables: Ref<'a, Vec<Variable>>,
    constraints: ModelConstraints<'a>,
    objective: Ref<'a, Option<Objective>>,
    kind: ModelKind,
}

/// A shareable arena snapshot and bounded, reuse-admitted extraction caches.
/// Each cache has 16 independent shards, each retaining at most 16 expressions
/// and 1,024 coefficient entries, evicting in insertion order. Before cache
/// initialization, first encounters return owned terms without locking or
/// allocating cache storage. Only repeated roots are admitted. A small
/// direct-mapped history recognizes reuse across interleaved roots, and
/// collisions may delay admission. Once initialized, cached entries are
/// always checked independently of admission history.
/// Larger decompositions are returned uncached.
#[derive(Debug)]
pub struct PreparedExpressions {
    arena: ExprArena,
    linear: ExtractionCache<LinearTerms<'static>>,
    quadratic: ExtractionCache<QuadraticTerms>,
}

const CACHE_EXPRESSIONS: usize = 256;
const CACHE_COEFFICIENTS: usize = 16_384;
const CACHE_SHARDS: usize = 16;

#[derive(Debug)]
struct CacheEntries<T> {
    values: FxHashMap<ExprId, Option<Arc<T>>>,
    order: VecDeque<(ExprId, usize)>,
    coefficients: usize,
}

#[derive(Debug)]
struct ExtractionCache<T> {
    recent: [AtomicU64; CACHE_SHARDS],
    shards: OnceLock<Box<[CacheShard<T>; CACHE_SHARDS]>>,
}

// Keep unrelated workers' cache locks on independent cache lines.
#[derive(Debug)]
#[repr(align(64))]
struct CacheShard<T> {
    entries: Mutex<CacheEntries<T>>,
}

impl<T> Default for ExtractionCache<T> {
    fn default() -> Self {
        Self { recent: std::array::from_fn(|_| AtomicU64::new(u64::MAX)), shards: OnceLock::new() }
    }
}

fn cache_shards<T>() -> Box<[CacheShard<T>; CACHE_SHARDS]> {
    Box::new(std::array::from_fn(|_| CacheShard {
        entries: Mutex::new(CacheEntries {
            values: FxHashMap::default(),
            order: VecDeque::new(),
            coefficients: 0,
        }),
    }))
}

impl<T> ExtractionCache<T> {
    fn extract(
        &self,
        expr: ExprId,
        extract: impl FnOnce() -> Option<T>,
        size: impl FnOnce(&T) -> usize,
    ) -> Option<Extracted<T>> {
        let key = u64::from(expr.0);
        // Low bits distribute admission slots independently of cache shards.
        let admit =
            || self.recent[expr.0 as usize % CACHE_SHARDS].swap(key, Ordering::Relaxed) == key;
        let Some(shards) = self.shards.get() else {
            if !admit() {
                return extract().map(Extracted::Owned);
            }
            let shards = self.shards.get_or_init(cache_shards);
            let shard = &shards[(expr.0.wrapping_mul(0x9e37_79b9) >> 28) as usize];
            return shard.extract(expr, extract, size, || true);
        };
        let shard = &shards[(expr.0.wrapping_mul(0x9e37_79b9) >> 28) as usize];
        shard.extract(expr, extract, size, admit)
    }
}

impl<T> CacheShard<T> {
    fn extract(
        &self,
        expr: ExprId,
        extract: impl FnOnce() -> Option<T>,
        size: impl FnOnce(&T) -> usize,
        admit: impl FnOnce() -> bool,
    ) -> Option<Extracted<T>> {
        if let Some(value) =
            self.entries.lock().expect("preparation cache poisoned").values.get(&expr)
        {
            return value.clone().map(Extracted::Shared);
        }
        // Admission hints should not hide an already cached value.
        if !admit() {
            return extract().map(Extracted::Owned);
        }
        // Independent misses can be evaluated concurrently. Concurrent misses
        // for the same expression may repeat work, but publish one shared value.
        let value = extract();
        let coefficients = value.as_ref().map_or(0, size);
        if coefficients > CACHE_COEFFICIENTS / CACHE_SHARDS {
            return value.map(Extracted::Owned);
        }
        let mut cache = self.entries.lock().expect("preparation cache poisoned");
        if let Some(existing) = cache.values.get(&expr) {
            return existing.clone().map(Extracted::Shared);
        }
        while cache.values.len() >= CACHE_EXPRESSIONS / CACHE_SHARDS
            || cache.coefficients + coefficients > CACHE_COEFFICIENTS / CACHE_SHARDS
        {
            let (old, size) = cache.order.pop_front().expect("nonempty cache");
            cache.values.remove(&old);
            cache.coefficients -= size;
        }
        cache.coefficients += coefficients;
        cache.order.push_back((expr, coefficients));
        let value = value.map(Arc::new);
        cache.values.insert(expr, value.clone());
        value.map(Extracted::Shared)
    }
}

impl<'a> LoweringContext<'a> {
    /// Validate the objective declaration and capture the current parameter values.
    ///
    /// # Errors
    /// Returns a core error if neither an objective nor feasibility was declared.
    pub fn new(model: &'a Model) -> Result<Self, SolverError> {
        model.ensure_objective_declared()?;
        Ok(Self {
            expressions: PreparedExpressions::new((*model.arena()).clone()),
            variables: model.variables(),
            constraints: model.constraints(),
            objective: model.objective(),
            kind: model.kind(),
        })
    }

    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }
    pub fn constraints(&self) -> &ModelConstraints<'a> {
        &self.constraints
    }
    pub fn objective(&self) -> Option<&Objective> {
        self.objective.as_ref()
    }
    pub fn kind(&self) -> ModelKind {
        self.kind
    }
    pub fn sense(&self) -> ObjectiveSense {
        self.objective().map_or(ObjectiveSense::Minimize, |o| o.sense)
    }

    /// Explain an unsupported expression using original variable names.
    #[cold]
    #[inline(never)]
    pub fn nonlinear_error(&self, expr: ExprId, location: impl Into<String>) -> SolverError {
        SolverError::Nonlinear {
            location: location.into(),
            term: oximo_expr::describe_nonlinear_term(self.arena(), expr, &|id| {
                oximo_core::var_name(self.variables(), id)
            })
            .unwrap_or_else(|| "<nonlinear>".into()),
        }
    }

    /// Require an affine expression.
    ///
    /// # Errors
    /// Returns a named nonlinear-expression error when extraction fails.
    #[inline]
    pub fn require_linear(
        &self,
        expr: ExprId,
        location: impl FnOnce() -> String,
    ) -> Result<AffineTerms<'_>, SolverError> {
        self.linear(expr).ok_or_else(|| self.nonlinear_error(expr, location()))
    }

    /// Require an affine expression for a one-use streaming consumer.
    ///
    /// This bypasses reuse admission and returns the expression crate's native
    /// borrowed/owned representation. Prefer [`Self::require_linear`] when the
    /// same root is likely to be requested again during this preparation.
    ///
    /// # Errors
    /// Returns a lazily named nonlinear-expression error when extraction fails.
    #[expect(clippy::inline_always, reason = "called once per streamed backend row")]
    #[inline(always)]
    pub fn require_linear_once(
        &self,
        expr: ExprId,
        location: impl FnOnce() -> String,
    ) -> Result<LinearTerms<'_>, SolverError> {
        extract_linear(&self.arena, expr).ok_or_else(|| self.nonlinear_error(expr, location()))
    }

    /// Require a polynomial of degree at most two.
    ///
    /// # Errors
    /// Returns a named nonlinear-expression error when extraction fails.
    #[inline]
    pub fn require_quadratic(
        &self,
        expr: ExprId,
        location: impl FnOnce() -> String,
    ) -> Result<Extracted<QuadraticTerms>, SolverError> {
        self.quadratic(expr).ok_or_else(|| self.nonlinear_error(expr, location()))
    }

    /// Require a polynomial while retaining the direct-affine borrowed path.
    /// Non-direct nodes are traversed once as quadratics, avoiding a failed
    /// affine pass before quadratic/SOC handling.
    ///
    /// # Errors
    /// Returns a lazily named nonlinear-expression error when extraction fails.
    pub fn require_polynomial(
        &self,
        expr: ExprId,
        location: impl FnOnce() -> String,
    ) -> Result<PolynomialTerms<'_>, SolverError> {
        if let ExprNode::Linear { coeffs, constant } = self.arena.get(expr) {
            return Ok(PolynomialTerms::Affine(LinearTerms::borrowed(coeffs, *constant)));
        }
        self.quadratic(expr)
            .map(PolynomialTerms::Quadratic)
            .ok_or_else(|| self.nonlinear_error(expr, location()))
    }

    /// An optional alternative, never a replacement for the original row.
    pub fn detected_soc(&self, row: &Constraint) -> Option<SocForm> {
        self.expressions.detected_soc(self.variables(), row)
    }
}

/// Bounds for a polynomial body with its constant removed. No row is dropped,
/// split, relaxed or reclassified as feasible/infeasible by this operation.
#[inline]
pub fn shifted_bounds(row: &Constraint, constant: f64) -> (f64, f64) {
    (row.lower - constant, row.upper - constant)
}

/// Optional single-sided view, in lower/upper order for a range. Free rows
/// produce no sides. The caller owns native row ordering and provenance.
#[inline]
pub fn row_sides(row: &Constraint) -> [Option<(Sense, f64)>; 2] {
    match row.as_single() {
        Some(single) => [Some(single), None],
        None if row.is_range() => [Some((Sense::Ge, row.lower)), Some((Sense::Le, row.upper))],
        None => [None, None],
    }
}

impl Deref for LoweringContext<'_> {
    type Target = PreparedExpressions;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.expressions
    }
}

impl Deref for PreparedExpressions {
    type Target = ExprArena;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.arena
    }
}

impl PreparedExpressions {
    pub fn new(arena: ExprArena) -> Self {
        Self { arena, linear: ExtractionCache::default(), quadratic: ExtractionCache::default() }
    }
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }
    /// Direct affine nodes retain their zero-copy path. Reused compound roots
    /// may be cached, including failed extractions, within this snapshot's budget.
    ///
    /// # Panics
    /// Panics for an expression outside this arena or a poisoned cache.
    #[inline]
    pub fn linear(&self, expr: ExprId) -> Option<AffineTerms<'_>> {
        if let ExprNode::Linear { coeffs, constant } = self.arena.get(expr) {
            return Some(AffineTerms::Borrowed(LinearTerms::borrowed(coeffs, *constant)));
        }
        self.compound_linear(expr)
    }

    // Keep cache admission out of the direct-affine hot loop in each adapter.
    #[inline(never)]
    fn compound_linear(&self, expr: ExprId) -> Option<AffineTerms<'_>> {
        self.linear
            .extract(
                expr,
                || extract_linear(&self.arena, expr).map(LinearTerms::into_owned),
                |t| t.coeffs.len(),
            )
            .map(|terms| match terms {
                Extracted::Owned(terms) => AffineTerms::Borrowed(terms),
                Extracted::Shared(terms) => AffineTerms::Shared(terms),
            })
    }

    /// Extract a quadratic polynomial, sharing terms only after observed reuse.
    ///
    /// # Panics
    /// Panics for an expression outside this arena or a poisoned cache.
    pub fn quadratic(&self, expr: ExprId) -> Option<Extracted<QuadraticTerms>> {
        self.quadratic.extract(
            expr,
            || extract_quadratic(&self.arena, expr),
            |t| t.linear.len() + t.hessian.len(),
        )
    }

    /// Detect an optional cone representation using cached polynomial terms.
    pub fn detected_soc(&self, variables: &[Variable], row: &Constraint) -> Option<SocForm> {
        let q = self.quadratic(row.lhs)?;
        oximo_core::__detect_soc_from_quadratic(variables, row, &q)
    }

    /// Extract an explicit cone in one streaming pass over its affine members.
    ///
    /// # Errors
    /// Returns a backend error for invalid affine cone members.
    pub fn explicit_soc(&self, soc: &SocConstraint) -> Result<SocForm, SolverError> {
        let affine = |expr| {
            extract_linear(&self.arena, expr).map(LinearTerms::into_owned).ok_or_else(|| {
                SolverError::Backend(format!(
                    "invalid affine expressions in SOC constraint {:?}",
                    soc.name
                ))
            })
        };
        Ok(SocForm {
            terms: soc.terms.iter().map(|&expr| affine(expr)).collect::<Result<_, _>>()?,
            bound: affine(soc.bound)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_roots_are_admitted_and_cached_hits_ignore_history() {
        let cache = ExtractionCache::default();
        let shard = |id: u32| id.wrapping_mul(0x9e37_79b9) >> 28;
        let a = 0;
        let b = (1..1000).find(|&id| shard(id) == shard(a) && id % 16 != a % 16).unwrap();
        for id in [a, b] {
            assert!(matches!(
                cache.extract(ExprId(id), || Some(vec![id]), Vec::len),
                Some(Extracted::Owned(_))
            ));
        }
        for id in [a, b, a, b] {
            assert!(matches!(
                cache.extract(ExprId(id), || Some(vec![id]), Vec::len),
                Some(Extracted::Shared(_))
            ));
        }
        // Change the admission history for a without evicting its cached value.
        cache.extract(ExprId(a + 16), || Some(vec![16]), Vec::len);
        let hit =
            cache.extract(ExprId(a), || panic!("cached root was re-extracted"), Vec::len).unwrap();
        assert_eq!(hit.as_slice(), &[a]);
    }

    #[test]
    fn unique_roots_do_not_allocate_cache_entries() {
        let cache = ExtractionCache::default();
        for index in 0..1000 {
            assert!(matches!(
                cache.extract(ExprId(index), || Some(vec![1]), Vec::len),
                Some(Extracted::Owned(_))
            ));
        }
        assert!(cache.shards.get().is_none());
    }

    #[test]
    fn cache_bounds_storage_and_keeps_returned_terms_alive() {
        let cache = ExtractionCache::default();
        let extract =
            |id, size| cache.extract(ExprId(id), || Some(vec![1; size]), Vec::len).unwrap();
        extract(0, 1);
        let first = extract(0, 1);
        assert!(matches!(first, Extracted::Shared(_)));
        for index in 1..1000 {
            extract(index, 100);
            extract(index, 100);
        }
        assert_eq!(first.len(), 1);
        let shards = cache.shards.get().unwrap();
        let entries: usize = shards.iter().map(|s| s.entries.lock().unwrap().values.len()).sum();
        let coefficients: usize =
            shards.iter().map(|s| s.entries.lock().unwrap().coefficients).sum();
        assert!(entries <= CACHE_EXPRESSIONS);
        assert!(coefficients <= CACHE_COEFFICIENTS);
        extract(1001, CACHE_COEFFICIENTS + 1);
        assert!(matches!(extract(1001, CACHE_COEFFICIENTS + 1), Extracted::Owned(_)));
        for index in 2000..3000 {
            for _ in 0..2 {
                assert!(cache.extract(ExprId(index), || None, Vec::len).is_none());
            }
        }
        assert!(shards
            .iter()
            .all(|s| s.entries.lock().unwrap().values.len() <= CACHE_EXPRESSIONS / CACHE_SHARDS));
    }

    #[test]
    fn cache_extracts_outside_lock_and_shares_concurrent_publication() {
        let single = ExtractionCache::default();
        single.extract(ExprId(0), || Some(vec![1]), Vec::len);
        single.extract(
            ExprId(0),
            || {
                assert!(single.shards.get().unwrap()[0].entries.try_lock().is_ok());
                Some(vec![1])
            },
            Vec::len,
        );
        let cache = ExtractionCache::default();
        cache.extract(ExprId(0), || Some(vec![1]), Vec::len);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let extract = || {
                let terms = cache
                    .extract(
                        ExprId(0),
                        || {
                            barrier.wait();
                            Some(vec![1])
                        },
                        Vec::len,
                    )
                    .unwrap();
                let Extracted::Shared(value) = terms else { panic!("reuse not admitted") };
                value
            };
            let a = scope.spawn(extract);
            let b = scope.spawn(extract);
            assert!(Arc::ptr_eq(&a.join().unwrap(), &b.join().unwrap()));
        });
    }
}
