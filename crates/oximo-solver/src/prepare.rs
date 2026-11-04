//! Solver-independent model snapshots and cached expression extraction.

use std::cell::Ref;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use oximo_core::{
    AlgebraicConstraint, Model, ModelConstraints, ModelKind, Objective, ObjectiveSense,
    ResolvedVariable, Sense, SocConstraint, SocForm,
};
use oximo_expr::{
    ExprArena, ExprId, ExprNode, LinearTerms, QuadraticTerms, extract_linear, extract_quadratic,
};
use rustc_hash::FxHashMap;

use crate::SolverError;

mod workset;

#[cfg(test)]
pub use workset::QuadraticAnalysisStats;
use workset::QuadraticWorkset;

/// Affine coefficients borrowed from the arena or shared with the preparation cache.
#[derive(Clone, Debug)]
pub enum AffineTerms<'a> {
    /// Borrowed affine nodes or owned, uncached terms.
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
    /// Take ownership, cloning shared terms only when needed.
    #[inline]
    pub fn into_owned(self) -> LinearTerms<'static> {
        match self {
            Self::Borrowed(t) => t.into_owned(),
            Self::Shared(t) => Arc::unwrap_or_clone(t),
        }
    }
}

/// An owned decomposition, or immutable terms shared within a preparation.
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

/// Model metadata and expression analysis for one solve.
/// Recreate after model or parameter changes. Adapters own native solver state.
#[derive(Debug)]
pub struct LoweringContext<'a> {
    key: oximo_core::model::ModelUpdateKey,
    expressions: PreparedExpressions,
    variables: PreparedVariables,
    constraints: ModelConstraints<'a>,
    objective: Ref<'a, Option<Objective>>,
    kind: ModelKind,
    semantic: oximo_core::SemanticConstraints<'a>,
}

#[derive(Debug)]
enum PreparedVariables {
    Owned(Vec<ResolvedVariable>),
    Shared(Arc<Vec<ResolvedVariable>>),
}

impl Deref for PreparedVariables {
    type Target = [ResolvedVariable];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(v) => v,
            Self::Shared(v) => v,
        }
    }
}

/// An arena snapshot with bounded extraction caches and optional resident reuse.
/// Preparations also share quadratic analysis from the first extraction.
#[derive(Debug)]
pub struct PreparedExpressions {
    arena: ExprArena,
    linear: ExtractionCache<LinearTerms<'static>>,
    quadratic: ExtractionCache<QuadraticTerms>,
    resident: Option<Arc<crate::resident::ResidentExpressions>>,
    workset: Option<QuadraticWorkset>,
    operators: OnceLock<Arc<crate::requirements::OperatorSummaries>>,
    #[cfg(test)]
    affine_extractions: AtomicU64,
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
    /// Look up successful terms without initializing storage or changing admission hints.
    fn lookup(&self, expr: ExprId) -> Option<Arc<T>> {
        let shards = self.shards.get()?;
        let shard = &shards[(expr.0.wrapping_mul(0x9e37_79b9) >> 28) as usize];
        shard.entries.lock().expect("preparation cache poisoned").values.get(&expr)?.clone()
    }

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

    /// Publish recovered terms, replacing a cached extraction failure within budget.
    fn publish(
        &self,
        expr: ExprId,
        value: Option<T>,
        size: impl FnOnce(&T) -> usize,
    ) -> Option<Extracted<T>> {
        let shards = self.shards.get_or_init(cache_shards);
        let shard = &shards[(expr.0.wrapping_mul(0x9e37_79b9) >> 28) as usize];
        shard.publish(expr, value, size)
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
        self.publish(expr, extract(), size)
    }

    fn publish(
        &self,
        expr: ExprId,
        value: Option<T>,
        size: impl FnOnce(&T) -> usize,
    ) -> Option<Extracted<T>> {
        let coefficients = value.as_ref().map_or(0, size);
        if coefficients > CACHE_COEFFICIENTS / CACHE_SHARDS {
            return value.map(Extracted::Owned);
        }
        let mut cache = self.entries.lock().expect("preparation cache poisoned");
        if let Some(existing) = cache.values.get(&expr) {
            if existing.is_some() || value.is_none() {
                return existing.clone().map(Extracted::Shared);
            }
            cache.values.remove(&expr);
            cache.order.retain(|&(root, _)| root != expr);
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
    /// Validate declarations, snapshot values, and classify the model using shared analysis.
    ///
    /// # Errors
    /// Returns a core error for unreformulated GDP or a missing objective declaration.
    pub fn new(model: &'a Model) -> Result<Self, SolverError> {
        model.ensure_gdp_reformulated()?;
        model.ensure_objective_declared()?;
        let arena = model.arena();
        let variables: Vec<_> = model.variables().iter().map(|v| v.resolved(&arena)).collect();
        let expressions = PreparedExpressions::for_preparation((*arena).clone());

        // Share classification results with compatibility checks and lowering.
        let kind = model.__kind_with_quadratic_at(expressions.arena(), &variables, &|root| {
            expressions.quadratic(root)
        });

        Ok(Self {
            key: model.__update_key(),
            expressions,
            variables: PreparedVariables::Owned(variables),
            constraints: model.constraints(),
            objective: model.objective(),
            kind,
            semantic: model.semantic_constraints_ref(),
        })
    }

    /// Combine resident expression analysis and resolved variables with model metadata.
    pub(crate) fn resident(
        model: &'a Model,
        expressions: PreparedExpressions,
        variables: Arc<Vec<ResolvedVariable>>,
        kind: ModelKind,
    ) -> Self {
        Self {
            key: model.__update_key(),
            expressions,
            variables: PreparedVariables::Shared(variables),
            constraints: model.constraints(),
            objective: model.objective(),
            kind,
            semantic: model.semantic_constraints_ref(),
        }
    }

    /// Reject a different model or changed model state or parameter values.
    pub(crate) fn ensure_matches(&self, model: &Model) -> Result<(), SolverError> {
        if self.key != model.__update_key()
            || !self.arena().parameter_snapshot().same_bits(&model.arena().parameter_snapshot())
        {
            return Err(SolverError::Backend(
                "prepared context no longer matches model state".into(),
            ));
        }
        Ok(())
    }

    pub fn semantic_constraints(&self) -> &oximo_core::SemanticConstraints<'a> {
        &self.semantic
    }

    /// Extract finite affine PSD entries in upper-column order.
    ///
    /// # Errors
    /// Returns an error for non-affine or nonfinite entries.
    pub fn explicit_psd(
        &self,
        psd: &oximo_core::PsdConstraint,
    ) -> Result<oximo_core::SymmetricMatrix<AffineTerms<'_>>, SolverError> {
        let entries = psd
            .matrix
            .upper_triangle()
            .iter()
            .map(|&expr| {
                let terms =
                    self.require_linear(expr, || format!("PSD constraint {:?}", psd.name))?;
                if !terms.constant.is_finite()
                    || terms.coeffs.iter().any(|&(_, value)| !value.is_finite())
                {
                    return Err(SolverError::Backend(format!(
                        "nonfinite affine entry in PSD constraint {:?}",
                        psd.name
                    )));
                }
                Ok(terms)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(oximo_core::SymmetricMatrix::from_upper_triangle(psd.matrix.side_dimension(), entries))
    }

    pub fn variables(&self) -> &[ResolvedVariable] {
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

    /// Borrow direct affine terms; otherwise reuse resident terms or extract uncached.
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
        if let ExprNode::Linear { coeffs, constant } = self.arena.get(expr) {
            return Ok(LinearTerms::borrowed(coeffs, *constant));
        }
        if self.expressions.resident.is_some() {
            return self.require_linear(expr, location).map(AffineTerms::into_owned);
        }
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

    /// Borrow direct affine terms or extract a polynomial of degree at most two.
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

    /// Detect a cone representation of the row using shared quadratic analysis.
    pub fn detected_soc(&self, row: &AlgebraicConstraint) -> Option<SocForm> {
        self.expressions.detected_soc(self.variables(), row)
    }
}

/// Subtract the body's constant from both row bounds.
#[inline]
pub fn shifted_bounds(row: &AlgebraicConstraint, constant: f64) -> (f64, f64) {
    (row.lower - constant, row.upper - constant)
}

/// Return single-sided bounds, lower first for ranges; free rows have no sides.
#[inline]
pub fn row_sides(row: &AlgebraicConstraint) -> [Option<(Sense, f64)>; 2] {
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
    /// Wrap an arena with empty reuse-admitted caches.
    pub fn new(arena: ExprArena) -> Self {
        Self {
            arena,
            linear: ExtractionCache::default(),
            quadratic: ExtractionCache::default(),
            resident: None,
            workset: None,
            operators: OnceLock::new(),
            #[cfg(test)]
            affine_extractions: AtomicU64::new(0),
        }
    }

    /// Enable cached classification and shared quadratic analysis for a solve.
    pub(crate) fn for_preparation(arena: ExprArena) -> Self {
        arena.__enable_preparation_classifications();
        Self { workset: Some(QuadraticWorkset::default()), ..Self::new(arena) }
    }

    /// Attach the resident cache for reuse across solves.
    pub(crate) fn with_resident(
        mut self,
        cache: Arc<crate::resident::ResidentExpressions>,
    ) -> Self {
        self.resident = Some(cache);
        self
    }

    #[cfg(test)]
    /// Return quadratic analysis counters, or zeros without a workset.
    pub fn quadratic_analysis_stats(&self) -> QuadraticAnalysisStats {
        self.workset.as_ref().map_or_else(QuadraticAnalysisStats::default, QuadraticWorkset::stats)
    }
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }

    /// Retain the first operator summaries supplied by requirement inference.
    pub(crate) fn set_operator_summaries(
        &self,
        summaries: Arc<crate::requirements::OperatorSummaries>,
    ) {
        // Compact and detailed checks prove the same operator facts.
        let _ = self.operators.set(summaries);
    }

    /// Check for sums, products, negation, and constant powers only, regardless of degree.
    /// Compound roots without inferred summaries return false.
    ///
    /// # Panics
    /// Panics for a root outside this arena.
    pub fn has_only_polynomial_operators(&self, root: ExprId) -> bool {
        if matches!(
            self.arena.get(root),
            ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
        ) {
            return true;
        }
        const MASK: u64 = crate::ExpressionOperator::Add.bit()
            | crate::ExpressionOperator::Multiply.bit()
            | crate::ExpressionOperator::Unary(oximo_expr::UnaryOp::Neg).bit()
            | crate::ExpressionOperator::ConstantPower.bit();
        self.operators
            .get()
            .and_then(|summaries| summaries.get(root))
            .is_some_and(|operators| operators & !MASK == 0)
    }
    /// Borrow direct affine terms, reuse cached polynomials, or extract through the affine cache.
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
        let quadratic = self.cached_quadratic(expr);
        if quadratic.as_ref().is_some_and(|terms| !terms.hessian.is_empty()) {
            return None;
        }

        let cached_affine = || {
            quadratic.as_ref().map(|terms| LinearTerms::owned(terms.linear.clone(), terms.constant))
        };
        let extract = || {
            if let Some(terms) = cached_affine() {
                return Some(terms);
            }

            #[cfg(test)]
            self.affine_extractions.fetch_add(1, Ordering::Relaxed);
            extract_linear(&self.arena, expr)
        };
        let affine = if let Some(cache) = &self.resident {
            if !cache.admits(expr) {
                return extract().map(AffineTerms::Borrowed);
            }
            resident_extract(&cache.linear, expr, || extract().map(LinearTerms::into_owned))
                .map(AffineTerms::Shared)
        } else {
            self.linear
                .extract(expr, || extract().map(LinearTerms::into_owned), |t| t.coeffs.len())
                .map(|terms| match terms {
                    Extracted::Owned(terms) => AffineTerms::Borrowed(terms),
                    Extracted::Shared(terms) => AffineTerms::Shared(terms),
                })
        };

        affine.or_else(|| {
            let recovered = cached_affine()?.into_owned();
            let terms = if let Some(cache) = &self.resident {
                let mut entries = cache.linear.lock().expect("resident cache poisoned");
                let terms =
                    entries.entry(expr).or_default().get_or_insert_with(|| Arc::new(recovered));
                Extracted::Shared(Arc::clone(terms))
            } else {
                self.linear.publish(expr, Some(recovered), |t| t.coeffs.len())?
            };
            Some(match terms {
                Extracted::Owned(terms) => AffineTerms::Borrowed(terms),
                Extracted::Shared(terms) => AffineTerms::Shared(terms),
            })
        })
    }

    /// Look up successful quadratic terms in the workset, resident cache, then reuse-admitted cache.
    /// Return `None` when no successful result is cached, without extracting coefficients.
    fn cached_quadratic(&self, expr: ExprId) -> Option<Extracted<QuadraticTerms>> {
        if let Some(workset) = &self.workset
            && let workset::Lookup::Ready(Some(terms)) = workset.lookup(expr)
        {
            return Some(terms);
        }
        if let Some(cache) = &self.resident
            && cache.admits(expr)
            && let Some(Some(terms)) =
                cache.quadratic.lock().expect("resident cache poisoned").get(&expr)
        {
            return Some(Extracted::Shared(Arc::clone(terms)));
        }
        self.quadratic.lookup(expr).map(Extracted::Shared)
    }

    /// Extract a polynomial of degree at most two, reusing cached results and failures.
    /// Preparations cache first results within budget. Standalone caches require reuse.
    ///
    /// # Panics
    /// Panics for an expression outside this arena or a poisoned cache.
    pub fn quadratic(&self, expr: ExprId) -> Option<Extracted<QuadraticTerms>> {
        if let Some(workset) = &self.workset {
            if let Some(cache) = &self.resident
                && cache.admits(expr)
                && let Some(value) =
                    cache.quadratic.lock().expect("resident cache poisoned").get(&expr)
            {
                return value.clone().map(Extracted::Shared);
            }
            return workset.extract(expr, || {
                if let Some(cache) = &self.resident
                    && cache.admits(expr)
                {
                    resident_extract(&cache.quadratic, expr, || {
                        extract_quadratic(&self.arena, expr)
                    })
                } else {
                    extract_quadratic(&self.arena, expr).map(Arc::new)
                }
            });
        }
        self.quadratic_unregistered(expr)
    }

    /// Reuse registered quadratic analysis or extract without registering the root.
    #[doc(hidden)]
    pub fn quadratic_once(&self, expr: ExprId) -> Option<Extracted<QuadraticTerms>> {
        if let Some(workset) = &self.workset {
            match workset.lookup(expr) {
                workset::Lookup::Ready(value) => return value,
                workset::Lookup::Retry => return self.quadratic(expr),
                workset::Lookup::Missing => {}
            }
        }
        if oximo_expr::classify(&self.arena, expr) == oximo_expr::ExprClass::Nonlinear {
            return None;
        }
        self.quadratic_unregistered(expr)
    }

    /// Extract through resident or reuse-admitted caches without registering analysis.
    fn quadratic_unregistered(&self, expr: ExprId) -> Option<Extracted<QuadraticTerms>> {
        let extract = || {
            #[cfg(test)]
            if let Some(workset) = &self.workset {
                workset.record_streamed_attempt();
            }
            extract_quadratic(&self.arena, expr)
        };
        if let Some(cache) = &self.resident {
            if !cache.admits(expr) {
                return extract().map(Extracted::Owned);
            }
            return resident_extract(&cache.quadratic, expr, extract).map(Extracted::Shared);
        }
        self.quadratic.extract(expr, extract, |t| t.linear.len() + t.hessian.len())
    }

    /// Detect an optional cone representation using cached polynomial terms.
    pub fn detected_soc(
        &self,
        variables: &[ResolvedVariable],
        row: &AlgebraicConstraint,
    ) -> Option<SocForm> {
        let q = self.quadratic(row.lhs)?;
        oximo_core::__detect_soc_from_quadratic(variables, row, &q)
    }

    /// Extract owned affine terms for an explicit cone, reusing cached terms.
    ///
    /// # Errors
    /// Returns a backend error for invalid affine cone members.
    pub fn explicit_soc(&self, soc: &SocConstraint) -> Result<SocForm, SolverError> {
        let affine = |expr| {
            self.linear(expr).map(AffineTerms::into_owned).ok_or_else(|| {
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

fn resident_extract<T>(
    cache: &Mutex<FxHashMap<ExprId, Option<Arc<T>>>>,
    expr: ExprId,
    extract: impl FnOnce() -> Option<T>,
) -> Option<Arc<T>> {
    if let Some(value) = cache.lock().expect("resident cache poisoned").get(&expr) {
        return value.clone();
    }
    // Fold outside the lock so independent dirty roots can run in parallel.
    let value = extract().map(Arc::new);
    cache.lock().expect("resident cache poisoned").entry(expr).or_insert(value).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polynomial_caches(arena: &ExprArena, root: ExprId) -> [PreparedExpressions; 2] {
        let caches = [
            PreparedExpressions::for_preparation(arena.clone()),
            PreparedExpressions::new(arena.clone()),
        ];
        for prepared in &caches {
            prepared.quadratic(root);
            prepared.quadratic(root);
        }
        caches
    }

    #[test]
    fn cached_nested_quadratics_skip_affine_extraction() {
        use oximo_expr::VarId;
        for coefficient in [1.0, 1e-100] {
            let mut arena = ExprArena::new();
            let var = arena.var(VarId(0));
            let scale = arena.push(ExprNode::Const(coefficient));
            let mut root = arena.push(ExprNode::Mul(vec![scale, var, var].into()));
            let affine = arena.linear(vec![(VarId(0), 2.0)], 3.0);
            for _ in 0..128 {
                root = arena.push(ExprNode::Add(vec![root, affine].into()));
            }

            for prepared in polynomial_caches(&arena, root) {
                let attempts = prepared.quadratic_analysis_stats().attempts;
                for _ in 0..3 {
                    assert!(prepared.linear(root).is_none());
                    assert_eq!(prepared.quadratic_once(root).unwrap().hessian.len(), 1);
                }
                assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), 0);
                assert_eq!(prepared.quadratic_analysis_stats().attempts, attempts);
                assert!(prepared.linear.shards.get().is_none());
            }
        }
    }

    #[test]
    fn cached_cancellation_yields_affine_terms_without_extraction() {
        use oximo_expr::{UnaryOp, VarId};
        let mut arena = ExprArena::new();
        let var = arena.var(VarId(0));
        let square = arena.push(ExprNode::Mul(vec![var, var].into()));
        let negated = arena.push(ExprNode::Unary(UnaryOp::Neg, square));
        let affine = arena.linear(vec![(VarId(0), 3.0)], 5.0);
        let sum = arena.push(ExprNode::Add(vec![square, affine].into()));
        let root = arena.push(ExprNode::Add(vec![sum, negated].into()));
        for prepared in polynomial_caches(&arena, root) {
            assert_eq!(prepared.quadratic(root).unwrap().hessian.len(), 0);
            let terms = prepared.linear(root).unwrap();
            assert_eq!(terms.coeffs.as_ref(), &[(VarId(0), 3.0)]);
            assert_eq!(terms.constant, 5.0);
            assert!(matches!(prepared.linear(root), Some(AffineTerms::Shared(_))));
            assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn cached_polynomial_affinity_overrides_prior_affine_failure() {
        use oximo_expr::{UnaryOp, VarId};
        let mut arena = ExprArena::new();
        let var = arena.var(VarId(0));
        let square = arena.push(ExprNode::Mul(vec![var, var].into()));
        let negated = arena.push(ExprNode::Unary(UnaryOp::Neg, square));
        let root = arena.push(ExprNode::Add(vec![square, negated, var].into()));
        for prepared in
            [PreparedExpressions::for_preparation(arena.clone()), PreparedExpressions::new(arena)]
        {
            assert!(prepared.linear(root).is_none());
            assert!(prepared.linear(root).is_none());
            prepared.quadratic(root).unwrap();
            prepared.quadratic(root).unwrap();
            let recovered = prepared.linear(root).unwrap();
            assert_eq!(recovered.coeffs.as_ref(), &[(VarId(0), 1.0)]);
            let AffineTerms::Shared(recovered) = recovered else {
                panic!("recovered affine terms were not cached")
            };
            for _ in 0..3 {
                let Some(AffineTerms::Shared(repeated)) = prepared.linear(root) else {
                    panic!("recovered affine terms were not reused")
                };
                assert!(Arc::ptr_eq(&recovered, &repeated));
            }
            assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), 2);
        }
    }

    #[test]
    fn direct_affine_borrows_and_uncached_compounds_still_extract() {
        use oximo_expr::{UnaryOp, VarId};
        let mut arena = ExprArena::new();
        let affine = arena.linear(vec![(VarId(0), 2.0)], 3.0);
        let negated = arena.push(ExprNode::Unary(UnaryOp::Neg, affine));
        let prepared = PreparedExpressions::for_preparation(arena);
        let terms = prepared.linear(affine).unwrap();
        let ExprNode::Linear { coeffs, .. } = prepared.arena.get(affine) else {
            panic!("direct affine node")
        };
        assert!(matches!(terms, AffineTerms::Borrowed(_)));
        assert!(std::ptr::eq(terms.coeffs.as_ptr(), coeffs.as_ptr()));
        assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), 0);
        assert_eq!(prepared.quadratic_analysis_stats().roots, 0);

        let terms = prepared.linear(negated).unwrap();
        assert_eq!(terms.coeffs.as_ref(), &[(VarId(0), -2.0)]);
        assert_eq!(terms.constant, -3.0);
        assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), 1);
        assert_eq!(prepared.quadratic_analysis_stats().attempts, 0);
        assert!(prepared.quadratic.shards.get().is_none());
    }

    #[test]
    fn cached_polynomial_failures_preserve_affine_and_nonlinear_fallback() {
        use oximo_expr::{UnaryOp, VarId};
        let mut arena = ExprArena::new();
        let var = arena.var(VarId(0));
        let two = arena.push(ExprNode::Const(2.0));
        let three = arena.push(ExprNode::Const(3.0));
        let cube = arena.push(ExprNode::Pow(two, three));
        let affine = arena.push(ExprNode::Add(vec![var, cube].into()));
        let nonlinear = arena.push(ExprNode::Unary(UnaryOp::Sin, var));
        for root in [affine, nonlinear] {
            for prepared in polynomial_caches(&arena, root) {
                assert!(prepared.quadratic(root).is_none());
                let terms = prepared.linear(root);
                if root == affine {
                    let terms = terms.unwrap();
                    assert_eq!(terms.coeffs.as_ref(), &[(VarId(0), 1.0)]);
                    assert_eq!(terms.constant, 8.0);
                } else {
                    assert!(terms.is_none());
                    assert!(prepared.quadratic_once(root).is_none());
                }
                assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), 1);
            }
        }
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact zero controls polynomial affinity")]
    fn parameter_changes_refresh_cached_polynomial_affinity() {
        use crate::resident::PersistentPreparation;
        use oximo_core::prelude::*;

        let model = Model::new("cached numeric affinity");
        variable!(model, x);
        param!(model, p = 1.0);
        let root = p * x * x + 3.0 * x + p;
        objective!(model, Min, root);

        let mut persistent = PersistentPreparation::default();
        for value in [1.0, 0.0, 1e-100, 2.0, 0.0] {
            model.set_param(p, value).unwrap();
            let fresh = LoweringContext::new(&model).unwrap();
            let resident = persistent.prepare(&model).unwrap();
            for prepared in [&*fresh, &*resident] {
                if value == 0.0 {
                    assert!(prepared.linear(root.id()).is_none());
                    assert!(prepared.linear(root.id()).is_none());
                }
                let attempts = prepared.affine_extractions.load(Ordering::Relaxed);
                let quadratic = prepared.quadratic(root.id()).unwrap();
                let affine = prepared.linear(root.id());
                if value == 0.0 {
                    assert_eq!(quadratic.hessian.len(), 0);
                    let affine = affine.unwrap();
                    assert_eq!(affine.coeffs.as_ref(), &[(x.var_id().unwrap(), 3.0)]);
                    assert_eq!(affine.constant, value);
                    if prepared.resident.as_ref().is_none_or(|cache| cache.admits(root.id())) {
                        let AffineTerms::Shared(recovered) = affine else {
                            panic!("recovered affine terms were not cached")
                        };
                        let Some(AffineTerms::Shared(repeated)) = prepared.linear(root.id()) else {
                            panic!("recovered affine terms were not reused")
                        };
                        assert!(Arc::ptr_eq(&recovered, &repeated));
                    }
                } else {
                    assert!(affine.is_none());
                    assert_eq!(quadratic.hessian[0].2, 2.0 * value);
                }
                assert_eq!(prepared.affine_extractions.load(Ordering::Relaxed), attempts);
            }
            drop(resident);
            persistent.commit();

            let unchanged = persistent.prepare(&model).unwrap();
            if value != 1.0 {
                assert!(unchanged.resident.as_ref().unwrap().admits(root.id()));
                assert_eq!(unchanged.linear(root.id()).is_some(), value == 0.0);
                assert_eq!(unchanged.affine_extractions.load(Ordering::Relaxed), 0);
                assert_eq!(unchanged.quadratic_analysis_stats().attempts, 0);
            }

            drop(unchanged);
            persistent.commit();
        }
    }

    #[test]
    fn classification_matching_and_translation_share_one_snapshot_extraction() {
        use oximo_core::prelude::*;
        let model = Model::new("shared analysis");
        variable!(model, x >= 0.0);
        variable!(model, t >= 0.0);
        param!(model, p = 1.0);
        let root = p * x * x - t * t;
        constraint!(model, cone, root <= 0.0);
        objective!(model, Min, t);

        let prepared = LoweringContext::new(&model).unwrap();
        assert_eq!(prepared.kind(), ModelKind::SOCP);

        let expected = crate::ModelRequirements::infer(&model);
        let profile = crate::SolverProfile::new("unsupported");
        let prepared_report = profile.check_prepared(&model, &prepared).unwrap();
        let standalone_report = profile.check(&expected);
        assert_eq!(format!("{prepared_report:?}"), format!("{standalone_report:?}"));
        assert!(prepared.quadratic_once(root.id()).is_some());

        let stats = prepared.quadratic_analysis_stats();
        assert_eq!(
            (stats.attempts, stats.roots, stats.re_extractions, stats.bypasses),
            (1, 1, 0, 0)
        );
        model.set_param(p, 0.0).unwrap();
        assert!(profile.check_prepared(&model, &prepared).is_err());
        assert_eq!(prepared.quadratic(root.id()).unwrap().hessian.len(), 2);

        drop(prepared);
        let prepared = LoweringContext::new(&model).unwrap();
        assert_eq!(prepared.kind(), ModelKind::QCP);
        assert_eq!(prepared.quadratic(root.id()).unwrap().hessian.len(), 1);
        assert_eq!(prepared.quadratic_analysis_stats().attempts, 1);
    }

    #[test]
    fn nonlinear_translation_roots_skip_speculative_coefficient_folds() {
        use oximo_core::prelude::*;
        let model = Model::new("streamed nonlinear failure");
        variable!(model, x);
        objective!(model, Min, x * x + x.sin());

        let prepared = LoweringContext::new(&model).unwrap();
        let root = prepared.objective().unwrap().expr;
        assert!(prepared.quadratic_once(root).is_none());

        let stats = prepared.quadratic_analysis_stats();
        assert_eq!((stats.attempts, stats.roots, stats.streamed_attempts), (0, 0, 0));
        assert_eq!(stats.retained_bytes, 0);
        assert!(prepared.quadratic(root).is_none());
        assert_eq!(prepared.quadratic_analysis_stats().attempts, 1);
        assert!(prepared.quadratic_once(root).is_none());
        assert_eq!(prepared.quadratic_analysis_stats().attempts, 1);
    }

    #[test]
    fn bounds_activity_and_objectives_are_reclassified_without_cross_model_reuse() {
        use oximo_core::prelude::*;
        let model = Model::new("snapshot validity");
        variable!(model, x);
        variable!(model, t >= 0.0);
        let row = constraint!(model, cone, x * x <= t * t);
        objective!(model, Min, t);
        assert_eq!(LoweringContext::new(&model).unwrap().kind(), ModelKind::SOCP);

        model.set_bounds(t, -1.0, 2.0).unwrap();
        assert_eq!(LoweringContext::new(&model).unwrap().kind(), ModelKind::QCP);

        model.set_constraint_active(row, false).unwrap();
        assert_eq!(LoweringContext::new(&model).unwrap().kind(), ModelKind::LP);

        objective!(model, Min, x * x);
        assert_eq!(LoweringContext::new(&model).unwrap().kind(), ModelKind::QP);

        let other = Model::new("another snapshot");
        let y = other.__var("y").integer().build();
        objective!(other, Min, y * y + 7.0);

        let prepared = LoweringContext::new(&other).unwrap();
        assert_eq!(prepared.kind(), ModelKind::MIQP);
        assert!(prepared.ensure_matches(&model).is_err());

        let root = other.objective().as_ref().unwrap().expr;
        assert_eq!(prepared.quadratic(root).unwrap().constant, 7.0);
    }

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
            assert!(cache.lookup(ExprId(index)).is_none());
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
    fn recovered_cache_entries_respect_budgets_and_reuse_publication() {
        let cache = ExtractionCache::default();
        let roots: Vec<_> = (0u32..1000)
            .filter(|id| id.wrapping_mul(0x9e37_79b9) >> 28 == 0)
            .take(CACHE_EXPRESSIONS / CACHE_SHARDS)
            .map(ExprId)
            .collect();
        for &root in &roots {
            for _ in 0..2 {
                assert!(cache.extract(root, || None, Vec::<u8>::len).is_none());
            }
        }
        for root in roots {
            let Some(Extracted::Shared(recovered)) =
                cache.publish(root, Some(vec![1; 128]), Vec::len)
            else {
                panic!("recovered terms were not cached")
            };
            let Some(Extracted::Shared(repeated)) =
                cache.extract(root, || panic!("recovered terms were re-extracted"), Vec::len)
            else {
                panic!("recovered terms were not reused")
            };
            assert!(Arc::ptr_eq(&recovered, &repeated));
            let entries = cache.shards.get().unwrap()[0].entries.lock().unwrap();
            assert_eq!(entries.values.len(), entries.order.len());
            assert!(entries.values.len() <= CACHE_EXPRESSIONS / CACHE_SHARDS);
            assert_eq!(entries.coefficients, entries.order.iter().map(|&(_, size)| size).sum());
            assert!(entries.coefficients <= CACHE_COEFFICIENTS / CACHE_SHARDS);
        }
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
