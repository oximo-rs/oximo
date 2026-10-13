//! Indexed family construction and worker-local expression preparation.

use super::constraints::{PendingConstraint, PendingSoc, PendingSos};
use super::variables::{PendingParam, PendingVar};
use super::{Model, arena_key, assert_expr_arena, validate_batch_names};
use crate::constraint::{ConstraintHandle, IntoRhs, RangeConstraintHandles, Relate};
use crate::domain::Domain;
use crate::function_set::{
    AlgebraicConstraintIr, Constraint, GreaterThan, Interval, IntoAffineConstraint,
    IntoAffineFunction, IntoFunction, LessThan, ScalarFunction, SecondOrderCone,
    VectorAffineFunction, validate_bounds,
};
use crate::indexed::{
    IndexedConstraint, IndexedFamily, IndexedIndicatorConstraint, IndexedParam,
    IndexedRangeConstraint, IndexedRangeIndicatorConstraint, IndexedVar, build_storage,
};
use crate::indicator::{IndicatorConstraintHandle, RangeIndicatorConstraintHandles};
use crate::set::{Axis, FromIndexKey, IndexKey, Set};
use crate::sos::{SosMember, SosType, validate_members};
use crate::{IntoSymmetricAffineFunction, PsdConstraint, PsdConstraintId};
use oximo_expr::{Degree, Expr, ExprClass, ModelMismatchError};
use rayon::prelude::*;
use smol_str::SmolStr;
use std::fmt;
use std::marker::PhantomData;

const PAR_INDEXED_METADATA_THRESHOLD: usize = 1_024;
const PAR_INDEXED_ALGEBRAIC_THRESHOLD: usize = 512;
const PAR_INDEXED_RANGE_THRESHOLD: usize = 512;
const PAR_INDEXED_SOC_THRESHOLD: usize = 256;
const PAR_INDEXED_SOS_THRESHOLD: usize = 512;

fn indexed_parallel(len: usize, forced: Option<bool>, threshold: usize) -> bool {
    forced.unwrap_or(len >= threshold && rayon::current_num_threads() > 1)
}

fn indexed_chunk_size(len: usize) -> usize {
    let chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    len.div_ceil(chunks).max(1)
}

#[derive(Debug)]
struct PendingRangeBatch {
    rows: Vec<PendingConstraint>,
    row_counts: Vec<u8>,
}

fn prepare_range<
    'a,
    D: crate::function_set::FunctionDegree,
    B1: IntoRhs<'a, D>,
    B2: IntoRhs<'a, D>,
>(
    name: String,
    mid: Expr<'a, D>,
    lo: B1,
    hi: B2,
) -> (PendingConstraint, Option<PendingConstraint>) {
    if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound()) {
        let function = mid.into_function();
        validate_bounds(lower, upper);
        if function.is_affine() {
            (
                PendingConstraint::from_expr(
                    name.into(),
                    Constraint::new(function, Interval { lower, upper }),
                ),
                None,
            )
        } else {
            (
                PendingConstraint::from_expr(
                    format!("{name}_lo").into(),
                    Constraint::new(function, GreaterThan(lower)),
                ),
                Some(PendingConstraint::from_expr(
                    format!("{name}_hi").into(),
                    Constraint::new(function, LessThan(upper)),
                )),
            )
        }
    } else {
        (
            PendingConstraint::from_expr(format!("{name}_lo").into(), mid.ge(lo)),
            Some(PendingConstraint::from_expr(format!("{name}_hi").into(), mid.le(hi))),
        )
    }
}

fn prepare_soc<'a, A: IntoAffineFunction<'a>, B: IntoAffineFunction<'a>>(
    name: SmolStr,
    terms: impl IntoIterator<Item = A>,
    bound: B,
    expected_arena: usize,
) -> PendingSoc {
    let function = VectorAffineFunction::new(
        std::iter::once(bound.into_affine_function())
            .chain(terms.into_iter().map(IntoAffineFunction::into_affine_function)),
    );
    assert_expr_arena(function.first, expected_arena);
    let dimension = function.dimension();
    let ir = Constraint::new(function, SecondOrderCone { dimension }).into_ir();
    PendingSoc { name, terms: ir.0.rest, bound: ir.0.first.id }
}

impl Model {
    /// Macro-facing entry point backing the indexed form of the `variable!`
    /// macro. Not part of the stable public API.
    #[doc(hidden)]
    pub fn __indexed_var<'a, K>(
        &'a self,
        name: impl Into<String>,
        set: &Set<K>,
    ) -> IndexedVarBuilder<'a, K> {
        IndexedVarBuilder {
            model: self,
            base_name: name.into(),
            keys: set.iter().collect(),
            axes: set.axes().map(Box::from),
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
            lb_by: None,
            ub_by: None,
            domain: Domain::Real,
            parallel: None,
            _k: PhantomData,
        }
    }

    /// Macro-facing entry point backing the indexed form of the `param!` macro
    /// (`param!(m, cost[i in items] = data[i])`). Registers one scalar parameter
    /// per key, evaluating `value` on the typed key, and returns an
    /// [`IndexedParam`]. Not part of the stable public API.
    /// Large families may evaluate `value` concurrently, so it must be safe to
    /// call from multiple worker threads.
    ///
    /// # Panics
    ///
    /// Panics if a per-key parameter name collides with one already registered.
    #[doc(hidden)]
    pub fn __indexed_param<'a, K, F>(
        &'a self,
        name: impl Into<String>,
        set: &Set<K>,
        value: F,
    ) -> IndexedParam<'a, K>
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + Send + Sync,
    {
        self.indexed_param_with(name.into(), set, &value, None)
    }

    pub(super) fn indexed_param_with<'a, K, F>(
        &'a self,
        base: String,
        set: &Set<K>,
        value: &F,
        forced_parallel: Option<bool>,
    ) -> IndexedParam<'a, K>
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + Send + Sync,
    {
        let axes = set.axes().map(Box::from);
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_METADATA_THRESHOLD) {
            let handles = keys
                .iter()
                .map(|key| {
                    let name = format_index_name_small(&base, key);
                    self.register_param(name, value(K::from_index_key(key)))
                })
                .collect();
            let storage = build_storage(keys, axes, handles);
            return IndexedFamily { storage, model_id: self.id(), _marker: PhantomData };
        }
        let prepare = |key: &IndexKey| PendingParam {
            name: format_index_name_small(&base, key),
            value: value(K::from_index_key(key)),
        };
        let prepared: Vec<_> = keys.par_iter().map(prepare).collect();
        let handles = self.register_params_batch(&prepared);
        drop(prepared);
        let storage = build_storage(keys, axes, handles);
        IndexedFamily { storage, model_id: self.id(), _marker: PhantomData }
    }

    /// Re-bind the parameter at `key` of an indexed family to `value`. Takes
    /// effect on the next solve.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not present in the family.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `params` belongs to another model.
    pub fn set_param_idx<K, Q: Into<IndexKey>>(
        &self,
        params: &IndexedParam<'_, K>,
        key: Q,
        value: f64,
    ) -> std::result::Result<(), ModelMismatchError> {
        self.ensure_model_id(params.model_id())?;
        let e = params.get(key).expect("set_param_idx: key not present in indexed parameter");
        let id = e.param_id().expect("indexed parameter entry is not a parameter handle");
        self.set_param_id(id, value);
        Ok(())
    }

    /// Current value bound to the parameter at `key` of an indexed family.
    /// Returns `Ok(None)` if the key is absent and [`ModelMismatchError`] if the
    /// family belongs to another model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMismatchError`] if `params` belongs to another model.
    pub fn param_value_idx<K, Q: Into<IndexKey>>(
        &self,
        params: &IndexedParam<'_, K>,
        key: Q,
    ) -> std::result::Result<Option<f64>, ModelMismatchError> {
        self.ensure_model_id(params.model_id())?;
        params.get(key).map_or(Ok(None), |e| self.param_value_of(e))
    }

    /// Macro-facing entry point backing the indexed-family form of the
    /// `constraint!` macro. The closure receives the index as a typed value `K`
    /// (any [`FromIndexKey`]: `i64`, `i32`, `usize`, `String`, raw `IndexKey`, or
    /// tuples up to arity 4). Not part of the stable public API.
    #[doc(hidden)]
    pub fn __add_constraints_over<'a, K, F, C>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: F,
    ) -> IndexedConstraint<K>
    where
        K: FromIndexKey,
        F: Fn(K) -> C + Send + Sync,
        C: Into<AlgebraicConstraintIr<'a>>,
    {
        self.add_constraints_over_with(name_prefix, set, &rule, None)
    }

    pub(super) fn add_constraints_over_with<'a, K, F, C>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: &F,
        forced_parallel: Option<bool>,
    ) -> IndexedConstraint<K>
    where
        K: FromIndexKey,
        F: Fn(K) -> C + Send + Sync,
        C: Into<AlgebraicConstraintIr<'a>>,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_ALGEBRAIC_THRESHOLD) {
            let mut ids = Vec::with_capacity(keys.len());
            for key in &keys {
                let constraint: AlgebraicConstraintIr = rule(K::from_index_key(key)).into();
                self.assert_expr_belongs(constraint.lhs);
                let name: SmolStr = format_index_name(name_prefix, key).into();
                ids.push(self.__add_constraint(name, constraint));
            }
            return IndexedConstraint::new(keys, set.axes(), ids);
        }

        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let mut forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    chunk
                        .iter()
                        .map(|key| {
                            let name = format_index_name(name_prefix, key).into();
                            let constraint: AlgebraicConstraintIr =
                                rule(K::from_index_key(key)).into();
                            assert_expr_arena(constraint.lhs, expected_arena);
                            PendingConstraint::from_expr(name, constraint)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        drop(snapshot);
        {
            let existing = self.constraint_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.iter().map(|item| &item.name)),
                "constraint",
                keys.len(),
            );
        }
        let remaps = batch.merge(&mut forks);
        let mut pending = Vec::with_capacity(keys.len());
        for (mut fork, remap) in forks.into_iter().zip(remaps) {
            for item in &mut fork.value {
                item.remap(remap);
            }
            pending.extend(fork.value);
        }
        drop(batch);
        let ids = self
            .register_prevalidated_constraints_batch(pending)
            .into_iter()
            .map(|id| ConstraintHandle::new(id, self.id()))
            .collect();
        IndexedConstraint::new(keys, set.axes(), ids)
    }

    /// Macro-facing entry point for a two-sided range family. Each key maps to
    /// one interval row or separate lower/upper rows (see [`Self::__add_range`]).
    #[doc(hidden)]
    pub fn __add_range_constraints_over<'a, K, B1, B2, F, D: crate::function_set::FunctionDegree>(
        &'a self,
        name: &str,
        set: &Set<K>,
        rule: F,
    ) -> IndexedRangeConstraint<K>
    where
        K: FromIndexKey,
        B1: IntoRhs<'a, D>,
        B2: IntoRhs<'a, D>,
        F: Fn(K) -> (Expr<'a, D>, B1, B2) + Send + Sync,
    {
        self.add_range_constraints_over_with(name, set, &rule, None)
    }

    pub(super) fn add_range_constraints_over_with<
        'a,
        K,
        B1,
        B2,
        F,
        D: crate::function_set::FunctionDegree,
    >(
        &'a self,
        name: &str,
        set: &Set<K>,
        rule: &F,
        forced_parallel: Option<bool>,
    ) -> IndexedRangeConstraint<K>
    where
        K: FromIndexKey,
        B1: IntoRhs<'a, D>,
        B2: IntoRhs<'a, D>,
        F: Fn(K) -> (Expr<'a, D>, B1, B2) + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_RANGE_THRESHOLD) {
            let mut ids = Vec::with_capacity(keys.len());
            for key in &keys {
                let (mid, lo, hi) = rule(K::from_index_key(key));
                self.assert_expr_belongs(mid);
                ids.push(self.__add_range(&format_index_name(name, key), mid, lo, hi));
            }
            return IndexedRangeConstraint::new(keys, set.axes(), ids);
        }

        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let mut forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    let mut prepared = PendingRangeBatch {
                        rows: Vec::with_capacity(chunk.len()),
                        row_counts: Vec::with_capacity(chunk.len()),
                    };
                    for key in chunk {
                        let (mid, lo, hi) = rule(K::from_index_key(key));
                        assert_expr_arena(mid, expected_arena);
                        let (first, upper) =
                            prepare_range(format_index_name(name, key), mid, lo, hi);
                        prepared.row_counts.push(if upper.is_some() { 2 } else { 1 });
                        prepared.rows.push(first);
                        prepared.rows.extend(upper);
                    }
                    prepared
                })
            })
            .collect();
        drop(snapshot);
        let row_count = forks.iter().map(|fork| fork.value.rows.len()).sum();
        {
            let existing = self.constraint_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.rows.iter().map(|item| &item.name)),
                "constraint",
                row_count,
            );
        }
        let remaps = batch.merge(&mut forks);
        let mut pending = Vec::with_capacity(row_count);
        let mut row_counts = Vec::with_capacity(keys.len());
        for (mut fork, remap) in forks.into_iter().zip(remaps) {
            for item in &mut fork.value.rows {
                item.remap(remap);
            }
            row_counts.extend(fork.value.row_counts);
            pending.extend(fork.value.rows);
        }
        drop(batch);
        let mut ids = self.register_prevalidated_constraints_batch(pending).into_iter();
        let groups = row_counts
            .into_iter()
            .map(|count| {
                let first = ids.next().expect("range row ID missing");
                match count {
                    1 => RangeConstraintHandles::Interval(ConstraintHandle::new(first, self.id())),
                    2 => RangeConstraintHandles::Split {
                        lower: ConstraintHandle::new(first, self.id()),
                        upper: ConstraintHandle::new(
                            ids.next().expect("upper range row ID missing"),
                            self.id(),
                        ),
                    },
                    _ => unreachable!("range must lower to one or two rows"),
                }
            })
            .collect();
        IndexedRangeConstraint::new(keys, set.axes(), groups)
    }

    /// Transactional indexed PSD registration with deterministic worker arena remapping.
    #[doc(hidden)]
    pub fn __add_psd_constraints_over<'a, K, M, F>(&'a self, prefix: &str, set: &Set<K>, rule: F)
    where
        K: FromIndexKey,
        M: IntoSymmetricAffineFunction<'a>,
        F: Fn(K) -> M + Send + Sync,
    {
        let keys: Vec<_> = set.iter().collect();
        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let prepare = |chunk: &[IndexKey]| {
            arena.__with_fork(snapshot.clone(), || {
                chunk
                    .iter()
                    .map(|key| {
                        let function =
                            rule(K::from_index_key(key)).__into_symmetric_affine_function(arena);
                        for &entry in function.matrix().upper_triangle() {
                            assert_expr_arena(entry, expected_arena);
                        }
                        (
                            SmolStr::from(format_index_name(prefix, key)),
                            function.matrix().map(|entry| entry.id()),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        };
        let mut forks: Vec<_> = if indexed_parallel(keys.len(), None, PAR_INDEXED_SOC_THRESHOLD) {
            keys.par_chunks(indexed_chunk_size(keys.len())).map(prepare).collect()
        } else {
            vec![prepare(&keys)]
        };
        drop(snapshot);
        let mut names = self.psd_names.borrow_mut();
        validate_batch_names(
            &names,
            forks.iter().flat_map(|fork| fork.value.iter().map(|(name, _)| name)),
            "PSD constraint",
            keys.len(),
        );
        let mut all = self.psd_constraints.borrow_mut();
        let final_count = all.len().checked_add(keys.len()).expect("PSD count overflow");
        if final_count > 0 {
            u32::try_from(final_count - 1).expect("PSD count overflow");
        }
        let remaps = batch.merge(&mut forks);
        drop(batch);
        for (fork, remap) in forks.into_iter().zip(remaps) {
            for (name, matrix) in fork.value {
                let id = PsdConstraintId(u32::try_from(all.len()).expect("PSD count overflow"));
                all.push(PsdConstraint {
                    name: name.clone(),
                    matrix: matrix.map(|&entry| remap.apply(entry)),
                    active: true,
                });
                names.insert(name, id);
            }
        }
        self.invalidate_kind();
    }

    /// Macro-facing entry point backing the indexed-family form of the
    /// `soc_constraint!` macro: one cone per key, named `{prefix}[{key}]`. The
    /// closure returns the cone's `(terms, bound)` pair for each typed key.
    /// Not part of the stable public API.
    #[doc(hidden)]
    pub fn __add_soc_constraints_over<
        'a,
        K,
        T,
        F,
        A: IntoAffineFunction<'a>,
        B: IntoAffineFunction<'a>,
    >(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = A>,
        F: Fn(K) -> (T, B) + Send + Sync,
    {
        self.add_soc_constraints_over_with(name_prefix, set, &rule, None);
    }

    pub(super) fn add_soc_constraints_over_with<
        'a,
        K,
        T,
        F,
        A: IntoAffineFunction<'a>,
        B: IntoAffineFunction<'a>,
    >(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        rule: &F,
        forced_parallel: Option<bool>,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = A>,
        F: Fn(K) -> (T, B) + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_SOC_THRESHOLD) {
            for key in &keys {
                let name: SmolStr = format_index_name(name_prefix, key).into();
                let (terms, bound) = rule(K::from_index_key(key));
                self.add_soc_constraint(name, terms, bound);
            }
            return;
        }

        let arena = &self.arena;
        let expected_arena = arena_key(arena);
        let mut batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let mut forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    chunk
                        .iter()
                        .map(|key| {
                            let name = format_index_name(name_prefix, key).into();
                            let (terms, bound) = rule(K::from_index_key(key));
                            prepare_soc(name, terms, bound, expected_arena)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        drop(snapshot);
        {
            let existing = self.soc_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.iter().map(|item| &item.name)),
                "SOC constraint",
                keys.len(),
            );
        }
        let remaps = batch.merge(&mut forks);
        let mut pending = Vec::with_capacity(keys.len());
        for (mut fork, remap) in forks.into_iter().zip(remaps) {
            for item in &mut fork.value {
                item.remap(remap);
            }
            pending.extend(fork.value);
        }
        drop(batch);
        self.register_prevalidated_soc_batch(pending);
    }

    fn add_pending_sos_over(
        &self,
        keys: Vec<IndexKey>,
        sos_type: SosType,
        forced_parallel: Option<bool>,
        prepare: impl Fn(&IndexKey) -> PendingSos + Send + Sync,
    ) {
        if !indexed_parallel(keys.len(), forced_parallel, PAR_INDEXED_SOS_THRESHOLD) {
            for key in &keys {
                self.register_sos_one(sos_type, prepare(key));
            }
            return;
        }

        let arena = &self.arena;
        let batch = arena.__begin_batch();
        let snapshot = batch.snapshot();
        let chunk_size = indexed_chunk_size(keys.len());
        let forks: Vec<_> = keys
            .par_chunks(chunk_size)
            .map(|chunk| {
                arena.__with_fork(snapshot.clone(), || {
                    chunk.iter().map(&prepare).collect::<Vec<_>>()
                })
            })
            .collect();
        drop(snapshot);
        {
            let existing = self.sos_names.borrow();
            validate_batch_names(
                &existing,
                forks.iter().flat_map(|fork| fork.value.iter().map(|item| &item.name)),
                "SOS constraint",
                keys.len(),
            );
        }
        // Valid SOS members contain only VarIds, so no expression-root remap is needed.
        drop(batch);
        let pending = forks.into_iter().flat_map(|fork| fork.value).collect();
        self.register_prevalidated_sos_batch(sos_type, pending);
    }

    #[doc(hidden)]
    pub fn __add_sos_constraints_over<'a, K, T, F, D: Degree>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = (Expr<'a, D>, f64)>,
        F: Fn(K) -> T + Send + Sync,
    {
        self.add_sos_constraints_over_with(name_prefix, set, sos_type, &rule, None);
    }

    pub(super) fn add_sos_constraints_over_with<'a, K, T, F, D: Degree>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: &F,
        forced_parallel: Option<bool>,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = (Expr<'a, D>, f64)>,
        F: Fn(K) -> T + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        let expected_arena = arena_key(&self.arena);
        self.add_pending_sos_over(keys, sos_type, forced_parallel, |key| {
            let name: SmolStr = format_index_name(name_prefix, key).into();
            let members: Vec<_> = rule(K::from_index_key(key))
                .into_iter()
                .map(|(expr, weight)| {
                    assert_expr_arena(expr, expected_arena);
                    let variable = expr.var_id().expect("SOS members must be bare variables");
                    SosMember { variable, weight }
                })
                .collect();
            validate_members(&name, &members);
            PendingSos { name, members }
        });
    }

    #[doc(hidden)]
    pub fn __add_sos_constraints_over_auto_weights<'a, K, T, F, D: Degree>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: F,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a, D>>,
        F: Fn(K) -> T + Send + Sync,
    {
        self.add_sos_constraints_over_auto_weights_with(name_prefix, set, sos_type, &rule, None);
    }

    pub(super) fn add_sos_constraints_over_auto_weights_with<'a, K, T, F, D: Degree>(
        &'a self,
        name_prefix: &str,
        set: &Set<K>,
        sos_type: SosType,
        rule: &F,
        forced_parallel: Option<bool>,
    ) where
        K: FromIndexKey,
        T: IntoIterator<Item = Expr<'a, D>>,
        F: Fn(K) -> T + Send + Sync,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        let expected_arena = arena_key(&self.arena);
        self.add_pending_sos_over(keys, sos_type, forced_parallel, |key| {
            let name: SmolStr = format_index_name(name_prefix, key).into();
            let members: Vec<_> = rule(K::from_index_key(key))
                .into_iter()
                .enumerate()
                .map(|(index, expr)| {
                    let weight = index
                        .checked_add(1)
                        .and_then(|index| u32::try_from(index).ok())
                        .expect("SOS member count exceeds positional weight range");
                    assert_expr_arena(expr, expected_arena);
                    let variable = expr.var_id().expect("SOS members must be bare variables");
                    SosMember { variable, weight: f64::from(weight) }
                })
                .collect();
            validate_members(&name, &members);
            PendingSos { name, members }
        });
    }

    #[doc(hidden)]
    pub fn __add_indicator_constraints_over<'a, K, F, C, D: Degree>(
        &'a self,
        prefix: &str,
        set: &Set<K>,
        rule: F,
    ) -> IndexedIndicatorConstraint<'a, K>
    where
        K: FromIndexKey,
        F: Fn(K) -> (Expr<'a, D>, bool, C),
        C: IntoAffineConstraint<'a>,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        let pending: Vec<_> = keys
            .iter()
            .map(|key| {
                let (trigger, value, consequent) = rule(K::from_index_key(key));
                let trigger = self.indicator_trigger(trigger.erase());
                self.prepare_indicator(
                    format_index_name(prefix, key).into(),
                    trigger,
                    value,
                    consequent,
                )
            })
            .collect();
        let handles = self
            .register_indicators_batch(pending)
            .into_iter()
            .map(|id| IndicatorConstraintHandle { model: self, id })
            .collect();
        IndexedIndicatorConstraint::new(keys, set.axes(), handles)
    }

    #[doc(hidden)]
    pub fn __add_indicator_ranges_over<'a, K, B1, B2, F, D: Degree, E: Degree>(
        &'a self,
        prefix: &str,
        set: &Set<K>,
        rule: F,
    ) -> IndexedRangeIndicatorConstraint<'a, K>
    where
        K: FromIndexKey,
        B1: IntoRhs<'a>,
        B2: IntoRhs<'a>,
        F: Fn(K) -> (Expr<'a, D>, bool, Expr<'a, E>, B1, B2),
        Expr<'a, E>: IntoAffineFunction<'a>,
    {
        let keys: Vec<IndexKey> = set.iter().collect();
        let mut pending = Vec::new();
        let mut row_counts = Vec::with_capacity(keys.len());
        for key in &keys {
            let (trigger, value, mid, lo, hi) = rule(K::from_index_key(key));
            let mid = mid.into_affine_function().expression().erase();
            self.assert_expr_belongs(mid);
            let trigger = self.indicator_trigger(trigger.erase());
            let name = format_index_name(prefix, key);
            if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound())
                && mid.__class() == ExprClass::Linear
            {
                pending.push(self.prepare_indicator_interval(
                    name.into(),
                    trigger,
                    value,
                    mid.id,
                    lower,
                    upper,
                ));
                row_counts.push(1_u8);
            } else {
                pending.push(self.prepare_indicator(
                    format!("{name}_lo").into(),
                    trigger,
                    value,
                    mid.ge(lo).into_ir().into_affine(),
                ));
                pending.push(self.prepare_indicator(
                    format!("{name}_hi").into(),
                    trigger,
                    value,
                    mid.le(hi).into_ir().into_affine(),
                ));
                row_counts.push(2_u8);
            }
        }
        let mut ids = self.register_indicators_batch(pending).into_iter();
        let handles = row_counts
            .into_iter()
            .map(|count| {
                let lower = IndicatorConstraintHandle {
                    model: self,
                    id: ids.next().expect("indicator range ID missing"),
                };
                match count {
                    1 => RangeIndicatorConstraintHandles::Interval(lower),
                    2 => RangeIndicatorConstraintHandles::Split {
                        lower,
                        upper: IndicatorConstraintHandle {
                            model: self,
                            id: ids.next().expect("upper indicator range ID missing"),
                        },
                    },
                    _ => unreachable!("indicator range must lower to one or two rows"),
                }
            })
            .collect();
        IndexedRangeIndicatorConstraint::new(keys, set.axes(), handles)
    }
}

/// Builder for a collection of scalar variables indexed by a [`Set`].
///
/// For example, `flow[i]` for `i in 0..3` registers `flow[0]`, `flow[1]`, and
/// `flow[2]` as separate scalar variables in the model. Call `.build()` to get
/// an [`IndexedVar`] that maps each key to its [`Expr`] handle. Bounds and
/// domain set here apply uniformly to every scalar in the collection.
type BoundFn<'a> = Box<dyn Fn(&IndexKey) -> f64 + Send + Sync + 'a>;

#[must_use = "IndexedVarBuilder does nothing until you call .build()"]
pub struct IndexedVarBuilder<'a, K = IndexKey> {
    model: &'a Model,
    base_name: String,
    keys: Vec<IndexKey>,
    axes: Option<Box<[Axis]>>,
    lb: f64,
    ub: f64,
    lb_by: Option<BoundFn<'a>>,
    ub_by: Option<BoundFn<'a>>,
    domain: Domain,
    parallel: Option<bool>,
    _k: PhantomData<fn() -> K>,
}

impl<'a, K> std::fmt::Debug for IndexedVarBuilder<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedVarBuilder")
            .field("base_name", &self.base_name)
            .field("keys", &self.keys.len())
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("per_key_lb", &self.lb_by.is_some())
            .field("per_key_ub", &self.ub_by.is_some())
            .field("domain", &self.domain)
            .finish()
    }
}

impl<'a, K> IndexedVarBuilder<'a, K> {
    pub fn lb(mut self, v: f64) -> Self {
        self.lb = v;
        self
    }
    pub fn ub(mut self, v: f64) -> Self {
        self.ub = v;
        self
    }
    pub fn bounds(mut self, lb: f64, ub: f64) -> Self {
        self.lb = lb;
        self.ub = ub;
        self
    }
    /// Per-key lower bound. Overrides [`Self::lb`] when both are set.
    ///
    /// The closure receives a typed index value via [`FromIndexKey`].
    /// Annotate the argument to select the projection:
    /// ```ignore
    /// .lb_by(|(p, q): (String, String)| floor_for(&p, &q))
    /// .lb_by(|i: usize| lower_bounds[i])
    /// ```
    pub fn lb_by<F>(mut self, f: F) -> Self
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + 'a,
        F: Send + Sync,
    {
        self.lb_by = Some(Box::new(move |k: &IndexKey| f(K::from_index_key(k))));
        self
    }
    /// Per-key upper bound. Overrides [`Self::ub`] when both are set.
    ///
    /// The closure receives a typed index value via [`FromIndexKey`]; annotate
    /// the argument to select the projection:
    /// ```ignore
    /// .ub_by(|(p, q): (String, String)| capacity_for(&p, &q))
    /// .ub_by(|i: usize| upper_bounds[i])
    /// ```
    pub fn ub_by<F>(mut self, f: F) -> Self
    where
        K: FromIndexKey,
        F: Fn(K) -> f64 + 'a,
        F: Send + Sync,
    {
        self.ub_by = Some(Box::new(move |k: &IndexKey| f(K::from_index_key(k))));
        self
    }
    pub fn domain(mut self, d: Domain) -> Self {
        self.domain = d;
        self
    }
    pub fn integer(mut self) -> Self {
        self.domain = Domain::Integer;
        self
    }
    pub fn binary(mut self) -> Self {
        self.domain = Domain::Binary;
        self.lb = 0.0;
        self.ub = 1.0;
        self
    }

    /// Register one scalar variable per key and return the [`IndexedVar`] handle.
    ///
    /// # Panics
    /// Panics if a scalar variable name collides with one already registered.
    pub fn build(self) -> IndexedVar<'a, K> {
        let Self { model, base_name, keys, axes, lb, ub, lb_by, ub_by, domain, parallel, _k } =
            self;

        if !indexed_parallel(keys.len(), parallel, PAR_INDEXED_METADATA_THRESHOLD) {
            let handles = keys
                .iter()
                .map(|key| {
                    let scalar_name = format_index_name_small(&base_name, key);
                    let lo = lb_by.as_ref().map_or(lb, |f| f(key));
                    let hi = ub_by.as_ref().map_or(ub, |f| f(key));
                    model.__var(scalar_name).lb(lo).ub(hi).domain(domain).build()
                })
                .collect();
            let storage = build_storage(keys, axes, handles);
            return IndexedFamily { storage, model_id: self.model.id(), _marker: PhantomData };
        }

        let prepare = |key: &IndexKey| PendingVar {
            name: format_index_name_small(&base_name, key),
            lb: lb_by.as_ref().map_or(lb, |f| f(key)),
            ub: ub_by.as_ref().map_or(ub, |f| f(key)),
        };
        let prepared: Vec<_> = keys.par_iter().map(prepare).collect();
        let handles = model.register_vars_batch(&prepared, domain);
        drop(prepared);
        let storage = build_storage(keys, axes, handles);
        IndexedFamily { storage, model_id: self.model.id(), _marker: PhantomData }
    }

    #[cfg(any(test, feature = "benchmark-support"))]
    pub(super) fn parallel_for_benchmark(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }
}

pub(super) fn format_index_name(base: &str, key: &IndexKey) -> String {
    let mut out = String::with_capacity(base.len() + 4);
    out.push_str(base);
    out.push('[');
    write_key_parts(&mut out, key);
    out.push(']');
    out
}

// Indexed variable names usually fit SmolStr's inline storage.
pub(super) fn format_index_name_small(base: &str, key: &IndexKey) -> SmolStr {
    let mut out = smol_str::SmolStrBuilder::new();
    out.push_str(base);
    out.push('[');
    write_key_parts(&mut out, key);
    out.push(']');
    out.finish()
}

fn write_key_parts(out: &mut impl fmt::Write, key: &IndexKey) {
    match key {
        IndexKey::Int(i) => write!(out, "{i}").unwrap(),
        IndexKey::Str(s) => out.write_str(s).unwrap(),
        IndexKey::Tuple(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.write_char(',').unwrap();
                }
                write_key_parts(out, p);
            }
        }
    }
}

/// Public render of an `IndexKey`'s textual form, used when deriving
/// auto-generated names for indexed-family constraints.
pub fn display_index_key(key: &IndexKey) -> String {
    let mut out = String::new();
    write_key_parts(&mut out, key);
    out
}
