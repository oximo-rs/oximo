//! Owned preparation retained by persistent adapters.

use std::sync::{Arc, Mutex};

use oximo_core::{
    Model, ModelKind, ModelParameterSnapshot, ParameterDependencies, ParameterTarget,
    ResolvedVariable, model::ModelUpdateKey,
};
use oximo_expr::{ExprClass, ExprId, LinearTerms, QuadraticTerms, classify};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::{SolverError, prepare::LoweringContext};

/// Native source entities requiring numeric preparation.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparationChange {
    Full,
    Unchanged,
    Selected(SmallVec<[ParameterTarget; 4]>),
}

impl PreparationChange {
    pub fn contains(&self, target: ParameterTarget) -> bool {
        match self {
            Self::Full => true,
            Self::Unchanged => false,
            Self::Selected(targets) => targets.contains(&target),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ResidentExpressions {
    roots: FxHashSet<ExprId>,
    pub(crate) linear: Mutex<FxHashMap<ExprId, Option<Arc<LinearTerms<'static>>>>>,
    pub(crate) quadratic: Mutex<FxHashMap<ExprId, Option<Arc<QuadraticTerms>>>>,
}

impl ResidentExpressions {
    pub(crate) fn admits(&self, root: ExprId) -> bool {
        self.roots.contains(&root)
    }

    fn invalidate(&self, roots: impl IntoIterator<Item = ExprId>) {
        let mut linear = self.linear.lock().expect("resident cache poisoned");
        let mut quadratic = self.quadratic.lock().expect("resident cache poisoned");
        for root in roots {
            linear.remove(&root);
            quadratic.remove(&root);
        }
    }
}

#[derive(Debug)]
struct Baseline {
    key: ModelUpdateKey,
    parameters: ModelParameterSnapshot,
    variables: Arc<Vec<ResolvedVariable>>,
    kind: ModelKind,
}

/// Parameter dependency tracking and reusable numeric expression preparation.
/// Call `clear` after any failed native update/solve and `commit` after success.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct PersistentPreparation {
    baseline: Option<Baseline>,
    pending: Option<Baseline>,
    dependencies: Option<ParameterDependencies>,
    expressions: Arc<ResidentExpressions>,
    change: Option<PreparationChange>,
    kind_sensitive_rows: FxHashSet<oximo_core::ConstraintId>,
    soc_status: Option<FxHashMap<oximo_core::ConstraintId, Option<oximo_expr::VarId>>>,
}

impl PersistentPreparation {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn change(&self) -> &PreparationChange {
        self.change.as_ref().expect("prepare before inspecting changes")
    }

    pub fn commit(&mut self) {
        self.baseline = self.pending.take();
    }

    /// Capture a fresh view while sharing only unaffected owned decompositions.
    ///
    /// # Errors
    /// Returns the same declaration errors as `LoweringContext::new`.
    pub fn prepare<'a>(&mut self, model: &'a Model) -> Result<LoweringContext<'a>, SolverError> {
        model.ensure_gdp_reformulated()?;
        model.ensure_objective_declared()?;

        // A missing commit means a previous operation did not complete.
        if self.pending.is_some() {
            self.clear();
        }
        let key = model.__update_key();
        let (parameters, arena) = model.__capture_parameters();
        let change = match &self.baseline {
            Some(base) if base.key == key => {
                if base.parameters.parameters().same_bits(parameters.parameters()) {
                    PreparationChange::Unchanged
                } else {
                    let index =
                        self.dependencies.get_or_insert_with(|| model.parameter_dependencies());
                    let targets: SmallVec<[ParameterTarget; 4]> = index
                        .changed_targets_iter(&base.parameters, &parameters)
                        .expect("update key includes model identity")
                        .collect();
                    if targets.is_empty() {
                        PreparationChange::Unchanged
                    } else {
                        PreparationChange::Selected(targets)
                    }
                }
            }
            _ => PreparationChange::Full,
        };
        if change == PreparationChange::Full {
            self.dependencies = None;
            self.expressions = Arc::default();
            self.kind_sensitive_rows.clear();
            self.soc_status = None;
        } else if let PreparationChange::Selected(targets) = &change {
            if self.expressions.roots.is_empty() {
                self.expressions = Arc::new(ResidentExpressions {
                    roots: roots(model, None).into_iter().collect(),
                    ..ResidentExpressions::default()
                });
            } else {
                self.expressions.invalidate(roots(model, Some(targets)));
            }
        }
        let variables = if change == PreparationChange::Full {
            Arc::new(model.variables().iter().map(|v| v.resolved(&arena)).collect())
        } else {
            let mut variables = Arc::clone(&self.baseline.as_ref().expect("baseline").variables);
            if let PreparationChange::Selected(targets) = &change {
                let source = model.variables();
                for target in targets {
                    if let ParameterTarget::VariableBound(id) = target {
                        Arc::make_mut(&mut variables)[id.index()] =
                            source[id.index()].resolved(&arena);
                    }
                }
            }
            variables
        };
        let kind = self.prepare_kind(model, &arena, &variables, &change);
        self.pending = Some(Baseline { key, parameters, variables: Arc::clone(&variables), kind });
        self.change = Some(change);
        Ok(LoweringContext::resident(
            model,
            (*arena).clone(),
            variables,
            kind,
            Arc::clone(&self.expressions),
        ))
    }

    fn prepare_kind(
        &mut self,
        model: &Model,
        arena: &oximo_expr::ExprArena,
        variables: &[ResolvedVariable],
        change: &PreparationChange,
    ) -> ModelKind {
        // SOC recognition can depend on numeric coefficients and bounds.
        // Recheck it on used changes.
        let recheck_kind = match change {
            PreparationChange::Full => true,
            PreparationChange::Unchanged => false,
            PreparationChange::Selected(targets) => targets.iter().any(|target| match target {
                ParameterTarget::VariableBound(_) => !self.kind_sensitive_rows.is_empty(),
                ParameterTarget::Algebraic(id) => self.kind_sensitive_rows.contains(id),
                _ => false,
            }),
        };
        let kind = if matches!(change, PreparationChange::Full) {
            model.__kind_at(arena, variables)
        } else if recheck_kind {
            let constraints = model.constraints();
            let classify_row = |id: oximo_core::ConstraintId| {
                let row = &constraints.algebraic()[id.index()];
                oximo_expr::extract_quadratic(arena, row.lhs)
                    .and_then(|q| oximo_core::__soc_bound_variable(row, &q))
            };
            let all_rows = self.soc_status.is_none();
            let status = self.soc_status.get_or_insert_with(FxHashMap::default);
            if all_rows {
                for &id in &self.kind_sensitive_rows {
                    status.insert(id, classify_row(id));
                }
            } else if let PreparationChange::Selected(targets) = change {
                for target in targets {
                    if let ParameterTarget::Algebraic(id) = target
                        && self.kind_sensitive_rows.contains(id)
                    {
                        status.insert(*id, classify_row(*id));
                    }
                }
            }
            let discrete = matches!(
                self.baseline.as_ref().expect("baseline").kind,
                ModelKind::MIQCP | ModelKind::MISOCP | ModelKind::MISDP
            );
            if status
                .values()
                .any(|candidate| candidate.is_none_or(|bound| variables[bound.index()].lb < 0.0))
            {
                if discrete { ModelKind::MIQCP } else { ModelKind::QCP }
            } else if model.has_active_psd_constraints() {
                if discrete { ModelKind::MISDP } else { ModelKind::SDP }
            } else if discrete {
                ModelKind::MISOCP
            } else {
                ModelKind::SOCP
            }
        } else {
            self.baseline.as_ref().expect("baseline").kind
        };
        if matches!(change, PreparationChange::Full)
            && matches!(
                kind,
                ModelKind::QCP
                    | ModelKind::MIQCP
                    | ModelKind::SOCP
                    | ModelKind::MISOCP
                    | ModelKind::SDP
                    | ModelKind::MISDP
            )
        {
            self.kind_sensitive_rows = model
                .constraints()
                .algebraic()
                .iter()
                .enumerate()
                .filter(|(_, row)| row.active && classify(arena, row.lhs) == ExprClass::Quadratic)
                .map(|(i, _)| oximo_core::ConstraintId(u32::try_from(i).expect("row ID")))
                .collect();
        }
        kind
    }
}

fn roots(model: &Model, selected: Option<&[ParameterTarget]>) -> Vec<ExprId> {
    let mut roots = Vec::new();
    if let Some(targets) = selected {
        let constraints = model.constraints();
        for target in targets {
            match target {
                ParameterTarget::Objective => {
                    if let Some(objective) = model.objective().as_ref() {
                        roots.push(objective.expr);
                    }
                }
                ParameterTarget::Algebraic(id) => {
                    roots.push(constraints.algebraic()[id.index()].lhs);
                }
                ParameterTarget::SecondOrderCone(id) => {
                    let cone = &constraints.second_order_cones()[id.index()];
                    roots.extend(cone.terms.iter().copied());
                    roots.push(cone.bound);
                }
                ParameterTarget::PositiveSemidefinite(id) => roots.extend(
                    constraints.positive_semidefinite()[id.index()]
                        .matrix
                        .upper_triangle()
                        .iter()
                        .copied(),
                ),
                ParameterTarget::Indicator(id) => {
                    roots.push(constraints.indicators()[id.index()].lhs);
                }
                _ => {}
            }
        }
        return roots;
    }
    let include = |target| selected.is_none_or(|targets| targets.contains(&target));
    if include(ParameterTarget::Objective)
        && let Some(objective) = model.objective().as_ref()
    {
        roots.push(objective.expr);
    }
    let constraints = model.constraints();
    for (i, row) in constraints.algebraic().iter().enumerate() {
        if include(ParameterTarget::Algebraic(oximo_core::ConstraintId(
            u32::try_from(i).expect("row ID"),
        ))) {
            roots.push(row.lhs);
        }
    }
    for (i, cone) in constraints.second_order_cones().iter().enumerate() {
        if include(ParameterTarget::SecondOrderCone(oximo_core::SocConstraintId(
            u32::try_from(i).expect("cone ID"),
        ))) {
            roots.extend(cone.terms.iter().copied());
            roots.push(cone.bound);
        }
    }
    for (i, cone) in constraints.positive_semidefinite().iter().enumerate() {
        if include(ParameterTarget::PositiveSemidefinite(oximo_core::PsdConstraintId(
            u32::try_from(i).expect("PSD ID"),
        ))) {
            roots.extend(cone.matrix.upper_triangle().iter().copied());
        }
    }
    for (i, row) in constraints.indicators().iter().enumerate() {
        if include(ParameterTarget::Indicator(oximo_core::IndicatorConstraintId(
            u32::try_from(i).expect("indicator ID"),
        ))) {
            roots.push(row.lhs);
        }
    }
    roots
}

#[cfg(test)]
#[expect(clippy::many_single_char_names, reason = "small symbolic test models")]
mod tests {
    use super::*;

    #[test]
    fn unused_registration_retains_the_index_and_new_uses_invalidate_it() {
        use oximo_core::prelude::*;
        let m = Model::new("unused registration");

        param!(m, p = 1.0);
        variable!(m, x);
        objective!(m, Min, p * x);
        let mut cache = PersistentPreparation::default();
        drop(cache.prepare(&m).unwrap());
        cache.commit();
        p.set_param_value(2.0).unwrap();
        drop(cache.prepare(&m).unwrap());
        assert!(
            matches!(cache.change(), PreparationChange::Selected(targets) if !targets.spilled())
        );

        cache.commit();
        assert!(cache.dependencies.is_some());

        param!(m, unused = 7.0);
        drop(cache.prepare(&m).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Unchanged);
        assert!(cache.dependencies.is_some());

        cache.commit();
        objective!(m, Min, unused * x);
        drop(cache.prepare(&m).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Full);

        cache.commit();
        unused.set_param_value(8.0).unwrap();
        drop(cache.prepare(&m).unwrap());
        assert!(cache.change().contains(ParameterTarget::Objective));
    }
    use oximo_core::prelude::*;

    #[test]
    fn selective_kind_cache_tracks_zero_crossings_and_bound_signs() {
        let m = Model::new("kind metadata");

        param!(m, p = 1.0);
        param!(m, q = 1.0);
        param!(m, lower = 0.0);
        variable!(m,-1.0<=x<=1.0);
        variable!(m,-1.0<=y<=1.0);
        variable!(m,lower<=t<=2.0);
        variable!(m,0.0<=u<=2.0);
        constraint!(m, a, p * x * x - t * t <= 0.0);
        constraint!(m, b, q * y * y - u * u <= 0.0);
        objective!(m, Min, t + u);

        let mut cache = PersistentPreparation::default();
        for value in [1.0, 0.0, -1.0, 2.0] {
            p.set_param_value(value).unwrap();
            let prepared = cache.prepare(&m).unwrap();
            assert_eq!(prepared.kind(), m.kind());
            drop(prepared);
            cache.commit();
        }
        for value in [-1.0, 0.0] {
            lower.set_param_value(value).unwrap();
            let prepared = cache.prepare(&m).unwrap();
            assert_eq!(prepared.kind(), m.kind());
            drop(prepared);
            cache.commit();
        }
    }

    #[test]
    fn selective_refresh_preserves_unaffected_terms_and_result_snapshots() {
        let m = Model::new("resident cache");

        param!(m, p = 2.0);
        param!(m, q = 3.0);
        param!(m, unused = 7.0);
        variable!(m,0.0<=x<=5.0);
        let a = constraint!(m, a, p * x <= 4.0);
        let b = constraint!(m, b, q * x <= 6.0);
        objective!(m, Min, x);

        let mut cache = PersistentPreparation::default();
        drop(cache.prepare(&m).unwrap());
        cache.commit();
        p.set_param_value(3.0).unwrap();
        let first = cache.prepare(&m).unwrap();
        let ar = first.constraints().algebraic()[a.row_id().index()].lhs;
        let br = first.constraints().algebraic()[b.row_id().index()].lhs;
        let old_a = first.linear(ar).unwrap().into_owned();
        first.linear(br).unwrap();
        let old_b = cache.expressions.linear.lock().unwrap()[&br].clone().unwrap();
        let snapshot = first.arena().parameter_snapshot();
        drop(first);
        cache.commit();
        p.set_param_value(4.0).unwrap();
        let next = cache.prepare(&m).unwrap();
        assert_eq!(
            cache.change(),
            &PreparationChange::Selected(vec![ParameterTarget::Algebraic(a.row_id())].into())
        );
        assert_eq!(next.linear(ar).unwrap().coeffs[0].1, 4.0);
        assert_eq!(old_a.coeffs[0].1, 3.0);
        assert_eq!(snapshot.values()[p.param_id().unwrap().index()], 3.0);

        let new_b = cache.expressions.linear.lock().unwrap()[&br].clone().unwrap();
        assert!(Arc::ptr_eq(&old_b, &new_b));

        drop(next);
        cache.commit();
        unused.set_param_value(9.0).unwrap();
        let fresh = cache.prepare(&m).unwrap();
        assert_eq!(cache.change(), &PreparationChange::Unchanged);
        assert_eq!(
            fresh.arena().parameter_snapshot().values()[unused.param_id().unwrap().index()],
            9.0
        );
    }

    #[test]
    fn structure_edits_forks_and_uncommitted_preparation_invalidate() {
        let mut m = Model::new("key");

        param!(m, p = 2.0);
        variable!(m,0.0<=x<=5.0);
        let row = constraint!(m, row, p * x <= 4.0);
        objective!(m, Min, x);
        let original = m.__update_key();
        p.set_param_value(3.0).unwrap();
        assert_eq!(original, m.__update_key());

        std::hint::black_box(x + 3.0);
        assert_eq!(original, m.__update_key(), "unused arena append");

        m.set_initial(x, 1.0).unwrap();
        assert_ne!(original, m.__update_key());

        let mut cache = PersistentPreparation::default();
        drop(cache.prepare(&m).unwrap());
        cache.commit();
        m.set_constraint_active(row, false).unwrap();
        drop(cache.prepare(&m).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Full);

        cache.commit();
        objective!(m, Max, x);
        drop(cache.prepare(&m).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Full);

        cache.commit();
        m.name = "renamed".into();
        drop(cache.prepare(&m).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Full);

        cache.commit();
        let fork = m.fork();
        drop(cache.prepare(&fork).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Full);

        drop(cache.prepare(&fork).unwrap());
        assert_eq!(cache.change(), &PreparationChange::Full, "missing commit");
    }
}
