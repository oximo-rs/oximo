//! Model classification and cache invalidation.

use super::Model;
use crate::soc::is_detected_soc;
use crate::var::ResolvedVariable;
use oximo_expr::{ExprArena, ExprClass, ExprId, QuadraticTerms, classify};
use rayon::prelude::*;
use std::fmt;

pub(super) const PAR_KIND_THRESHOLD: usize = 256;

/// Estimate quadratic-analysis work to help decide whether to process rows in parallel.
///
/// Inspect at most sixteen nodes, descending through unary nodes and sums with
/// at most eight terms. Count larger sums and all products by their number of
/// children, affine nodes by their coefficient count, and other nodes as one
/// unit. Wide sums and products contribute their width without visiting their
/// children.
fn quadratic_work_hint(arena: &ExprArena, root: ExprId) -> usize {
    use oximo_expr::ExprNode;
    let mut pending: smallvec::SmallVec<[ExprId; 16]> = smallvec::smallvec![root];
    let mut work = 0_usize;
    for _ in 0..16 {
        let Some(id) = pending.pop() else { break };
        match arena.get(id) {
            ExprNode::Add(children) if children.len() <= 8 => {
                pending.extend_from_slice(children);
            }
            ExprNode::Add(children) | ExprNode::Mul(children) => {
                work = work.saturating_add(children.len());
            }
            ExprNode::Linear { coeffs, .. } => work = work.saturating_add(coeffs.len()),
            ExprNode::Unary(_, child) => pending.push(*child),
            _ => work = work.saturating_add(1),
        }
    }

    work
}

/// Decide whether quadratic rows warrant parallel SOC recognition during model classification.
fn parallel_quadratic_analysis(
    arena: &ExprArena,
    rows: &[&crate::AlgebraicConstraint],
    workers: usize,
) -> bool {
    if workers <= 1 || rows.len() < 4 {
        return false;
    }
    let mut work = 0_usize;
    rows.len() >= PAR_KIND_THRESHOLD
        || rows.iter().any(|row| {
            work = work.saturating_add(quadratic_work_hint(arena, row.lhs));
            work >= 4_096
        })
}

/// The kind of mathematical program a `Model` represents.
///
/// This is inferred from the variables and expressions in the model, not set
/// explicitly by the user. See [`Model::kind`] for the exact decision ladder.
///
/// The `MI*` variant of each class is picked when any variable has an integer or semicontinuous
/// domain. The continuous classes are, from most to least general:
///
/// - `NLP`: some expression is nonlinear (degree > 2, transcendental, division)
/// - `QCP`: some constraint is quadratic and not recognized as a second-order
///   cone
/// - `SOCP`: second-order cone constraints are present (explicit
///   [`crate::SocConstraint`]s or SOC-shaped quadratic constraints recognized
///   by the model's structural cone predicate). The objective may be linear or quadratic:
/// - `QP`: quadratic objective, linear constraints
/// - `LP`: everything linear
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModelKind {
    LP,
    MILP,
    QP,
    MIQP,
    QCP,
    MIQCP,
    SOCP,
    MISOCP,
    SDP,
    MISDP,
    NLP,
    MINLP,
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LP => "LP",
            Self::MILP => "MILP",
            Self::QP => "QP",
            Self::MIQP => "MIQP",
            Self::QCP => "QCP",
            Self::MIQCP => "MIQCP",
            Self::SOCP => "SOCP",
            Self::MISOCP => "MISOCP",
            Self::SDP => "SDP",
            Self::MISDP => "MISDP",
            Self::NLP => "NLP",
            Self::MINLP => "MINLP",
        })
    }
}

impl Model {
    pub(crate) fn invalidate_kind(&self) {
        self.cached_kind.set(None);
    }

    // Classification

    /// Infer the [`ModelKind`] from current variables and expressions.
    /// Result is cached and invalidated whenever variables, constraints, or the
    /// objective change.
    ///
    /// The decision ladder, top-down (any discrete variable domain picks the `MI*`
    /// column):
    ///
    /// 1. any nonlinear expression (objective or constraint) -> `NLP`
    /// 2. any quadratic constraint not recognized by the structural SOC
    ///    predicate -> `QCP`
    /// 3. active PSD matrices -> `SDP`
    /// 4. SOC cones present (explicit or detected) -> `SOCP`
    /// 5. quadratic objective -> `QP`
    /// 6. otherwise -> `LP`
    pub fn kind(&self) -> ModelKind {
        let arena = self.arena.borrow();
        let revision = arena.parameter_revision();
        if let Some(k) = self.cached_kind.get()
            && revision == self.cached_kind_parameter_revision.get()
        {
            return k;
        }
        let vars = self.variables.borrow().iter().map(|v| v.resolved(&arena)).collect::<Vec<_>>();
        self.__kind_at(&arena, &vars)
    }

    /// Classify metadata captured from this model at one parameter assignment.
    #[doc(hidden)]
    pub fn __kind_at(&self, arena: &ExprArena, vars: &[ResolvedVariable]) -> ModelKind {
        self.kind_at_with_soc(arena, vars, &|row| is_detected_soc(arena, vars, row))
    }

    /// Classify using snapshot-owned coefficients supplied by solver preparation.
    #[doc(hidden)]
    pub fn __kind_with_quadratic_at<Q: std::ops::Deref<Target = QuadraticTerms>>(
        &self,
        arena: &ExprArena,
        vars: &[ResolvedVariable],
        quadratic: &(impl Fn(ExprId) -> Option<Q> + Sync),
    ) -> ModelKind {
        self.infer_kind_impl(arena, vars, None, &|row| {
            quadratic(row.lhs)
                .is_some_and(|q| crate::soc::__is_detected_soc_from_quadratic(vars, row, &q))
        })
    }

    fn kind_at_with_soc(
        &self,
        arena: &ExprArena,
        vars: &[ResolvedVariable],
        is_soc: &(impl Fn(&crate::AlgebraicConstraint) -> bool + Sync),
    ) -> ModelKind {
        let revision = arena.parameter_revision();
        if let Some(k) = self.cached_kind.get()
            && revision == self.cached_kind_parameter_revision.get()
        {
            return k;
        }
        let k = self.infer_kind_impl(arena, vars, None, is_soc);
        self.cached_kind.set(Some(k));
        self.cached_kind_parameter_revision.set(revision);
        k
    }

    /// Infer the model kind without reading or updating the kind cache.
    ///
    /// Automatic inference keeps the initial classification scan serial, then
    /// parallelizes sufficiently many or sufficiently wide quadratic candidates
    /// for SOC recognition. This keeps LP/NLP models off the Rayon path.
    #[cfg(any(test, feature = "benchmark-support"))]
    pub(super) fn infer_kind_with(&self, parallel: Option<bool>) -> ModelKind {
        let arena = self.arena.borrow();
        let vars = self.variables.borrow().iter().map(|v| v.resolved(&arena)).collect::<Vec<_>>();
        self.infer_kind_impl(&arena, &vars, parallel, &|row| is_detected_soc(&arena, &vars, row))
    }

    /// Infer the kind with a forced serial or parallel SOC-recognition pass.
    /// This is used by benchmarks and parity tests.
    #[cfg(any(test, feature = "benchmark-support"))]
    pub(super) fn infer_kind(&self, parallel: bool) -> ModelKind {
        self.infer_kind_with(Some(parallel))
    }

    fn infer_kind_impl(
        &self,
        arena: &ExprArena,
        vars: &[ResolvedVariable],
        parallel: Option<bool>,
        is_soc: &(impl Fn(&crate::AlgebraicConstraint) -> bool + Sync),
    ) -> ModelKind {
        let has_int = vars.iter().any(|v| v.domain.requires_discrete_choice())
            || self.sos_constraints.borrow().iter().any(|constraint| constraint.active);
        let obj_class =
            self.objective.borrow().as_ref().map_or(ExprClass::Linear, |o| classify(arena, o.expr));

        let mut any_nonlinear = obj_class == ExprClass::Nonlinear;
        let mut plain_quad_con = false;
        let mut detected_soc = false;
        if !any_nonlinear {
            let constraints = self.constraints.borrow();
            let mut quadratic = Vec::new();
            for c in constraints.iter().filter(|c| c.active) {
                match classify(arena, c.lhs) {
                    ExprClass::Linear => {}
                    ExprClass::Quadratic => quadratic.push(c),
                    ExprClass::Nonlinear => {
                        any_nonlinear = true;
                        break;
                    }
                }
            }
            if !any_nonlinear {
                let use_parallel = parallel.unwrap_or_else(|| {
                    parallel_quadratic_analysis(arena, &quadratic, rayon::current_num_threads())
                });
                if use_parallel {
                    let (has_cone, has_plain) = quadratic
                        .par_iter()
                        .map(|c| {
                            let detected = is_soc(c);
                            (detected, !detected)
                        })
                        .reduce(
                            || (false, false),
                            |left, right| (left.0 || right.0, left.1 || right.1),
                        );
                    detected_soc = has_cone;
                    plain_quad_con = has_plain;
                } else {
                    for c in quadratic {
                        if is_soc(c) {
                            detected_soc = true;
                        } else {
                            plain_quad_con = true;
                        }
                    }
                }
            }
        }
        let has_soc = detected_soc || self.has_active_soc_constraints();

        let pick = |cont, int| if has_int { int } else { cont };
        if any_nonlinear {
            pick(ModelKind::NLP, ModelKind::MINLP)
        } else if plain_quad_con {
            pick(ModelKind::QCP, ModelKind::MIQCP)
        } else if self.has_active_psd_constraints() {
            pick(ModelKind::SDP, ModelKind::MISDP)
        } else if has_soc {
            pick(ModelKind::SOCP, ModelKind::MISOCP)
        } else if obj_class == ExprClass::Quadratic {
            pick(ModelKind::QP, ModelKind::MIQP)
        } else {
            pick(ModelKind::LP, ModelKind::MILP)
        }
    }
}

#[cfg(test)]
mod supplied_analysis_tests {
    use super::*;
    use crate::prelude::*;
    use std::sync::Arc;

    #[test]
    fn supplied_coefficients_do_not_read_or_update_kind_cache() {
        let model = Model::new("supplied SOC cache isolation");
        variable!(model, x);
        variable!(model, t >= 0.0);
        constraint!(model, x * x - t * t <= 0.0);

        let arena = model.arena();
        let variables = model.resolved_variables();
        let missing_coefficients = |_: ExprId| None::<Arc<QuadraticTerms>>;
        assert_eq!(
            model.__kind_with_quadratic_at(&arena, &variables, &missing_coefficients),
            ModelKind::QCP
        );
        assert_eq!(model.cached_kind.get(), None);
        assert_eq!(model.kind(), ModelKind::SOCP);
        assert_eq!(
            model.__kind_with_quadratic_at(&arena, &variables, &missing_coefficients),
            ModelKind::QCP
        );
        assert_eq!(model.kind(), ModelKind::SOCP);
    }

    #[test]
    fn few_wide_quadratic_rows_use_work_based_parallelism_with_kind_parity() {
        let model = Model::new("wide quadratic classification");
        let x: Vec<_> = (0..50).map(|i| model.__var(format!("x{i}")).build()).collect();
        for row in 0..10 {
            let terms = (0..650).map(|i| x[i % 50] * x[(i + row) % 50]);
            let lhs = Expr::__sum_terms_in(model.__sum_context(), terms);
            model.__add_constraint_auto(lhs.le(1.0));
        }
        objective!(model, Min, x[0]);

        let arena = model.arena();
        let constraints = model.constraints();
        let rows: Vec<_> = constraints.algebraic().iter().collect();
        assert!(!parallel_quadratic_analysis(&arena, &rows, 1));
        assert!(parallel_quadratic_analysis(&arena, &rows, 6));
        assert!(!parallel_quadratic_analysis(&arena, &rows[..1], 6));
        assert_eq!(model.infer_kind(false), ModelKind::QCP);
        assert_eq!(model.infer_kind(true), ModelKind::QCP);
        assert_eq!(model.infer_kind_with(None), ModelKind::QCP);

        let small = Model::new("small quadratic classification");
        variable!(small, y);
        for _ in 0..10 {
            constraint!(small, y * y <= 1.0);
        }
        objective!(small, Min, y);

        let arena = small.arena();
        let constraints = small.constraints();
        let rows: Vec<_> = constraints.algebraic().iter().collect();
        assert!(!parallel_quadratic_analysis(&arena, &rows, 6));
    }

    #[test]
    fn supplied_coefficients_preserve_forced_serial_and_parallel_classification() {
        let model = Model::new("supplied SOC analysis");
        variable!(model, x);
        variable!(model, t >= 0.0);
        param!(model, p = 1.0);
        for scale in 1..=300 {
            model.__add_constraint_auto((f64::from(scale) * p * x * x - t * t).le(0.0));
        }
        objective!(model, Min, t);

        for value in [1.0, 0.0, 2.0] {
            model.set_param(p, value).unwrap();
            let arena = model.arena();
            let variables = model.resolved_variables();
            let rows = model.constraints();
            let coefficients: rustc_hash::FxHashMap<_, _> = rows
                .algebraic()
                .iter()
                .map(|row| {
                    (row.lhs, Arc::new(oximo_expr::extract_quadratic(&arena, row.lhs).unwrap()))
                })
                .collect();
            let predicate = |row: &crate::AlgebraicConstraint| {
                crate::__is_detected_soc_from_quadratic(&variables, row, &coefficients[&row.lhs])
            };

            let serial = model.infer_kind_impl(&arena, &variables, Some(false), &predicate);
            let parallel = model.infer_kind_impl(&arena, &variables, Some(true), &predicate);
            assert_eq!(serial, parallel);
            assert_eq!(serial, model.infer_kind(false));
            assert_eq!(
                serial,
                model.__kind_with_quadratic_at(&arena, &variables, &|id| coefficients
                    .get(&id)
                    .cloned())
            );
        }
    }
}
