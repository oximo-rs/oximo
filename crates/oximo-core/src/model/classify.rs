//! Model classification and cache invalidation.

use super::Model;
use crate::soc::is_detected_soc;
use oximo_expr::{ExprClass, classify};
use rayon::prelude::*;
use std::fmt;

pub(super) const PAR_KIND_THRESHOLD: usize = 256;

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
        if let Some(k) = self.cached_kind.get() {
            return k;
        }
        let k = self.infer_kind_with(None);
        self.cached_kind.set(Some(k));
        k
    }

    /// Infer the model kind without reading or updating the kind cache.
    ///
    /// Automatic inference keeps the initial classification scan serial, then
    /// parallelizes only a large set of already-known quadratic candidates for
    /// SOC recognition. This keeps LP/NLP models off the Rayon path.
    pub(super) fn infer_kind_with(&self, parallel: Option<bool>) -> ModelKind {
        self.infer_kind_impl(parallel)
    }

    /// Infer the kind with a forced serial or parallel SOC-recognition pass.
    /// This is used by benchmarks and parity tests.
    #[cfg(any(test, feature = "benchmark-support"))]
    pub(super) fn infer_kind(&self, parallel: bool) -> ModelKind {
        self.infer_kind_impl(Some(parallel))
    }

    fn infer_kind_impl(&self, parallel: Option<bool>) -> ModelKind {
        let arena = self.arena.borrow();
        let vars = self.variables.borrow();
        let has_int = vars.iter().any(|v| v.domain.requires_discrete_choice())
            || self.sos_constraints.borrow().iter().any(|constraint| constraint.active);
        let obj_class = self
            .objective
            .borrow()
            .as_ref()
            .map_or(ExprClass::Linear, |o| classify(&arena, o.expr));

        let mut any_nonlinear = obj_class == ExprClass::Nonlinear;
        let mut plain_quad_con = false;
        let mut detected_soc = false;
        if !any_nonlinear {
            let constraints = self.constraints.borrow();
            let arena_ref = &*arena;
            let vars_ref = &*vars;
            let mut quadratic = Vec::new();
            for c in constraints.iter().filter(|c| c.active) {
                match classify(arena_ref, c.lhs) {
                    ExprClass::Linear => {}
                    ExprClass::Quadratic => quadratic.push(c),
                    ExprClass::Nonlinear => {
                        any_nonlinear = true;
                        break;
                    }
                }
            }
            if !any_nonlinear {
                let use_parallel = parallel.unwrap_or(
                    quadratic.len() >= PAR_KIND_THRESHOLD && rayon::current_num_threads() > 1,
                );
                if use_parallel {
                    let (has_cone, has_plain) = quadratic
                        .par_iter()
                        .map(|c| {
                            let is_soc = is_detected_soc(arena_ref, vars_ref, c);
                            (is_soc, !is_soc)
                        })
                        .reduce(
                            || (false, false),
                            |left, right| (left.0 || right.0, left.1 || right.1),
                        );
                    detected_soc = has_cone;
                    plain_quad_con = has_plain;
                } else {
                    for c in quadratic {
                        if is_detected_soc(arena_ref, vars_ref, c) {
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
