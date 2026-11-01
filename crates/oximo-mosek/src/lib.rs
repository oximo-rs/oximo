#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod options;
mod persistent;
mod translate;

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
pub use translate::benchmark_support;

pub use options::MosekOptions;
pub use persistent::MosekPersistent;
pub use translate::solve;

use oximo_core::{Model, ModelKind};
use oximo_solver::{
    ExpressionOperator, MODEL_KINDS, PersistentSolver, Solver, SolverError, SolverProfile,
    SolverResult,
};
use std::{borrow::Cow, sync::LazyLock};

/// MOSEK solver backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct Mosek;

pub(crate) const NAME: &str = "MOSEK";

pub(crate) const fn supported(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::SDP
            | ModelKind::LP
            | ModelKind::MILP
            | ModelKind::QP
            | ModelKind::MIQP
            | ModelKind::QCP
            | ModelKind::MIQCP
            | ModelKind::SOCP
            | ModelKind::MISOCP
    )
}

static PROFILE: LazyLock<SolverProfile> = LazyLock::new(build_profile);

fn build_profile() -> SolverProfile {
    use oximo_expr::ExprClass::{Linear, Quadratic};
    use oximo_solver::VariableDomain::{Binary, Integer, Real};

    SolverProfile::new(NAME)
        .with_scalar_constraints([Linear, Quadratic])
        .with_scalar_objectives([Linear, Quadratic])
        .with_soc(true)
        .with_kinds(MODEL_KINDS.into_iter().filter(|kind| supported(*kind)))
        .with_domains([Real, Integer, Binary])
        .with_operators(ExpressionOperator::ALL.iter().copied())
        .with_psd(true)
        .with_indicators(true)
        .reject_combination(
            |f| {
                f.has_psd
                    && (f.kind != ModelKind::SDP || f.has_indicators || f.objective_class != Linear)
            },
            "MOSEK PSD models require a continuous SDP with an affine objective and no indicators",
        )
}

impl Solver for Mosek {
    type Options = MosekOptions;

    fn profile(&self, _opts: &Self::Options) -> Cow<'_, SolverProfile> {
        Cow::Borrowed(&PROFILE)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn solve(&mut self, model: &Model, opts: &MosekOptions) -> Result<SolverResult, SolverError> {
        translate::solve(model, opts)
    }
}

impl PersistentSolver for Mosek {
    type Handle = MosekPersistent;

    fn persistent(&self) -> MosekPersistent {
        MosekPersistent::new()
    }
}

/// Native quadratic objectives are not included in the initial MOSEK SDP path.
pub(crate) fn psd_compatible(model: &Model) -> bool {
    if !model.has_active_psd_constraints() {
        return true;
    }
    model.kind() == ModelKind::SDP
        && !model.has_active_indicator_constraints()
        && model
            .objective()
            .as_ref()
            .is_none_or(|o| oximo_expr::extract_linear(&model.arena(), o.expr).is_some())
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;

    use super::*;

    fn assert_support(model: &Model, expected: bool) {
        assert_eq!(Mosek.validate_model(model, &MosekOptions::default()).is_supported(), expected);
        assert_eq!(
            MosekPersistent::new().validate_model(model, &MosekOptions::default()).is_supported(),
            expected
        );
    }

    #[test]
    fn shared_checks_accept_indicators_and_reject_sos() {
        let model = Model::new("shared capabilities");
        variable!(model, x);
        variable!(model, y, Bin);
        objective!(model, Min, x);
        assert_support(&model, true);
        model.add_indicator_constraint("conditional", y, true, x.le(1.0));
        assert_support(&model, true);
        model.add_sos_constraint("choice", SosType::Sos1, [(x, 1.0), (y, 2.0)]);
        assert_support(&model, false);
    }

    #[test]
    fn backend_hook_preserves_psd_restrictions() {
        let linear = Model::new("linear SDP");
        variable!(linear, x);
        psd_constraint!(linear, SymmetricMatrix::from_upper_triangle(1, [x]));
        objective!(linear, Min, x);
        assert_support(&linear, true);
        variable!(linear, y, Bin);
        linear.add_indicator_constraint("conditional", y, true, x.le(1.0));
        assert_support(&linear, false);

        let quadratic = Model::new("quadratic SDP");
        variable!(quadratic, z);
        psd_constraint!(quadratic, SymmetricMatrix::from_upper_triangle(1, [z]));
        objective!(quadratic, Min, z.square());
        assert_eq!(quadratic.kind(), ModelKind::SDP);
        assert_support(&quadratic, false);
    }
}
