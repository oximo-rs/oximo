#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod convex;
mod fbbt;
mod minima;
mod options;
mod persistent;
mod tnlp;
mod translate;

#[cfg(feature = "enzyme")]
mod exact;
#[cfg(not(feature = "enzyme"))]
mod hybrid;
#[cfg(not(feature = "enzyme"))]
mod stable;

#[cfg(all(feature = "benchmark-support", not(feature = "enzyme")))]
#[doc(hidden)]
pub use hybrid::benchmark_support;

pub use minima::{PounceMinimaMethod, PounceMinimaOptions};
pub use options::{
    MuStrategy, PounceAlgorithm, PounceOptionValue, PounceOptions, PounceSolverSelection,
};
pub use persistent::PouncePersistent;

/// The POUNCE backend:
/// specialized convex LP/QP/SOCP engines with a general IPOPT-lineage
/// QCP/NLP fallback.
///
/// On stable Rust an all-linear/quadratic model gets exact analytic
/// derivatives (including the Hessian).
/// A model with nonlinear functions is handed to POUNCE's builder, which
/// finite-differences them with an L-BFGS Hessian. With the `enzyme`
/// feature everything is exact via `oximo-autodiff`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pounce;

impl oximo_solver::Solver for Pounce {
    type Options = PounceOptions;

    fn profile(&self, opts: &Self::Options) -> std::borrow::Cow<'_, oximo_solver::SolverProfile> {
        std::borrow::Cow::Owned({
            use oximo_core::ModelKind;
            use oximo_expr::ExprClass::{Linear, Nonlinear, Quadratic};
            use oximo_solver::VariableDomain::Real;
            let selection = crate::translate::selected_solver(opts).ok();
            let algorithm = crate::translate::selected_algorithm(opts).ok();
            let profile = oximo_solver::SolverProfile::new(format!(
                "pounce ({})",
                selection.map_or("invalid route", |s| s.as_str())
            ))
            .with_scalar_constraints([Linear, Quadratic, Nonlinear])
            .with_scalar_objectives([Linear, Quadratic, Nonlinear])
            .with_soc(true)
            .with_kinds(oximo_solver::MODEL_KINDS.into_iter().filter(|kind| {
                let (Some(algorithm), Some(selection)) = (algorithm, selection) else {
                    return true;
                };
                algorithm.supports(*kind)
                    && match selection {
                        PounceSolverSelection::Auto | PounceSolverSelection::Nlp => true,
                        PounceSolverSelection::LpIpm => *kind == ModelKind::LP,
                        PounceSolverSelection::QpIpm | PounceSolverSelection::QpActiveSet => {
                            matches!(kind, ModelKind::LP | ModelKind::QP)
                        }
                        PounceSolverSelection::Socp => {
                            matches!(kind, ModelKind::LP | ModelKind::QP | ModelKind::SOCP)
                        }
                    }
            }))
            .with_domains([Real])
            .with_operators(oximo_solver::ExpressionOperator::ALL.iter().copied());
            if opts.minima.is_some()
                || selection == Some(PounceSolverSelection::Nlp)
                || (selection == Some(PounceSolverSelection::Auto)
                    && algorithm == Some(PounceAlgorithm::ActiveSetSqp))
            {
                profile.reject_combination(
                    |f| f.has_explicit_soc,
                    "the selected NLP/search route does not accept explicit SOC constraints",
                )
            } else {
                profile
            }
        })
    }

    fn name(&self) -> &str {
        "pounce"
    }

    fn validate_requirements(
        &self,
        model: &oximo_core::Model,
        opts: &Self::Options,
        requirements: &oximo_solver::ModelRequirements,
    ) -> Vec<oximo_solver::RequirementDiagnostic> {
        convex::validate_route(model, opts, requirements)
            .err()
            .map(|error| {
                oximo_solver::RequirementDiagnostic::adapter("POUNCE route", error.to_string())
            })
            .into_iter()
            .collect()
    }

    fn solve(
        &mut self,
        model: &oximo_core::Model,
        opts: &Self::Options,
    ) -> Result<oximo_solver::SolverResult, oximo_solver::SolverError> {
        translate::solve(model, opts)
    }
}

impl oximo_solver::PersistentSolver for Pounce {
    type Handle = PouncePersistent;

    fn persistent(&self) -> Self::Handle {
        PouncePersistent::new()
    }
}
