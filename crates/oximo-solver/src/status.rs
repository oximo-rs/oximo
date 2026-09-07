use thiserror::Error;

/// Why a solver stopped, independent of whether a usable point was returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminationStatus {
    /// Proven globally optimal.
    Optimal,
    /// A local optimum.
    LocallyOptimal,
    /// Stopped with a feasible point, without any optimality claim.
    Feasible,
    /// Proven infeasible.
    Infeasible,
    /// Proven unbounded.
    Unbounded,
    /// Infeasible or unbounded, the backend can't differentiate.
    InfeasibleOrUnbounded,
    /// Stopped at an iteration limit.
    IterationLimit,
    /// Stopped at a time limit.
    TimeLimit,
    /// Stopped at a branch-and-bound node limit.
    NodeLimit,
    /// Stopped after reaching an objective cutoff or target.
    ObjectiveLimit,
    /// Stopped after finding the requested number of solutions.
    SolutionLimit,
    /// Stopped at a solver-defined work limit.
    WorkLimit,
    /// Stopped because the solver exhausted its memory limit.
    MemoryLimit,
    /// Converged to a locally infeasible point.
    LocallyInfeasible,
    /// The solver could not run because no usable license was available.
    LicenseError,
    /// Stopped by a genuine user or external interrupt.
    Interrupted,
    /// The solver hit a numerical problem (singular basis, presolve error, ...).
    NumericError,
    /// No solve has been attempted yet.
    NotSolved,
    /// A backend status with no direct mapping. Carries the raw label.
    Other(String),
}

impl TerminationStatus {
    /// Whether a solver that stopped for this reason may still return a usable
    /// primal point. `true` for optimality, plain feasibility, and the various
    /// limits (which keep the best incumbent found so far), `false` for
    /// infeasible/unbounded/error/unsolved states.
    pub fn admits_primal(&self) -> bool {
        matches!(
            self,
            Self::Optimal
                | Self::LocallyOptimal
                | Self::Feasible
                | Self::IterationLimit
                | Self::TimeLimit
                | Self::NodeLimit
                | Self::ObjectiveLimit
                | Self::SolutionLimit
                | Self::WorkLimit
                | Self::MemoryLimit
                | Self::Interrupted
        )
    }

    /// Whether this status proves the model infeasible, so a conflict/IIS
    /// diagnosis is meaningful. Treats the ambiguous [`InfeasibleOrUnbounded`]
    /// as infeasible.
    ///
    /// [`InfeasibleOrUnbounded`]: Self::InfeasibleOrUnbounded
    #[must_use]
    pub fn is_infeasible(&self) -> bool {
        matches!(self, Self::Infeasible | Self::InfeasibleOrUnbounded)
    }
}

/// The status of the primal point held in a [`crate::SolverResult`].
///
/// Decoupled from [`TerminationStatus`] so a result that stopped at a limit can
/// still carry a usable incumbent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrimalStatus {
    /// No primal point is available.
    NoSolution,
    /// A feasible point is available, but not proven optimal.
    FeasiblePoint,
    /// A proven-optimal point is available.
    OptimalPoint,
}

impl PrimalStatus {
    /// Classify the primal status from the termination reason and whether a
    /// point was actually returned. `Optimal` termination with a point yields
    /// [`PrimalStatus::OptimalPoint`]; any other termination with a point yields
    /// [`PrimalStatus::FeasiblePoint`]; no point yields
    /// [`PrimalStatus::NoSolution`].
    pub fn infer(termination: &TerminationStatus, has_point: bool) -> Self {
        if !has_point {
            Self::NoSolution
        } else if matches!(termination, TerminationStatus::Optimal) {
            Self::OptimalPoint
        } else {
            Self::FeasiblePoint
        }
    }

    /// Whether a usable primal point is available.
    pub fn has_solution(self) -> bool {
        !matches!(self, Self::NoSolution)
    }
}

#[derive(Error)]
pub enum SolverError {
    #[error("solver does not support model kind {kind:?} (supported: {})", format_kinds(supported))]
    UnsupportedKind { kind: oximo_core::ModelKind, supported: &'static [oximo_core::ModelKind] },
    #[error("solver does not support native {0} constraints")]
    UnsupportedConstraint(&'static str),
    #[error(
        "solver does not support native SOS1/SOS2 constraints. Explicitly reformulate them with \
         Model::reformulate_sos or Model::to_reformulated_sos_model"
    )]
    UnsupportedSos,
    #[error("model is missing an objective")]
    NoObjective,
    #[error("{location} contains a nonlinear term unsupported by this backend: {term}")]
    Nonlinear { location: String, term: String },
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Core(#[from] oximo_core::Error),
}

impl SolverError {
    pub fn unsupported_kind(
        kind: oximo_core::ModelKind,
        supported: &'static [oximo_core::ModelKind],
    ) -> Self {
        Self::UnsupportedKind { kind, supported }
    }
}

// Mirror `Display` in `Debug`. When a `main` returning `Result` propagates an
// error, Rust's `Termination` impl prints it with `{:?}`. The derived `Debug`
// would escape newlines in `Backend` messages (e.g. multi-line GAMS reports)
// onto a single line. These messages are human-facing, so render them as-is.
impl std::fmt::Debug for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

fn format_kinds(kinds: &[oximo_core::ModelKind]) -> String {
    kinds.iter().map(|k| format!("{:?}", k)).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_sos_error_points_to_explicit_reformulation_methods() {
        let message = SolverError::UnsupportedSos.to_string();
        assert!(message.contains("Model::reformulate_sos"));
        assert!(message.contains("Model::to_reformulated_sos_model"));
    }

    /// The contract for a single termination: whether it admits a primal point
    /// ([`TerminationStatus::admits_primal`]) and what [`PrimalStatus::infer`]
    /// yields when a point is present.
    fn contract(t: &TerminationStatus) -> (bool, PrimalStatus) {
        use TerminationStatus as T;
        match t {
            T::Optimal => (true, PrimalStatus::OptimalPoint),
            T::LocallyOptimal
            | T::Feasible
            | T::IterationLimit
            | T::TimeLimit
            | T::NodeLimit
            | T::ObjectiveLimit
            | T::SolutionLimit
            | T::WorkLimit
            | T::MemoryLimit
            | T::Interrupted => (true, PrimalStatus::FeasiblePoint),
            T::Infeasible
            | T::Unbounded
            | T::InfeasibleOrUnbounded
            | T::LocallyInfeasible
            | T::LicenseError
            | T::NumericError
            | T::NotSolved
            | T::Other(_) => (false, PrimalStatus::FeasiblePoint),
        }
    }

    fn all_terminations() -> Vec<TerminationStatus> {
        use TerminationStatus as T;
        vec![
            T::Optimal,
            T::LocallyOptimal,
            T::Feasible,
            T::Infeasible,
            T::Unbounded,
            T::InfeasibleOrUnbounded,
            T::IterationLimit,
            T::TimeLimit,
            T::NodeLimit,
            T::ObjectiveLimit,
            T::SolutionLimit,
            T::WorkLimit,
            T::MemoryLimit,
            T::LocallyInfeasible,
            T::LicenseError,
            T::Interrupted,
            T::NumericError,
            T::NotSolved,
            T::Other("backend_specific".into()),
        ]
    }

    #[test]
    fn admits_primal_and_infer_match_contract() {
        for t in all_terminations() {
            let (admits, with_point) = contract(&t);
            assert_eq!(t.admits_primal(), admits, "admits_primal for {t:?}");
            assert_eq!(PrimalStatus::infer(&t, true), with_point, "infer(.., true) for {t:?}");
            assert_eq!(
                PrimalStatus::infer(&t, false),
                PrimalStatus::NoSolution,
                "infer(.., false) for {t:?}"
            );
        }
    }

    #[test]
    fn admits_primal_drives_inference_for_status_driven_backends() {
        for t in all_terminations() {
            let has_point = t.admits_primal();
            let primal = PrimalStatus::infer(&t, has_point);
            assert_eq!(
                primal.has_solution(),
                has_point,
                "has_solution mirrors admits_primal for {t:?}"
            );
            let expected = match (has_point, &t) {
                (false, _) => PrimalStatus::NoSolution,
                (true, TerminationStatus::Optimal) => PrimalStatus::OptimalPoint,
                (true, _) => PrimalStatus::FeasiblePoint,
            };
            assert_eq!(primal, expected, "inferred primal for {t:?}");
        }
    }

    #[test]
    fn is_infeasible_covers_infeasible_and_ambiguous() {
        use TerminationStatus as T;
        for t in all_terminations() {
            let expected = matches!(t, T::Infeasible | T::InfeasibleOrUnbounded);
            assert_eq!(t.is_infeasible(), expected, "is_infeasible for {t:?}");
        }
    }

    #[test]
    fn primal_status_has_solution() {
        assert!(!PrimalStatus::NoSolution.has_solution());
        assert!(PrimalStatus::FeasiblePoint.has_solution());
        assert!(PrimalStatus::OptimalPoint.has_solution());
    }
}
