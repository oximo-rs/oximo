use thiserror::Error;

/// Why a solver stopped, independent of whether a usable point was returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminationStatus {
    /// Proven globally optimal.
    Optimal,
    /// A local optimum.
    LocallyOptimal,
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
    /// Stopped by an external interrupt or solver-specific user limit.
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
    /// primal point. `true` for optimality and for the various limits (which
    /// keep the best incumbent found so far), `false` for infeasible/unbounded/
    /// error/unsolved states.
    pub fn admits_primal(&self) -> bool {
        matches!(
            self,
            Self::Optimal
                | Self::LocallyOptimal
                | Self::IterationLimit
                | Self::TimeLimit
                | Self::NodeLimit
                | Self::Interrupted
        )
    }
}

/// The status of the primal point held in a [`SolverResult`].
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
    #[error("solver does not support model kind {0:?}")]
    UnsupportedKind(oximo_core::ModelKind),
    #[error("model is missing an objective")]
    NoObjective,
    #[error("nonlinear constructs are not supported by this backend")]
    Nonlinear,
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Core(#[from] oximo_core::Error),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract for a single termination: whether it admits a primal point
    /// ([`TerminationStatus::admits_primal`]) and what [`PrimalStatus::infer`]
    /// yields when a point is present.
    fn contract(t: &TerminationStatus) -> (bool, PrimalStatus) {
        use TerminationStatus as T;
        match t {
            T::Optimal => (true, PrimalStatus::OptimalPoint),
            T::LocallyOptimal
            | T::IterationLimit
            | T::TimeLimit
            | T::NodeLimit
            | T::Interrupted => (true, PrimalStatus::FeasiblePoint),
            T::Infeasible
            | T::Unbounded
            | T::InfeasibleOrUnbounded
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
            T::Infeasible,
            T::Unbounded,
            T::InfeasibleOrUnbounded,
            T::IterationLimit,
            T::TimeLimit,
            T::NodeLimit,
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
    fn primal_status_has_solution() {
        assert!(!PrimalStatus::NoSolution.has_solution());
        assert!(PrimalStatus::FeasiblePoint.has_solution());
        assert!(PrimalStatus::OptimalPoint.has_solution());
    }
}
