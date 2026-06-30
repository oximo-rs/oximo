use crate::solver::Solver;

/// A [`Solver`] that can hand out a stateful handle keeping the built backend model
/// resident across solves.
///
/// The returned [`Handle`](PersistentSolver::Handle) is itself a [`Solver`]: build it
/// once, then call [`solve`](Solver::solve) on it repeatedly. When only objective
/// coefficients or variable bounds changed between calls it updates the resident
/// model in place and warm-starts from the previous basis. Any structural change
/// rebuilds transparently, so results always match a one-shot solve.
pub trait PersistentSolver: Solver {
    /// The stateful, resident handle. Solving the same (or a structurally identical)
    /// model on it repeatedly reuses the build.
    type Handle: Solver<Options = Self::Options>;

    /// Create a fresh resident handle with no model loaded yet.
    /// The first [`solve`](Solver::solve) builds it.
    fn persistent(&self) -> Self::Handle;
}
