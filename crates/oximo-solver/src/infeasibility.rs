use oximo_core::{ConstraintId, Model, SocConstraintId, VarId};

use crate::result::SolverResult;
use crate::solver::Solver;
use crate::status::SolverError;

/// Which side of a variable's bound participates in an infeasibility.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VarBoundKind {
    /// The variable's lower bound.
    Lower,
    /// The variable's upper bound.
    Upper,
}

/// An irreducible infeasible subsystem (IIS).
///
/// A minimal set of constraints and variable bounds that together make the model
/// infeasible. Removing any single member makes the remaining subsystem feasible.
///
/// Backends that can diagnose infeasibility return one via
/// [`InfeasibilityDiagnosis::compute_iis`]. The members are keyed by the same ids the
/// model assigns ([`ConstraintId`], [`SocConstraintId`], [`VarId`]), so
/// [`Iis::report`] can name them against the [`Model`].
#[derive(Clone, Debug, Default)]
pub struct Iis {
    /// Algebraic constraints in the IIS.
    pub constraints: Vec<ConstraintId>,
    /// Second-order-cone constraints in the IIS.
    pub soc_constraints: Vec<SocConstraintId>,
    /// Variable bounds in the IIS, each `(variable, which bound)`.
    pub var_bounds: Vec<(VarId, VarBoundKind)>,
}

impl Iis {
    /// Whether the IIS carries no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty() && self.soc_constraints.is_empty() && self.var_bounds.is_empty()
    }

    /// The total number of members (constraints, SOC constraints, and variable
    /// bounds) in the IIS.
    #[must_use]
    pub fn len(&self) -> usize {
        self.constraints.len() + self.soc_constraints.len() + self.var_bounds.len()
    }

    /// A human-readable, model-aware listing of this IIS.
    ///
    /// It names every constraint and variable bound in the subsystem using the
    /// model's own names. Render with [`ToString::to_string`] or by printing.
    #[must_use]
    pub fn report<'a>(&'a self, model: &'a Model) -> IisReport<'a> {
        IisReport { iis: self, model }
    }
}

/// A printable, model-aware listing of an [`Iis`]. Created by [`Iis::report`].
#[derive(Debug)]
pub struct IisReport<'a> {
    iis: &'a Iis,
    model: &'a Model,
}

impl std::fmt::Display for IisReport<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let iis = self.iis;
        let m = self.model;

        writeln!(f, "irreducible infeasible subsystem ({} members)", iis.len())?;

        if !iis.constraints.is_empty() {
            let cons = m.constraints();
            writeln!(f, "\nconstraints ({})", iis.constraints.len())?;
            for id in &iis.constraints {
                match cons.get(id.index()) {
                    Some(c) => writeln!(f, "  {}", c.name)?,
                    None => writeln!(f, "  <constraint #{}>", id.index())?,
                }
            }
        }

        if !iis.soc_constraints.is_empty() {
            let socs = m.soc_constraints();
            writeln!(f, "\nsoc constraints ({})", iis.soc_constraints.len())?;
            for id in &iis.soc_constraints {
                match socs.get(id.index()) {
                    Some(s) => writeln!(f, "  {}", s.name)?,
                    None => writeln!(f, "  <soc #{}>", id.index())?,
                }
            }
        }

        if !iis.var_bounds.is_empty() {
            let vars = m.variables();
            writeln!(f, "\nvariable bounds ({})", iis.var_bounds.len())?;
            for (id, kind) in &iis.var_bounds {
                let side = match kind {
                    VarBoundKind::Lower => "lower",
                    VarBoundKind::Upper => "upper",
                };
                match vars.get(id.index()) {
                    Some(v) => writeln!(f, "  {} ({side} bound)", v.name)?,
                    None => writeln!(f, "  <var #{}> ({side} bound)", id.index())?,
                }
            }
        }

        Ok(())
    }
}

/// A [`Solver`] that can diagnose why a model is infeasible by computing an
/// irreducible infeasible subsystem ([`Iis`]).
///
/// Implemented only by backends whose underlying solver exposes a native IIS/
/// conflict-refiner options. Solve the model and if infeasible call
/// [`compute_iis`](InfeasibilityDiagnosis::compute_iis) to get
/// the minimal conflicting set.
pub trait InfeasibilityDiagnosis: Solver {
    /// Compute an irreducible infeasible subsystem for `model`.
    ///
    /// The backend solves `model` (with `opts`) and, if it is infeasible, returns the
    /// minimal set of constraints and variable bounds responsible.
    ///
    /// # Errors
    ///
    /// Returns a [`SolverError`] if the solve fails, the backend cannot represent the
    /// model, or the model is not actually infeasible.
    fn compute_iis(&mut self, model: &Model, opts: &Self::Options) -> Result<Iis, SolverError>;
}

/// Helper for backends, whether a [`SolverResult`] indicates the model is infeasible.
/// Treats the ambiguous `InfeasibleOrUnbounded` as infeasible.
#[must_use]
pub fn is_infeasible(result: &SolverResult) -> bool {
    result.termination.is_infeasible()
}

#[cfg(test)]
mod tests {
    use oximo_core::{constraint, variable};

    use super::*;

    #[test]
    fn report_names_members() {
        let m = Model::new("infeas");
        variable!(m, x >= 0.0);
        let lo = constraint!(m, floor, x >= 2.0);
        let hi = constraint!(m, ceil, x <= 1.0);

        let iis = Iis {
            constraints: vec![lo, hi],
            soc_constraints: Vec::new(),
            var_bounds: vec![(x.var_id().unwrap(), VarBoundKind::Lower)],
        };

        assert_eq!(iis.len(), 3);
        assert!(!iis.is_empty());

        let out = iis.report(&m).to_string();
        assert!(out.contains("irreducible infeasible subsystem (3 members)"), "{out}");
        assert!(out.contains("floor"), "{out}");
        assert!(out.contains("ceil"), "{out}");
        assert!(out.contains("x (lower bound)"), "{out}");
    }

    #[test]
    fn empty_iis_reports_zero() {
        let m = Model::new("ok");
        let iis = Iis::default();
        assert!(iis.is_empty());
        assert_eq!(iis.len(), 0);
        let out = iis.report(&m).to_string();
        assert!(out.contains("(0 members)"), "{out}");
    }
}
