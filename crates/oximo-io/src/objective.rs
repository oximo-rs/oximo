//! Objective handling shared by the LP and MPS writers.

use oximo_core::{Model, ObjectiveSense, Variable, var_name};
use oximo_expr::{ExprArena, QuadraticTerms, describe_nonlinear_term, extract_quadratic};

use crate::error::IoError;

/// The objective sense and quadratic terms an exporter should write.
///
/// A feasibility model (`objective!(m, Feasibility)`) has no objective to
/// export, so it is written as a minimized zero objective - the same encoding
/// the solver backends translate it to. A model that declared no direction at
/// all is still [`IoError::NoObjective`].
///
/// # Errors
///
/// Returns [`IoError::NoObjective`] when the model declares neither an
/// objective nor feasibility, and [`IoError::Nonlinear`] when the objective is
/// not linear or quadratic.
pub(crate) fn export_terms(
    model: &Model,
    arena: &ExprArena,
    vars: &[Variable],
) -> Result<(ObjectiveSense, QuadraticTerms), IoError> {
    model.ensure_gdp_reformulated().map_err(|_| IoError::UnreformulatedGdp)?;
    model.ensure_objective_declared().map_err(|_| IoError::NoObjective)?;
    let Some(objective) = model.objective().clone() else {
        return Ok((ObjectiveSense::Minimize, QuadraticTerms::default()));
    };
    let terms = extract_quadratic(arena, objective.expr).ok_or_else(|| IoError::Nonlinear {
        location: "the objective".into(),
        term: describe_nonlinear_term(arena, objective.expr, &|v| var_name(vars, v))
            .unwrap_or_else(|| "<nonlinear>".into()),
    })?;
    Ok((objective.sense, terms))
}
