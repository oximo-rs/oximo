//! Read Boolean decisions and Hull activations from algebraic solver results.

use oximo_core::{
    ModelMismatchError,
    gdp::{BooleanHandle, GdpHullActivationHandle},
};
use oximo_solver::{SolutionPoint, SolverResult};
use thiserror::Error;

const BOOLEAN_VALUE_TOLERANCE: f64 = 1e-5;

/// A foreign Boolean handle or a selection value that cannot represent a Boolean.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum BooleanValueError {
    #[error(transparent)]
    ModelMismatch(#[from] ModelMismatchError),
    #[error("selection value {value} is not within {tolerance} of zero or one")]
    NonBooleanValue { value: f64, tolerance: f64 },
}

/// GDP value queries for [`SolverResult`] and individual [`SolutionPoint`]s.
pub trait GdpResultExt {
    /// Read a model-bound Hull activation, preserving fractional relaxation values.
    /// Returns `Ok(None)` when a required variable or primal solution is absent.
    /// Obtain the handle from the reformulated model's report.
    ///
    /// # Errors
    /// Returns [`ModelMismatchError`] for a foreign handle, even without a solution.
    /// For [`SolverResult`], also checks the best solution's model identity.
    fn hull_activation_value_of(
        &self,
        activation: GdpHullActivationHandle,
    ) -> Result<Option<f64>, ModelMismatchError>;

    /// Read a Boolean decision using its associated binary variable.
    ///
    /// Values within an absolute tolerance of `1e-5` of zero or one become
    /// `false` or `true`, respectively. Returns `Ok(None)` when no primal value
    /// is available. For [`SolverResult`], queries the best solution.
    /// Use [`SolverResult::solution`] to query another point.
    ///
    /// # Errors
    /// Returns [`BooleanValueError::ModelMismatch`] for a foreign handle, even
    /// when no solution is available. Returns [`BooleanValueError::NonBooleanValue`]
    /// for fractional values outside the tolerance or non-finite values.
    /// Use the binary expression's numeric `value_of` when inspecting relaxations.
    fn boolean_value_of(
        &self,
        boolean: BooleanHandle<'_>,
    ) -> Result<Option<bool>, BooleanValueError>;
}

impl GdpResultExt for SolutionPoint {
    fn hull_activation_value_of(
        &self,
        activation: GdpHullActivationHandle,
    ) -> Result<Option<f64>, ModelMismatchError> {
        ensure_activation_model(self.model_id(), activation)?;

        Ok(activation.kind().value_with(|id| self.value(id)))
    }

    fn boolean_value_of(
        &self,
        boolean: BooleanHandle<'_>,
    ) -> Result<Option<bool>, BooleanValueError> {
        boolean_value(self.value_of(boolean.binary())?)
    }
}

impl GdpResultExt for SolverResult {
    fn hull_activation_value_of(
        &self,
        activation: GdpHullActivationHandle,
    ) -> Result<Option<f64>, ModelMismatchError> {
        ensure_activation_model(self.model_id(), activation)?;

        self.solutions.first().map_or(Ok(None), |point| point.hull_activation_value_of(activation))
    }

    fn boolean_value_of(
        &self,
        boolean: BooleanHandle<'_>,
    ) -> Result<Option<bool>, BooleanValueError> {
        boolean_value(self.value_of(boolean.binary())?)
    }
}

fn boolean_value(value: Option<f64>) -> Result<Option<bool>, BooleanValueError> {
    value
        .map(|value| {
            if value.abs() <= BOOLEAN_VALUE_TOLERANCE {
                Ok(false)
            } else if (value - 1.0).abs() <= BOOLEAN_VALUE_TOLERANCE {
                Ok(true)
            } else {
                Err(BooleanValueError::NonBooleanValue {
                    value,
                    tolerance: BOOLEAN_VALUE_TOLERANCE,
                })
            }
        })
        .transpose()
}

fn ensure_activation_model(
    expected: oximo_core::ModelId,
    activation: GdpHullActivationHandle,
) -> Result<(), ModelMismatchError> {
    if activation.model_id() == expected {
        Ok(())
    } else {
        Err(ModelMismatchError::new(expected, activation.model_id()))
    }
}
