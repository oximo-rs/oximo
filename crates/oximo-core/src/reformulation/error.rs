//! Errors shared by the core model reformulations.

use smol_str::SmolStr;
use thiserror::Error;

/// Why an explicit model reformulation could not be constructed.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReformulationError {
    #[error("SOS constraint #{0} does not exist on this model")]
    UnknownSosConstraint(usize),
    #[error("fallback Big-M must be finite and positive, got {0}")]
    InvalidFallbackBigM(f64),
    #[error(
        "cannot reformulate SOS constraint {constraint:?}: variable {variable:?} has no finite \
         {side} bound; provide SosReformulationOptions::with_fallback_big_m(...)"
    )]
    MissingFiniteBound { constraint: SmolStr, variable: SmolStr, side: &'static str },
    #[error("indicator constraint #{0} does not exist on this model")]
    UnknownIndicatorConstraint(usize),
    #[error(
        "cannot reformulate indicator constraint {constraint:?}: its body depends on a parameter"
    )]
    ParameterDependentIndicator { constraint: SmolStr },
    #[error(
        "cannot reformulate indicator constraint {constraint:?}: body is not a finite affine expression"
    )]
    InvalidIndicatorExpression { constraint: SmolStr },
    #[error(
        "cannot reformulate indicator constraint {constraint:?}: lower bound is +infinity or upper bound is -infinity"
    )]
    InvalidIndicatorBounds { constraint: SmolStr },
    #[error(
        "cannot derive finite Big-M for indicator constraint {constraint:?} {side} side; provide IndicatorReformulationOptions::with_fallback_big_m(...)"
    )]
    MissingIndicatorBigM { constraint: SmolStr, side: &'static str },
}
