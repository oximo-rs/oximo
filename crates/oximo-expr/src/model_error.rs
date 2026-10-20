use crate::{ModelMismatchError, ParamId, VarId};

/// Recoverable errors from imperative model writes.
#[derive(Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum ModelError {
    #[error(transparent)]
    ModelMismatch(#[from] ModelMismatchError),
    #[error("unknown parameter {0:?}")]
    UnknownParam(ParamId),
    #[error("unknown variable {0:?}")]
    UnknownVariable(VarId),
    #[error("expected a bare {0} handle")]
    InvalidHandle(&'static str),
    #[error("parameter {param:?} is frozen by reformulation {source_name:?}")]
    FrozenByReformulation { param: ParamId, source_name: String },
    #[error("variable {0:?} has bounds frozen by a reformulation")]
    FrozenVariable(VarId),
    #[error("parameter {param:?} must be finite, received {value}")]
    ParamDomain { param: ParamId, value: f64 },
    #[error("parameter {0:?} occurs more than once in a batch")]
    DuplicateParam(ParamId),
    #[error("invalid bounds [{lower}, {upper}] for variable {variable:?}")]
    BoundsInfeasible { variable: VarId, lower: f64, upper: f64 },
    #[error("invalid {operation} for variable {variable:?}: {value}")]
    VariableValue { variable: VarId, operation: &'static str, value: f64 },
    #[error("parameter bounds are unsupported for this variable domain")]
    ParameterBoundDomain,
}

// TODO: Make param expressions beyond a bare parameter?

/// A variable-free bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bound {
    Constant(f64),
    Parameter(ParamId),
}

impl Bound {
    /// Resolve against the supplied parameter assignment.
    pub fn resolve(self, arena: &crate::ExprArena) -> f64 {
        match self {
            Self::Constant(v) => v,
            Self::Parameter(p) => arena.param_value(p),
        }
    }

    pub fn parameter(self) -> Option<ParamId> {
        match self {
            Self::Constant(_) => None,
            Self::Parameter(p) => Some(p),
        }
    }
}

impl From<f64> for Bound {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

/// Bound-ordering validation stored beside parameter values, including handle writes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundCheck {
    pub variable: VarId,
    pub lower: Bound,
    pub upper: Bound,
}
