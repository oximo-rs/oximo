use thiserror::Error;

/// Errors produced while building derivatives from a model.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AutodiffError {
    #[error("model has no objective")]
    NoObjective,
    #[error("point has {got} entries but the model has {expected} variables")]
    DimensionMismatch { expected: usize, got: usize },
}
