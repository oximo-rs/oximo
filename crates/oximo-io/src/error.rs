use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("model has no objective")]
    NoObjective,
    #[error("nonlinear nodes are not representable in this format")]
    Nonlinear,
    #[error("second-order cone constraints cannot be represented in this format")]
    Conic,
    #[error("unsupported expression node in NL writer: {0}")]
    UnsupportedNode(&'static str),
    #[error("variable domain {0} is not representable in NL")]
    UnsupportedDomain(&'static str),
    #[error("non-finite numeric value in expression")]
    InvalidNumber,
    #[error("binary NL output is not UTF-8, write it to a byte sink with write_nl_with")]
    BinaryToString,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
