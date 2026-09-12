use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("model has no objective")]
    NoObjective,
    #[error(
        "expected a linear or quadratic expression in {location}, found nonlinear term: {term}"
    )]
    Nonlinear { location: String, term: String },
    #[error("second-order cone constraints cannot be represented in this format")]
    Conic,
    #[error("unsupported expression node in NL writer: {0}")]
    UnsupportedNode(&'static str),
    #[error("AMPL .nl does not document an encoding for nonlinear operator {operator}")]
    UnsupportedNonlinearOperator { operator: &'static str },
    #[error("variable domain {0} is not representable in NL")]
    UnsupportedDomain(&'static str),
    #[error("variable {0} is used in an expression but was not added to this model")]
    UnknownVar(String),
    #[error("non-finite numeric value {value} in {location}")]
    InvalidNumber { value: f64, location: String },
    #[error("binary NL output is not UTF-8, write it to a byte sink with write_nl_with")]
    BinaryToString,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid NL file in {section}: {message}")]
    InvalidNl { section: String, message: String },
    #[error("unsupported NL feature in {section}: {feature}")]
    UnsupportedNl { section: String, feature: String },
    #[error("invalid LP file at {line}:{column}: {message}")]
    InvalidLp { line: usize, column: usize, message: String },
    #[error("unsupported LP feature in {section}: {feature}")]
    UnsupportedLp { section: String, feature: String },
    #[error("invalid MPS file at {line}:{column}: {message}")]
    InvalidMps { line: usize, column: usize, message: String },
    #[error("unsupported MPS feature in {section}: {feature}")]
    UnsupportedMps { section: String, feature: String },
}
