#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod error;
pub mod lp;
pub mod mps;
pub mod nl;

pub use error::IoError;
pub use lp::{to_lp_string, write_lp};
pub use mps::{to_mps_string, write_mps};
pub use nl::{
    Complementarity, DefinedVar, ImportedFunction, NlFormat, SuffixData, SuffixFlavour, SuffixKind,
    WriteOptions, to_nl_string, to_nl_string_with, write_nl, write_nl_files, write_nl_with,
};
