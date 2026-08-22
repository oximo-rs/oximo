#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod error;
pub mod lp;
pub mod mps;
pub mod nl;

pub use error::IoError;
pub use lp::{read_lp, read_lp_file, to_lp_string, write_lp};
pub use mps::{
    MpsQuadraticFormat, MpsReadOptions, MpsWriteOptions, read_mps, read_mps_file,
    read_mps_file_with, read_mps_with, to_mps_string, to_mps_string_with, write_mps,
    write_mps_with,
};
pub use nl::{
    Complementarity, DefinedVar, ImportedFunction, NlFormat, NlReadData, SuffixData, SuffixFlavour,
    SuffixKind, WriteOptions, read_nl, read_nl_data, read_nl_file, read_nl_file_data, to_nl_string,
    to_nl_string_with, write_nl, write_nl_files, write_nl_with,
};
