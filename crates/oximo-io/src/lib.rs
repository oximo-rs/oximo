#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod error;
pub mod lp;
pub mod mps;
pub mod nl;
mod objective;

/// Reserve ordinary text models without sizing nonlinear expressions in advance.
fn text_buffer(model: &oximo_core::Model) -> Vec<u8> {
    let estimate = model.num_constraints().saturating_add(model.num_variables()).saturating_mul(32);
    if estimate < 4096 {
        Vec::new()
    } else {
        let capacity = estimate.min(64 * 1024 * 1024).next_power_of_two();
        Vec::with_capacity(capacity)
    }
}

pub use error::IoError;
pub use lp::{read_lp, read_lp_file, to_lp_string, write_lp};
pub use mps::{
    MpsQuadraticFormat, MpsReadOptions, MpsWriteOptions, read_mps, read_mps_file,
    read_mps_file_with, read_mps_with, to_mps_string, to_mps_string_with, write_mps,
    write_mps_with,
};
pub use nl::{
    Complementarity, DefinedVar, ImportedFunction, NlFormat, SuffixData, SuffixFlavour, SuffixKind,
    WriteOptions, read_nl, read_nl_file, to_nl_string, to_nl_string_with, write_nl, write_nl_files,
    write_nl_with,
};
