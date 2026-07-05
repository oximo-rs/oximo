//! AMPL `.nl` file format writer.
//!
//! `.nl` is the de facto interchange format for nonlinear and mixed-integer
//! nonlinear models. Any ASL-linked solver (Ipopt, BARON, Couenne, Bonmin,
//! Knitro) reads it.
//!
//! Three entry points:
//! - [`write_nl`] / [`to_nl_string`]: default ASCII output with comments.
//! - [`write_nl_with`] / [`to_nl_string_with`]: caller controls
//!   format / precision / comments / NaN-Inf handling / F-S-V-d hooks via
//!   [`WriteOptions`].
//! - [`write_nl_files`]: write `<stub>.nl` plus optional `<stub>.row` and
//!   `<stub>.col` sidecar name files. Required by tools that look up
//!   user-readable names by index.
//!
//! References:
//! - D. M. Gay, "Writing .nl Files," Sandia National Laboratories, Albuquerque, NM, USA, Tech. Rep.
//!   SAND2005-7907P; 526059, Nov. 2005. \[Online]\. Available: <https://ampl.github.io/nlwrite.pdf>
//!   (accessed June 2, 2026).
//! - AMPL Optimization, Inc. "nl-writer2.cc," in mp, GitHub repository. \[Online]\.
//!   Available: <https://github.com/ampl/mp/blob/develop/nl-writer2/src/nl-writer2.cc>
//!   (accessed: June 2, 2026).
//! - AMPL Optimization, Inc. "Hooking Your Solver to AMPL," by David M. Gay. \[Online]\.
//!   Available: <https://ampl.com/wp-content/uploads/Hooking-Your-Solver-to-AMPL-by-David-M.-Gay.pdf>
//!   (accessed: June 3, 2026).

// TODO: Missing features:
// - Hollerith strings

mod analyze;
mod emit_expr;
mod header;
mod options;
mod permute;
mod segments;
mod writer;

pub use options::{
    Complementarity, DefinedVar, ImportedFunction, NlFormat, SuffixData, SuffixFlavour, SuffixKind,
    WriteOptions,
};

use std::io::Write;
use std::path::Path;

use oximo_core::Model;

use crate::error::IoError;
use writer::Writer;

/// Write `model` to `out` using default options (ASCII, comments on,
/// shortest-round-trip numbers, error on NaN/Inf).
///
/// # Errors
///
/// Returns [`IoError`] on missing objective, unsupported nodes, or I/O failure.
pub fn write_nl<W: Write>(model: &Model, out: &mut W) -> Result<(), IoError> {
    write_nl_with(model, out, &WriteOptions::default())
}

/// Convenience: render the `.nl` file into a `String` using default options
/// (ASCII, so the result is always valid UTF-8).
///
/// # Errors
///
/// Returns [`IoError`] on missing objective, unsupported nodes, or I/O failure.
pub fn to_nl_string(model: &Model) -> Result<String, IoError> {
    to_nl_string_with(model, &WriteOptions::default())
}

/// Write `model` to `out`, honouring `opts`.
///
/// # Errors
///
/// Returns [`IoError`] on missing objective, unsupported nodes, second-order
/// cone constraints ([`IoError::Conic`]; the NL format has no conic segment),
/// or I/O failure.
pub fn write_nl_with<W: Write>(
    model: &Model,
    out: &mut W,
    opts: &WriteOptions,
) -> Result<(), IoError> {
    if model.num_soc_constraints() > 0 {
        return Err(IoError::Conic);
    }
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(|_| IoError::NoObjective)?;

    let arena = model.arena();
    let analysis =
        analyze::Analysis::build(&arena, &vars, &constraints, &objective, opts.nonfinite_strings)?;
    let perm = permute::Permutation::build(&vars, &analysis);
    let stats = header::Stats::build(&vars, &constraints, &analysis, &perm, opts);

    let mut w = Writer::new(out, opts);
    header::write_header(&mut w, model, &stats, opts)?;
    segments::write_segments(
        &mut w,
        &arena,
        &vars,
        &constraints,
        &objective,
        &analysis,
        &perm,
        &stats,
        opts,
    )?;
    Ok(())
}

/// Convenience: render the `.nl` file into a `String`. ASCII output only.
///
/// # Errors
///
/// Returns [`IoError::BinaryToString`] when `opts.format` is
/// [`NlFormat::Binary`] (binary output is not UTF-8, so use [`write_nl_with`]
/// with a byte sink instead), or [`IoError`] on missing objective, unsupported
/// nodes, or I/O failure.
pub fn to_nl_string_with(model: &Model, opts: &WriteOptions) -> Result<String, IoError> {
    if opts.format == NlFormat::Binary {
        return Err(IoError::BinaryToString);
    }
    let mut buf = Vec::new();
    write_nl_with(model, &mut buf, opts)?;
    // ASCII output is always valid UTF-8
    // TODO: See if there is a better approach
    String::from_utf8(buf).map_err(|_| IoError::BinaryToString)
}

/// Write `<stub>.nl`, plus the `<stub>.row` (constraint names) and
/// `<stub>.col` (variable names) sidecars when `opts.aux_files` is set.
/// Variable/constraint names are written in NL (permuted) order, one per line,
/// matching the convention expected by AMPL.
///
/// The sidecars are always written alongside the `.nl` (same stub, `.row` /
/// `.col` extensions).
///
/// # Errors
///
/// Returns [`IoError`] on I/O failure or model-content errors.
pub fn write_nl_files(model: &Model, stub: &Path, opts: &WriteOptions) -> Result<(), IoError> {
    let nl_path = stub.with_extension("nl");
    {
        let mut f = std::fs::File::create(&nl_path)?;
        write_nl_with(model, &mut f, opts)?;
    }
    if opts.aux_files {
        write_aux_files(model, stub, opts.nonfinite_strings)?;
    }
    Ok(())
}

fn write_aux_files(model: &Model, stub: &Path, nonfinite_strings: bool) -> Result<(), IoError> {
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(|_| IoError::NoObjective)?;

    let arena = model.arena();
    let analysis =
        analyze::Analysis::build(&arena, &vars, &constraints, &objective, nonfinite_strings)?;
    let perm = permute::Permutation::build(&vars, &analysis);

    let row_path = stub.with_extension("row");
    {
        let mut f = std::fs::File::create(&row_path)?;
        for &orig in &perm.con_order {
            writeln!(f, "{}", constraints[orig].name)?;
        }
        // AMPL `.row` includes objective names after constraint names.
        // oximo has at most one objective (for now).
        writeln!(f, "{}", model.name)?;
    }
    let col_path = stub.with_extension("col");
    {
        let mut f = std::fs::File::create(&col_path)?;
        for &vid in &perm.var_order {
            writeln!(f, "{}", vars[vid.index()].name)?;
        }
    }
    Ok(())
}
