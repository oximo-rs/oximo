//! MPS file format import and export.
//!
//! MPS is a widely supported text format for linear optimization problems. It can
//! represent LP and MILP models, but not nonlinear or conic problems. The format is
//! somewhat idiosyncratic and has some limitations (e.g. only one objective row
//! named `OBJ`, no native support for free variables), but is a common lingua franca for
//! exchanging linear models between tools.
//!
//! This module provides functions to write an oximo [`Model`] to MPS format.
//! The main function is [`write_mps`], which writes to any `std::io::Write`.
//!
//! References:
//! - "MPS file format," lp_solve. <https://lpsolve.sourceforge.net/5.5/mps-format.htm> (accessed May 09, 2026).

use std::io::Write;

use oximo_core::{Model, ObjectiveSense, Sense};
use oximo_expr::{LinearTerms, VarId, extract_linear};
use rustc_hash::FxHashMap;

use crate::error::IoError;

/// Write `model` to `out` in fixed-format MPS.
///
/// MPS only represents linear LP / MILP. Nonlinear expressions in the
/// objective or constraints raise [`IoError::Nonlinear`]. The objective row
/// is named `OBJ`. Constraint rows take their oximo names.
///
/// # Errors
///
/// Returns [`IoError`] if there is an error writing the MPS data or if the model contains unsupported features.
///
#[allow(clippy::too_many_lines)]
pub fn write_mps<W: Write>(model: &Model, out: &mut W) -> Result<(), IoError> {
    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(|_| IoError::NoObjective)?;

    let obj_terms = extract_linear(&arena, objective.expr).ok_or(IoError::Nonlinear)?;

    // Pre-compute constraint linear terms once, reused for COLUMNS and RHS.
    let con_terms: Vec<LinearTerms> = constraints
        .iter()
        .map(|c| extract_linear(&arena, c.lhs).ok_or(IoError::Nonlinear))
        .collect::<Result<_, _>>()?;

    // Build column index: VarId to [(row_name, coef)] in row order (OBJ first, then constraints).
    let mut col_index: FxHashMap<VarId, Vec<(&str, f64)>> = FxHashMap::default();
    for (v, c) in &obj_terms.coeffs {
        col_index.entry(*v).or_default().push(("OBJ", *c));
    }
    for (constr, terms) in constraints.iter().zip(con_terms.iter()) {
        for (v, coef) in &terms.coeffs {
            col_index.entry(*v).or_default().push((constr.name.as_str(), *coef));
        }
    }

    // Per the MPS spec, max problems are negated, since most solvers assume
    // minimization. Tag the sense in a comment so re-importers can recover it.
    writeln!(out, "* OXIMO MPS export")?;
    writeln!(
        out,
        "* sense: {}",
        match objective.sense {
            ObjectiveSense::Minimize => "minimize",
            ObjectiveSense::Maximize => "maximize",
        }
    )?;
    writeln!(out, "NAME          {}", model.name)?;

    writeln!(out, "ROWS")?;
    writeln!(out, " N  OBJ")?;
    for c in constraints.iter() {
        let tag = match c.sense {
            Sense::Le => 'L',
            Sense::Ge => 'G',
            Sense::Eq => 'E',
        };
        writeln!(out, " {tag}  {}", c.name)?;
    }

    writeln!(out, "COLUMNS")?;
    let mut int_open = false;
    for v in vars.iter() {
        let needs_marker = v.domain.is_integer();
        if needs_marker && !int_open {
            writeln!(out, "    MARKER                 'MARKER'                 'INTORG'")?;
            int_open = true;
        } else if !needs_marker && int_open {
            writeln!(out, "    MARKER                 'MARKER'                 'INTEND'")?;
            int_open = false;
        }
        if let Some(entries) = col_index.get(&v.id) {
            for (row_name, coef) in entries {
                writeln!(out, "    {:<10}{:<10}{}", v.name, row_name, coef)?;
            }
        }
    }
    if int_open {
        writeln!(out, "    MARKER                 'MARKER'                 'INTEND'")?;
    }

    writeln!(out, "RHS")?;
    let obj_constant = obj_terms.constant;
    if obj_constant != 0.0 {
        writeln!(out, "    RHS       OBJ       {}", -obj_constant)?;
    }
    for (c, t) in constraints.iter().zip(con_terms.iter()) {
        let adjusted = c.rhs - t.constant;
        if adjusted != 0.0 {
            writeln!(out, "    RHS       {:<10}{}", c.name, adjusted)?;
        }
    }

    writeln!(out, "BOUNDS")?;
    for v in vars.iter() {
        let lb = v.lb;
        let ub = v.ub;
        let infinite_lo = lb == f64::NEG_INFINITY;
        let infinite_hi = ub == f64::INFINITY;
        match (infinite_lo, infinite_hi) {
            (true, true) => writeln!(out, " FR BND       {}", v.name)?,
            (true, false) => {
                writeln!(out, " MI BND       {}", v.name)?;
                writeln!(out, " UP BND       {:<10}{}", v.name, ub)?;
            }
            (false, true) => {
                if lb != 0.0 {
                    writeln!(out, " LO BND       {:<10}{}", v.name, lb)?;
                }
            }
            (false, false) => {
                if lb != 0.0 {
                    writeln!(out, " LO BND       {:<10}{}", v.name, lb)?;
                }
                writeln!(out, " UP BND       {:<10}{}", v.name, ub)?;
            }
        }
    }

    writeln!(out, "ENDATA")?;
    Ok(())
}

/// Convenience: render the MPS into a `String`.
///
/// # Errors
///
/// Returns [`IoError`] if writing the MPS data fails.
///
/// # Panics
///
/// Panics if the MPS writer internal buffer does not produce valid UTF-8 data.
pub fn to_mps_string(model: &Model) -> Result<String, IoError> {
    let mut buf = Vec::new();
    write_mps(model, &mut buf)?;
    Ok(String::from_utf8(buf).expect("MPS writer emits ASCII"))
}
