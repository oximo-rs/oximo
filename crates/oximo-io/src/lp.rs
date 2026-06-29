//! CPLEX LP file format import and export.
//!
//! CPLEX LP is a widely supported text format for linear optimization problems. It can
//! represent LP and MILP models, but not nonlinear or conic problems. The format only
//! supports a single objective and assumes variables are non-negative by default
//! (free variables must be declared explicitly in the `Bounds` section), but is a
//! common lingua franca for exchanging linear models between tools.
//!
//! This module provides functions to write an oximo [`Model`] to CPLEX LP format.
//! The main function is [`write_lp`], which writes to any `std::io::Write`.
//!
//! References:
//! - "CPLEX lp files," lp_solve. <https://lpsolve.sourceforge.net/5.5/CPLEX-format.htm> (accessed May 11, 2026).

use std::io::Write;

use oximo_core::{Domain, Model, ObjectiveSense, Sense};
use oximo_expr::{LinearTerms, extract_linear};

use crate::error::IoError;

/// Write `model` to `out` in LP format.
///
/// Sections emitted:
/// - `\* ... *\` header comment with model name and original sense
/// - `Minimize` / `Maximize` with `obj:` row
/// - `Subject To` with each constraint
/// - `Bounds` (only non-default bounds)
/// - `General` (non-binary integer vars)
/// - `Binaries` (binary vars)
/// - `End`
///
/// LP only represents linear LP/MILP. Nonlinear nodes raise [`IoError::Nonlinear`].
///
/// # Errors
///
/// Returns [`IoError`] on I/O failure, missing objective, or nonlinear constructs.
#[allow(clippy::too_many_lines)]
pub fn write_lp<W: Write>(model: &Model, out: &mut W) -> Result<(), IoError> {
    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(|_| IoError::NoObjective)?;

    let obj_terms = extract_linear(&arena, objective.expr).ok_or(IoError::Nonlinear)?;

    writeln!(out, "\\* OXIMO LP export - model: {} *\\", model.name)?;

    let sense_kw = match objective.sense {
        ObjectiveSense::Minimize => "Minimize",
        ObjectiveSense::Maximize => "Maximize",
    };
    writeln!(out, "{sense_kw}")?;
    write!(out, " obj:")?;
    write_linear(out, &obj_terms, &vars)?;
    writeln!(out)?;
    if obj_terms.constant != 0.0 {
        writeln!(out, "\\* objective constant: {} *\\", obj_terms.constant)?;
    }

    writeln!(out, "Subject To")?;
    for c in constraints.iter() {
        let t = extract_linear(&arena, c.lhs).ok_or(IoError::Nonlinear)?;
        if let Some((sense, rhs)) = c.as_single() {
            let op = match sense {
                Sense::Le => "<=",
                Sense::Ge => ">=",
                Sense::Eq => "=",
            };
            let adjusted_rhs = rhs - t.constant;
            write!(out, " {}:", c.name)?;
            write_linear(out, &t, &vars)?;
            writeln!(out, " {op} {adjusted_rhs}")?;
        } else {
            let lo = c.lower - t.constant;
            let hi = c.upper - t.constant;
            write!(out, " {}_lo:", c.name)?;
            write_linear(out, &t, &vars)?;
            writeln!(out, " >= {lo}")?;
            write!(out, " {}_hi:", c.name)?;
            write_linear(out, &t, &vars)?;
            writeln!(out, " <= {hi}")?;
        }
    }

    let mut wrote_bounds_header = false;
    for v in vars.iter() {
        if matches!(v.domain, Domain::Binary) {
            continue;
        }
        // Semicont/semiint: the gap floor (`threshold`) is the LP lower bound.
        // The `Semi-Continuous` section below marks the gap.
        if let Some(thr) = v.domain.semi_threshold() {
            if !wrote_bounds_header {
                writeln!(out, "Bounds")?;
                wrote_bounds_header = true;
            }
            if v.ub == f64::INFINITY {
                writeln!(out, " {} >= {}", v.name, thr)?;
            } else {
                writeln!(out, " {} <= {} <= {}", thr, v.name, v.ub)?;
            }
            continue;
        }
        if v.lb.is_finite() && (v.lb - v.ub).abs() < f64::EPSILON {
            if !wrote_bounds_header {
                writeln!(out, "Bounds")?;
                wrote_bounds_header = true;
            }
            writeln!(out, " {} <= {} <= {}", v.lb, v.name, v.ub)?;
            continue;
        }
        let lb_default = v.lb == 0.0;
        let ub_default = v.ub == f64::INFINITY;
        if lb_default && ub_default {
            continue;
        }
        if !wrote_bounds_header {
            writeln!(out, "Bounds")?;
            wrote_bounds_header = true;
        }
        if v.lb == f64::NEG_INFINITY && ub_default {
            writeln!(out, " {} free", v.name)?;
        } else if v.lb == f64::NEG_INFINITY {
            writeln!(out, " -inf <= {} <= {}", v.name, v.ub)?;
        } else if ub_default {
            writeln!(out, " {} >= {}", v.name, v.lb)?;
        } else {
            writeln!(out, " {} <= {} <= {}", v.lb, v.name, v.ub)?;
        }
    }

    let general_vars: Vec<&str> = vars
        .iter()
        .filter(|v| matches!(v.domain, Domain::Integer | Domain::SemiInteger { .. }))
        .map(|v| v.name.as_str())
        .collect();
    if !general_vars.is_empty() {
        writeln!(out, "General")?;
        writeln!(out, " {}", general_vars.join(" "))?;
    }

    let binary_vars: Vec<&str> = vars
        .iter()
        .filter(|v| matches!(v.domain, Domain::Binary))
        .map(|v| v.name.as_str())
        .collect();
    if !binary_vars.is_empty() {
        writeln!(out, "Binaries")?;
        writeln!(out, " {}", binary_vars.join(" "))?;
    }

    // Semicontinuous and semi-integer vars. A var that is also in `General`
    // (the SemiInteger filter above) is read back as semi-integer.
    let semi_vars: Vec<&str> = vars
        .iter()
        .filter(|v| v.domain.semi_threshold().is_some())
        .map(|v| v.name.as_str())
        .collect();
    if !semi_vars.is_empty() {
        writeln!(out, "Semi-Continuous")?;
        writeln!(out, " {}", semi_vars.join(" "))?;
    }

    writeln!(out, "End")?;
    Ok(())
}

/// Convenience: render the LP into a `String`.
///
/// # Errors
///
/// Returns [`IoError`] if writing fails.
///
/// # Panics
///
/// Panics if the writer's internal buffer is not valid UTF-8
pub fn to_lp_string(model: &Model) -> Result<String, IoError> {
    let mut buf = Vec::new();
    write_lp(model, &mut buf)?;
    Ok(String::from_utf8(buf).expect("LP writer emits ASCII"))
}

/// Write a linear expression as a sequence of `+/- coeff varname` terms.
/// Skips zero coefficients; coefficient `1` and `-1` are written without the
/// magnitude (LP format permits `+ x` and `- x`).
fn write_linear<W: Write>(
    out: &mut W,
    t: &LinearTerms,
    vars: &[oximo_core::Variable],
) -> std::io::Result<()> {
    let mut first = true;
    for (v, coef) in &t.coeffs {
        if *coef == 0.0 {
            continue;
        }
        let name = vars[v.index()].name.as_str();
        let (sign, mag) = if *coef < 0.0 { ("-", -coef) } else { ("+", *coef) };
        if first {
            if sign == "-" {
                if (mag - 1.0).abs() < f64::EPSILON {
                    write!(out, " - {name}")?;
                } else {
                    write!(out, " -{mag} {name}")?;
                }
            } else if (mag - 1.0).abs() < f64::EPSILON {
                write!(out, " {name}")?;
            } else {
                write!(out, " {mag} {name}")?;
            }
            first = false;
        } else if (mag - 1.0).abs() < f64::EPSILON {
            write!(out, " {sign} {name}")?;
        } else {
            write!(out, " {sign} {mag} {name}")?;
        }
    }
    if first {
        write!(out, " 0")?;
    }
    Ok(())
}
