//! Emit the body segments of the `.nl` file in canonical ASL order (D. M. Gay,
//! Table 13): F, S, V, C, L, O, d, x, r, b, k, J, G. F / S / V / d are emitted
//! only when the caller provides data via `WriteOptions`, oximo's `Model`
//! carries no source for them. L (logical constraints) is always empty in
//! oximo for now.

use std::io::Write;

use oximo_core::{Constraint, Objective, ObjectiveSense, Sense, Variable};
use oximo_expr::{ExprArena, VarId};
use rustc_hash::FxHashMap;

use super::analyze::{Analysis, Row};
use super::emit_expr::emit_residual;
use super::header::Stats;
use super::options::{Complementarity, SuffixFlavour, WriteOptions};
use super::permute::Permutation;
use super::writer::Writer;
use crate::error::IoError;

#[expect(clippy::too_many_arguments)]
pub(crate) fn write_segments<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    vars: &[Variable],
    constraints: &[Constraint],
    objective: &Objective,
    analysis: &Analysis,
    perm: &Permutation,
    _stats: &Stats,
    opts: &WriteOptions,
) -> Result<(), IoError> {
    write_f_segments(w, &opts.functions)?;
    write_s_segments(w, &opts.suffixes)?;
    write_v_segments(w, &opts.defined_vars)?;
    write_c_segments(w, arena, perm, analysis)?;
    write_o_segment(w, arena, perm, objective, &analysis.obj)?;
    write_d_segment(w, &opts.dual_init)?;
    write_x_segment(w, vars, perm)?;
    write_r_segment(w, constraints, perm, analysis, &opts.complementarity)?;
    write_b_segment(w, vars, perm)?;
    write_k_segment(w, vars, perm, analysis)?;
    write_j_segments(w, perm, analysis)?;
    write_g_segment(w, perm, analysis)?;
    Ok(())
}

fn write_f_segments<W: Write>(
    w: &mut Writer<'_, W>,
    functions: &[super::options::ImportedFunction],
) -> Result<(), IoError> {
    for (i, f) in functions.iter().enumerate() {
        w.seg_header(
            b'F',
            &[
                i64::try_from(i).expect("F index"),
                i64::from(f.allow_string_args),
                i64::from(f.n_args),
            ],
            Some(&f.name),
        )?;
    }
    Ok(())
}

fn write_s_segments<W: Write>(
    w: &mut Writer<'_, W>,
    suffixes: &[super::options::SuffixData],
) -> Result<(), IoError> {
    for s in suffixes {
        let kind_word: i64 =
            (s.kind as i64) | (if matches!(s.flavour, SuffixFlavour::Real) { 4 } else { 0 });
        w.seg_header(
            b'S',
            &[kind_word, i64::try_from(s.values.len()).expect("S n")],
            Some(&s.name),
        )?;
        for (off, val) in &s.values {
            w.int(i64::from(*off))?;
            w.sep()?;
            w.dbl(*val)?;
            w.eor()?;
        }
    }
    Ok(())
}

fn write_v_segments<W: Write>(
    w: &mut Writer<'_, W>,
    defined_vars: &[super::options::DefinedVar],
) -> Result<(), IoError> {
    for d in defined_vars {
        w.seg_header(
            b'V',
            &[
                i64::from(d.nl_index),
                i64::try_from(d.linear.len()).expect("V nlin"),
                i64::from(d.appearance),
            ],
            None,
        )?;
        for (col, coef) in &d.linear {
            w.int(i64::from(*col))?;
            w.sep()?;
            w.dbl(*coef)?;
            w.eor()?;
        }
        if d.nonlinear_polish.is_empty() {
            w.num(0.0)?;
        } else {
            w.write_text(&d.nonlinear_polish)?;
            if !d.nonlinear_polish.ends_with('\n') {
                w.eor()?;
            }
        }
    }
    Ok(())
}

fn write_c_segments<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    perm: &Permutation,
    analysis: &Analysis,
) -> Result<(), IoError> {
    for (nl_idx, &orig_idx) in perm.con_order.iter().enumerate() {
        w.seg_header(b'C', &[i64::try_from(nl_idx).expect("Cidx")], None)?;
        let residual = &analysis.cons[orig_idx].residual;
        if residual.is_empty() {
            w.num(0.0)?;
        } else {
            emit_residual(w, arena, &perm.var_index, residual)?;
        }
    }
    Ok(())
}

fn write_o_segment<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    perm: &Permutation,
    objective: &Objective,
    obj: &Row,
) -> Result<(), IoError> {
    let sense_flag: i64 = match objective.sense {
        ObjectiveSense::Minimize => 0,
        ObjectiveSense::Maximize => 1,
    };
    w.seg_header(b'O', &[0, sense_flag], None)?;
    if obj.residual.is_empty() {
        w.num(obj.linear.constant)?;
    } else {
        emit_residual(w, arena, &perm.var_index, &obj.residual)?;
    }
    Ok(())
}

fn write_x_segment<W: Write>(
    w: &mut Writer<'_, W>,
    vars: &[Variable],
    perm: &Permutation,
) -> Result<(), IoError> {
    let count = vars.iter().filter(|v| v.initial.is_some()).count();
    if count == 0 {
        return Ok(());
    }
    w.seg_header(b'x', &[i64::try_from(count).expect("x n")], None)?;
    for &vid in &perm.var_order {
        let v = &vars[vid.index()];
        if let Some(val) = v.initial {
            let nl_idx = perm.var_index[&vid];
            w.int(i64::from(nl_idx))?;
            w.sep()?;
            w.dbl(val)?;
            w.eor()?;
        }
    }
    Ok(())
}

fn write_d_segment<W: Write>(w: &mut Writer<'_, W>, duals: &[(u32, f64)]) -> Result<(), IoError> {
    if duals.is_empty() {
        return Ok(());
    }
    w.seg_header(b'd', &[i64::try_from(duals.len()).expect("d n")], None)?;
    for (i, val) in duals {
        w.int(i64::from(*i))?;
        w.sep()?;
        w.dbl(*val)?;
        w.eor()?;
    }
    Ok(())
}

/// `r` segment (D. M. Gay, Table 17).
fn write_r_segment<W: Write>(
    w: &mut Writer<'_, W>,
    constraints: &[Constraint],
    perm: &Permutation,
    analysis: &Analysis,
    complementarity: &[(usize, Complementarity)],
) -> Result<(), IoError> {
    w.seg_header(b'r', &[], None)?;
    let comp_map: FxHashMap<usize, Complementarity> = complementarity.iter().copied().collect();
    for &orig_idx in &perm.con_order {
        if let Some(comp) = comp_map.get(&orig_idx) {
            w.int(5)?;
            w.sep()?;
            w.int(i64::from(comp.k))?;
            w.sep()?;
            w.int(i64::from(comp.i))?;
            w.eor()?;
            continue;
        }
        let c = &constraints[orig_idx];
        let lin_const = analysis.cons[orig_idx].linear.constant;
        // `r`-segment line types (D. M. Gay, Table 17): 0 = `lo <= body <= hi`
        // (range, two values), 1 = upper, 2 = lower, 3 = free, 4 = equality.
        match c.as_single() {
            Some((Sense::Le, rhs)) => {
                w.int(1)?;
                w.sep()?;
                w.dbl(rhs - lin_const)?;
                w.eor()?;
            }
            Some((Sense::Ge, rhs)) => {
                w.int(2)?;
                w.sep()?;
                w.dbl(rhs - lin_const)?;
                w.eor()?;
            }
            Some((Sense::Eq, rhs)) => {
                w.int(4)?;
                w.sep()?;
                w.dbl(rhs - lin_const)?;
                w.eor()?;
            }
            None if c.is_range() => {
                w.int(0)?;
                w.sep()?;
                w.dbl(c.lower - lin_const)?;
                w.sep()?;
                w.dbl(c.upper - lin_const)?;
                w.eor()?;
            }
            None => {
                w.int(3)?;
                w.eor()?;
            }
        }
    }
    Ok(())
}

/// `b` segment (D. M. Gay, Table 17)
fn write_b_segment<W: Write>(
    w: &mut Writer<'_, W>,
    vars: &[Variable],
    perm: &Permutation,
) -> Result<(), IoError> {
    w.seg_header(b'b', &[], None)?;
    for &vid in &perm.var_order {
        let v = &vars[vid.index()];
        let lb = v.lb;
        let ub = v.ub;
        let lo_inf = lb == f64::NEG_INFINITY;
        let hi_inf = ub == f64::INFINITY;
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() == 0.0 {
            w.int(4)?;
            w.sep()?;
            w.dbl(lb)?;
            w.eor()?;
        } else {
            match (lo_inf, hi_inf) {
                (true, true) => {
                    w.int(3)?;
                    w.eor()?;
                }
                (true, false) => {
                    w.int(1)?;
                    w.sep()?;
                    w.dbl(ub)?;
                    w.eor()?;
                }
                (false, true) => {
                    w.int(2)?;
                    w.sep()?;
                    w.dbl(lb)?;
                    w.eor()?;
                }
                (false, false) => {
                    w.int(0)?;
                    w.sep()?;
                    w.dbl(lb)?;
                    w.sep()?;
                    w.dbl(ub)?;
                    w.eor()?;
                }
            }
        }
    }
    Ok(())
}

/// Per-row var lists in NL-index order, with associated J coefficient.
fn row_entries(
    row: &Row,
    row_vars: &[VarId],
    var_index: &FxHashMap<VarId, u32>,
) -> Vec<(u32, f64)> {
    let lin_map: FxHashMap<VarId, f64> = row.linear.coeffs.iter().copied().collect();
    let mut entries: Vec<(u32, f64)> = row_vars
        .iter()
        .map(|v| {
            let coef = lin_map.get(v).copied().unwrap_or(0.0);
            (var_index[v], coef)
        })
        .collect();
    entries.sort_by_key(|(col, _)| *col);
    entries
}

fn write_k_segment<W: Write>(
    w: &mut Writer<'_, W>,
    vars: &[Variable],
    perm: &Permutation,
    analysis: &Analysis,
) -> Result<(), IoError> {
    let n_var = vars.len();
    if n_var == 0 {
        return Ok(());
    }
    let mut col_counts = vec![0usize; n_var];
    for row_vars in &analysis.cons_vars {
        for v in row_vars {
            let nl_col = perm.var_index[v] as usize;
            col_counts[nl_col] += 1;
        }
    }
    w.seg_header(b'k', &[i64::try_from(n_var - 1).expect("k n")], None)?;
    let mut acc = 0usize;
    for count in col_counts.iter().take(n_var - 1) {
        acc += count;
        w.int(i64::try_from(acc).expect("k acc"))?;
        w.eor()?;
    }
    Ok(())
}

fn write_j_segments<W: Write>(
    w: &mut Writer<'_, W>,
    perm: &Permutation,
    analysis: &Analysis,
) -> Result<(), IoError> {
    for (nl_idx, &orig_idx) in perm.con_order.iter().enumerate() {
        let row_vars = &analysis.cons_vars[orig_idx];
        if row_vars.is_empty() {
            continue;
        }
        let entries = row_entries(&analysis.cons[orig_idx], row_vars, &perm.var_index);
        w.seg_header(
            b'J',
            &[i64::try_from(nl_idx).expect("J idx"), i64::try_from(entries.len()).expect("J nz")],
            None,
        )?;
        for (col, coef) in entries {
            w.int(i64::from(col))?;
            w.sep()?;
            w.dbl(coef)?;
            w.eor()?;
        }
    }
    Ok(())
}

fn write_g_segment<W: Write>(
    w: &mut Writer<'_, W>,
    perm: &Permutation,
    analysis: &Analysis,
) -> Result<(), IoError> {
    if analysis.obj_vars.is_empty() {
        return Ok(());
    }
    let entries = row_entries(&analysis.obj, &analysis.obj_vars, &perm.var_index);
    w.seg_header(b'G', &[0, i64::try_from(entries.len()).expect("G nz")], None)?;
    for (col, coef) in entries {
        w.int(i64::from(col))?;
        w.sep()?;
        w.dbl(coef)?;
        w.eor()?;
    }
    Ok(())
}
