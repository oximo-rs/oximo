//! Ten-line `.nl` header (always ASCII, even when the body is binary).

use std::io::Write;

use oximo_core::{Constraint, Domain, Model, Sense, Variable};

use super::analyze::Analysis;
use super::options::{NlFormat, WriteOptions};
use super::permute::Permutation;
use super::writer::Writer;
use crate::error::IoError;

#[derive(Debug)]
pub(crate) struct Stats {
    pub(crate) n_var: usize,
    pub(crate) n_con: usize,
    pub(crate) n_obj: usize,
    pub(crate) n_ranges: usize,
    pub(crate) n_eqns: usize,
    pub(crate) nl_con: usize,
    pub(crate) nl_obj: usize,
    pub(crate) nl_vars_in_c: usize,
    pub(crate) nl_vars_in_o: usize,
    pub(crate) nl_vars_in_both: usize,
    pub(crate) n_bin: usize,
    pub(crate) n_int_other: usize,
    pub(crate) nl_int_b: usize,
    pub(crate) nl_int_c: usize,
    pub(crate) nl_int_o: usize,
    pub(crate) nnz_jac: usize,
    pub(crate) nnz_grad: usize,
    pub(crate) max_con_name_len: usize,
    pub(crate) max_var_name_len: usize,
}

impl Stats {
    pub(crate) fn build(
        vars: &[Variable],
        constraints: &[Constraint],
        analysis: &Analysis,
        _perm: &Permutation,
        opts: &WriteOptions,
    ) -> Self {
        let n_eqns = constraints.iter().filter(|c| c.sense == Sense::Eq).count();
        let nl_con = analysis.cons.iter().filter(|r| r.is_nonlinear()).count();
        let nl_obj = usize::from(analysis.obj.is_nonlinear());

        let cv = &analysis.nl_vars_c;
        let ov = &analysis.nl_vars_o;
        let in_both = cv.iter().filter(|v| ov.contains(v)).count();

        // Per Gay Table 3 these buckets are disjoint: nbv/niv count only
        // *linearly* used binary/integer variables; integer variables that
        // appear nonlinearly are counted exclusively in nl_int_{b,c,o}.
        let mut n_bin = 0usize;
        let mut n_int_other = 0usize;
        let mut nl_int_b = 0usize;
        let mut nl_int_c = 0usize;
        let mut nl_int_o = 0usize;
        for v in vars {
            if !v.domain.is_integer() {
                continue;
            }
            let in_c = cv.contains(&v.id);
            let in_o = ov.contains(&v.id);
            if in_c && in_o {
                nl_int_b += 1;
            } else if in_c {
                nl_int_c += 1;
            } else if in_o {
                nl_int_o += 1;
            } else if matches!(v.domain, Domain::Binary) {
                n_bin += 1;
            } else {
                n_int_other += 1;
            }
        }

        let nnz_jac: usize = analysis.cons_vars.iter().map(Vec::len).sum();
        let nnz_grad = analysis.obj_vars.len();

        let (max_con_name_len, max_var_name_len) = if opts.aux_files.is_some() {
            (
                constraints.iter().map(|c| c.name.len()).max().unwrap_or(0),
                vars.iter().map(|v| v.name.len()).max().unwrap_or(0),
            )
        } else {
            (0, 0)
        };

        Self {
            n_var: vars.len(),
            n_con: constraints.len(),
            n_obj: 1,
            n_ranges: 0,
            n_eqns,
            nl_con,
            nl_obj,
            nl_vars_in_c: cv.len(),
            nl_vars_in_o: ov.len(),
            nl_vars_in_both: in_both,
            n_bin,
            n_int_other,
            nl_int_b,
            nl_int_c,
            nl_int_o,
            nnz_jac,
            nnz_grad,
            max_con_name_len,
            max_var_name_len,
        }
    }
}

pub(crate) fn write_header<W: Write>(
    w: &mut Writer<'_, W>,
    model: &Model,
    s: &Stats,
    opts: &WriteOptions,
) -> Result<(), IoError> {
    let leader = match opts.format {
        NlFormat::Ascii => "g3 1 1 0",
        NlFormat::Binary => "b3 1 1 0",
    };
    let n_funcs = opts.functions.len();

    write_header_line(w, leader, &format!("problem {}", model.name))?;
    write_header_line(
        w,
        &format!(" {} {} {} {} {}", s.n_var, s.n_con, s.n_obj, s.n_ranges, s.n_eqns),
        "vars, constraints, objectives, ranges, eqns",
    )?;
    write_header_line(
        w,
        &format!(" {} {}", s.nl_con, s.nl_obj),
        "nonlinear constraints, objectives",
    )?;
    write_header_line(w, " 0 0", "network constraints: nonlinear, linear")?;
    write_header_line(
        w,
        &format!(" {} {} {}", s.nl_vars_in_c, s.nl_vars_in_o, s.nl_vars_in_both),
        "nonlinear vars in constraints, objectives, both",
    )?;
    write_header_line(
        w,
        &format!(" 0 {n_funcs} 0 1"),
        "linear network variables; functions; arith, flags",
    )?;
    write_header_line(
        w,
        &format!(" {} {} {} {} {}", s.n_bin, s.n_int_other, s.nl_int_b, s.nl_int_c, s.nl_int_o),
        "discrete variables: binary, integer, nonlinear (b,c,o)",
    )?;
    write_header_line(
        w,
        &format!(" {} {}", s.nnz_jac, s.nnz_grad),
        "nonzeros in Jacobian, gradients",
    )?;
    write_header_line(
        w,
        &format!(" {} {}", s.max_con_name_len, s.max_var_name_len),
        "max name lengths: constraints, variables",
    )?;
    write_header_line(w, " 0 0 0 0 0", "common exprs: b,c,o,c1,o1")?;
    Ok(())
}

fn write_header_line<W: Write>(
    w: &mut Writer<'_, W>,
    body: &str,
    comment: &str,
) -> Result<(), IoError> {
    w.write_text(body)?;
    w.header_eol(comment)?;
    Ok(())
}
