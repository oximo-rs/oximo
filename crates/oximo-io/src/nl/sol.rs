//! Minimal `.sol` (AMPL solution) file parser.
//!
//! Format (Gay / ASL conventions, as written by the AMPL `ASL` library):
//! - one or more message lines
//! - blank line
//! - line `Options`
//! - integer `nopt`, then `nopt` option ints, one per line
//! - a four-integer dimension preamble: `n_con`, `n_dual`, `n_var`, `n_primal`.
//!   The counts that govern the body are `n_dual` and `n_primal`, how many
//!   dual / primal values actually follow. A solver may report fewer than the
//!   model's totals: a MIP / nonconvex solve returns no duals, so `n_dual == 0`
//!   even when `n_con > 0`.
//! - `n_dual` dual values, one per line
//! - `n_primal` primal values, one per line
//! - optional trailing `objno 0 <solve_result_num>` line
//!
//! Only used by the integration-test harness, not exposed in the public API.

use std::str::FromStr;

#[derive(Debug, Default)]
pub(crate) struct SolFile {
    pub(crate) primals: Vec<f64>,
    pub(crate) duals: Vec<f64>,
    pub(crate) status: Option<i32>,
}

#[derive(Debug)]
pub(crate) enum SolParseError {
    Truncated,
    BadNumber(String),
}

impl std::fmt::Display for SolParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => f.write_str("sol: truncated"),
            Self::BadNumber(s) => write!(f, "sol: bad number {s:?}"),
        }
    }
}

impl std::error::Error for SolParseError {}

pub(crate) fn parse_sol(text: &str) -> Result<SolFile, SolParseError> {
    let mut lines = text.lines();

    // Skip message lines until a blank line.
    for line in lines.by_ref() {
        if line.trim().is_empty() {
            break;
        }
    }

    // Find the Options block, then skip the option array.
    loop {
        let line = lines.next().ok_or(SolParseError::Truncated)?;
        if line.trim() == "Options" {
            break;
        }
    }
    let n_opts = read_count(&mut lines)?;
    for _ in 0..n_opts {
        let _ = parse_line::<i64>(&mut lines)?;
    }

    // Dimension preamble: n_con, n_dual, n_var, n_primal. Only the second and
    // fourth, the number of values that actually follow, drive the reads.
    let _n_con = read_count(&mut lines)?;
    let n_dual = read_count(&mut lines)?;
    let _n_var = read_count(&mut lines)?;
    let n_primal = read_count(&mut lines)?;

    let mut duals = Vec::with_capacity(n_dual);
    for _ in 0..n_dual {
        duals.push(parse_line::<f64>(&mut lines)?);
    }
    let mut primals = Vec::with_capacity(n_primal);
    for _ in 0..n_primal {
        primals.push(parse_line::<f64>(&mut lines)?);
    }

    // Optional `objno 0 <status>` trailer.
    let mut status = None;
    for line in lines {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("objno") {
            let mut parts = rest.split_whitespace();
            let _ = parts.next();
            if let Some(s) = parts.next() {
                status = s.parse().ok();
            }
            break;
        }
    }

    Ok(SolFile { primals, duals, status })
}

fn read_count(lines: &mut std::str::Lines<'_>) -> Result<usize, SolParseError> {
    let n = parse_line::<i64>(lines)?;
    Ok(usize::try_from(n.max(0)).unwrap_or(0))
}

fn parse_line<T: FromStr>(lines: &mut std::str::Lines<'_>) -> Result<T, SolParseError> {
    let line = lines.next().ok_or(SolParseError::Truncated)?;
    line.trim().parse::<T>().map_err(|_| SolParseError::BadNumber(line.trim().to_string()))
}
