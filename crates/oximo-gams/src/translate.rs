use std::fmt::Write as FmtWrite;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::{fs, io};

static SOLVE_ID: AtomicU64 = AtomicU64::new(0);

use oximo_core::{Domain, Model, ModelKind, ObjectiveSense, Sense, VarId};
use oximo_expr::{LinearTerms, extract_linear};
use oximo_solver::{SolverError, SolverResult, SolverStatus};
use rustc_hash::FxHashMap;

use crate::GamsOptions;
use crate::options::write_options;

/// Write `model` to a temporary GAMS `.gms` file, execute the GAMS solver, and
/// return the parsed [`SolverResult`].
///
/// `exec` is an optional override for the GAMS executable path; `None` uses
/// `"gams"` resolved from `PATH`.
///
/// # Errors
///
/// Returns [`SolverError`] on unsupported model kind, nonlinear expressions,
/// a missing GAMS executable, GAMS compilation errors, or I/O failures.
///
/// # Panics
///
/// Panics if variable indices overflow `u32`.
#[allow(clippy::too_many_lines)]
pub fn solve(
    model: &Model,
    opts: &GamsOptions,
    exec: Option<&str>,
) -> Result<SolverResult, SolverError> {
    let kind = model.kind();
    if !matches!(kind, ModelKind::LP | ModelKind::MILP) {
        return Err(SolverError::UnsupportedKind(kind));
    }

    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(SolverError::Core)?;

    let obj_terms = extract_linear(&arena, objective.expr).ok_or(SolverError::Nonlinear)?;

    let mut con_terms = Vec::with_capacity(constraints.len());
    for c in constraints.iter() {
        con_terms.push(extract_linear(&arena, c.lhs).ok_or(SolverError::Nonlinear)?);
    }

    let solve_type = if matches!(kind, ModelKind::MILP) { "MIP" } else { "LP" };
    let sense_kw = match objective.sense {
        ObjectiveSense::Minimize => "minimizing",
        ObjectiveSense::Maximize => "maximizing",
    };

    // Pre-compute solver opt file content so we know whether to inject optfile=1.
    let solver_opt: Option<(String, String)> = opts.solver.as_ref().and_then(|cfg| {
        let mut buf = String::new();
        if cfg.write_opt_file(&mut buf) {
            let fname = format!("{}.opt", cfg.gams_name().to_ascii_lowercase());
            Some((fname, buf))
        } else {
            None
        }
    });

    // - Build the .gms file
    let mut gms = String::with_capacity(4096);

    writeln!(gms, "$title oximo_model").unwrap();
    writeln!(gms, "$offSymList").unwrap();
    writeln!(gms, "$offSymXRef").unwrap();
    writeln!(gms, "option solprint = off;").unwrap();
    writeln!(gms, "option limrow = 0;").unwrap();
    writeln!(gms, "option limcol = 0;").unwrap();
    writeln!(gms).unwrap();

    // Variable declarations, split by domain
    let (mut cont_vars, mut bin_vars, mut int_vars) = (Vec::new(), Vec::new(), Vec::new());
    for v in vars.iter() {
        match v.domain {
            Domain::Binary => bin_vars.push(v),
            Domain::Integer | Domain::SemiInteger { .. } => int_vars.push(v),
            _ => cont_vars.push(v),
        }
    }

    // Continuous + objective variable
    write!(gms, "Variables\n    v_obj").unwrap();
    for v in &cont_vars {
        write!(gms, ", v{}", v.id.index()).unwrap();
    }
    writeln!(gms, ";").unwrap();

    if !bin_vars.is_empty() {
        write!(gms, "Binary Variables").unwrap();
        for (k, v) in bin_vars.iter().enumerate() {
            if k == 0 {
                write!(gms, "\n    v{}", v.id.index()).unwrap();
            } else {
                write!(gms, ", v{}", v.id.index()).unwrap();
            }
        }
        writeln!(gms, ";").unwrap();
    }

    if !int_vars.is_empty() {
        write!(gms, "Integer Variables").unwrap();
        for (k, v) in int_vars.iter().enumerate() {
            if k == 0 {
                write!(gms, "\n    v{}", v.id.index()).unwrap();
            } else {
                write!(gms, ", v{}", v.id.index()).unwrap();
            }
        }
        writeln!(gms, ";").unwrap();
    }
    writeln!(gms).unwrap();

    // Bounds
    for v in vars.iter() {
        let i = v.id.index();
        if matches!(v.domain, Domain::Binary) {
            // Default binary bounds are [0, 1], only emit when overridden.
            if (v.lb - v.ub).abs() < f64::EPSILON {
                writeln!(gms, "v{i}.fx = {};", fmt(v.lb)).unwrap();
            } else {
                if v.lb.abs() > f64::EPSILON {
                    writeln!(gms, "v{i}.lo = {};", fmt(v.lb)).unwrap();
                }
                if (v.ub - 1.0).abs() > f64::EPSILON {
                    writeln!(gms, "v{i}.up = {};", fmt(v.ub)).unwrap();
                }
            }
            continue;
        }
        // Lower bound
        if v.lb == f64::NEG_INFINITY {
            writeln!(gms, "v{i}.lo = -Inf;").unwrap();
        } else if v.lb.is_finite() {
            writeln!(gms, "v{i}.lo = {};", fmt(v.lb)).unwrap();
        }
        // Upper bound (+Inf is the GAMS default, only write when finite)
        if v.ub.is_finite() {
            writeln!(gms, "v{i}.up = {};", fmt(v.ub)).unwrap();
        }
    }
    writeln!(gms).unwrap();

    // Equations declaration
    write!(gms, "Equations\n    eq_obj").unwrap();
    for i in 0..constraints.len() {
        write!(gms, ", eq_c{i}").unwrap();
    }
    writeln!(gms, ";").unwrap();
    writeln!(gms).unwrap();

    // Objective equation: v_obj =e= <full expr including constant>
    write!(gms, "eq_obj..  v_obj =e=").unwrap();
    write_expr(&mut gms, &obj_terms, true);
    writeln!(gms, ";").unwrap();

    // Constraint equations: variable terms only, constant folded into RHS
    for (ci, (c, t)) in constraints.iter().zip(con_terms.iter()).enumerate() {
        let adjusted_rhs = c.rhs - t.constant;
        let sense_str = match c.sense {
            Sense::Le => "=l=",
            Sense::Ge => "=g=",
            Sense::Eq => "=e=",
        };
        write!(gms, "eq_c{ci}..").unwrap();
        write_expr(&mut gms, t, false);
        writeln!(gms, " {sense_str} {};", fmt(adjusted_rhs)).unwrap();
    }
    writeln!(gms).unwrap();

    // Options (time limit, MIP gap, sub-solver, etc.)
    write_options(&mut gms, opts, solve_type);

    // Model and solve statements
    writeln!(gms, "Model oximo_m / all /;").unwrap();
    if solver_opt.is_some() {
        writeln!(gms, "oximo_m.optfile = 1;").unwrap();
    }
    writeln!(gms, "Solve oximo_m using {solve_type} {sense_kw} v_obj;").unwrap();
    writeln!(gms).unwrap();

    // - Temp directory
    // Combine timestamp with a per-process atomic counter so concurrent
    // invocations (e.g. parallel threads) never share a directory.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let id = SOLVE_ID.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir().join(format!("oximo_gams_{ts}_{id}"));
    fs::create_dir_all(&tmp_dir)
        .map_err(|e| SolverError::Backend(format!("cannot create temp dir: {e}")))?;

    let sol_path = tmp_dir.join("solution.txt");
    writeln!(gms, "File oximo_sol / 'solution.txt' /;").unwrap();
    writeln!(gms, "Put oximo_sol;").unwrap();
    writeln!(gms, "Put 'STATUS=' oximo_m.modelstat:0:0 /;").unwrap();
    writeln!(gms, "Put 'SOLVESTAT=' oximo_m.solvestat:0:0 /;").unwrap();
    writeln!(gms, "Put 'OBJVAL=' v_obj.l:0:15 /;").unwrap();
    for i in 0..vars.len() {
        writeln!(gms, "Put '{i}=' v{i}.l:0:15 /;").unwrap();
    }
    writeln!(gms, "Putclose oximo_sol;").unwrap();

    drop(arena);
    drop(vars);
    drop(constraints);

    // - Write .gms file
    let gms_path = tmp_dir.join("model.gms");
    fs::write(&gms_path, &gms)
        .map_err(|e| SolverError::Backend(format!("cannot write .gms file: {e}")))?;

    // - Write solver opt file (if any)
    if let Some((ref fname, ref content)) = solver_opt {
        fs::write(tmp_dir.join(fname), content)
            .map_err(|e| SolverError::Backend(format!("cannot write solver opt file: {e}")))?;
    }

    // - Execute GAMS
    let gams_exec =
        opts.gams_path.as_deref().and_then(std::path::Path::to_str).or(exec).unwrap_or("gams");

    let verbose = opts.universal.verbose.unwrap_or(false);

    let started = Instant::now();
    let mut cmd = std::process::Command::new(gams_exec);
    cmd.arg(&gms_path);
    if !verbose {
        cmd.arg("lo=0");
    }
    cmd.current_dir(&tmp_dir);

    let output = cmd.output().map_err(|e| {
        let _ = fs::remove_dir_all(&tmp_dir);
        if e.kind() == io::ErrorKind::NotFound {
            SolverError::Backend(format!(
                "GAMS executable '{gams_exec}' not found. \
                Install GAMS and ensure it is on PATH, or set the 'gams_path' option."
            ))
        } else {
            SolverError::Backend(format!("failed to launch GAMS: {e}"))
        }
    })?;
    let elapsed = started.elapsed();

    let raw_log = if verbose || !output.status.success() {
        let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.stderr.is_empty() {
            log.push('\n');
            log.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        Some(log)
    } else {
        None
    };

    // - Parse solution file
    // Check the solution file before the exit code: GAMS may return a
    // non-zero exit on infeasible/unbounded models while still writing a
    // valid modelstat to the PUT file.
    let result = if sol_path.exists() {
        let content = fs::read_to_string(&sol_path)
            .map_err(|e| SolverError::Backend(format!("cannot read solution file: {e}")))?;
        parseoximo_solution(&content, elapsed, raw_log)
    } else {
        // No solution file. GAMS must have failed before the Solve statement
        // (compilation error, license error, etc.).  Fall back to the listing.
        let listing = fs::read_to_string(tmp_dir.join("model.lst")).unwrap_or_default();
        let _ = fs::remove_dir_all(&tmp_dir);
        let detail = if output.status.success() {
            format!(
                "GAMS did not produce a solution file. \
                Check the .gms listing for compilation errors.\n{listing}"
            )
        } else {
            format!("GAMS exited with code {:?}.\n{listing}", output.status.code())
        };
        return Err(SolverError::Backend(detail));
    };

    let _ = fs::remove_dir_all(&tmp_dir);
    Ok(result)
}

/// Parse the PUT-generated solution file.
fn parseoximo_solution(
    content: &str,
    elapsed: std::time::Duration,
    raw_log: Option<String>,
) -> SolverResult {
    let mut modelstat: Option<i32> = None;
    let mut solvestat: Option<i32> = None;
    let mut obj_val: Option<f64> = None;
    let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();

    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("STATUS=") {
            modelstat = parse_gams_int(rest);
        } else if let Some(rest) = line.strip_prefix("SOLVESTAT=") {
            solvestat = parse_gams_int(rest);
        } else if let Some(rest) = line.strip_prefix("OBJVAL=") {
            obj_val = parse_gams_float(rest);
        } else if let Some(eq) = line.find('=') {
            let key = line[..eq].trim();
            if let Ok(idx) = key.parse::<u32>() {
                if let Some(val) = parse_gams_float(line[eq + 1..].trim()) {
                    primal.insert(VarId(idx), val);
                }
            }
        }
    }

    let status = map_status(modelstat.unwrap_or(13), solvestat.unwrap_or(0));

    SolverResult {
        objective: if status.has_solution() { obj_val } else { None },
        primal: if status.has_solution() { primal } else { FxHashMap::default() },
        dual: FxHashMap::default(),
        reduced_costs: FxHashMap::default(),
        status,
        solve_time: elapsed,
        iterations: 0,
        raw_log,
    }
}

/// Map GAMS model-status to `SolverStatus`.
///
/// Full modelstat table (codes 1-19):
///  1 = Optimal,
///  2 = Locally Optimal,
///  3 = Unbounded,
///  4 = Infeasible,
///  5 = Locally Infeasible,
///  6 = Intermediate Infeasible,
///  7 = Feasible Solution,
///  8 = Integer Solution,
///  9 = Intermediate Non-integer,
///  10 = Integer Infeasible,
///  11 = Lic Problem No Solution,
///  12 = Error Unknown,
///  13 = Error No Solution,
///  14 = No Solution Returned,
///  15 = Solved Unique,
///  16 = Solved,
///  17 = Solved Singular,
///  18 = Unbounded-No Solution,
///  19 = Infeasible-No Solution.
///
/// References:
/// "GAMS Output - Model Status," GAMS Development Corporation.
/// <https://www.gams.com/latest/docs/UG_GAMSOutput.html#UG_GAMSOutput_ModelStatus> (accessed May 14, 2026).
fn map_status(modelstat: i32, solvestat: i32) -> SolverStatus {
    // TODO: Could refine this mapping if we modify the `SolverStatus` enum.
    match modelstat {
        1 | 15 | 16 => SolverStatus::Optimal,
        2 | 7 | 9 | 17 => SolverStatus::Feasible,
        3 | 18 => SolverStatus::Unbounded,
        4 | 5 | 6 | 10 | 19 => SolverStatus::Infeasible,
        // MIP integer solution: proven optimal when solvestat == 1 (normal completion)
        8 => {
            if solvestat == 1 {
                SolverStatus::Optimal
            } else {
                SolverStatus::Feasible
            }
        }
        11 => SolverStatus::Other("gams_license_error".into()),
        _ => SolverStatus::Other(format!("gams_modelstat_{modelstat}")),
    }
}

// - Helpers

/// Append the linear expression `t` to `gms`.
/// When `include_constant` is true, the constant term is included; otherwise
/// only variable terms are emitted (used for constraints where the constant is
/// folded into the RHS).
fn write_expr(gms: &mut String, t: &LinearTerms, include_constant: bool) {
    let mut first = true;
    for (v, coef) in &t.coeffs {
        if *coef == 0.0 {
            continue;
        }
        let idx = v.index();
        if first {
            write!(gms, " {}*v{idx}", fmt(*coef)).unwrap();
            first = false;
        } else if *coef < 0.0 {
            write!(gms, " - {}*v{idx}", fmt(-coef)).unwrap();
        } else {
            write!(gms, " + {}*v{idx}", fmt(*coef)).unwrap();
        }
    }
    if include_constant && t.constant != 0.0 {
        if first {
            write!(gms, " {}", fmt(t.constant)).unwrap();
            first = false;
        } else if t.constant < 0.0 {
            write!(gms, " - {}", fmt(-t.constant)).unwrap();
        } else {
            write!(gms, " + {}", fmt(t.constant)).unwrap();
        }
    }
    if first {
        write!(gms, " 0").unwrap();
    }
}

/// Format an `f64` for use in a GAMS file.
fn fmt(v: f64) -> String {
    if v == f64::INFINITY {
        return "+Inf".into();
    }
    if v == f64::NEG_INFINITY {
        return "-Inf".into();
    }
    format!("{v}")
}

/// Parse a GAMS-formatted integer (may be written as `"1"` or `"1.000"`).
fn parse_gams_int(s: &str) -> Option<i32> {
    let trimmed = s.trim();
    if let Ok(n) = trimmed.parse::<i32>() {
        return Some(n);
    }
    // Fall back through f64 for GAMS formats like "1.000".
    #[allow(clippy::cast_possible_truncation)]
    trimmed.parse::<f64>().ok().map(|f| f.round() as i32)
}

/// Parse a GAMS-formatted float, tolerating GAMS special tokens.
fn parse_gams_float(s: &str) -> Option<f64> {
    match s.trim() {
        "INF" | "+INF" | "Inf" | "+Inf" => Some(f64::INFINITY),
        "-INF" | "-Inf" => Some(f64::NEG_INFINITY),
        "NA" | "UNDF" | "EPS" => Some(0.0),
        other => other.parse().ok(),
    }
}
