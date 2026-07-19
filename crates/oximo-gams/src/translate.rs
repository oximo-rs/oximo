use std::borrow::Cow;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::{fs, io};

static SOLVE_ID: AtomicU64 = AtomicU64::new(0);

use oximo_core::{
    Constraint, ConstraintId, Domain, Model, ModelKind, Objective, ObjectiveSense, Sense,
    SocConstraint, SocConstraintId, VarId, Variable,
};
use oximo_expr::{ExprArena, ExprId, ExprNode, LinearTerms, extract_linear};
use oximo_solver::{PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
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
#[expect(clippy::too_many_lines)]
pub fn solve(
    model: &Model,
    opts: &GamsOptions,
    exec: Option<&str>,
) -> Result<SolverResult, SolverError> {
    model.ensure_objective_declared().map_err(SolverError::Core)?;
    let kind = model.kind();
    validate_solver(opts, kind)?;
    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let socs = model.soc_constraints();
    let objective = model.objective();
    let sense = objective.as_ref().map_or(ObjectiveSense::Minimize, |o| o.sense);

    let sense_kw = match sense {
        ObjectiveSense::Minimize => "minimizing",
        ObjectiveSense::Maximize => "maximizing",
    };

    let mut gms = String::with_capacity(4096);
    let solver_opt = build_model_section(
        &mut gms,
        kind,
        &arena,
        &vars,
        &constraints,
        &socs,
        objective.as_ref(),
        sense_kw,
        opts,
    );

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
    writeln!(gms, "Put 'ITER=' oximo_m.iterusd:0:0 /;").unwrap();
    writeln!(gms, "Put 'OBJVAL=' v_obj.l:0:15 /;").unwrap();
    for i in 0..vars.len() {
        writeln!(gms, "Put '{i}=' v{i}.l:0:15 /;").unwrap();
    }

    // Marginals are emitted for every model kind and GAMS decides
    // which ones to populate based on the solve type and solver capabilities.
    for i in 0..vars.len() {
        writeln!(gms, "Put 'R{i}=' v{i}.m:0:15 /;").unwrap();
    }
    // A two-sided range is emitted as two equations (`_lo`/`_hi`), summing their
    // marginals into one `D{i}` keeps the dual keyed by the single `ConstraintId`.
    for (i, c) in constraints.iter().enumerate() {
        if c.is_range() {
            writeln!(gms, "Put 'D{i}=' (eq_c{i}_lo.m + eq_c{i}_hi.m):0:15 /;").unwrap();
        } else {
            writeln!(gms, "Put 'D{i}=' eq_c{i}.m:0:15 /;").unwrap();
        }
    }
    write_soc_marginal_puts(&mut gms, socs.len());
    writeln!(gms, "Putclose oximo_sol;").unwrap();

    // The affine bound side of each explicit SOC constraint, kept to rescale
    // the squared-form `eq_soc{i}` marginal back to the norm form after the
    // solve (see `parseoximo_solution`).
    let soc_bounds: Vec<LinearTerms> = socs
        .iter()
        .map(|s| extract_linear(&arena, s.bound).expect("SOC bound is validated affine"))
        .collect();

    drop(arena);
    drop(vars);
    drop(constraints);
    drop(socs);

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

    // When verbose, inherit stdio so that GAMS writes directly to the terminal in
    // real time. When silent, capture output so errors can be surfaced later.
    let launch_err = |e: io::Error| {
        let _ = fs::remove_dir_all(&tmp_dir);
        if e.kind() == io::ErrorKind::NotFound {
            SolverError::Backend(format!(
                "GAMS executable '{gams_exec}' not found. \
                Install GAMS and ensure it is on PATH, or set the 'gams_path' option."
            ))
        } else {
            SolverError::Backend(format!("failed to launch GAMS: {e}"))
        }
    };

    let (exit_ok, raw_log) = if verbose {
        let status =
            cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit()).status().map_err(launch_err)?;
        (status.success(), None)
    } else {
        let out = cmd.output().map_err(launch_err)?;
        let log = if out.status.success() {
            None
        } else {
            let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
            if !out.stderr.is_empty() {
                s.push('\n');
                s.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            Some(s)
        };
        (out.status.success(), log)
    };
    let elapsed = started.elapsed();

    // - Parse solution file
    // Check the solution file before the exit code: GAMS may return a
    // non-zero exit on infeasible/unbounded models while still writing a
    // valid modelstat to the PUT file.
    let mut result = if sol_path.exists() {
        let content = fs::read_to_string(&sol_path)
            .map_err(|e| SolverError::Backend(format!("cannot read solution file: {e}")))?;
        let mut result = parseoximo_solution(&content, &soc_bounds, elapsed, raw_log);
        // If a sub-solver wrote a solution pool (e.g. CPLEX `solnpool`), surface
        // every pooled point. The model itself emits no GDX, so any pool GDX in
        // the run directory came from the user's option file.
        if result.has_solution() {
            let pool = read_solution_pool(&tmp_dir, gams_exec, sense);
            if !pool.is_empty() {
                result.solutions = pool;
            }
        }
        result
    } else {
        // No solution file. GAMS must have failed before the Solve statement
        // (compilation error, license error, etc.).  Fall back to the listing.
        let listing = fs::read_to_string(tmp_dir.join("model.lst")).unwrap_or_default();
        let _ = fs::remove_dir_all(&tmp_dir);
        let report = summarize_listing(&listing);
        let detail = if exit_ok {
            format!("GAMS did not produce a solution file.\n{report}")
        } else {
            format!("GAMS exited with a non-zero exit code.\n{report}")
        };
        return Err(SolverError::Backend(detail));
    };

    let _ = fs::remove_dir_all(&tmp_dir);
    result.solver_name = Some(gams_solver_label(opts));
    Ok(result)
}

/// Backend label for the result: `GAMS/<sub-solver>` when a sub-solver is
/// configured, otherwise just `GAMS`. When none is set, GAMS selects its own
/// default solver for the model type, whose name we do not resolve here.
fn gams_solver_label(opts: &GamsOptions) -> Cow<'static, str> {
    match &opts.solver {
        Some(cfg) => Cow::Owned(format!("{}/{}", crate::NAME, cfg.gams_name())),
        None => Cow::Borrowed(crate::NAME),
    }
}

/// Extract the compilation-error section from a GAMS `.lst` listing.
///
/// A raw listing echoes every source line of the model with the diagnostics
/// buried among them on lines prefixed by `****`.
/// This keeps those `****` marker lines word for word, each preceded by the source
/// line it points at, and drops the unrelated echo. The markers are kept as GAMS
/// prints them so the `$<code>` carets stay aligned under the offending column,
/// and the listing's `\r\n` endings are normalized to `\n` so the result renders
/// across multiple lines instead of collapsing onto one.
/// If no `****` markers are found (e.g. a license or runtime failure)
/// the original listing is returned so no detail is lost.
fn summarize_listing(listing: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut last_src: Option<&str> = None;
    let mut src_emitted = false;

    for raw in listing.lines() {
        let line = raw.trim_end();
        if let Some(rest) = line.strip_prefix("****") {
            if rest.contains("ERROR(S)") || rest.contains("WARNING(S)") {
                out.push(line);
                continue;
            }
            if !src_emitted {
                if let Some(src) = last_src {
                    out.push(src);
                }
                src_emitted = true;
            }
            out.push(line);
        } else if !line.trim().is_empty() {
            last_src = Some(line);
            src_emitted = false;
        }
    }

    if out.is_empty() {
        return listing.trim_end().to_string();
    }
    out.join("\n")
}

/// Emit `Z{i}=` PUT lines reading each lowered SOC row's marginal, mirroring
/// the `D{i}` constraint-dual lines.
fn write_soc_marginal_puts(gms: &mut String, n: usize) {
    for i in 0..n {
        writeln!(gms, "Put 'Z{i}=' eq_soc{i}.m:0:15 /;").unwrap();
    }
}

/// Parse the PUT-generated solution file.
///
/// `soc_bounds` holds the affine bound side of each explicit SOC constraint
/// (in `SocConstraintId` order), the squared-row marginal parsed is rescaled.
fn parseoximo_solution(
    content: &str,
    soc_bounds: &[LinearTerms],
    elapsed: std::time::Duration,
    raw_log: Option<String>,
) -> SolverResult {
    let mut modelstat: Option<i32> = None;
    let mut solvestat: Option<i32> = None;
    let mut obj_val: Option<f64> = None;
    let mut iterations: u64 = 0;
    let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut dual: FxHashMap<ConstraintId, f64> = FxHashMap::default();
    let mut soc_marginals: FxHashMap<u32, f64> = FxHashMap::default();
    let mut reduced_costs: FxHashMap<VarId, f64> = FxHashMap::default();

    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("STATUS=") {
            modelstat = parse_gams_int(rest);
        } else if let Some(rest) = line.strip_prefix("SOLVESTAT=") {
            solvestat = parse_gams_int(rest);
        } else if let Some(rest) = line.strip_prefix("OBJVAL=") {
            obj_val = parse_gams_float(rest);
        } else if let Some(rest) = line.strip_prefix("ITER=") {
            if let Some(n) = parse_gams_u64(rest) {
                iterations = n;
            }
        } else if let Some(rest) = line.strip_prefix('R') {
            if let Some(eq) = rest.find('=') {
                if let Ok(idx) = rest[..eq].parse::<u32>() {
                    if let Some(val) = parse_gams_float(rest[eq + 1..].trim()) {
                        reduced_costs.insert(VarId(idx), val);
                    }
                }
            }
        } else if let Some(rest) = line.strip_prefix('D') {
            if let Some(eq) = rest.find('=') {
                if let Ok(idx) = rest[..eq].parse::<u32>() {
                    if let Some(val) = parse_gams_float(rest[eq + 1..].trim()) {
                        dual.insert(ConstraintId(idx), val);
                    }
                }
            }
        } else if let Some(rest) = line.strip_prefix('Z') {
            if let Some(eq) = rest.find('=') {
                if let Ok(idx) = rest[..eq].parse::<u32>() {
                    if let Some(val) = parse_gams_float(rest[eq + 1..].trim()) {
                        soc_marginals.insert(idx, val);
                    }
                }
            }
        } else if let Some(eq) = line.find('=') {
            let key = line[..eq].trim();
            if let Ok(idx) = key.parse::<u32>() {
                if let Some(val) = parse_gams_float(line[eq + 1..].trim()) {
                    primal.insert(VarId(idx), val);
                }
            }
        }
    }

    let modelstat = modelstat.unwrap_or(13);
    let termination = map_status(modelstat, solvestat.unwrap_or(0));
    let has_sol = modelstat_has_solution(modelstat);

    // Rescale each lowered SOC row's marginal to the norm-form bound
    // multiplier using the bound's value at the primal point.
    let mut soc_dual: FxHashMap<SocConstraintId, f64> = FxHashMap::default();
    if has_sol {
        for (i, bound) in soc_bounds.iter().enumerate() {
            let idx = u32::try_from(i).expect("SOC count overflow");
            if let Some(m) = soc_marginals.get(&idx) {
                let b_val = bound.constant
                    + bound
                        .coeffs
                        .iter()
                        .map(|&(v, c)| c * primal.get(&v).copied().unwrap_or(0.0))
                        .sum::<f64>();
                soc_dual.insert(SocConstraintId(idx), 2.0 * b_val * m.abs());
            }
        }
    }

    // The PUT solution file holds only the incumbent. `solve` augments this with
    // a sub-solver solution pool (if one was written) read from the run dir's GDX.
    let solutions =
        if has_sol { vec![SolutionPoint { primal, objective: obj_val }] } else { Vec::new() };
    let primal_status = PrimalStatus::infer(&termination, !solutions.is_empty());
    SolverResult {
        solutions,
        dual: if has_sol { dual } else { FxHashMap::default() },
        soc_dual,
        reduced_costs: if has_sol { reduced_costs } else { FxHashMap::default() },
        termination,
        primal_status,
        best_bound: None,
        gap: None,
        solve_time: elapsed,
        iterations,
        raw_log,
        solver_name: Some(crate::NAME.into()),
    }
}

/// Read a sub-solver solution pool from the GAMS run directory.
///
/// A sub-solver `solnpool` option (CPLEX/Gurobi/Xpress) writes each alternative
/// solution to its own GDX file plus an index GDX. oximo's generated model emits
/// no GDX of its own, so every `*.gdx` in `tmp_dir` belongs to the pool. Each is
/// dumped with `gdxdump` and parsed for the model's `v{i}` variable levels.
///
/// Returns the points best-first by `sense`, empty when no pool was written.
fn read_solution_pool(
    tmp_dir: &Path,
    gams_exec: &str,
    sense: ObjectiveSense,
) -> Vec<SolutionPoint> {
    let gdxdump = gdxdump_path(gams_exec);
    let Ok(entries) = fs::read_dir(tmp_dir) else {
        return Vec::new();
    };
    let mut members: Vec<SolutionPoint> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("gdx") {
            continue;
        }
        let Ok(out) = std::process::Command::new(&gdxdump).arg(&path).output() else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let dump = String::from_utf8_lossy(&out.stdout);
        if let Some(pt) = parse_pool_member(&dump) {
            members.push(pt);
        }
    }
    members.sort_by(|a, b| {
        let ord = match sense {
            ObjectiveSense::Maximize => b.objective.partial_cmp(&a.objective),
            ObjectiveSense::Minimize => a.objective.partial_cmp(&b.objective),
        };
        ord.unwrap_or(std::cmp::Ordering::Equal)
    });
    members
}

/// `gdxdump` lives beside the `gams` executable.
/// Fall back to `PATH` when only a bare command name is known.
fn gdxdump_path(gams_exec: &str) -> PathBuf {
    let exe = if cfg!(windows) { "gdxdump.exe" } else { "gdxdump" };
    match Path::new(gams_exec).parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(exe),
        _ => PathBuf::from("gdxdump"),
    }
}

/// Parse one pool member's `gdxdump` text into a [`SolutionPoint`].
///
/// Each model variable is dumped as `<type> Variable v{i} /L <level> /;` (or an
/// empty record `/ /` for a default-zero level). The objective variable `v_obj`
/// carries the point's objective. Returns `None` for a GDX with no `v{i}`
/// symbols (e.g. the pool index file).
fn parse_pool_member(dump: &str) -> Option<SolutionPoint> {
    let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut objective: Option<f64> = None;
    for line in dump.lines() {
        let Some(pos) = line.find("Variable ") else {
            continue;
        };
        let rest = &line[pos + "Variable ".len()..];
        let name_end = rest.find(|c: char| c.is_whitespace() || c == '/').unwrap_or(rest.len());
        let name = &rest[..name_end];
        if name == "v_obj" {
            objective = Some(parse_gdx_level(line));
        } else if let Some(idx) = name.strip_prefix('v').and_then(|d| d.parse::<u32>().ok()) {
            primal.insert(VarId(idx), parse_gdx_level(line));
        }
    }
    if primal.is_empty() { None } else { Some(SolutionPoint { primal, objective }) }
}

/// Extract the `L` (level) field from a `gdxdump` variable record line, e.g.
/// `binary Variable v0 /L 1 /;` -> `1.0`. An empty record (`/ /`) or a missing
/// `L` field means the default level of `0.0`.
fn parse_gdx_level(line: &str) -> f64 {
    let (Some(start), Some(end)) = (line.find('/'), line.rfind('/')) else {
        return 0.0;
    };
    if end <= start {
        return 0.0;
    }
    let mut tokens = line[start + 1..end].split_whitespace();
    while let Some(tok) = tokens.next() {
        if tok == "L" {
            return tokens.next().map_or(0.0, |v| v.trim_end_matches(',').parse().unwrap_or(0.0));
        }
    }
    0.0
}

/// Map a GAMS solve to a [`TerminationStatus`].
///
/// GAMS reports two orthogonal codes, which line up with our two-axis result:
/// `solvestat` (the *solver termination condition* — why the run stopped) drives
/// the [`TerminationStatus`], while `modelstat` (the *model status* — what kind
/// of solution exists) drives feasibility/primal status (see
/// [`modelstat_has_solution`]) and disambiguates the outcome on normal
/// completion.
///
/// `solvestat` table:
///  1 = Normal, 2 = Iteration, 3 = Resource (time), 4 = Terminated by solver,
///  5 = Evaluation error, 6 = Capability (model beyond solver), 7 = License,
///  8 = User (interrupt), 9 = Setup error, 10 = Solver error,
///  11 = Internal error, 12 = Skipped, 13 = System error.
///
/// On a normal completion the `solvestat` carries no outcome, so we defer to
/// [`modelstat_termination`].
///
/// Reference: GAMS `SolveStat` codes,
/// <https://www.gams.com/latest/docs/apis/python/classgams_1_1control_1_1workspace_1_1SolveStat.html>
fn map_status(modelstat: i32, solvestat: i32) -> TerminationStatus {
    match solvestat {
        1 => modelstat_termination(modelstat),
        2 => TerminationStatus::IterationLimit,
        3 => TerminationStatus::TimeLimit,
        // "Terminated by solver" / "User request": abnormal early stops that may
        // still carry an incumbent.
        4 | 8 => TerminationStatus::Interrupted,
        // Evaluation / solver / internal / system errors.
        5 | 10 | 11 | 13 => TerminationStatus::NumericError,
        6 => TerminationStatus::Other("gams_solver_capability".into()),
        7 => TerminationStatus::Other("gams_license_error".into()),
        9 => TerminationStatus::Other("gams_setup_error".into()),
        12 => TerminationStatus::NotSolved,
        n => TerminationStatus::Other(format!("gams_solvestat_{n}")),
    }
}

/// Termination from the GAMS `modelstat` when the solver completed normally
/// (`solvestat == 1`), so the model status is the authoritative outcome.
///
/// Full modelstat table (codes 1-19):
///  1 = Optimal, 2 = Locally Optimal, 3 = Unbounded, 4 = Infeasible,
///  5 = Locally Infeasible, 6 = Intermediate Infeasible, 7 = Feasible Solution,
///  8 = Integer Solution, 9 = Intermediate Non-integer, 10 = Integer Infeasible,
///  11 = Lic Problem No Solution, 12 = Error Unknown, 13 = Error No Solution,
///  14 = No Solution Returned, 15 = Solved Unique, 16 = Solved,
///  17 = Solved Singular, 18 = Unbounded-No Solution, 19 = Infeasible-No Solution.
///
/// Reference: GAMS `ModelStat` codes,
/// <https://www.gams.com/latest/docs/apis/python/classgams_1_1control_1_1workspace_1_1ModelStat.html>
fn modelstat_termination(modelstat: i32) -> TerminationStatus {
    match modelstat {
        1 | 8 | 15 | 16 | 17 => TerminationStatus::Optimal,
        2 | 7 | 9 => TerminationStatus::LocallyOptimal,
        3 | 18 => TerminationStatus::Unbounded,
        4 | 5 | 6 | 10 | 19 => TerminationStatus::Infeasible,
        11 => TerminationStatus::Other("gams_license_error".into()),
        n => TerminationStatus::Other(format!("gams_modelstat_{n}")),
    }
}

/// GAMS model-status codes that carry a usable primal point.
fn modelstat_has_solution(modelstat: i32) -> bool {
    matches!(modelstat, 1 | 2 | 7 | 8 | 9 | 15 | 16 | 17)
}

// - Helpers

/// Write the formulation portion of the `.gms` file: title, variables, bounds,
/// equations, options, model, and solve statement. Returns the solve type
/// (`"LP"` / `"MIP"` / `"NLP"` / `"MINLP"` / `"QCP"` / `"MIQCP"`) and any
/// solver-options file pair `(filename, content)` the caller should also
/// persist alongside the `.gms`.
#[expect(clippy::too_many_arguments)]
fn build_model_section(
    gms: &mut String,
    kind: ModelKind,
    arena: &ExprArena,
    vars: &[Variable],
    constraints: &[Constraint],
    socs: &[SocConstraint],
    objective: Option<&Objective>,
    sense_kw: &str,
    opts: &GamsOptions,
) -> Option<(String, String)> {
    let solve_type = gams_solve_type(kind);
    let solver_opt = build_solver_opt(opts);

    write_preamble(gms);
    write_var_declarations(gms, vars);
    write_bounds_and_initials(gms, vars);
    write_equations(gms, arena, constraints, socs, objective);
    write_options(gms, opts, solve_type);
    write_model_and_solve(gms, solve_type, sense_kw, solver_opt.is_some());

    solver_opt
}

pub(crate) fn gams_solve_type(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::LP => "LP",
        ModelKind::MILP => "MIP",
        ModelKind::QP | ModelKind::QCP | ModelKind::SOCP => "QCP",
        ModelKind::MIQP | ModelKind::MIQCP | ModelKind::MISOCP => "MIQCP",
        ModelKind::NLP => "NLP",
        ModelKind::MINLP => "MINLP",
    }
}

/// Reject an explicitly selected sub-solver that cannot handle `kind` before
/// invoking GAMS, so the caller gets a clear error naming the solver and model
/// type instead of a downstream GAMS compilation failure.
fn validate_solver(opts: &GamsOptions, kind: ModelKind) -> Result<(), SolverError> {
    if let Some(cfg) = &opts.solver {
        if !cfg.supports(kind) {
            let solve_type = gams_solve_type(kind);
            return Err(SolverError::Backend(format!(
                "GAMS solver {} does not support {solve_type} models (model kind {kind:?}); \
                select a solver that supports {solve_type}",
                cfg.gams_name()
            )));
        }
    }
    Ok(())
}

fn build_solver_opt(opts: &GamsOptions) -> Option<(String, String)> {
    opts.solver.as_ref().and_then(|cfg| {
        let mut buf = String::new();
        cfg.write_opt_file(&mut buf)
            .then(|| (format!("{}.opt", cfg.gams_name().to_ascii_lowercase()), buf))
    })
}

fn write_preamble(gms: &mut String) {
    writeln!(gms, "$title oximo_model").unwrap();
    writeln!(gms, "$offSymList").unwrap();
    writeln!(gms, "$offSymXRef").unwrap();
    writeln!(gms, "option solprint = off;").unwrap();
    writeln!(gms, "option limrow = 0;").unwrap();
    writeln!(gms, "option limcol = 0;").unwrap();
    writeln!(gms).unwrap();
}

/// Emit `Variables`, `Binary Variables`, `Integer Variables`,
/// `Semicont Variables`, `Semiint Variables` sections.
fn write_var_declarations(gms: &mut String, vars: &[Variable]) {
    let (mut cont, mut bin, mut int, mut semicont, mut semiint) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for v in vars {
        match v.domain {
            Domain::Binary => bin.push(v),
            Domain::Integer => int.push(v),
            Domain::SemiContinuous { .. } => semicont.push(v),
            Domain::SemiInteger { .. } => semiint.push(v),
            Domain::Real => cont.push(v),
        }
    }

    write!(gms, "Variables\n    v_obj").unwrap();
    for v in &cont {
        write!(gms, ", v{}", v.id.index()).unwrap();
    }
    writeln!(gms, ";").unwrap();

    write_typed_var_section(gms, "Binary Variables", &bin);
    write_typed_var_section(gms, "Integer Variables", &int);
    write_typed_var_section(gms, "Semicont Variables", &semicont);
    write_typed_var_section(gms, "Semiint Variables", &semiint);
    writeln!(gms).unwrap();
}

fn write_typed_var_section(gms: &mut String, header: &str, vars: &[&Variable]) {
    if vars.is_empty() {
        return;
    }
    write!(gms, "{header}\n    ").unwrap();
    for (k, v) in vars.iter().enumerate() {
        if k > 0 {
            write!(gms, ", ").unwrap();
        }
        write!(gms, "v{}", v.id.index()).unwrap();
    }
    writeln!(gms, ";").unwrap();
}

fn write_bounds_and_initials(gms: &mut String, vars: &[Variable]) {
    for v in vars {
        write_var_bounds(gms, v);
    }
    for v in vars {
        if let Some(val) = v.initial {
            writeln!(gms, "v{}.l = {};", v.id.index(), fmt(val)).unwrap();
        }
    }
    writeln!(gms).unwrap();
}

fn write_var_bounds(gms: &mut String, v: &Variable) {
    let i = v.id.index();
    if matches!(v.domain, Domain::Binary) {
        // Default binary bounds are [0, 1], only emit when overridden or fixed.
        if (v.lb - v.ub).abs() < f64::EPSILON {
            writeln!(gms, "v{i}.fx = {};", fmt(v.lb)).unwrap();
            return;
        }
        if v.lb.abs() > f64::EPSILON {
            writeln!(gms, "v{i}.lo = {};", fmt(v.lb)).unwrap();
        }
        if (v.ub - 1.0).abs() > f64::EPSILON {
            writeln!(gms, "v{i}.up = {};", fmt(v.ub)).unwrap();
        }
        return;
    }
    // Semicont/semiint variables: GAMS reads `.lo` as the gap floor (the value
    // is 0 or in `[.lo, .up]`), so emit the threshold there rather than `lb`.
    if let Some(thr) = v.domain.semi_threshold() {
        writeln!(gms, "v{i}.lo = {};", fmt(thr)).unwrap();
        if v.ub.is_finite() {
            writeln!(gms, "v{i}.up = {};", fmt(v.ub)).unwrap();
        }
        return;
    }
    if v.lb == f64::NEG_INFINITY {
        writeln!(gms, "v{i}.lo = -Inf;").unwrap();
    } else if v.lb.is_finite() {
        writeln!(gms, "v{i}.lo = {};", fmt(v.lb)).unwrap();
    }
    if v.ub.is_finite() {
        writeln!(gms, "v{i}.up = {};", fmt(v.ub)).unwrap();
    }
}

fn write_equations(
    gms: &mut String,
    arena: &ExprArena,
    constraints: &[Constraint],
    socs: &[SocConstraint],
    objective: Option<&Objective>,
) {
    write!(gms, "Equations\n    eq_obj").unwrap();
    for (i, c) in constraints.iter().enumerate() {
        if c.as_single().is_some() {
            write!(gms, ", eq_c{i}").unwrap();
        } else {
            write!(gms, ", eq_c{i}_lo, eq_c{i}_hi").unwrap();
        }
    }
    for i in 0..socs.len() {
        write!(gms, ", eq_soc{i}, eq_soc{i}_sign").unwrap();
    }
    writeln!(gms, ";").unwrap();
    writeln!(gms).unwrap();

    match objective {
        None => writeln!(gms, "eq_obj..  v_obj =e= 0;").unwrap(),
        Some(obj) => {
            let obj_form = ExprForm::from(arena, obj.expr);
            write!(gms, "eq_obj..  v_obj =e=").unwrap();
            write_form(gms, arena, &obj_form, true);
            writeln!(gms, ";").unwrap();
        }
    }

    for (ci, c) in constraints.iter().enumerate() {
        if let Some((sense, rhs)) = c.as_single() {
            let sense_str = match sense {
                Sense::Le => "=l=",
                Sense::Ge => "=g=",
                Sense::Eq => "=e=",
            };
            write!(gms, "eq_c{ci}..").unwrap();
            match ExprForm::from(arena, c.lhs) {
                ExprForm::Linear(t) => {
                    let adjusted_rhs = rhs - t.constant;
                    write_linear(gms, &t, false);
                    writeln!(gms, " {sense_str} {};", fmt(adjusted_rhs)).unwrap();
                }
                ExprForm::Nonlinear(id) => {
                    write_gams_expr(gms, arena, id, true);
                    writeln!(gms, " {sense_str} {};", fmt(rhs)).unwrap();
                }
            }
        } else {
            match ExprForm::from(arena, c.lhs) {
                ExprForm::Linear(t) => {
                    let lo = c.lower - t.constant;
                    let hi = c.upper - t.constant;
                    write!(gms, "eq_c{ci}_lo..").unwrap();
                    write_linear(gms, &t, false);
                    writeln!(gms, " =g= {};", fmt(lo)).unwrap();
                    write!(gms, "eq_c{ci}_hi..").unwrap();
                    write_linear(gms, &t, false);
                    writeln!(gms, " =l= {};", fmt(hi)).unwrap();
                }
                ExprForm::Nonlinear(id) => {
                    write!(gms, "eq_c{ci}_lo..").unwrap();
                    write_gams_expr(gms, arena, id, true);
                    writeln!(gms, " =g= {};", fmt(c.lower)).unwrap();
                    write!(gms, "eq_c{ci}_hi..").unwrap();
                    write_gams_expr(gms, arena, id, true);
                    writeln!(gms, " =l= {};", fmt(c.upper)).unwrap();
                }
            }
        }
    }
    write_soc_equations(gms, arena, socs);
    writeln!(gms).unwrap();
}

/// Emit each explicit SOC constraint `||terms||_2 <= bound` as the quadratic
/// row `sqr(term_1) + ... =l= sqr(bound)` plus the sign row `bound =g= 0`
/// (squaring loses the sign of the bound side).
fn write_soc_equations(gms: &mut String, arena: &ExprArena, socs: &[SocConstraint]) {
    for (i, s) in socs.iter().enumerate() {
        write!(gms, "eq_soc{i}.. ").unwrap();
        for (k, &term) in s.terms.iter().enumerate() {
            if k > 0 {
                write!(gms, " +").unwrap();
            }
            let t = extract_linear(arena, term).expect("SOC members are validated affine");
            write!(gms, " sqr(").unwrap();
            write_linear(gms, &t, true);
            write!(gms, " )").unwrap();
        }
        let b = extract_linear(arena, s.bound).expect("SOC bound is validated affine");
        write!(gms, " =l= sqr(").unwrap();
        write_linear(gms, &b, true);
        writeln!(gms, " );").unwrap();

        write!(gms, "eq_soc{i}_sign..").unwrap();
        write_linear(gms, &b, true);
        writeln!(gms, " =g= 0;").unwrap();
    }
}

fn write_model_and_solve(gms: &mut String, solve_type: &str, sense_kw: &str, has_opt: bool) {
    writeln!(gms, "Model oximo_m / all /;").unwrap();
    if has_opt {
        writeln!(gms, "oximo_m.optfile = 1;").unwrap();
    }
    writeln!(gms, "Solve oximo_m using {solve_type} {sense_kw} v_obj;").unwrap();
    writeln!(gms).unwrap();
}

/// Captured form of an expression for GAMS emission.
enum ExprForm {
    Linear(LinearTerms),
    Nonlinear(ExprId),
}

impl ExprForm {
    fn from(arena: &ExprArena, id: ExprId) -> Self {
        match extract_linear(arena, id) {
            Some(t) => ExprForm::Linear(t),
            None => ExprForm::Nonlinear(id),
        }
    }
}

/// Append a captured expression form to `gms`.
fn write_form(gms: &mut String, arena: &ExprArena, form: &ExprForm, include_constant: bool) {
    match form {
        ExprForm::Linear(t) => write_linear(gms, t, include_constant),
        ExprForm::Nonlinear(id) => write_gams_expr(gms, arena, *id, true),
    }
}

/// Append the linear expression `t` to `gms`.
/// When `include_constant` is true, the constant term is included; otherwise
/// only variable terms are emitted (used for constraints where the constant is
/// folded into the RHS).
fn write_linear(gms: &mut String, t: &LinearTerms, include_constant: bool) {
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

/// Recursive infix printer for a GAMS-compatible expression.
fn write_gams_expr(gms: &mut String, arena: &ExprArena, id: ExprId, leading_space: bool) {
    if leading_space {
        write!(gms, " ").unwrap();
    }
    match arena.get(id) {
        ExprNode::Const(c) => write!(gms, "{}", fmt(*c)).unwrap(),
        ExprNode::Var(v) => write!(gms, "v{}", v.index()).unwrap(),
        ExprNode::Param(p) => write!(gms, "{}", fmt(arena.param_value(*p))).unwrap(),
        ExprNode::Linear { coeffs, constant } => {
            let t = LinearTerms { coeffs: coeffs.clone(), constant: *constant };
            write!(gms, "(").unwrap();
            write_linear(gms, &t, true);
            write!(gms, " )").unwrap();
        }
        ExprNode::Neg(inner) => {
            write!(gms, "(-").unwrap();
            write_gams_expr(gms, arena, *inner, true);
            write!(gms, ")").unwrap();
        }
        ExprNode::Add(children) => {
            write!(gms, "(").unwrap();
            for (i, c) in children.iter().enumerate() {
                if i > 0 {
                    write!(gms, " +").unwrap();
                }
                write_gams_expr(gms, arena, *c, true);
            }
            write!(gms, ")").unwrap();
        }
        ExprNode::Mul(children) => {
            write!(gms, "(").unwrap();
            for (i, c) in children.iter().enumerate() {
                if i > 0 {
                    write!(gms, " *").unwrap();
                }
                write_gams_expr(gms, arena, *c, true);
            }
            write!(gms, ")").unwrap();
        }
        ExprNode::Pow(base, exp) => {
            // GAMS's `**` lowers to `rPower(x, r)`, which rejects negative
            // bases. For small integer constant exponents emit `power(x, n)`
            // (accepts any real base), otherwise fall back to `**`.
            //
            // The 1e9 cap keeps the cast safe and rejects nonsense huge exponents
            // that would still satisfy the integer check after f64 rounding.
            if let ExprNode::Const(c) = arena.get(*exp) {
                if (c - c.round()).abs() < f64::EPSILON && c.abs() <= 1e9 {
                    write!(gms, "power(").unwrap();
                    write_gams_expr(gms, arena, *base, false);
                    write!(gms, ", {:.0})", c.round()).unwrap();
                    return;
                }
            }
            write!(gms, "(").unwrap();
            write_gams_expr(gms, arena, *base, false);
            write!(gms, " **").unwrap();
            write_gams_expr(gms, arena, *exp, true);
            write!(gms, ")").unwrap();
        }
        ExprNode::Div(num, den) => {
            write!(gms, "(").unwrap();
            write_gams_expr(gms, arena, *num, false);
            write!(gms, " /").unwrap();
            write_gams_expr(gms, arena, *den, true);
            write!(gms, ")").unwrap();
        }
        ExprNode::Sin(a) => {
            write!(gms, "sin(").unwrap();
            write_gams_expr(gms, arena, *a, false);
            write!(gms, ")").unwrap();
        }
        ExprNode::Cos(a) => {
            write!(gms, "cos(").unwrap();
            write_gams_expr(gms, arena, *a, false);
            write!(gms, ")").unwrap();
        }
        ExprNode::Exp(a) => {
            write!(gms, "exp(").unwrap();
            write_gams_expr(gms, arena, *a, false);
            write!(gms, ")").unwrap();
        }
        ExprNode::Log(a) => {
            write!(gms, "log(").unwrap();
            write_gams_expr(gms, arena, *a, false);
            write!(gms, ")").unwrap();
        }
        ExprNode::Abs(a) => {
            write!(gms, "abs(").unwrap();
            write_gams_expr(gms, arena, *a, false);
            write!(gms, ")").unwrap();
        }
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
    // GAMS writes modelstat/solvestat with the `:0:0` PUT format, so we
    // normally see a bare integer.
    let head = trimmed.split_once('.').map_or(trimmed, |(int, _)| int);
    head.parse::<i32>().ok()
}

/// Parse a non-negative GAMS integer (e.g. `iterusd` under `:0:0`) into `u64`,
/// tolerating a trailing `.0` fraction.
fn parse_gams_u64(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    let head = trimmed.split_once('.').map_or(trimmed, |(int, _)| int);
    head.parse::<u64>().ok()
}

/// Parse a GAMS-formatted float, tolerating GAMS special tokens.
fn parse_gams_float(s: &str) -> Option<f64> {
    match s.trim() {
        "INF" | "+INF" | "Inf" | "+Inf" => Some(f64::INFINITY),
        "-INF" | "-Inf" => Some(f64::NEG_INFINITY),
        "EPS" => Some(0.0),
        "NA" | "UNDF" => None,
        other => other.parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::prelude::*;

    fn render(model: &Model, opts: &GamsOptions) -> String {
        let arena = model.arena();
        let vars = model.variables();
        let constraints = model.constraints();
        let socs = model.soc_constraints();
        let objective = model.objective();
        let sense_kw = match objective.as_ref().map_or(ObjectiveSense::Minimize, |o| o.sense) {
            ObjectiveSense::Minimize => "minimizing",
            ObjectiveSense::Maximize => "maximizing",
        };
        let mut gms = String::new();
        build_model_section(
            &mut gms,
            model.kind(),
            &arena,
            &vars,
            &constraints,
            &socs,
            objective.as_ref(),
            sense_kw,
            opts,
        );
        gms
    }

    #[test]
    fn parses_iterations_from_put_file() {
        // The PUT solution file emits `ITER=` from `oximo_m.iterusd`.
        let content = "STATUS=1\nSOLVESTAT=1\nITER=42\nOBJVAL=10.0\n0=2.5\n";
        let r = parseoximo_solution(content, &[], std::time::Duration::ZERO, None);
        assert_eq!(r.termination, TerminationStatus::Optimal);
        assert_eq!(r.iterations, 42);
    }

    #[test]
    fn soc_marginal_is_rescaled_to_norm_form() {
        let content = "STATUS=1\nSOLVESTAT=1\nOBJVAL=-1.0\n0=-1.0\n1=1.5\nZ0=-0.75\n";
        let bounds = vec![LinearTerms { coeffs: vec![(VarId(1), 1.0)], constant: 0.5 }];
        let r = parseoximo_solution(content, &bounds, std::time::Duration::ZERO, None);
        let z0 = r.soc_dual_of(SocConstraintId(0)).expect("SOC dual missing");
        assert!((z0 - 3.0).abs() < 1e-9, "z0 = {z0}");
    }

    #[test]
    fn put_section_reads_soc_marginals() {
        let mut gms = String::new();
        write_soc_marginal_puts(&mut gms, 2);
        assert!(gms.contains("Put 'Z0=' eq_soc0.m:0:15 /;"), "{gms}");
        assert!(gms.contains("Put 'Z1=' eq_soc1.m:0:15 /;"), "{gms}");
    }

    #[test]
    fn termination_is_driven_by_solvestat() {
        // solvestat (args are `(modelstat, solvestat)`).
        // Normal completion defers to modelstat.
        assert_eq!(map_status(1, 1), TerminationStatus::Optimal);
        assert_eq!(map_status(2, 1), TerminationStatus::LocallyOptimal);
        assert_eq!(map_status(8, 1), TerminationStatus::Optimal);
        assert_eq!(map_status(4, 1), TerminationStatus::Infeasible);
        assert_eq!(map_status(3, 1), TerminationStatus::Unbounded);
        assert_eq!(map_status(8, 2), TerminationStatus::IterationLimit);
        assert_eq!(map_status(7, 3), TerminationStatus::TimeLimit);
        assert_eq!(map_status(8, 8), TerminationStatus::Interrupted);
        assert_eq!(map_status(8, 4), TerminationStatus::Interrupted);
        assert_eq!(map_status(1, 5), TerminationStatus::NumericError);
        assert_eq!(map_status(1, 12), TerminationStatus::NotSolved);
        assert_eq!(map_status(1, 7), TerminationStatus::Other("gams_license_error".into()));
    }

    #[test]
    fn solver_label_names_subsolver() {
        use crate::solver_options::{GamsCplexOptions, GamsSolverConfig};
        let opts =
            GamsOptions::default().solver(GamsSolverConfig::Cplex(GamsCplexOptions::default()));
        assert_eq!(gams_solver_label(&opts).as_ref(), "GAMS/CPLEX");
        // No sub-solver configured: GAMS picks its own default, so just "GAMS".
        assert_eq!(gams_solver_label(&GamsOptions::default()).as_ref(), "GAMS");
    }

    #[test]
    fn parse_gams_u64_tolerates_trailing_fraction() {
        assert_eq!(parse_gams_u64("1234"), Some(1234));
        assert_eq!(parse_gams_u64("  88.0 "), Some(88));
        assert_eq!(parse_gams_u64("NA"), None);
    }

    #[test]
    fn linear_objective_uses_lp_solve_type() {
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 10.0);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, c, x + y <= 5.0);
        objective!(m, Min, x + 2.0 * y);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using LP minimizing v_obj;"), "got:\n{gms}");
    }

    #[test]
    fn feasibility_minimizes_zero_with_lp_solve_type() {
        let m = Model::new("feas");
        variable!(m, 0.0 <= x <= 10.0);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, c, x + y == 5.0);
        objective!(m, Feasibility);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("eq_obj..  v_obj =e= 0;"), "got:\n{gms}");
        assert!(gms.contains("Solve oximo_m using LP minimizing v_obj;"), "got:\n{gms}");
    }

    #[test]
    fn nlp_uses_transcendental_and_picks_nlp_solve_type() {
        let m = Model::new("nlp");
        variable!(m, -std::f64::consts::PI <= x <= std::f64::consts::PI);
        objective!(m, Min, x.sin() + x.exp());
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using NLP minimizing v_obj;"), "got:\n{gms}");
        assert!(gms.contains("sin("), "expected sin(...) in objective:\n{gms}");
        assert!(gms.contains("exp("), "expected exp(...) in objective:\n{gms}");
    }

    #[test]
    fn abs_objective_emits_abs_func() {
        let m = Model::new("absnlp");
        variable!(m, -5.0 <= x <= 5.0);
        objective!(m, Min, x.abs());
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using NLP minimizing v_obj;"), "got:\n{gms}");
        assert!(gms.contains("abs("), "expected abs(...) in objective:\n{gms}");
    }

    #[test]
    fn minlp_nonlinear_knapsack_routes_to_minlp_solve_type() {
        let m = Model::new("minlp");
        variable!(m, x, Bin);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, budget, x + y <= 8.0);
        objective!(m, Max, (1.0 + y).log() + 2.0 * x);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using MINLP maximizing v_obj;"), "got:\n{gms}");
        assert!(gms.contains("log("), "expected log(...) in objective:\n{gms}");
    }

    #[test]
    fn solve_type_map_is_total_over_all_kinds() {
        assert_eq!(gams_solve_type(ModelKind::LP), "LP");
        assert_eq!(gams_solve_type(ModelKind::MILP), "MIP");
        assert_eq!(gams_solve_type(ModelKind::QP), "QCP");
        assert_eq!(gams_solve_type(ModelKind::MIQP), "MIQCP");
        assert_eq!(gams_solve_type(ModelKind::QCP), "QCP");
        assert_eq!(gams_solve_type(ModelKind::MIQCP), "MIQCP");
        assert_eq!(gams_solve_type(ModelKind::SOCP), "QCP");
        assert_eq!(gams_solve_type(ModelKind::MISOCP), "MIQCP");
        assert_eq!(gams_solve_type(ModelKind::NLP), "NLP");
        assert_eq!(gams_solve_type(ModelKind::MINLP), "MINLP");
    }

    #[test]
    fn explicit_soc_emits_sqr_rows_and_sign_row() {
        let m = Model::new("socp");
        variable!(m, -10.0 <= x <= 10.0);
        variable!(m, -10.0 <= y <= 10.0);
        variable!(m, t >= 0.0);
        m.add_soc_constraint("cone", [x, y], t);
        objective!(m, Min, x + y);
        assert_eq!(m.kind(), ModelKind::SOCP);

        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("eq_soc0, eq_soc0_sign"), "declares SOC equations:\n{gms}");
        assert!(gms.contains("eq_soc0.. "), "emits SOC row:\n{gms}");
        assert!(gms.contains("sqr("), "uses sqr():\n{gms}");
        assert!(gms.contains("=l= sqr("), "bound side squared:\n{gms}");
        assert!(gms.contains("eq_soc0_sign.."), "emits sign row:\n{gms}");
        assert!(gms.contains("=g= 0;"), "sign row nonneg:\n{gms}");
        assert!(gms.contains("Solve oximo_m using QCP minimizing v_obj;"), "got:\n{gms}");
    }

    #[test]
    fn subsolver_capabilities_cover_new_kinds() {
        use crate::GamsSolver;
        use crate::solver_options::GamsSolverConfig;
        let cplex = GamsSolverConfig::from(GamsSolver::Cplex);
        assert!(cplex.supports(ModelKind::QCP));
        assert!(cplex.supports(ModelKind::SOCP), "SOCP routes through QCP");
        assert!(cplex.supports(ModelKind::MISOCP));
        let highs = GamsSolverConfig::from(GamsSolver::Highs);
        assert!(!highs.supports(ModelKind::QCP), "HiGHS under GAMS is LP/MIP only");
        assert!(!highs.supports(ModelKind::SOCP));
    }

    #[test]
    fn quadratic_constraint_emits_full_expression_against_rhs() {
        let m = Model::new("qcp");
        variable!(m, 0.0 <= x <= 5.0);
        variable!(m, 0.0 <= y <= 5.0);
        constraint!(m, xy, x * y <= 4.0);
        objective!(m, Min, x + y);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using QCP minimizing v_obj;"), "got:\n{gms}");
        // The product term must appear on the LHS, the user RHS untouched.
        assert!(gms.contains("v0") && gms.contains("v1"), "vars missing:\n{gms}");
        assert!(gms.contains("=l= 4"), "expected =l= 4 on the right:\n{gms}");
    }

    #[test]
    fn integer_power_uses_power_func() {
        let m = Model::new("pow");
        variable!(m, -10.0 <= x <= 10.0);
        objective!(m, Min, x.powi(3));
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("power("), "expected power(...) for int Pow:\n{gms}");
        assert!(gms.contains(", 3)"), "expected exponent 3:\n{gms}");
        assert!(gms.contains("Solve oximo_m using NLP minimizing v_obj;"), "got:\n{gms}");
    }

    #[test]
    fn real_power_falls_back_to_double_star() {
        let m = Model::new("rpow");
        variable!(m, 0.1 <= x <= 10.0);
        objective!(m, Min, x.powf(0.5));
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains(" **"), "expected ** for real Pow:\n{gms}");
    }

    #[test]
    fn validate_solver_rejects_lp_only_solver_on_nlp() {
        use crate::GamsSolver;
        use crate::solver_options::{GamsCplexOptions, GamsSolverConfig};
        let o = GamsOptions::default().solver(GamsSolverConfig::Cplex(GamsCplexOptions::default()));
        let err = validate_solver(&o, ModelKind::NLP).unwrap_err();
        match err {
            SolverError::Backend(msg) => {
                assert!(msg.contains("CPLEX"), "names solver: {msg}");
                assert!(msg.contains("NLP"), "names solve type: {msg}");
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
        // A named LP/MIP solver is rejected the same way.
        let o = GamsOptions::default().solver(GamsSolver::Highs);
        assert!(validate_solver(&o, ModelKind::MINLP).is_err());
    }

    #[test]
    fn validate_solver_accepts_compatible_solver() {
        use crate::solver_options::{GamsIpoptOptions, GamsSolverConfig};
        let o = GamsOptions::default().solver(GamsSolverConfig::Ipopt(GamsIpoptOptions::default()));
        assert!(validate_solver(&o, ModelKind::LP).is_ok());
        assert!(validate_solver(&o, ModelKind::NLP).is_ok());
        assert!(validate_solver(&o, ModelKind::QP).is_ok(), "QP routes through QCP");
        // IPOPT does LP/NLP/QCP only, so the integer kinds must be rejected.
        assert!(validate_solver(&o, ModelKind::MILP).is_err());
        assert!(validate_solver(&o, ModelKind::MIQP).is_err());
    }

    #[test]
    fn validate_solver_noop_without_solver_and_for_custom() {
        use crate::GamsSolver;
        // No explicit solver: GAMS picks its default, nothing to validate.
        assert!(validate_solver(&GamsOptions::default(), ModelKind::MINLP).is_ok());
        // Unknown/custom names can't be validated, so they pass.
        let o = GamsOptions::default().solver(GamsSolver::Custom("MYSOLVER".into()));
        assert!(validate_solver(&o, ModelKind::MINLP).is_ok());
    }
}
