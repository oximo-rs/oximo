use std::fmt::Write as FmtWrite;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::{fs, io};

static SOLVE_ID: AtomicU64 = AtomicU64::new(0);

use oximo_core::{
    Constraint, ConstraintId, Domain, Model, ModelKind, Objective, ObjectiveSense, Sense, VarId,
    Variable,
};
use oximo_expr::{ExprArena, ExprId, ExprNode, LinearTerms, extract_linear};
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
    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let objective = model.try_objective().map_err(SolverError::Core)?;

    let sense_kw = match objective.sense {
        ObjectiveSense::Minimize => "minimizing",
        ObjectiveSense::Maximize => "maximizing",
    };

    let mut gms = String::with_capacity(4096);
    let (solve_type, solver_opt) = build_model_section(
        &mut gms,
        kind,
        &arena,
        &vars,
        &constraints,
        &objective,
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
    writeln!(gms, "Put 'OBJVAL=' v_obj.l:0:15 /;").unwrap();
    for i in 0..vars.len() {
        writeln!(gms, "Put '{i}=' v{i}.l:0:15 /;").unwrap();
    }
    // Marginals are well-defined only for LP. For MIP, GAMS returns NA/0 for
    // `.m`, so we skip them and leave the dual/reduced-cost maps empty.
    let emit_marginals = solve_type == "LP";
    if emit_marginals {
        for i in 0..vars.len() {
            writeln!(gms, "Put 'R{i}=' v{i}.m:0:15 /;").unwrap();
        }
        for i in 0..constraints.len() {
            writeln!(gms, "Put 'D{i}=' eq_c{i}.m:0:15 /;").unwrap();
        }
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
    let result = if sol_path.exists() {
        let content = fs::read_to_string(&sol_path)
            .map_err(|e| SolverError::Backend(format!("cannot read solution file: {e}")))?;
        parseoximo_solution(&content, elapsed, raw_log)
    } else {
        // No solution file. GAMS must have failed before the Solve statement
        // (compilation error, license error, etc.).  Fall back to the listing.
        let listing = fs::read_to_string(tmp_dir.join("model.lst")).unwrap_or_default();
        let _ = fs::remove_dir_all(&tmp_dir);
        let detail = if exit_ok {
            format!(
                "GAMS did not produce a solution file. \
                Check the .gms listing for compilation errors.\n{listing}"
            )
        } else {
            format!("GAMS exited with a non-zero exit code.\n{listing}")
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
    let mut dual: FxHashMap<ConstraintId, f64> = FxHashMap::default();
    let mut reduced_costs: FxHashMap<VarId, f64> = FxHashMap::default();

    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("STATUS=") {
            modelstat = parse_gams_int(rest);
        } else if let Some(rest) = line.strip_prefix("SOLVESTAT=") {
            solvestat = parse_gams_int(rest);
        } else if let Some(rest) = line.strip_prefix("OBJVAL=") {
            obj_val = parse_gams_float(rest);
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
    let has_sol = status.has_solution();

    SolverResult {
        objective: if has_sol { obj_val } else { None },
        primal: if has_sol { primal } else { FxHashMap::default() },
        dual: if has_sol { dual } else { FxHashMap::default() },
        reduced_costs: if has_sol { reduced_costs } else { FxHashMap::default() },
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

/// Write the formulation portion of the `.gms` file: title, variables, bounds,
/// equations, options, model, and solve statement. Returns the solve type
/// (`"LP"` / `"MIP"` / `"NLP"` / `"MINLP"`) and any solver-options file pair
/// `(filename, content)` the caller should also persist alongside the `.gms`.
#[allow(clippy::too_many_arguments)]
fn build_model_section(
    gms: &mut String,
    kind: ModelKind,
    arena: &ExprArena,
    vars: &[Variable],
    constraints: &[Constraint],
    objective: &Objective,
    sense_kw: &str,
    opts: &GamsOptions,
) -> (&'static str, Option<(String, String)>) {
    let solve_type = gams_solve_type(kind);
    let solver_opt = build_solver_opt(opts);

    write_preamble(gms);
    write_var_declarations(gms, vars);
    write_bounds_and_initials(gms, vars);
    write_equations(gms, arena, constraints, objective);
    write_options(gms, opts, solve_type);
    write_model_and_solve(gms, solve_type, sense_kw, solver_opt.is_some());

    (solve_type, solver_opt)
}

fn gams_solve_type(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::LP => "LP",
        ModelKind::MILP => "MIP",
        ModelKind::QP | ModelKind::NLP => "NLP",
        ModelKind::MIQP | ModelKind::MINLP => "MINLP",
    }
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

/// Emit `Variables`, `Binary Variables`, `Integer Variables` sections.
fn write_var_declarations(gms: &mut String, vars: &[Variable]) {
    let (mut cont, mut bin, mut int) = (Vec::new(), Vec::new(), Vec::new());
    for v in vars {
        match v.domain {
            Domain::Binary => bin.push(v),
            Domain::Integer | Domain::SemiInteger { .. } => int.push(v),
            _ => cont.push(v),
        }
    }

    write!(gms, "Variables\n    v_obj").unwrap();
    for v in &cont {
        write!(gms, ", v{}", v.id.index()).unwrap();
    }
    writeln!(gms, ";").unwrap();

    write_typed_var_section(gms, "Binary Variables", &bin);
    write_typed_var_section(gms, "Integer Variables", &int);
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
    objective: &Objective,
) {
    write!(gms, "Equations\n    eq_obj").unwrap();
    for i in 0..constraints.len() {
        write!(gms, ", eq_c{i}").unwrap();
    }
    writeln!(gms, ";").unwrap();
    writeln!(gms).unwrap();

    let obj_form = ExprForm::from(arena, objective.expr);
    write!(gms, "eq_obj..  v_obj =e=").unwrap();
    write_form(gms, arena, &obj_form, true);
    writeln!(gms, ";").unwrap();

    for (ci, c) in constraints.iter().enumerate() {
        let sense_str = match c.sense {
            Sense::Le => "=l=",
            Sense::Ge => "=g=",
            Sense::Eq => "=e=",
        };
        write!(gms, "eq_c{ci}..").unwrap();
        match ExprForm::from(arena, c.lhs) {
            ExprForm::Linear(t) => {
                let adjusted_rhs = c.rhs - t.constant;
                write_linear(gms, &t, false);
                writeln!(gms, " {sense_str} {};", fmt(adjusted_rhs)).unwrap();
            }
            ExprForm::Nonlinear(id) => {
                write_gams_expr(gms, arena, id, true);
                writeln!(gms, " {sense_str} {};", fmt(c.rhs)).unwrap();
            }
        }
    }
    writeln!(gms).unwrap();
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
        ExprNode::Param(_) => {
            // Since params not yet passed into GAMS emission, we emit a placeholder
            // so downstream errors are clear.
            write!(gms, "0 /* unsupported: param */").unwrap();
        }
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

/// Parse a GAMS-formatted float, tolerating GAMS special tokens.
fn parse_gams_float(s: &str) -> Option<f64> {
    match s.trim() {
        "INF" | "+INF" | "Inf" | "+Inf" => Some(f64::INFINITY),
        "-INF" | "-Inf" => Some(f64::NEG_INFINITY),
        "NA" | "UNDF" | "EPS" => Some(0.0),
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
        let objective = model.try_objective().expect("objective set");
        let sense_kw = match objective.sense {
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
            &objective,
            sense_kw,
            opts,
        );
        gms
    }

    #[test]
    fn linear_objective_uses_lp_solve_type() {
        let m = Model::new("lp");
        let x = m.var("x").lb(0.0).ub(10.0).build();
        let y = m.var("y").lb(0.0).ub(10.0).build();
        m.constraint("c", (x + y).le(5.0));
        m.minimize(x + 2.0 * y);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using LP minimizing v_obj;"), "got:\n{gms}");
    }

    #[test]
    fn nlp_uses_transcendental_and_picks_nlp_solve_type() {
        let m = Model::new("nlp");
        let x = m.var("x").lb(-std::f64::consts::PI).ub(std::f64::consts::PI).build();
        m.minimize(x.sin() + x.exp());
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using NLP minimizing v_obj;"), "got:\n{gms}");
        assert!(gms.contains("sin("), "expected sin(...) in objective:\n{gms}");
        assert!(gms.contains("exp("), "expected exp(...) in objective:\n{gms}");
    }

    #[test]
    fn minlp_nonlinear_knapsack_routes_to_minlp_solve_type() {
        let m = Model::new("minlp");
        let x = m.var("x").binary().build();
        let y = m.var("y").lb(0.0).ub(10.0).build();
        m.constraint("budget", (x + y).le(8.0));
        let one = Expr::constant(x.arena, 1.0);
        m.maximize((one + y).log() + 2.0 * x);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using MINLP maximizing v_obj;"), "got:\n{gms}");
        assert!(gms.contains("log("), "expected log(...) in objective:\n{gms}");
    }

    #[test]
    fn quadratic_constraint_emits_full_expression_against_rhs() {
        let m = Model::new("qcp");
        let x = m.var("x").lb(0.0).ub(5.0).build();
        let y = m.var("y").lb(0.0).ub(5.0).build();
        m.constraint("xy", (x * y).le(4.0));
        m.minimize(x + y);
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("Solve oximo_m using NLP minimizing v_obj;"), "got:\n{gms}");
        // The product term must appear on the LHS, the user RHS untouched.
        assert!(gms.contains("v0") && gms.contains("v1"), "vars missing:\n{gms}");
        assert!(gms.contains("=l= 4"), "expected =l= 4 on the right:\n{gms}");
    }

    #[test]
    fn integer_power_uses_power_func() {
        let m = Model::new("pow");
        let x = m.var("x").lb(-10.0).ub(10.0).build();
        m.minimize(x.powi(3));
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains("power("), "expected power(...) for int Pow:\n{gms}");
        assert!(gms.contains(", 3)"), "expected exponent 3:\n{gms}");
        assert!(gms.contains("Solve oximo_m using NLP minimizing v_obj;"), "got:\n{gms}");
    }

    #[test]
    fn real_power_falls_back_to_double_star() {
        let m = Model::new("rpow");
        let x = m.var("x").lb(0.1).ub(10.0).build();
        m.minimize(x.powf(0.5));
        let gms = render(&m, &GamsOptions::default());
        assert!(gms.contains(" **"), "expected ** for real Pow:\n{gms}");
    }
}
