use std::fmt::Write as FmtWrite;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::{fs, io};

use oximo_core::{
    Constraint, ConstraintId, Domain, Model, Objective, ObjectiveSense, Sense, SocConstraint,
    SocConstraintId, VarId, Variable,
};
use oximo_expr::{ExprArena, ExprId, ExprNode, LinearTerms, extract_linear};
use oximo_solver::{PrimalStatus, SolutionPoint, SolverError, SolverResult, TerminationStatus};
use rustc_hash::FxHashMap;

use crate::BaronOptions;
use crate::options::write_options;

static SOLVE_ID: AtomicU64 = AtomicU64::new(0);

const RES_NAME: &str = "res.lst";
const TIM_NAME: &str = "tim.lst";
const BAR_NAME: &str = "problem.bar";

/// Write `model` to a temporary BARON `.bar` file, execute the `baron`
/// executable, and return the parsed [`SolverResult`].
///
/// `exec` is an optional override for the BARON executable path. `None` uses
/// `"baron"` resolved from `PATH`. [`BaronOptions::baron_path`] takes precedence
/// over `exec`.
///
/// # Errors
///
/// Returns [`SolverError`] on constructs BARON's `.bar` format cannot represent
/// (`sin`/`cos`, semicontinuous/semi-integer variables), a missing BARON
/// executable, a BARON run that produced no times file, or I/O failures.
///
/// # Panics
///
/// Panics if variable indices overflow `u32`.
pub fn solve(
    model: &Model,
    opts: &BaronOptions,
    exec: Option<&str>,
) -> Result<SolverResult, SolverError> {
    let sense = model.objective().as_ref().map_or(ObjectiveSense::Minimize, |o| o.sense);
    let (bar, var_order, con_order, soc_bounds) = build_bar(model, opts)?;

    // - Temp directory. Combine a timestamp with a per-process atomic counter so
    //   concurrent invocations never share a directory.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let id = SOLVE_ID.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir().join(format!("oximo_baron_{ts}_{id}"));
    fs::create_dir_all(&tmp_dir)
        .map_err(|e| SolverError::Backend(format!("cannot create temp dir: {e}")))?;

    let bar_path = tmp_dir.join(BAR_NAME);
    fs::write(&bar_path, &bar)
        .map_err(|e| SolverError::Backend(format!("cannot write .bar file: {e}")))?;

    // - Execute BARON.
    let baron_exec =
        opts.baron_path.as_deref().and_then(std::path::Path::to_str).or(exec).unwrap_or("baron");
    let verbose = opts.universal.verbose.unwrap_or(false);

    let started = Instant::now();
    let mut cmd = std::process::Command::new(baron_exec);
    cmd.arg(BAR_NAME);
    cmd.current_dir(&tmp_dir);

    let launch_err = |e: io::Error| {
        let _ = fs::remove_dir_all(&tmp_dir);
        if e.kind() == io::ErrorKind::NotFound {
            SolverError::Backend(format!(
                "BARON executable '{baron_exec}' not found. \
                Install BARON and ensure it is on PATH, or set the 'baron_path' option."
            ))
        } else {
            SolverError::Backend(format!("failed to launch BARON: {e}"))
        }
    };

    // When verbose, stream BARON's output to the terminal. Otherwise capture it
    // so we can surface it on failure.
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

    // - Parse the times file (status) and results file (primal).
    let tim_path = tmp_dir.join(TIM_NAME);
    if !tim_path.exists() {
        // BARON never reached the solve (syntax/license error). Surface its log.
        let _ = fs::remove_dir_all(&tmp_dir);
        let detail = raw_log.unwrap_or_else(|| {
            if exit_ok {
                "BARON produced no times file and emitted no output.".to_string()
            } else {
                "BARON exited with a non-zero exit code and produced no times file.".to_string()
            }
        });
        return Err(SolverError::Backend(format!("BARON did not produce a solution.\n{detail}")));
    }

    let tim = fs::read_to_string(&tim_path)
        .map_err(|e| SolverError::Backend(format!("cannot read times file: {e}")))?;
    let res = fs::read_to_string(tmp_dir.join(RES_NAME)).unwrap_or_default();
    let result =
        parse_solution(&tim, &res, sense, elapsed, raw_log, &var_order, &con_order, &soc_bounds);

    let _ = fs::remove_dir_all(&tmp_dir);
    Ok(result)
}

/// Everything `solve` needs from the `.bar` writer: the file text, the
/// variable declaration order (see [`write_var_declarations`]) used to decode
/// BARON's numeric solution-pool blocks, the constraint emit-order (see
/// [`write_equations`]) used to fold range duals back onto their
/// `ConstraintId`, and the affine bound side of each explicit SOC constraint
/// used to rescale its squared-row price to the norm form.
type BarParts = (String, Vec<VarId>, Vec<ConstraintId>, Vec<LinearTerms>);

fn build_bar(model: &Model, opts: &BaronOptions) -> Result<BarParts, SolverError> {
    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();
    let socs = model.soc_constraints();
    let objective = model.objective();

    let mut bar = String::with_capacity(4096);
    write_options(&mut bar, opts, RES_NAME, TIM_NAME);
    let var_order = write_var_declarations(&mut bar, &vars)?;
    write_bounds(&mut bar, &vars);
    let con_order = write_equations(&mut bar, &arena, &constraints, &socs)?;
    write_objective(&mut bar, &arena, objective.as_ref())?;
    write_starting_point(&mut bar, &vars);
    let soc_bounds = socs
        .iter()
        .map(|s| extract_linear(&arena, s.bound).expect("SOC bound is validated affine"))
        .collect();
    Ok((bar, var_order, con_order, soc_bounds))
}

// - .bar writers

/// Emit the `BINARY_VARIABLES` / `INTEGER_VARIABLES` / `POSITIVE_VARIABLES` /
/// `VARIABLES` declaration sections.
///
/// Returns the variable `VarId`s in the order they are declared (binary,
/// integer, positive, free). BARON numbers variables 1-based in this order, so
/// the result maps "Variable no. `k`" back to a `VarId` when parsing the
/// enumerated solution pool.
fn write_var_declarations(bar: &mut String, vars: &[Variable]) -> Result<Vec<VarId>, SolverError> {
    let (mut bin, mut int, mut pos, mut free) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for v in vars {
        match v.domain {
            Domain::Binary => bin.push(v),
            Domain::Integer => int.push(v),
            // A continuous variable with a zero lower bound is exactly BARON's
            // POSITIVE_VARIABLES; everything else (free, negative, finite lb) is
            // a general VARIABLES with explicit bounds emitted below.
            Domain::Real if v.lb == 0.0 => pos.push(v),
            Domain::Real => free.push(v),
            Domain::SemiContinuous { .. } | Domain::SemiInteger { .. } => {
                return Err(SolverError::Backend(format!(
                    "variable x{} has a semicontinuous/semi-integer domain, \
                    which BARON's .bar format cannot represent",
                    v.id.index()
                )));
            }
        }
    }
    write_var_section(bar, "BINARY_VARIABLES", &bin);
    write_var_section(bar, "INTEGER_VARIABLES", &int);
    write_var_section(bar, "POSITIVE_VARIABLES", &pos);
    write_var_section(bar, "VARIABLES", &free);
    writeln!(bar).unwrap();
    let order = bin.iter().chain(&int).chain(&pos).chain(&free).map(|v| v.id).collect();
    Ok(order)
}

fn write_var_section(bar: &mut String, header: &str, vars: &[&Variable]) {
    if vars.is_empty() {
        return;
    }
    write!(bar, "{header} ").unwrap();
    for (k, v) in vars.iter().enumerate() {
        if k > 0 {
            write!(bar, ", ").unwrap();
        }
        write!(bar, "x{}", v.id.index()).unwrap();
    }
    writeln!(bar, ";").unwrap();
}

fn write_bounds(bar: &mut String, vars: &[Variable]) {
    let mut lo = String::new();
    let mut hi = String::new();
    for v in vars {
        let i = v.id.index();
        if let Some(lb) = lower_bound_to_emit(v) {
            writeln!(lo, "x{i}: {};", fmt(lb)).unwrap();
        }
        if let Some(ub) = upper_bound_to_emit(v) {
            writeln!(hi, "x{i}: {};", fmt(ub)).unwrap();
        }
    }
    if !lo.is_empty() {
        writeln!(bar, "LOWER_BOUNDS{{").unwrap();
        bar.push_str(&lo);
        writeln!(bar, "}}").unwrap();
        writeln!(bar).unwrap();
    }
    if !hi.is_empty() {
        writeln!(bar, "UPPER_BOUNDS{{").unwrap();
        bar.push_str(&hi);
        writeln!(bar, "}}").unwrap();
        writeln!(bar).unwrap();
    }
}

/// The lower bound to write, or `None` when it equals the implied default for
/// the variable's declaration section (so we avoid redundant lines).
fn lower_bound_to_emit(v: &Variable) -> Option<f64> {
    if !v.lb.is_finite() {
        return None;
    }
    match v.domain {
        // Binary defaults to [0, 1]. Real with lb == 0 sits in POSITIVE_VARIABLES.
        Domain::Binary | Domain::Real if v.lb == 0.0 => None,
        _ => Some(v.lb),
    }
}

fn upper_bound_to_emit(v: &Variable) -> Option<f64> {
    if !v.ub.is_finite() {
        return None;
    }
    match v.domain {
        // Binary defaults to an upper bound of 1. Only emit when overridden.
        Domain::Binary if (v.ub - 1.0).abs() < f64::EPSILON => None,
        _ => Some(v.ub),
    }
}

/// Write the `EQUATIONS` block, returning the emit-order map.
/// The `ConstraintId` behind each emitted equation, in BARON's 1-based
/// "Constraint no." order. BARON has no two-sided equation,
/// so a range emits two rows (`c{i}_lo`/`c{i}_hi`) that both map back
/// to the one `ConstraintId`. The dual parser folds them together so
/// positional indices stay correct.
fn write_equations(
    bar: &mut String,
    arena: &ExprArena,
    constraints: &[Constraint],
    socs: &[SocConstraint],
) -> Result<Vec<ConstraintId>, SolverError> {
    let mut emit_map: Vec<ConstraintId> = Vec::with_capacity(constraints.len());
    if constraints.is_empty() && socs.is_empty() {
        return Ok(emit_map);
    }

    let mut names: Vec<String> = Vec::with_capacity(constraints.len() + 2 * socs.len());
    for (i, c) in constraints.iter().enumerate() {
        let id = ConstraintId(u32::try_from(i).expect("constraint count overflow"));
        if c.is_range() {
            names.push(format!("c{i}_lo"));
            names.push(format!("c{i}_hi"));
            emit_map.push(id);
            emit_map.push(id);
        } else {
            names.push(format!("c{i}"));
            emit_map.push(id);
        }
    }
    for i in 0..socs.len() {
        names.push(format!("soc{i}"));
        names.push(format!("soc{i}_sign"));
    }
    writeln!(bar, "EQUATIONS {};", names.join(", ")).unwrap();

    for (i, c) in constraints.iter().enumerate() {
        // BARON rejects a constraint whose expression evaluates to a constant,
        // so surface a clear error instead of emitting an invalid `.bar`.
        if !expr_has_var(arena, c.lhs) {
            return Err(SolverError::Backend(format!(
                "constraint '{}' has no variables (its left-hand side is constant); \
                BARON requires every constraint to contain at least one variable",
                c.name
            )));
        }
        if let Some((sense, rhs)) = c.as_single() {
            let op = match sense {
                Sense::Le => "<=",
                Sense::Ge => ">=",
                Sense::Eq => "==",
            };
            write!(bar, "c{i}: ").unwrap();
            write_constraint_body(bar, arena, c.lhs, op, rhs)?;
        } else {
            // Two-sided range -> `_lo` (>= lower) and `_hi` (<= upper).
            write!(bar, "c{i}_lo: ").unwrap();
            write_constraint_body(bar, arena, c.lhs, ">=", c.lower)?;
            write!(bar, "c{i}_hi: ").unwrap();
            write_constraint_body(bar, arena, c.lhs, "<=", c.upper)?;
        }
    }
    write_soc_rows(bar, arena, socs);
    writeln!(bar).unwrap();
    Ok(emit_map)
}

/// Emit each explicit SOC constraint `||terms||_2 <= bound` as the polynomial
/// row `(term_1)^2 + ... - (bound)^2 <= 0` plus the sign row `bound >= 0`
/// (squaring loses the sign of the bound side).
fn write_soc_rows(bar: &mut String, arena: &ExprArena, socs: &[SocConstraint]) {
    for (i, s) in socs.iter().enumerate() {
        write!(bar, "soc{i}: ").unwrap();
        for (k, &term) in s.terms.iter().enumerate() {
            if k > 0 {
                write!(bar, " + ").unwrap();
            }
            let t = extract_linear(arena, term).expect("SOC members are validated affine");
            write!(bar, "(").unwrap();
            write_linear(bar, &t, true);
            write!(bar, ")^2").unwrap();
        }
        let b = extract_linear(arena, s.bound).expect("SOC bound is validated affine");
        write!(bar, " - (").unwrap();
        write_linear(bar, &b, true);
        writeln!(bar, ")^2 <= 0;").unwrap();

        write!(bar, "soc{i}_sign: ").unwrap();
        write_linear(bar, &b, true);
        writeln!(bar, " >= 0;").unwrap();
    }
}

/// Emit one constraint body `<lhs> <op> <rhs>;`. Linear constraints fold the
/// constant into the RHS (canonical `lhs <op> rhs`). Nonlinear ones emit the
/// full LHS and keep the original RHS.
fn write_constraint_body(
    bar: &mut String,
    arena: &ExprArena,
    lhs: ExprId,
    op: &str,
    rhs: f64,
) -> Result<(), SolverError> {
    if let Some(t) = extract_linear(arena, lhs) {
        let adjusted_rhs = rhs - t.constant;
        write_linear(bar, &t, false);
        writeln!(bar, " {op} {};", fmt(adjusted_rhs)).unwrap();
    } else {
        write_bar_expr(bar, arena, lhs)?;
        writeln!(bar, " {op} {};", fmt(rhs)).unwrap();
    }
    Ok(())
}

fn write_objective(
    bar: &mut String,
    arena: &ExprArena,
    objective: Option<&Objective>,
) -> Result<(), SolverError> {
    write!(bar, "OBJ: ").unwrap();
    match objective {
        // BARON requires an objective. A feasibility problem minimizes a constant.
        None => writeln!(bar, "minimize 0;").unwrap(),
        Some(o) => {
            let kw = match o.sense {
                ObjectiveSense::Minimize => "minimize",
                ObjectiveSense::Maximize => "maximize",
            };
            write!(bar, "{kw} ").unwrap();
            if let Some(t) = extract_linear(arena, o.expr) {
                write_linear(bar, &t, true);
            } else {
                write_bar_expr(bar, arena, o.expr)?;
            }
            writeln!(bar, ";").unwrap();
        }
    }
    writeln!(bar).unwrap();
    Ok(())
}

fn write_starting_point(bar: &mut String, vars: &[Variable]) {
    if !vars.iter().any(|v| v.initial.is_some()) {
        return;
    }
    writeln!(bar, "STARTING_POINT{{").unwrap();
    for v in vars {
        if let Some(val) = v.initial {
            writeln!(bar, "x{}: {};", v.id.index(), fmt(val)).unwrap();
        }
    }
    writeln!(bar, "}}").unwrap();
    writeln!(bar).unwrap();
}

/// Append the linear expression `t` to `bar` as a standalone BARON expression.
fn write_linear(bar: &mut String, t: &LinearTerms, include_constant: bool) {
    let mut first = true;
    for (v, coef) in &t.coeffs {
        if *coef == 0.0 {
            continue;
        }
        let idx = v.index();
        if first {
            write!(bar, "{}*x{idx}", fmt(*coef)).unwrap();
            first = false;
        } else if *coef < 0.0 {
            write!(bar, " - {}*x{idx}", fmt(-coef)).unwrap();
        } else {
            write!(bar, " + {}*x{idx}", fmt(*coef)).unwrap();
        }
    }
    if include_constant && t.constant != 0.0 {
        if first {
            write!(bar, "{}", fmt(t.constant)).unwrap();
            first = false;
        } else if t.constant < 0.0 {
            write!(bar, " - {}", fmt(-t.constant)).unwrap();
        } else {
            write!(bar, " + {}", fmt(t.constant)).unwrap();
        }
    }
    if first {
        write!(bar, "0").unwrap();
    }
}

/// Recursive infix printer for a BARON-compatible expression.
fn write_bar_expr(bar: &mut String, arena: &ExprArena, id: ExprId) -> Result<(), SolverError> {
    match arena.get(id) {
        ExprNode::Const(c) => write!(bar, "{}", fmt(*c)).unwrap(),
        ExprNode::Var(v) => write!(bar, "x{}", v.index()).unwrap(),
        ExprNode::Param(p) => write!(bar, "{}", fmt(arena.param_value(*p))).unwrap(),
        ExprNode::Linear { coeffs, constant } => {
            let t = LinearTerms { coeffs: coeffs.clone(), constant: *constant };
            write!(bar, "(").unwrap();
            write_linear(bar, &t, true);
            write!(bar, ")").unwrap();
        }
        ExprNode::Neg(inner) => {
            write!(bar, "(-").unwrap();
            write_bar_expr(bar, arena, *inner)?;
            write!(bar, ")").unwrap();
        }
        ExprNode::Add(children) => {
            write!(bar, "(").unwrap();
            for (i, c) in children.iter().enumerate() {
                if i > 0 {
                    write!(bar, " + ").unwrap();
                }
                write_bar_expr(bar, arena, *c)?;
            }
            write!(bar, ")").unwrap();
        }
        ExprNode::Mul(children) => {
            write!(bar, "(").unwrap();
            for (i, c) in children.iter().enumerate() {
                if i > 0 {
                    write!(bar, " * ").unwrap();
                }
                write_bar_expr(bar, arena, *c)?;
            }
            write!(bar, ")").unwrap();
        }
        ExprNode::Pow(base, exp) => {
            // BARON natively supports `x^a` for a constant exponent `a` and
            // `b^x` for a constant base `b`. Only a variable-on-variable power
            // needs the `exp(y*log(x))` rewrite.
            let exp_is_const = matches!(arena.get(*exp), ExprNode::Const(_));
            let base_is_const = matches!(arena.get(*base), ExprNode::Const(_));
            if exp_is_const || base_is_const {
                write!(bar, "(").unwrap();
                write_bar_expr(bar, arena, *base)?;
                write!(bar, " ^ ").unwrap();
                write_bar_expr(bar, arena, *exp)?;
                write!(bar, ")").unwrap();
            } else {
                write!(bar, "exp((").unwrap();
                write_bar_expr(bar, arena, *exp)?;
                write!(bar, ") * log(").unwrap();
                write_bar_expr(bar, arena, *base)?;
                write!(bar, "))").unwrap();
            }
        }
        ExprNode::Div(num, den) => {
            write!(bar, "(").unwrap();
            write_bar_expr(bar, arena, *num)?;
            write!(bar, " / ").unwrap();
            write_bar_expr(bar, arena, *den)?;
            write!(bar, ")").unwrap();
        }
        ExprNode::Exp(a) => {
            write!(bar, "exp(").unwrap();
            write_bar_expr(bar, arena, *a)?;
            write!(bar, ")").unwrap();
        }
        ExprNode::Log(a) => {
            write!(bar, "log(").unwrap();
            write_bar_expr(bar, arena, *a)?;
            write!(bar, ")").unwrap();
        }
        ExprNode::Sin(_) => {
            return Err(SolverError::Backend(
                "BARON does not support sin(); the .bar format has no trigonometric intrinsics"
                    .into(),
            ));
        }
        ExprNode::Cos(_) => {
            return Err(SolverError::Backend(
                "BARON does not support cos(); the .bar format has no trigonometric intrinsics"
                    .into(),
            ));
        }
        ExprNode::Abs(a) => {
            // BARON has no abs() intrinsic.
            // We reformulate: |x| = (x^2)^(1/2),
            // As suggested by the BARON user manual.
            write!(bar, "(((").unwrap();
            write_bar_expr(bar, arena, *a)?;
            write!(bar, ") ^ 2) ^ 0.5)").unwrap();
        }
    }
    Ok(())
}

/// Whether `id` references at least one variable (vs. evaluating to a constant).
/// Used to enforce BARON's rule that every constraint must contain a non-constant
/// expression.
fn expr_has_var(arena: &ExprArena, id: ExprId) -> bool {
    match arena.get(id) {
        ExprNode::Var(_) => true,
        ExprNode::Const(_) | ExprNode::Param(_) => false,
        ExprNode::Linear { coeffs, .. } => coeffs.iter().any(|(_, c)| *c != 0.0),
        ExprNode::Neg(a)
        | ExprNode::Sin(a)
        | ExprNode::Cos(a)
        | ExprNode::Exp(a)
        | ExprNode::Log(a)
        | ExprNode::Abs(a) => expr_has_var(arena, *a),
        ExprNode::Pow(a, b) | ExprNode::Div(a, b) => {
            expr_has_var(arena, *a) || expr_has_var(arena, *b)
        }
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            children.iter().any(|c| expr_has_var(arena, *c))
        }
    }
}

/// Format an `f64` for use in a `.bar` file.
fn fmt(v: f64) -> String {
    if v == f64::INFINITY {
        return "1e51".into();
    }
    if v == f64::NEG_INFINITY {
        return "-1e51".into();
    }
    format!("{v}")
}

// - Result parsing

/// Parse BARON's times file (`tim.lst`) and results file (`res.lst`).
///
/// The times file is a single whitespace-separated line. Field positions follow
/// the BARON convention (0-indexed here):
/// `[5]` lower bound, `[6]` upper bound, `[8]` model status, `[10]`
/// branch-and-reduce iterations, `[11]` node where the optimum was found
/// (`-3` => no solution), last = wall time. The termination reason comes from
/// the `res.lst` banner (see [`map_status`]), not the numeric `[7]` solver status.
#[allow(clippy::too_many_arguments)]
fn parse_solution(
    tim: &str,
    res: &str,
    sense: ObjectiveSense,
    elapsed: std::time::Duration,
    raw_log: Option<String>,
    var_order: &[VarId],
    con_order: &[ConstraintId],
    soc_bounds: &[LinearTerms],
) -> SolverResult {
    let tokens: Vec<&str> = tim.split_whitespace().collect();
    let int_at = |i: usize| tokens.get(i).and_then(|s| s.parse::<i64>().ok());
    let float_at = |i: usize| tokens.get(i).and_then(|s| parse_baron_float(s));

    let model_status = int_at(8).unwrap_or(5);
    let lower = float_at(5);
    let upper = float_at(6);
    let iterations = int_at(10).and_then(|n| u64::try_from(n).ok()).unwrap_or(0);
    let nodeopt = int_at(11);

    let termination = map_status(res, model_status);
    // BARON has a usable point when the model status says optimal (`1`) or
    // intermediate-feasible (`4`). The solution node (`nodeopt`) only
    // controls whether a primal vector is available.
    let has_sol = matches!(model_status, 1 | 4);

    // For minimization the incumbent is the upper bound, for maximization the
    // lower bound (the other field is the dual/relaxation bound).
    let objective = match sense {
        ObjectiveSense::Minimize => upper,
        ObjectiveSense::Maximize => lower,
    };

    let mut solutions = if has_sol && nodeopt != Some(-3) {
        parse_results(res, var_order, sense)
    } else {
        Vec::new()
    };

    // BARON prints each solution's exact objective. Only when the status says a
    // solution exists but no primal block was parsed do we fall back to the times
    // file so `result_count` still reflects that a solution exists.
    if has_sol && solutions.is_empty() {
        solutions.push(SolutionPoint { primal: FxHashMap::default(), objective });
    }

    let (dual, reduced_costs, soc_prices) = if has_sol {
        parse_dual_solution(res, var_order, con_order)
    } else {
        (FxHashMap::default(), FxHashMap::default(), FxHashMap::default())
    };

    // Rescale each SOC squared-row price to the norm-form bound multiplier
    // using the bound's value at the best primal point.
    let mut soc_dual: FxHashMap<SocConstraintId, f64> = FxHashMap::default();
    if let Some(best) = solutions.first() {
        for (i, bound) in soc_bounds.iter().enumerate() {
            if let Some(price) = soc_prices.get(&i) {
                let Some(b_val) = bound.coeffs.iter().try_fold(bound.constant, |acc, &(v, c)| {
                    best.primal.get(&v).map(|value| acc + c * *value)
                }) else {
                    continue;
                };
                soc_dual.insert(
                    SocConstraintId(u32::try_from(i).expect("SOC count overflow")),
                    2.0 * b_val * price.abs(),
                );
            }
        }
    }

    // A point is only usable if it carries primal values.
    let has_usable_primal = solutions.iter().any(|s| !s.primal.is_empty());
    let primal_status = PrimalStatus::infer(&termination, has_usable_primal);
    let best_bound = match sense {
        ObjectiveSense::Minimize => lower,
        ObjectiveSense::Maximize => upper,
    };
    let gap = relative_gap(lower, upper);

    SolverResult {
        solutions,
        dual,
        soc_dual,
        reduced_costs,
        termination,
        primal_status,
        best_bound,
        gap,
        solve_time: elapsed,
        iterations,
        raw_log,
        solver_name: Some(crate::NAME.into()),
    }
}

/// Relative optimality gap from BARON's lower/upper bounds, `None` unless both
/// are finite. Normalized by the larger-magnitude bound (+ epsilon) so it is
/// comparable across models.
fn relative_gap(lower: Option<f64>, upper: Option<f64>) -> Option<f64> {
    match (lower, upper) {
        (Some(lo), Some(hi)) if lo.is_finite() && hi.is_finite() => {
            let g = (hi - lo).abs() / (lo.abs().max(hi.abs()) + 1e-10);
            g.is_finite().then_some(g)
        }
        _ => None,
    }
}

/// Determine the [`TerminationStatus`] from BARON's results-file termination banner
/// (BARON manual, Section 5.2 "Termination messages, model and solver statuses").
///
/// BARON prints exactly one `*** ... ***` banner at the end of `res.lst`. We map
/// it to a termination reason. `Normal completion` carries no limit/error reason,
/// so it (and any unrecognized output) defers to the numeric model status via
/// [`model_status_termination`]. Whether a usable point exists is tracked
/// separately via [`PrimalStatus`], so a limit- or heuristic-terminated run still
/// keeps its incumbent.
fn map_status(res: &str, model_status: i64) -> TerminationStatus {
    let log = res.to_ascii_lowercase();
    let has = |needle: &str| log.contains(needle);

    if has("nodes in memory") {
        // *** Max. allowable nodes in memory reached ***
        TerminationStatus::NodeLimit
    } else if has("bar iterations") {
        // *** Max. allowable BaR iterations reached ***
        TerminationStatus::IterationLimit
    } else if has("time exceeded") {
        // *** Max. allowable time exceeded ***
        TerminationStatus::TimeLimit
    } else if has("numerically sensitive") {
        // *** Problem is numerically sensitive ***
        TerminationStatus::NumericError
    } else if has("search interrupted by user") || has("access violation") {
        // *** Search interrupted by user *** / *** ... access violation ... ***
        TerminationStatus::Interrupted
    } else if has("heuristic termination") {
        // *** Heuristic termination ***, feasible found, global optimality not
        // guaranteed (DeltaTerm).
        TerminationStatus::Interrupted
    } else if has("insufficient memory") {
        // *** Insufficient Memory for Data structures ***
        TerminationStatus::Other("baron_insufficient_memory".into())
    } else if has("appropriate variable bounds") {
        // *** User did not provide appropriate variable bounds ***, relaxation
        // bounds may be invalid, so neither globality nor infeasibility is
        // guaranteed. Feasibility, if any, is still reflected in the primal status.
        TerminationStatus::Other("baron_missing_bounds".into())
    } else {
        // *** Normal completion *** or no recognizable banner.
        // The model status is the authoritative outcome.
        model_status_termination(model_status)
    }
}

/// Outcome implied by BARON's numeric model status, used on normal completion or
/// when `res.lst` carries no recognizable termination banner. `1`=optimal,
/// `2`=infeasible, `3`=unbounded, `4`=intermediate feasible (a normal completion
/// that reports a feasible point is a solved global optimum), `5`=unknown.
fn model_status_termination(model_status: i64) -> TerminationStatus {
    match model_status {
        1 | 4 => TerminationStatus::Optimal,
        2 => TerminationStatus::Infeasible,
        3 => TerminationStatus::Unbounded,
        n => TerminationStatus::Other(format!("baron_model_status_{n}")),
    }
}

/// Parse the primal solutions from BARON's results file (`res.lst`), best first.
///
/// When BARON enumerates a pool (`NumSol > 1`) it prints the distinct points
/// after `*** Normal completion ***` as numeric blocks (see
/// [`parse_solution_pool`]). Each has its own objective, but in the order it
/// found them. We need to sort the pool by objective per the
/// optimization `sense`, so index `0` is the incumbent.
fn parse_results(res: &str, var_order: &[VarId], sense: ObjectiveSense) -> Vec<SolutionPoint> {
    let mut pool = parse_solution_pool(res, var_order);
    if pool.is_empty() {
        return parse_best_table(res).into_iter().collect();
    }
    pool.sort_by(|a, b| {
        let ord = match sense {
            ObjectiveSense::Maximize => b.objective.partial_cmp(&a.objective),
            ObjectiveSense::Minimize => a.objective.partial_cmp(&b.objective),
        };
        ord.unwrap_or(std::cmp::Ordering::Equal)
    });
    pool
}

/// Parse the enumerated solution pool BARON prints after
/// `*** Normal completion ***`:
///
/// ```text
///  >>> Objective value is:        <obj>
///  >>> Corresponding solution vector is:
///  >>> Variable no.       Value
///  >>>      1             <v1>
///  >>>      2             <v2>
/// ```
///
/// one block per distinct point, best first. Variable numbers are 1-based
/// positions in the `.bar` declaration order, so `var_order[k - 1]` is the
/// `VarId` for "Variable no. `k`". Only the post-completion region is scanned.
///
/// Returns empty when BARON printed no pool (e.g. a time-limited run).
fn parse_solution_pool(res: &str, var_order: &[VarId]) -> Vec<SolutionPoint> {
    let Some(pos) = res.find("Normal completion") else {
        return Vec::new();
    };
    let strip = |l: &str| l.trim_start().trim_start_matches(">>>").trim().to_string();

    let mut out: Vec<SolutionPoint> = Vec::new();
    let mut last_obj: Option<f64> = None;
    let mut lines = res[pos..].lines().peekable();

    while let Some(line) = lines.next() {
        let t = strip(line);
        if t.starts_with("Objective value is") {
            last_obj = t.rsplit(':').next().and_then(parse_baron_float);
        } else if t.starts_with("Corresponding dual solution")
            || t.starts_with("The best solution found")
        {
            break;
        } else if t.starts_with("Corresponding solution vector") {
            let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();
            while let Some(peek) = lines.peek() {
                let p = strip(peek);
                if p.is_empty()
                    || p.starts_with("Objective value is")
                    || p.starts_with("Corresponding")
                    || p.starts_with("The best solution found")
                {
                    break;
                }
                lines.next();
                let parts: Vec<&str> = p.split_whitespace().collect();
                if let (Some(k), Some(val)) = (
                    parts.first().and_then(|s| s.parse::<usize>().ok()),
                    parts.get(1).and_then(|s| parse_baron_float(s)),
                ) {
                    if (1..=var_order.len()).contains(&k) {
                        primal.insert(var_order[k - 1], val);
                    }
                }
            }
            if !primal.is_empty() {
                out.push(SolutionPoint { primal, objective: last_obj });
            }
        }
    }
    out
}

/// `(dual, reduced_costs, soc_prices)` parsed from BARON's dual section.
/// `soc_prices` keys the raw price of each appended SOC quadratic row by cone
/// index (its sign row is skipped).
type DualParts = (FxHashMap<ConstraintId, f64>, FxHashMap<VarId, f64>, FxHashMap<usize, f64>);

/// Parse the dual solution BARON prints
///
/// Variable marginals are reduced costs, keyed by 1-based position in the
/// `.bar` declaration order (`var_order`, as in [`parse_solution_pool`]).
/// Constraint prices are duals, keyed by 1-based position in the `EQUATIONS`
/// order, which is the `ConstraintId` index plus one; positions past
/// `con_order` are the appended SOC `(quadratic row, sign row)` pairs in cone
/// order. Values are passed through with BARON's sign convention.
///
/// Returns empty maps when the section is absent (e.g. `WantDual` off, or
/// BARON reported "No dual information is available").
fn parse_dual_solution(res: &str, var_order: &[VarId], con_order: &[ConstraintId]) -> DualParts {
    let mut dual: FxHashMap<ConstraintId, f64> = FxHashMap::default();
    let mut reduced_costs: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut soc_prices: FxHashMap<usize, f64> = FxHashMap::default();
    // BARON may print the dual vector more than once (e.g. for a solution
    // found during preprocessing and again after `*** Normal completion ***`);
    // the last block is the one for the final best point.
    let Some(pos) = res.rfind("Corresponding dual solution vector") else {
        return (dual, reduced_costs, soc_prices);
    };
    let strip = |l: &str| l.trim_start().trim_start_matches(">>>").trim().to_string();

    // Marginals first, then prices once the `Constraint no. Price` header is
    // seen. Both subsections are optional
    let mut in_prices = false;
    for line in res[pos..].lines().skip(1) {
        let t = strip(line);
        if t.is_empty() || t.starts_with("The best solution found") {
            break;
        }
        if t.contains("Price") {
            in_prices = true;
            continue;
        }
        if t.contains("Marginal") {
            continue;
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        let (Some(k), Some(val)) = (
            parts.first().and_then(|s| s.parse::<usize>().ok()),
            parts.get(1).and_then(|s| parse_baron_float(s)),
        ) else {
            break;
        };
        if k == 0 {
            continue;
        }
        if in_prices {
            if let Some(&id) = con_order.get(k - 1) {
                *dual.entry(id).or_insert(0.0) += val;
            } else {
                // Past the ConstraintId rows: the appended SOC pairs, two
                // rows per cone. Keep the quadratic row's price, skip the
                // sign row's.
                let rel = k - con_order.len() - 1;
                if rel % 2 == 0 {
                    soc_prices.insert(rel / 2, val);
                }
            }
        } else if k <= var_order.len() {
            reduced_costs.insert(var_order[k - 1], val);
        }
    }
    (dual, reduced_costs, soc_prices)
}

/// Parse the single `"The best solution found is:"` table: rows of
/// `x{VarId} xlo xbest xup`, with the value in the third whitespace column and a
/// trailing `"... objective value of: X"` line. Returns `None` when the banner
/// is absent.
fn parse_best_table(res: &str) -> Option<SolutionPoint> {
    let pos = res.find("The best solution found")?;
    let mut primal: FxHashMap<VarId, f64> = FxHashMap::default();
    let mut objective = None;
    let mut started = false;
    for line in res[pos..].lines().skip(1) {
        if line.contains("objective value of") {
            objective = line.split_whitespace().last().and_then(parse_baron_float);
            break;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if let (Some(&p0), Some(&p2)) = (parts.first(), parts.get(2)) {
            if let (Some(idx), Some(val)) = (extract_index(p0), parse_baron_float(p2)) {
                primal.insert(VarId(idx), val);
                started = true;
                continue;
            }
        }
        if started {
            break;
        }
    }
    (!primal.is_empty() || objective.is_some()).then_some(SolutionPoint { primal, objective })
}

/// Recover the variable index from a synthetic `x{i}` name by reading its digits.
fn extract_index(name: &str) -> Option<u32> {
    let digits: String =
        name.chars().skip_while(|c| !c.is_ascii_digit()).take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Parse a BARON-formatted float, tolerating its infinity sentinels and the
/// leading-decimal-point form it prints for values in `(-1, 1)` (e.g. `-.9999`).
fn parse_baron_float(s: &str) -> Option<f64> {
    match s.trim() {
        "" => None,
        "inf" | "Inf" | "+inf" | "+Inf" => Some(f64::INFINITY),
        "-inf" | "-Inf" => Some(f64::NEG_INFINITY),
        other => {
            if let Some(rest) = other.strip_prefix("-.") {
                format!("-0.{rest}").parse().ok()
            } else if let Some(rest) = other.strip_prefix('.') {
                format!("0.{rest}").parse().ok()
            } else {
                other.parse().ok()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;

    use super::*;

    fn render(model: &Model) -> String {
        build_bar(model, &BaronOptions::default()).expect("build_bar").0
    }

    #[test]
    fn lp_emits_minimize_and_positive_vars() {
        let m = Model::new("lp");
        variable!(m, 0.0 <= x <= 10.0);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, c, x + y <= 5.0);
        objective!(m, Min, x + 2.0 * y);
        let bar = render(&m);
        assert!(bar.contains("POSITIVE_VARIABLES x0, x1;"), "{bar}");
        assert!(bar.contains("OBJ: minimize"), "{bar}");
        assert!(bar.contains("EQUATIONS c0;"), "{bar}");
        assert!(bar.contains("<= 5"), "{bar}");
        assert!(bar.contains("UPPER_BOUNDS{"), "{bar}");
    }

    #[test]
    fn free_variable_emits_lower_and_upper_bounds() {
        let m = Model::new("free");
        variable!(m, -5.0 <= x <= 5.0);
        objective!(m, Min, x * x);
        let bar = render(&m);
        assert!(bar.contains("VARIABLES x0;"), "{bar}");
        assert!(bar.contains("LOWER_BOUNDS{"), "{bar}");
        assert!(bar.contains("x0: -5;"), "{bar}");
        assert!(bar.contains("x0: 5;"), "{bar}");
    }

    #[test]
    fn nlp_emits_exp_and_log() {
        let m = Model::new("nlp");
        variable!(m, 0.1 <= x <= 10.0);
        objective!(m, Min, (1.0 + x).log() + x.exp());
        let bar = render(&m);
        assert!(bar.contains("log("), "{bar}");
        assert!(bar.contains("exp("), "{bar}");
    }

    #[test]
    fn minlp_partitions_binary_integer_and_continuous() {
        let m = Model::new("minlp");
        variable!(m, b, Bin);
        variable!(m, 0.0 <= n <= 5.0, Int);
        variable!(m, 0.0 <= y <= 10.0);
        constraint!(m, budget, b + n + y <= 8.0);
        objective!(m, Max, (1.0 + y).log() + 2.0 * b + n);
        let bar = render(&m);
        assert!(bar.contains("BINARY_VARIABLES x0;"), "{bar}");
        assert!(bar.contains("INTEGER_VARIABLES x1;"), "{bar}");
        assert!(bar.contains("POSITIVE_VARIABLES x2;"), "{bar}");
        assert!(bar.contains("OBJ: maximize"), "{bar}");
    }

    #[test]
    fn explicit_soc_emits_squared_rows_and_sign_row() {
        let m = Model::new("socp");
        variable!(m, -10.0 <= x <= 10.0);
        variable!(m, -10.0 <= y <= 10.0);
        variable!(m, t >= 0.0);
        m.add_soc_constraint("cone", [x, y], t);
        constraint!(m, c, x + y >= 1.0);
        objective!(m, Min, t);
        assert_eq!(m.kind(), ModelKind::SOCP);

        let bar = render(&m);
        assert!(bar.contains("EQUATIONS c0, soc0, soc0_sign;"), "declares SOC rows last:\n{bar}");
        assert!(bar.contains("soc0: "), "emits SOC row:\n{bar}");
        assert!(bar.contains(")^2"), "squares members:\n{bar}");
        assert!(bar.contains(")^2 <= 0;"), "row against 0:\n{bar}");
        assert!(bar.contains("soc0_sign: "), "emits sign row:\n{bar}");
        assert!(bar.contains(">= 0;"), "sign row nonneg:\n{bar}");
    }

    #[test]
    fn abs_reformulated_as_square_root() {
        let m = Model::new("absbar");
        variable!(m, -10.0 <= x <= 10.0);
        objective!(m, Min, x.abs());
        let bar = render(&m);
        // BARON has no abs(), reformulate |x| = (x^2)^(1/2).
        assert!(bar.contains(") ^ 2) ^ 0.5)"), "expected abs rewrite:\n{bar}");
        assert!(!bar.contains("abs("), "must not emit a literal abs():\n{bar}");
    }

    #[test]
    fn integer_power_uses_caret() {
        let m = Model::new("pow");
        variable!(m, -10.0 <= x <= 10.0);
        objective!(m, Min, x.powi(3));
        let bar = render(&m);
        assert!(bar.contains(" ^ 3)"), "expected caret power:\n{bar}");
    }

    #[test]
    fn constant_base_uses_native_caret() {
        let m = Model::new("cbpow");
        variable!(m, 0.0 <= x <= 5.0);
        let two = Expr::constant(x.arena, 2.0);
        objective!(m, Min, two.pow(x));
        let bar = render(&m);
        assert!(bar.contains("2 ^ x0"), "expected native b^x:\n{bar}");
        assert!(!bar.contains("exp("), "constant base must not rewrite to exp/log:\n{bar}");
    }

    #[test]
    fn variable_exponent_rewrites_to_exp_log() {
        let m = Model::new("vpow");
        variable!(m, 0.1 <= x <= 10.0);
        variable!(m, 0.1 <= y <= 10.0);
        objective!(m, Min, x.pow(y));
        let bar = render(&m);
        assert!(bar.contains("exp("), "{bar}");
        assert!(bar.contains("log("), "{bar}");
        assert!(!bar.contains('^'), "must not emit caret for variable exponent:\n{bar}");
    }

    #[test]
    fn quadratic_constraint_keeps_rhs() {
        let m = Model::new("qcp");
        variable!(m, 0.0 <= x <= 5.0);
        variable!(m, 0.0 <= y <= 5.0);
        constraint!(m, xy, x * y <= 4.0);
        objective!(m, Min, x + y);
        let bar = render(&m);
        assert!(bar.contains("x0") && bar.contains("x1"), "{bar}");
        assert!(bar.contains("<= 4;"), "{bar}");
    }

    #[test]
    fn feasibility_problem_minimizes_zero() {
        let m = Model::new("feas");
        variable!(m, 0.0 <= x <= 1.0);
        constraint!(m, c, x <= 1.0);
        let bar = render(&m);
        assert!(bar.contains("OBJ: minimize 0;"), "{bar}");
    }

    #[test]
    fn sin_is_rejected() {
        let m = Model::new("trig");
        variable!(m, -1.0 <= x <= 1.0);
        objective!(m, Min, x.sin());
        let err = build_bar(&m, &BaronOptions::default()).unwrap_err();
        match err {
            SolverError::Backend(msg) => assert!(msg.contains("sin"), "{msg}"),
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    #[test]
    fn semicontinuous_is_rejected() {
        let m = Model::new("semi");
        variable!(m, x <= 10.0, SemiCont(1.0));
        objective!(m, Min, x);
        let err = build_bar(&m, &BaronOptions::default()).unwrap_err();
        assert!(matches!(err, SolverError::Backend(_)));
    }

    #[test]
    fn constant_constraint_is_rejected() {
        let m = Model::new("constc");
        variable!(m, 0.0 <= x <= 5.0);
        // x - x folds to a constant left-hand side, which BARON would reject.
        constraint!(m, trivial, x - x <= 1.0);
        objective!(m, Min, x);
        let err = build_bar(&m, &BaronOptions::default()).unwrap_err();
        match err {
            SolverError::Backend(msg) => assert!(msg.contains("no variables"), "{msg}"),
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    #[test]
    fn binary_fixed_to_one_emits_lower_bound() {
        let m = Model::new("fix1");
        variable!(m, b, Bin);
        m.fix(b, 1.0);
        objective!(m, Min, b);
        let bar = render(&m);
        assert!(bar.contains("BINARY_VARIABLES x0;"), "{bar}");
        // lb=1 differs from the binary default 0, so it must be emitted.
        assert!(bar.contains("LOWER_BOUNDS{"), "{bar}");
        assert!(bar.contains("x0: 1;"), "fixed-to-1 binary must pin lb:\n{bar}");
    }

    #[test]
    fn binary_fixed_to_zero_emits_upper_bound() {
        let m = Model::new("fix0");
        variable!(m, b, Bin);
        m.fix(b, 0.0);
        objective!(m, Min, b);
        let bar = render(&m);
        // ub=0 differs from the binary default 1, so it must be emitted.
        assert!(bar.contains("UPPER_BOUNDS{"), "{bar}");
        assert!(bar.contains("x0: 0;"), "fixed-to-0 binary must pin ub:\n{bar}");
    }

    #[test]
    fn starting_point_emitted_when_initial_set() {
        let m = Model::new("start");
        variable!(m, 0.0 <= x <= 10.0);
        m.set_initial(x, 3.5);
        objective!(m, Min, x * x);
        let bar = render(&m);
        assert!(bar.contains("STARTING_POINT{"), "{bar}");
        assert!(bar.contains("x0: 3.5;"), "{bar}");
    }

    #[test]
    fn termination_from_banner() {
        // `map_status(res, model_status)`. The results-file banner
        // (manual, Section 5.2) drives the termination reason,
        // the model status is the fallback.
        let nc = "*** Normal completion ***";
        assert_eq!(map_status(nc, 1), TerminationStatus::Optimal);
        assert_eq!(map_status(nc, 2), TerminationStatus::Infeasible);
        assert_eq!(map_status(nc, 3), TerminationStatus::Unbounded);
        // Limit / interrupt banners are authoritative regardless of model status,
        // and a feasible incumbent is kept via the primal status.
        assert_eq!(
            map_status("*** Max. allowable nodes in memory reached ***", 4),
            TerminationStatus::NodeLimit
        );
        assert_eq!(
            map_status("*** Max. allowable BaR iterations reached ***", 4),
            TerminationStatus::IterationLimit
        );
        assert_eq!(
            map_status("*** Max. allowable time exceeded ***", 4),
            TerminationStatus::TimeLimit
        );
        assert_eq!(
            map_status("*** Problem is numerically sensitive ***", 4),
            TerminationStatus::NumericError
        );
        assert_eq!(
            map_status("*** Search interrupted by user ***", 4),
            TerminationStatus::Interrupted
        );
        assert_eq!(map_status("*** Heuristic termination ***", 4), TerminationStatus::Interrupted);
        assert_eq!(
            map_status("*** User did not provide appropriate variable bounds ***", 4),
            TerminationStatus::Other("baron_missing_bounds".into())
        );
        // No recognizable banner falls back to the model status.
        assert_eq!(map_status("", 1), TerminationStatus::Optimal);
    }

    #[test]
    fn parse_tim_picks_objective_by_sense() {
        // name ncon nvar a b lower upper solver model c iters nodeopt ... wall
        let tim = "m 1 2 0 0 1.5 9.5 1 1 0 42 7 0 0 0.42";
        let r = parse_solution(
            tim,
            "",
            ObjectiveSense::Minimize,
            std::time::Duration::ZERO,
            None,
            &[],
            &[],
            &[],
        );
        assert_eq!(r.termination, TerminationStatus::Optimal);
        assert_eq!(r.objective(), Some(9.5)); // upper bound for minimize
        assert_eq!(r.iterations, 42); // branch-and-reduce iterations from tim[10]
        let r = parse_solution(
            tim,
            "",
            ObjectiveSense::Maximize,
            std::time::Duration::ZERO,
            None,
            &[],
            &[],
            &[],
        );
        assert_eq!(r.objective(), Some(1.5)); // lower bound for maximize
    }

    #[test]
    fn parse_res_extracts_primal() {
        let res = "\
junk line
The best solution found is:

  variable     xlo      xbest    xup
  x0           0.0      1.25     10
  x1           0.0      3.50     10

The above solution has an objective value of:    0.0
";
        let blocks = parse_results(res, &[VarId(0), VarId(1)], ObjectiveSense::Minimize);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].primal.get(&VarId(0)), Some(&1.25));
        assert_eq!(blocks[0].primal.get(&VarId(1)), Some(&3.5));
    }

    #[test]
    fn parse_res_extracts_multiple_solutions() {
        let res = "\
                         *** Normal completion ***

 >>> Objective value is:           0.0000000000000000000000
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1             1.0000000010231691049967
 >>>       2             1.0000000010231691049967

 >>> Objective value is:           0.0000000000000000000000
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1            -.99999999996972754878755
 >>>       2            0.99999999997690125486116

 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             0.0000000000000000000000

The best solution found is:

  variable     xlo      xbest                          xup
  x0           -2       1.00000000102316910499667e+00  2
  x1           -2       1.00000000102316910499667e+00  2

The above solution has an objective value of:  0.0000000000000000000000
";
        let blocks = parse_results(res, &[VarId(0), VarId(1)], ObjectiveSense::Minimize);
        assert_eq!(blocks.len(), 2, "expected two solution blocks: {blocks:?}");
        assert!((blocks[0].primal[&VarId(0)] - 1.0).abs() < 1e-6);
        assert!((blocks[0].primal[&VarId(1)] - 1.0).abs() < 1e-6);
        assert!((blocks[1].primal[&VarId(0)] + 1.0).abs() < 1e-6);
        assert!((blocks[1].primal[&VarId(1)] - 1.0).abs() < 1e-6);
        assert_eq!(blocks[1].objective, Some(0.0));
    }

    #[test]
    fn parse_dual_extracts_marginals_and_prices() {
        let res = "\
                         *** Normal completion ***

 >>> Objective value is:           5.0000000000000000000000
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1             1.0000000000000000000000
 >>>       2             4.0000000000000000000000

 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             0.0000000000000000000000
 >>>       2            -.50000000000000000000000
 >>> Constraint no.             Price
 >>>       1             2.0000000000000000000000
 >>>       2            -.25000000000000000000000

The best solution found is:

  variable     xlo      xbest    xup
  x0           0.0      1.0      10
  x1           0.0      4.0      10

The above solution has an objective value of:    5.0
";
        let (dual, rc, _soc) =
            parse_dual_solution(res, &[VarId(0), VarId(1)], &[ConstraintId(0), ConstraintId(1)]);
        assert_eq!(rc.get(&VarId(0)), Some(&0.0));
        assert_eq!(rc.get(&VarId(1)), Some(&-0.5));
        assert_eq!(dual.get(&ConstraintId(0)), Some(&2.0));
        assert_eq!(dual.get(&ConstraintId(1)), Some(&-0.25));
        assert_eq!(dual.len(), 2);
        assert_eq!(rc.len(), 2);
    }

    #[test]
    fn parse_dual_takes_last_block_when_printed_twice() {
        let res = "\
 Solving bounding LP
 >>> Preprocessing found feasible solution
 >>> Objective value is:           6.0
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1             6.0000000000000000000000

 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             9.0000000000000000000000
 >>> Constraint no.            Price
 >>>       1             9.0000000000000000000000


                         *** Normal completion ***

 >>> Objective value is:           5.0
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1             5.0000000000000000000000

 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             0.0000000000000000000000
 >>> Constraint no.            Price
 >>>       1             1.0000000000000000000000

The best solution found is:
";
        let (dual, rc, _soc) = parse_dual_solution(res, &[VarId(0)], &[ConstraintId(0)]);
        assert_eq!(rc.get(&VarId(0)), Some(&0.0));
        assert_eq!(dual.get(&ConstraintId(0)), Some(&1.0));
    }

    #[test]
    fn parse_dual_folds_range_rows_onto_one_constraint() {
        let res = "\
 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             0.0000000000000000000000
 >>> Constraint no.             Price
 >>>       1             2.0000000000000000000000
 >>>       2            -.50000000000000000000000
";
        let (dual, _rc, _soc) =
            parse_dual_solution(res, &[VarId(0)], &[ConstraintId(0), ConstraintId(0)]);
        assert_eq!(dual.len(), 1);
        assert_eq!(dual.get(&ConstraintId(0)), Some(&1.5)); // 2.0 + (-0.5)
    }

    #[test]
    fn parse_dual_maps_soc_rows_past_constraint_order() {
        let res = "\
 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             0.0000000000000000000000
 >>> Constraint no.             Price
 >>>       1             2.0000000000000000000000
 >>>       2            -.75000000000000000000000
 >>>       3             0.0000000000000000000000
";
        let (dual, _rc, soc) = parse_dual_solution(res, &[VarId(0)], &[ConstraintId(0)]);
        assert_eq!(dual.get(&ConstraintId(0)), Some(&2.0));
        assert_eq!(dual.len(), 1);
        assert_eq!(soc.get(&0), Some(&-0.75));
        assert_eq!(soc.len(), 1);
    }

    #[test]
    fn parse_dual_absent_section_leaves_maps_empty() {
        let res = "\
                         *** Normal completion ***

 >>> Objective value is:           5.0
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1             1.0

No dual information is available.

The best solution found is:
";
        let (dual, rc, soc) = parse_dual_solution(res, &[VarId(0)], &[ConstraintId(0)]);
        assert!(dual.is_empty());
        assert!(rc.is_empty());
        assert!(soc.is_empty());
    }

    #[test]
    fn parse_solution_populates_duals_alongside_pool() {
        let tim = "m 1 2 0 0 0 0 1 1 0 42 7 0 0 0.42";
        let res = "\
                         *** Normal completion ***

 >>> Objective value is:           0.0000000000000000000000
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1             1.0000000000000000000000
 >>>       2             1.0000000000000000000000

 >>> Objective value is:           0.0000000000000000000000
 >>> Corresponding solution vector is:
 >>> Variable no.              Value
 >>>       1            -.99999999996972754878755
 >>>       2            0.99999999997690125486116

 >>> Corresponding dual solution vector is:
 >>> Variable no.              Marginal
 >>>       1             0.0000000000000000000000
 >>>       2             1.5000000000000000000000
 >>> Constraint no.             Price
 >>>       1            -3.0000000000000000000000

The best solution found is:

  variable     xlo      xbest    xup
  x0           -2       1.0      2
  x1           -2       1.0      2

The above solution has an objective value of:  0.0
";
        let r = parse_solution(
            tim,
            res,
            ObjectiveSense::Minimize,
            std::time::Duration::ZERO,
            None,
            &[VarId(0), VarId(1)],
            &[ConstraintId(0)],
            &[],
        );
        assert_eq!(r.result_count(), 2);
        assert_eq!(r.reduced_costs.get(&VarId(1)), Some(&1.5));
        assert_eq!(r.dual.get(&ConstraintId(0)), Some(&-3.0));
        assert_eq!(r.dual.len(), 1);
    }

    #[test]
    fn no_solution_node_minus_three_leaves_primal_empty() {
        // model=1 (optimal) but nodeopt = -3 => no solution vector. We surface a
        // single objective-only point, but it carries no primal data,
        // so it must NOT count as a usable solution.
        let tim = "m 1 1 0 0 0 0 1 1 0 0 -3 0 0 0.01";
        let res = "The best solution found is:\n\n\n  x0  0  9.9\n";
        let r = parse_solution(
            tim,
            res,
            ObjectiveSense::Minimize,
            std::time::Duration::ZERO,
            None,
            &[VarId(0)],
            &[],
            &[],
        );
        assert_eq!(r.result_count(), 1);
        assert!(
            r.solution(0).unwrap().primal.is_empty(),
            "nodeopt -3 must skip primal: {:?}",
            r.solution(0)
        );
        assert_eq!(r.primal_status, PrimalStatus::NoSolution);
        assert!(!r.has_solution(), "nodeopt -3 must not report a usable solution");
        let primal = r.primal().expect("objective-only point is present");
        assert!(primal.is_empty(), "nodeopt -3 must expose no primal values: {primal:?}");
        assert!(r.value(VarId(0)).is_none(), "nodeopt -3 must not yield a variable value");
    }
}
