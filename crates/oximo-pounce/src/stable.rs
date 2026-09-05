//! Stable-Rust path
//!
//! One resident [`HybridOracle`], two solve surfaces.
//! A model whose objective and constraints are all linear/quadratic solves on
//! POUNCE's low-level `TNLP` surface with exact derivatives.
//! A model with a nonlinear function solves through POUNCE's `builder`
//! surface instead, where POUNCE's finite differences cover what the oracle
//! cannot fill exactly, and the Hessian is limited-memory L-BFGS.

use std::cell::RefCell;
use std::rc::Rc;

use oximo_core::Model;
use oximo_solver::{DualStatus, SolverError};
use pounce_rs::IpoptApplication;
use pounce_rs::builder::{Nlp, Problem};

use crate::hybrid::HybridOracle;
use crate::options::{PounceOptionValue, PounceOptions};
use crate::tnlp::{self, DerivativeOracle};
use crate::translate::{
    Outcome, Prepared, WarmStart, apply_options, map_status, mu_strategy_str, print_level,
};

/// The resident derivative oracle, shared between the handle and the solve.
pub(crate) type Oracle = Rc<RefCell<HybridOracle>>;

// `Result` for signature parity with the exact path, which can fail to build.
#[expect(clippy::unnecessary_wraps)]
pub(crate) fn build(model: &Model) -> Result<Oracle, SolverError> {
    Ok(Rc::new(RefCell::new(HybridOracle::new(model))))
}

/// Reuse the resident oracle when the model's expression graph is unchanged,
/// re-extracting the slots at the current parameter values.
pub(crate) fn try_reuse(oracle: &Oracle, model: &Model) -> bool {
    let mut o = oracle.borrow_mut();
    if o.matches(model) {
        o.refresh(model);
        true
    } else {
        false
    }
}

/// Solve on the exact `TNLP` path when the whole model is closed-form,
/// otherwise through POUNCE's builder.
pub(crate) fn run(
    model: &Model,
    oracle: &Oracle,
    prep: &Prepared,
    opts: &PounceOptions,
    warm: Option<&WarmStart>,
) -> Result<Outcome, SolverError> {
    if oracle.borrow().all_closed_form() {
        tnlp::run(oracle, prep, opts, warm)
    } else {
        run_builder(model, oracle, prep, opts, warm)
    }
}

/// The oracle behind POUNCE's builder [`Problem`]: values from the slots, and
/// exact derivatives for whatever is closed-form (`false` hands the rest to
/// POUNCE's finite differences).
struct OximoProblem {
    oracle: Oracle,
    sign: f64,
    m: usize,
    fbbt_constraints: Vec<Option<pounce_rs::FbbtTape>>,
}

impl Problem for OximoProblem {
    fn objective(&self, x: &[f64]) -> f64 {
        self.sign * self.oracle.borrow_mut().eval_objective(x)
    }

    fn n_constraints(&self) -> usize {
        self.m
    }

    fn constraints(&self, x: &[f64], out: &mut [f64]) {
        self.oracle.borrow_mut().eval_constraints(x, out);
    }

    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
        let filled = self.oracle.borrow().try_exact_objective_gradient(x, grad);
        if filled {
            for g in grad.iter_mut() {
                *g *= self.sign;
            }
        }
        filled
    }

    fn jacobian(&self, x: &[f64], jac: &mut [f64]) -> bool {
        self.oracle.borrow().try_exact_dense_jacobian(x, jac)
    }

    fn constraint_expression(&self, index: usize) -> Option<pounce_rs::FbbtTape> {
        self.fbbt_constraints.get(index).and_then(Clone::clone)
    }
}

fn run_builder(
    model: &Model,
    oracle: &Oracle,
    prep: &Prepared,
    opts: &PounceOptions,
    warm: Option<&WarmStart>,
) -> Result<Outcome, SolverError> {
    // The builder silently discards option errors, so validate the options
    // against a fresh application first.
    let mut validator = IpoptApplication::new();
    validator.initialize().map_err(|e| SolverError::Backend(format!("pounce init: {e:?}")))?;
    apply_options(validator.options_mut(), opts, false)?;

    let m = oracle.borrow().num_constraints();
    let fbbt_constraints = if opts.effective_bool("presolve").unwrap_or(false)
        && opts.effective_bool("presolve_fbbt").unwrap_or(false)
    {
        crate::fbbt::constraint_tapes(model)
    } else {
        Vec::new()
    };
    let problem = OximoProblem { oracle: Rc::clone(oracle), sign: prep.sign, m, fbbt_constraints };

    // The builder path can only warm-start the primal point.
    let x0 = warm.map_or_else(|| prep.x0.clone(), |w| w.x.clone());
    let mut nlp = Nlp::new(problem)
        .var_bounds(&prep.x_l, &prep.x_u)
        .constraint_bounds(&prep.g_l, &prep.g_u)
        .x0(&x0)
        .option_int("print_level", print_level(opts));

    if opts.universal.verbose == Some(true) {
        nlp = nlp.capture_iterations();
    }

    if let Some(tol) = opts.tol {
        nlp = nlp.option_num("tol", tol);
    }
    if let Some(n) = opts.max_iter {
        nlp = nlp.option_int("max_iter", i32::try_from(n).unwrap_or(i32::MAX));
    }
    if let Some(limit) = opts.universal.time_limit {
        nlp = nlp.option_num("max_wall_time", limit.as_secs_f64());
    }
    if let Some(s) = opts.mu_strategy {
        nlp = nlp.option_str("mu_strategy", mu_strategy_str(s));
    }
    if let Some(algorithm) = opts.algorithm {
        nlp = nlp.option_str("algorithm", algorithm.as_str());
    }
    for &(name, v) in opts.num_opts() {
        if !crate::translate::convex_only_option(name) {
            nlp = nlp.option_num(name, v);
        }
    }
    for &(name, v) in opts.int_opts() {
        if !crate::translate::convex_only_option(name) {
            nlp = nlp.option_int(name, v);
        }
    }
    for (name, v) in opts.str_opts() {
        nlp = nlp.option_str(name, v);
    }
    for &(name, v) in opts.bool_opts() {
        if !crate::translate::convex_only_option(name) {
            nlp = nlp.option_str(name, if v { "yes" } else { "no" });
        }
    }
    for (name, value) in &opts.extra {
        if crate::translate::convex_only_option(name) {
            continue;
        }
        nlp = match value {
            PounceOptionValue::Num(v) => nlp.option_num(name, *v),
            PounceOptionValue::Int(v) => nlp.option_int(name, *v),
            PounceOptionValue::Str(v) => nlp.option_str(name, v),
            PounceOptionValue::Bool(v) => nlp.option_str(name, if *v { "yes" } else { "no" }),
        };
    }

    let sol = nlp
        .try_solve()
        .map_err(|error| SolverError::Backend(format!("pounce builder: {error}")))?;

    let termination = map_status(sol.status);
    let raw_log = (opts.universal.verbose == Some(true))
        .then(|| crate::tnlp::format_raw_log(&sol.stats, sol.status));
    let iterations = u64::try_from(sol.stats.iteration_count.max(0)).unwrap_or(0);
    let reduced = sol.z_l.iter().zip(&sol.z_u).map(|(&zl, &zu)| zl - zu).collect();
    let warm = sol.success.then(|| WarmStart {
        x: sol.x.clone(),
        z_l: sol.z_l.clone(),
        z_u: sol.z_u.clone(),
        lambda: sol.multipliers.clone(),
        sqp_working: None,
    });
    Ok(Outcome {
        termination,
        dual_status: if matches!(sol.status, pounce_rs::ApplicationReturnStatus::SolveSucceeded) {
            DualStatus::FeasiblePoint
        } else {
            DualStatus::Unknown
        },
        raw_status: format!("{:?}", sol.status),
        x: sol.x,
        lambda: sol.multipliers,
        soc_dual: Vec::new(),
        reduced: Some(reduced),
        objective: Some(sol.objective),
        iterations,
        warm,
        raw_log,
    })
}
