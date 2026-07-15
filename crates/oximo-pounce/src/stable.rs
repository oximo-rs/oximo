//! Finite-difference path: POUNCE's `builder` surface. We supply objective and
//! constraint values only, and POUNCE finite-differences the gradient/Jacobian
//! and uses a limited-memory (L-BFGS) Hessian. The value oracle lives in a
//! shared cell so a persistent handle can reuse the compiled tapes across solves.

use std::cell::RefCell;
use std::rc::Rc;

use oximo_core::Model;
use oximo_solver::SolverError;
use pounce_rs::IpoptApplication;
use pounce_rs::builder::{Nlp, Problem};

use crate::options::{PounceOptionValue, PounceOptions};
use crate::translate::{
    Outcome, Prepared, WarmStart, apply_options, map_status, mu_strategy_str, print_level,
};
use crate::values::ValueOracle;

/// The resident value oracle, shared between the handle and the `Problem`.
pub(crate) type Oracle = Rc<RefCell<ValueOracle>>;

// `Result` for signature parity with the exact path, which can fail to build.
#[expect(clippy::unnecessary_wraps)]
pub(crate) fn build(model: &Model) -> Result<Oracle, SolverError> {
    Ok(Rc::new(RefCell::new(ValueOracle::new(model))))
}

/// Reuse the resident tapes when the model's expression graph is unchanged,
/// refreshing only the parameter snapshot.
pub(crate) fn try_reuse(oracle: &Oracle, model: &Model) -> bool {
    let mut o = oracle.borrow_mut();
    if o.matches(model) {
        o.refresh_params(model);
        true
    } else {
        false
    }
}

struct OximoProblem {
    oracle: Oracle,
    sign: f64,
    m: usize,
}

impl Problem for OximoProblem {
    fn objective(&self, x: &[f64]) -> f64 {
        self.sign * self.oracle.borrow().objective(x)
    }

    fn n_constraints(&self) -> usize {
        self.m
    }

    fn constraints(&self, x: &[f64], g: &mut [f64]) {
        let o = self.oracle.borrow();
        for (i, out) in g.iter_mut().enumerate() {
            *out = o.constraint(i, x);
        }
    }
}

pub(crate) fn run(
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
    let problem = OximoProblem { oracle: Rc::clone(oracle), sign: prep.sign, m };

    // The builder finite-difference path can only warm-start the primal point.
    let x0 = warm.map_or_else(|| prep.x0.clone(), |w| w.x.clone());
    let mut nlp = Nlp::new(problem)
        .var_bounds(&prep.x_l, &prep.x_u)
        .constraint_bounds(&prep.g_l, &prep.g_u)
        .x0(&x0)
        .option_int("print_level", print_level(opts));

    if let Some(tol) = opts.tol {
        nlp = nlp.option_num("tol", tol);
    }
    if let Some(n) = opts.max_iter {
        nlp = nlp.option_int("max_iter", i32::try_from(n).unwrap_or(i32::MAX));
    }
    if let Some(limit) = opts.universal.time_limit {
        nlp = nlp.option_num("max_cpu_time", limit.as_secs_f64());
    }
    if let Some(s) = opts.mu_strategy {
        nlp = nlp.option_str("mu_strategy", mu_strategy_str(s));
    }
    for &(name, v) in opts.num_opts() {
        nlp = nlp.option_num(name, v);
    }
    for &(name, v) in opts.int_opts() {
        nlp = nlp.option_int(name, v);
    }
    for (name, v) in opts.str_opts() {
        nlp = nlp.option_str(name, v);
    }
    for &(name, v) in opts.bool_opts() {
        nlp = nlp.option_str(name, if v { "yes" } else { "no" });
    }
    for (name, value) in &opts.extra {
        nlp = match value {
            PounceOptionValue::Num(v) => nlp.option_num(name, *v),
            PounceOptionValue::Int(v) => nlp.option_int(name, *v),
            PounceOptionValue::Str(v) => nlp.option_str(name, v),
            PounceOptionValue::Bool(v) => nlp.option_str(name, if *v { "yes" } else { "no" }),
        };
    }

    let sol = nlp.solve();
    let termination = map_status(sol.status);
    let warm = sol.success.then(|| WarmStart {
        x: sol.x.clone(),
        z_l: Vec::new(),
        z_u: Vec::new(),
        lambda: sol.multipliers.clone(),
    });
    Ok(Outcome {
        termination,
        x: sol.x,
        lambda: sol.multipliers,
        reduced: None,
        objective: Some(sol.objective),
        iterations: 0,
        warm,
    })
}
