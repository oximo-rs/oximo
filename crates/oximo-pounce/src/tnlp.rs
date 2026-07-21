//! The shared POUNCE `TNLP` adapter and solve driver.
//!
//! Both derivative paths ([`crate::exact`] on enzyme, [`crate::hybrid`] on
//! stable) plug a [`DerivativeOracle`] into the one [`OximoTnlp`] adapter and
//! solve through [`run`], so warm starts, option handling, statistics, and log
//! capture behave identically regardless of where derivatives come from.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;

use oximo_solver::SolverError;
use pounce_rs::pounce_nlp::solve_statistics::SolveStatistics;
use pounce_rs::{
    ApplicationReturnStatus, BoundsInfo, Index, IndexStyle, IpoptApplication, IpoptCq, IpoptData,
    NlpInfo, Solution, SparsityRequest, StartingPoint, TNLP,
};

use crate::options::PounceOptions;
use crate::translate::{Outcome, Prepared, WarmStart, apply_options, map_status, set_str};

/// A derivative source for [`OximoTnlp`].
///
/// Values and derivatives are all in POUNCE's minimization sense before the
/// maximize sign flip.
pub(crate) trait DerivativeOracle {
    fn num_variables(&self) -> usize;
    fn num_constraints(&self) -> usize;
    /// Sparse `(constraint, variable)` Jacobian pattern in row-major order.
    fn jacobian_structure(&self) -> &[(usize, usize)];
    /// Sorted lower-triangle Hessian-of-the-Lagrangian pattern.
    fn hessian_structure(&self) -> &[(usize, usize)];
    /// Whether [`Self::eval_hessian_lagrangian`] provides exact values for
    /// [`Self::hessian_structure`].
    fn has_exact_hessian(&self) -> bool;
    fn eval_objective(&mut self, x: &[f64]) -> f64;
    fn eval_objective_gradient(&mut self, x: &[f64], grad: &mut [f64]);
    fn eval_constraints(&mut self, x: &[f64], g: &mut [f64]);
    /// Jacobian values aligned with [`Self::jacobian_structure`].
    fn eval_constraint_jacobian(&mut self, x: &[f64], vals: &mut [f64]);
    /// Hessian values aligned with [`Self::hessian_structure`].
    fn eval_hessian_lagrangian(
        &mut self,
        x: &[f64],
        obj_factor: f64,
        lambda: &[f64],
        vals: &mut [f64],
    );
}

fn to_index(v: usize) -> Index {
    Index::try_from(v).expect("index exceeds i32")
}

/// Drive one POUNCE solve of `oracle` over the snapshot in `prep`.
///
/// Applies options onto the live application.
/// Warm-starts from `warm` when given, and reads iteration
/// statistics back off the application after the solve.
pub(crate) fn run<O: DerivativeOracle + 'static>(
    oracle: &Rc<RefCell<O>>,
    prep: &Prepared,
    opts: &PounceOptions,
    warm: Option<&WarmStart>,
) -> Result<Outcome, SolverError> {
    let tnlp = Rc::new(RefCell::new(OximoTnlp {
        oracle: Rc::clone(oracle),
        sign: prep.sign,
        x_l: prep.x_l.clone(),
        x_u: prep.x_u.clone(),
        g_l: prep.g_l.clone(),
        g_u: prep.g_u.clone(),
        x0: prep.x0.clone(),
        warm: warm.cloned(),
        captured: None,
    }));

    let mut app = IpoptApplication::new();
    app.initialize().map_err(|e| SolverError::Backend(format!("pounce init: {e:?}")))?;
    if !oracle.borrow().has_exact_hessian() {
        set_str(app.options_mut(), "hessian_approximation", "limited-memory")?;
    }
    apply_options(app.options_mut(), opts, warm.is_some())?;

    let status = app.optimize_tnlp(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>);
    let termination = map_status(status);
    let stats = app.statistics();
    let iterations = u64::try_from(stats.iteration_count.max(0)).unwrap_or(0);
    let raw_log = (opts.universal.verbose == Some(true)).then(|| format_raw_log(&stats, status));

    let t = tnlp.borrow();
    Ok(match &t.captured {
        Some(c) => Outcome {
            termination,
            x: c.warm.x.clone(),
            lambda: c.warm.lambda.clone(),
            reduced: Some(c.reduced.clone()),
            objective: Some(c.obj),
            iterations,
            warm: Some(c.warm.clone()),
            raw_log,
        },
        None => Outcome {
            termination,
            x: Vec::new(),
            lambda: Vec::new(),
            reduced: None,
            objective: None,
            iterations,
            warm: None,
            raw_log,
        },
    })
}

/// The Ipopt-style end-of-solve report off the application's statistics
/// (values are in POUNCE's minimization sense).
fn format_raw_log(stats: &SolveStatistics, status: ApplicationReturnStatus) -> String {
    let mut log = String::new();
    let _ = writeln!(log, "Number of Iterations....: {}", stats.iteration_count);
    let _ =
        writeln!(log, "\n                                   (scaled)                 (unscaled)");
    let _ = writeln!(
        log,
        "Objective...............: {:24.16e} {:24.16e}",
        stats.final_scaled_objective, stats.final_objective
    );
    let _ = writeln!(
        log,
        "Dual infeasibility......: {:24.16e} {:24.16e}",
        stats.final_dual_inf, stats.final_unscaled_dual_inf
    );
    let _ = writeln!(
        log,
        "Constraint violation....: {:24.16e} {:24.16e}",
        stats.final_constr_viol, stats.final_unscaled_constr_viol
    );
    let _ = writeln!(
        log,
        "Complementarity.........: {:24.16e} {:24.16e}",
        stats.final_compl, stats.final_unscaled_compl
    );
    let _ = writeln!(
        log,
        "Overall NLP error.......: {:24.16e} {:24.16e}",
        stats.final_kkt_error, stats.final_unscaled_kkt_error
    );
    let _ = writeln!(log);
    let _ = writeln!(
        log,
        "Number of objective function evaluations             = {}",
        stats.num_obj_evals
    );
    let _ = writeln!(
        log,
        "Number of objective gradient evaluations             = {}",
        stats.num_obj_grad_evals
    );
    let _ = writeln!(
        log,
        "Number of constraint evaluations                     = {}",
        stats.num_constr_evals
    );
    let _ = writeln!(
        log,
        "Number of constraint Jacobian evaluations            = {}",
        stats.num_constr_jac_evals
    );
    let _ = writeln!(
        log,
        "Number of Lagrangian Hessian evaluations             = {}",
        stats.num_hess_evals
    );
    if stats.restoration_calls > 0 {
        let _ = writeln!(
            log,
            "Restoration phase calls                              = {} ({} outer / {} inner iters, {:.3} s)",
            stats.restoration_calls,
            stats.restoration_outer_iters,
            stats.restoration_inner_iters,
            stats.restoration_wall_secs
        );
    }
    let _ = writeln!(
        log,
        "Total seconds (CPU)                                  = {:.3}",
        stats.total_cpu_time_secs
    );
    let _ = writeln!(
        log,
        "Total seconds (wallclock)                            = {:.3}",
        stats.total_wallclock_time_secs
    );
    let _ = writeln!(log, "\nEXIT: {status:?}");
    log
}

/// Captured final iterate from `finalize_solution`.
struct Captured {
    warm: WarmStart,
    /// `z_l − z_u` per variable (bound multipliers), for reduced costs.
    reduced: Vec<f64>,
    obj: f64,
}

/// POUNCE `TNLP` backed by a shared derivative oracle.
/// `sign` is `-1.0` for a Maximize model.
struct OximoTnlp<O> {
    oracle: Rc<RefCell<O>>,
    sign: f64,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    warm: Option<WarmStart>,
    captured: Option<Captured>,
}

impl<O: DerivativeOracle> TNLP for OximoTnlp<O> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let e = self.oracle.borrow();
        Some(NlpInfo {
            n: to_index(e.num_variables()),
            m: to_index(e.num_constraints()),
            nnz_jac_g: to_index(e.jacobian_structure().len()),
            nnz_h_lag: to_index(e.hessian_structure().len()),
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }

    fn get_starting_point(&mut self, mut sp: StartingPoint<'_>) -> bool {
        match &self.warm {
            Some(w) => {
                sp.x.copy_from_slice(&w.x);
                if w.z_l.len() == sp.z_l.len() && w.z_u.len() == sp.z_u.len() {
                    sp.init_z = true;
                    sp.z_l.copy_from_slice(&w.z_l);
                    sp.z_u.copy_from_slice(&w.z_u);
                }
                if w.lambda.len() == sp.lambda.len() {
                    sp.init_lambda = true;
                    sp.lambda.copy_from_slice(&w.lambda);
                }
            }
            None => sp.x.copy_from_slice(&self.x0),
        }
        true
    }

    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some(self.sign * self.oracle.borrow_mut().eval_objective(x))
    }

    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad_f: &mut [f64]) -> bool {
        self.oracle.borrow_mut().eval_objective_gradient(x, grad_f);
        for g in grad_f.iter_mut() {
            *g *= self.sign;
        }
        true
    }

    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        self.oracle.borrow_mut().eval_constraints(x, g);
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let e = self.oracle.borrow();
                for (k, &(r, c)) in e.jacobian_structure().iter().enumerate() {
                    irow[k] = to_index(r);
                    jcol[k] = to_index(c);
                }
            }
            SparsityRequest::Values { values } => {
                self.oracle
                    .borrow_mut()
                    .eval_constraint_jacobian(x.expect("jacobian values need x"), values);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[f64]>,
        _new_x: bool,
        obj_factor: f64,
        lambda: Option<&[f64]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        if !self.oracle.borrow().has_exact_hessian() {
            return false;
        }
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let e = self.oracle.borrow();
                for (k, &(r, c)) in e.hessian_structure().iter().enumerate() {
                    irow[k] = to_index(r);
                    jcol[k] = to_index(c);
                }
            }
            SparsityRequest::Values { values } => {
                self.oracle.borrow_mut().eval_hessian_lagrangian(
                    x.expect("hessian values need x"),
                    self.sign * obj_factor,
                    lambda.expect("hessian values need lambda"),
                    values,
                );
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        let reduced = sol.z_l.iter().zip(sol.z_u).map(|(&zl, &zu)| zl - zu).collect::<Vec<f64>>();
        self.captured = Some(Captured {
            warm: WarmStart {
                x: sol.x.to_vec(),
                z_l: sol.z_l.to_vec(),
                z_u: sol.z_u.to_vec(),
                lambda: sol.lambda.to_vec(),
            },
            reduced,
            obj: sol.obj_value,
        });
    }
}
