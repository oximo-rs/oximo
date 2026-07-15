//! Exact-derivative path (enzyme feature): drives POUNCE's low-level [`TNLP`]
//! surface with exact gradient, sparse Jacobian, and the sparse Hessian of the
//! Lagrangian from `oximo-autodiff`'s [`NlpEvaluator`].

use std::cell::RefCell;
use std::rc::Rc;

use oximo_autodiff::NlpEvaluator;
use oximo_core::Model;
use oximo_solver::SolverError;
use pounce_rs::{
    BoundsInfo, Index, IndexStyle, IpoptApplication, IpoptCq, IpoptData, IterStats, NlpInfo,
    Solution, SparsityRequest, StartingPoint, TNLP,
};

use crate::options::PounceOptions;
use crate::translate::{Outcome, Prepared, WarmStart, apply_options, map_status};

/// The resident derivative oracle, shared between the handle and the `TNLP`.
pub(crate) type Oracle = Rc<RefCell<NlpEvaluator>>;

fn to_index(v: usize) -> Index {
    Index::try_from(v).expect("index exceeds i32")
}

/// Build a fresh evaluator.
///
/// Fails only when the model has no objective, which
/// [`crate::translate::setup`] already verifies.
pub(crate) fn build(model: &Model) -> Result<Oracle, SolverError> {
    let eval =
        NlpEvaluator::new(model).map_err(|e| SolverError::Backend(format!("autodiff: {e}")))?;
    Ok(Rc::new(RefCell::new(eval)))
}

/// Reuse the resident evaluator for `model` when the structure is unchanged.
pub(crate) fn try_reuse(oracle: &Oracle, model: &Model) -> bool {
    oracle.borrow_mut().try_refresh(model)
}

pub(crate) fn run(
    oracle: &Oracle,
    prep: &Prepared,
    opts: &PounceOptions,
    warm: Option<&WarmStart>,
) -> Result<Outcome, SolverError> {
    let tnlp = Rc::new(RefCell::new(OximoTnlp {
        eval: Rc::clone(oracle),
        sign: prep.sign,
        x_l: prep.x_l.clone(),
        x_u: prep.x_u.clone(),
        g_l: prep.g_l.clone(),
        g_u: prep.g_u.clone(),
        x0: prep.x0.clone(),
        warm: warm.cloned(),
        captured: None,
        iterations: 0,
    }));

    let mut app = IpoptApplication::new();
    app.initialize().map_err(|e| SolverError::Backend(format!("pounce init: {e:?}")))?;
    apply_options(app.options_mut(), opts, warm.is_some())?;

    let status = app.optimize_tnlp(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>);
    let termination = map_status(status);

    let t = tnlp.borrow();
    Ok(match &t.captured {
        Some(c) => Outcome {
            termination,
            x: c.warm.x.clone(),
            lambda: c.warm.lambda.clone(),
            reduced: Some(c.reduced.clone()),
            objective: Some(c.obj),
            iterations: t.iterations,
            warm: Some(c.warm.clone()),
        },
        None => Outcome {
            termination,
            x: Vec::new(),
            lambda: Vec::new(),
            reduced: None,
            objective: None,
            iterations: t.iterations,
            warm: None,
        },
    })
}

/// Captured final iterate from `finalize_solution`.
struct Captured {
    warm: WarmStart,
    /// `z_l − z_u` per variable (bound multipliers), for reduced costs.
    reduced: Vec<f64>,
    obj: f64,
}

/// POUNCE `TNLP` backed by the shared derivative oracle.
/// `sign` is `-1.0` for a Maximize model.
struct OximoTnlp {
    eval: Oracle,
    sign: f64,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    warm: Option<WarmStart>,
    captured: Option<Captured>,
    iterations: u64,
}

impl TNLP for OximoTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let e = self.eval.borrow();
        Some(NlpInfo {
            n: to_index(e.num_variables()),
            m: to_index(e.num_constraints()),
            nnz_jac_g: to_index(e.jacobian_structure().len()),
            nnz_h_lag: to_index(e.hessian_lagrangian_structure().len()),
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
        Some(self.sign * self.eval.borrow().eval_objective(x))
    }

    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad_f: &mut [f64]) -> bool {
        self.eval.borrow().eval_objective_gradient(x, grad_f);
        for g in grad_f.iter_mut() {
            *g *= self.sign;
        }
        true
    }

    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        self.eval.borrow().eval_constraint(x, g);
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        let e = self.eval.borrow();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in e.jacobian_structure().iter().enumerate() {
                    irow[k] = to_index(r);
                    jcol[k] = to_index(c);
                }
            }
            SparsityRequest::Values { values } => {
                e.eval_constraint_jacobian(x.expect("jacobian values need x"), values);
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
        let e = self.eval.borrow();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in e.hessian_lagrangian_structure().iter().enumerate() {
                    irow[k] = to_index(r);
                    jcol[k] = to_index(c);
                }
            }
            SparsityRequest::Values { values } => {
                e.eval_hessian_lagrangian(
                    x.expect("hessian values need x"),
                    self.sign * obj_factor,
                    lambda.expect("hessian values need lambda"),
                    values,
                );
            }
        }
        true
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        _ip_data: &IpoptData,
        _ip_cq: &IpoptCq,
    ) -> bool {
        self.iterations = u64::try_from(stats.iter).unwrap_or(0);
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
