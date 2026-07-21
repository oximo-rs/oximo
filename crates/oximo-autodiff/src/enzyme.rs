//! The `std::autodiff` (Enzyme) wrappers around the tape interpreter.
//!
//! [`eval_tape_ad`] is differentiated once in reverse mode for gradients,
//! [`grad_eval_tape`] (which runs the reverse pass) is differentiated in
//! forward mode for Hessian-vector products (forward-over-reverse).
//!
//! An `Active` return is implemented via an `enzyme_primal_return`
//! marker global that Enzyme cannot shadow when the forward-over-reverse
//! pass runs in a downstream crate's fat-LTO step.

#![expect(clippy::too_many_arguments)]

use std::autodiff::{autodiff_forward, autodiff_reverse};

use crate::tape::Tape;

/// Reverse-differentiated interpreter: `d_eval_tape(.., x, dx, .., regs,
/// dregs, out, dout)` accumulates `dout[0] * (partial f)/(partial x)` into `dx`.
#[autodiff_reverse(
    d_eval_tape,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Duplicated,
    Const,
    Const,
    Duplicated,
    Duplicated
)]
#[expect(clippy::too_many_arguments)]
#[inline(never)]
pub fn eval_tape_ad(
    ops: &[u32],
    a: &[u32],
    b: &[u32],
    consts: &[f64],
    lin_vars: &[u32],
    lin_coeffs: &[f64],
    x: &[f64],
    params: &[f64],
    mults: &[f64],
    regs: &mut [f64],
    out: &mut [f64],
) {
    crate::tape::eval_tape(ops, a, b, consts, lin_vars, lin_coeffs, x, params, mults, regs, out);
}

/// Gradient entry point: zeroes the shadows, seeds `dout[0] = 1.0`, then runs
/// one reverse sweep. `grad_out` receives `(partial f/partial x)` (dense, indexed by
/// variable) and `out[0]` the primal value.
///
/// Forward-differentiated as `hvp_eval_tape` for Hessian-vector products:
/// seeding the `x` tangent with a direction `v` makes the `grad_out` tangent
/// come back as `H*v`.
#[autodiff_forward(
    hvp_eval_tape,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Dual,
    Const,
    Const,
    Dual,
    Dual,
    Dual,
    Dual,
    Dual
)]
#[expect(clippy::too_many_arguments)]
#[inline(never)]
pub fn grad_eval_tape(
    ops: &[u32],
    a: &[u32],
    b: &[u32],
    consts: &[f64],
    lin_vars: &[u32],
    lin_coeffs: &[f64],
    x: &[f64],
    params: &[f64],
    mults: &[f64],
    regs: &mut [f64],
    dregs: &mut [f64],
    out: &mut [f64],
    dout: &mut [f64],
    grad_out: &mut [f64],
) {
    // This function is the primal that `hvp_eval_tape` forward-differentiates, so
    // it is kept deliberately free of a `memset` intrinsic that Enzyme's type
    // analysis would otherwise have to see through inside the differentiated
    // region.
    for g in grad_out.iter_mut() {
        *g = 0.0;
    }
    for d in dregs.iter_mut() {
        *d = 0.0;
    }
    dout[0] = 1.0;
    d_eval_tape(
        ops, a, b, consts, lin_vars, lin_coeffs, x, grad_out, params, mults, regs, dregs, out, dout,
    );
}

/// Compute the dense gradient of `tape` at `x` into `grad_out` (overwritten).
/// `regs`/`dregs` are scratch of length [`Tape::n_regs`].
pub(crate) fn tape_gradient(
    tape: &Tape,
    x: &[f64],
    params: &[f64],
    mults: &[f64],
    regs: &mut [f64],
    dregs: &mut [f64],
    grad_out: &mut [f64],
) {
    let (ops, a, b, consts, lin_vars, lin_coeffs) = tape.parts();
    let mut out = [0.0];
    let mut dout = [0.0];
    grad_eval_tape(
        ops, a, b, consts, lin_vars, lin_coeffs, x, params, mults, regs, dregs, &mut out,
        &mut dout, grad_out,
    );
}

/// Forward-over-reverse Hessian-vector product of `tape` at `x` along `dir`.
/// `grad_out` receives the gradient, `hv_out` receives `H*dir` (both
/// overwritten).
#[expect(clippy::too_many_arguments)]
pub(crate) fn tape_hvp(
    tape: &Tape,
    x: &[f64],
    dir: &[f64],
    params: &[f64],
    mults: &[f64],
    regs: &mut [f64],
    regs_t: &mut [f64],
    dregs: &mut [f64],
    dregs_t: &mut [f64],
    grad_out: &mut [f64],
    hv_out: &mut [f64],
) {
    let (ops, a, b, consts, lin_vars, lin_coeffs) = tape.parts();
    regs_t.fill(0.0);
    dregs_t.fill(0.0);
    hv_out.fill(0.0);
    let mut out = [0.0];
    let mut out_t = [0.0];
    let mut dout = [0.0];
    let mut dout_t = [0.0];
    hvp_eval_tape(
        ops,
        a,
        b,
        consts,
        lin_vars,
        lin_coeffs,
        x,
        dir,
        params,
        mults,
        regs,
        regs_t,
        dregs,
        dregs_t,
        &mut out,
        &mut out_t,
        &mut dout,
        &mut dout_t,
        grad_out,
        hv_out,
    );
}
