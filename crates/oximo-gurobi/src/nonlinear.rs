//! Lower an oximo expression tree onto Gurobi's native types.
//!
//! Gurobi 12 supports linear, quadratic, and a fixed set of general nonlinear
//! function constraints (exp/log/sin/cos/pow/...) where each one takes the form
//! `y = f(x)` between two variables. Anything more complex must be flattened
//! into a DAG of auxiliary variables connected by those primitive equalities.
//! This module is that flattening pass.

// TODO: This will change once we can get support for V13 in the grb crate
// since we will be able to use generic expresion trees

use grb::expr::{LinExpr, QuadExpr};
use grb::prelude::*;
use oximo_expr::{ExprArena, ExprId, ExprNode, VarId};
use oximo_solver::SolverError;

pub(crate) enum LoweredExpr {
    Linear(LinExpr),
    Quadratic(QuadExpr),
    Var(Var),
}

pub(crate) struct LoweringCtx<'a> {
    pub model: &'a mut Model,
    pub gurobi_vars: &'a [Var],
    pub aux_counter: u32,
}

impl LoweringCtx<'_> {
    fn next_name(&mut self, tag: &str) -> String {
        let n = self.aux_counter;
        self.aux_counter += 1;
        format!("aux_{tag}_{n}")
    }

    fn new_aux(&mut self, tag: &str, lb: f64, ub: f64) -> Result<Var, grb::Error> {
        let name = self.next_name(tag);
        let m = &mut *self.model;
        #[expect(clippy::unnecessary_cast)]
        let v = add_ctsvar!(m, name: &name, bounds: lb..ub)?;
        Ok(v)
    }
}

fn map_grb(e: grb::Error) -> SolverError {
    SolverError::Backend(format!("Gurobi: {e}"))
}

fn linear_from_var(v: Var) -> LinExpr {
    let mut e = LinExpr::new();
    e.add_term(1.0, v);
    e
}

fn linear_constant(c: f64) -> LinExpr {
    let mut e = LinExpr::new();
    e.add_constant(c);
    e
}

fn quad_from_linear(e: LinExpr) -> QuadExpr {
    let mut q = QuadExpr::new();
    q.add_constant(e.get_offset());
    for (v, c) in e.into_parts().0 {
        q.add_term(c, v);
    }
    q
}

fn add_linears(mut a: LinExpr, b: LinExpr) -> LinExpr {
    let (coeffs, offset) = b.into_parts();
    a.add_constant(offset);
    for (v, c) in coeffs {
        a.add_term(c, v);
    }
    a
}

fn add_quads(mut a: QuadExpr, b: QuadExpr) -> QuadExpr {
    let (qcoeffs, linexpr) = b.into_parts();
    a.add_constant(linexpr.get_offset());
    for (v, c) in linexpr.into_parts().0 {
        a.add_term(c, v);
    }
    for ((x, y), c) in qcoeffs {
        a.add_qterm(c, x, y);
    }
    a
}

fn scale_linear(mut e: LinExpr, k: f64) -> LinExpr {
    e.mul_scalar(k);
    e
}

fn scale_quad(mut e: QuadExpr, k: f64) -> QuadExpr {
    e.mul_scalar(k);
    e
}

/// Convert any lowered form to a `LinExpr`.
///
/// Panics if quadratic, callers must only invoke when the value
/// is known linear (i.e. degree <= 1).
fn into_linexpr(l: LoweredExpr) -> LinExpr {
    match l {
        LoweredExpr::Linear(e) => e,
        LoweredExpr::Var(v) => linear_from_var(v),
        LoweredExpr::Quadratic(_) => {
            panic!("internal: into_linexpr called on Quadratic LoweredExpr")
        }
    }
}

/// Combine two lowered values additively, promoting to the highest order.
fn lowered_add(a: LoweredExpr, b: LoweredExpr) -> LoweredExpr {
    use LoweredExpr::{Linear, Quadratic, Var};
    match (a, b) {
        (Linear(x), Linear(y)) => Linear(add_linears(x, y)),
        (Var(v), Linear(y)) | (Linear(y), Var(v)) => Linear(add_linears(y, linear_from_var(v))),
        (Var(v), Var(w)) => Linear(add_linears(linear_from_var(v), linear_from_var(w))),
        (Quadratic(x), Quadratic(y)) => Quadratic(add_quads(x, y)),
        (Quadratic(x), other) | (other, Quadratic(x)) => {
            let y = into_linexpr(other);
            Quadratic(add_quads(x, quad_from_linear(y)))
        }
    }
}

fn lowered_scale(l: LoweredExpr, k: f64) -> LoweredExpr {
    use LoweredExpr::{Linear, Quadratic, Var};
    if k == 0.0 {
        return Linear(linear_constant(0.0));
    }
    match l {
        Linear(e) => Linear(scale_linear(e, k)),
        Quadratic(e) => Quadratic(scale_quad(e, k)),
        Var(v) => Linear(scale_linear(linear_from_var(v), k)),
    }
}

fn lowered_neg(l: LoweredExpr) -> LoweredExpr {
    lowered_scale(l, -1.0)
}

/// Materialize `lowered` into a single Gurobi variable, returning the existing
/// one if it is already opaque. Bounds default to +-inf, callers should tighten
/// when they know more about the value's range.
fn as_aux_var(lowered: LoweredExpr, ctx: &mut LoweringCtx<'_>) -> Result<Var, SolverError> {
    match lowered {
        LoweredExpr::Var(v) => Ok(v),
        LoweredExpr::Linear(e) => {
            let aux = ctx.new_aux("v", f64::NEG_INFINITY, f64::INFINITY).map_err(map_grb)?;
            let name = ctx.next_name("eq");
            ctx.model.add_constr(&name, c!(aux == e)).map_err(map_grb)?;
            Ok(aux)
        }
        LoweredExpr::Quadratic(e) => {
            let aux = ctx.new_aux("v", f64::NEG_INFINITY, f64::INFINITY).map_err(map_grb)?;
            let name = ctx.next_name("qeq");
            ctx.model.add_qconstr(&name, c!(aux == e)).map_err(map_grb)?;
            Ok(aux)
        }
    }
}

/// True if `id` is a literal constant or a parameter, returns the value if so.
fn as_const(arena: &ExprArena, id: ExprId) -> Option<f64> {
    match arena.get(id) {
        ExprNode::Const(c) => Some(*c),
        ExprNode::Param(p) => Some(arena.param_value(*p)),
        _ => None,
    }
}

pub(crate) fn lower(
    arena: &ExprArena,
    id: ExprId,
    ctx: &mut LoweringCtx<'_>,
) -> Result<LoweredExpr, SolverError> {
    match arena.get(id) {
        ExprNode::Const(c) => Ok(LoweredExpr::Linear(linear_constant(*c))),
        ExprNode::Var(v) => Ok(LoweredExpr::Linear(linear_from_var(grb_var(ctx, *v)))),
        ExprNode::Param(p) => Ok(LoweredExpr::Linear(linear_constant(arena.param_value(*p)))),
        ExprNode::Linear { coeffs, constant } => {
            let mut e = linear_constant(*constant);
            for (v, c) in coeffs {
                e.add_term(*c, grb_var(ctx, *v));
            }
            Ok(LoweredExpr::Linear(e))
        }
        ExprNode::Neg(inner) => {
            let l = lower(arena, *inner, ctx)?;
            Ok(lowered_neg(l))
        }
        ExprNode::Add(children) => {
            let mut acc = LoweredExpr::Linear(linear_constant(0.0));
            for c in children {
                let l = lower(arena, *c, ctx)?;
                acc = lowered_add(acc, l);
            }
            Ok(acc)
        }
        ExprNode::Mul(children) => lower_mul(arena, children, ctx),
        ExprNode::Pow(base, exp) => lower_pow(arena, *base, *exp, ctx),
        ExprNode::Div(num, den) => lower_div(arena, *num, *den, ctx),
        ExprNode::Sin(a) => lower_unary(arena, *a, ctx, UnaryFn::Sin),
        ExprNode::Cos(a) => lower_unary(arena, *a, ctx, UnaryFn::Cos),
        ExprNode::Exp(a) => lower_unary(arena, *a, ctx, UnaryFn::Exp),
        ExprNode::Log(a) => lower_unary(arena, *a, ctx, UnaryFn::Log),
        ExprNode::Abs(a) => lower_unary(arena, *a, ctx, UnaryFn::Abs),
    }
}

fn grb_var(ctx: &LoweringCtx<'_>, v: VarId) -> Var {
    ctx.gurobi_vars[v.index()]
}

fn lower_mul(
    arena: &ExprArena,
    children: &[ExprId],
    ctx: &mut LoweringCtx<'_>,
) -> Result<LoweredExpr, SolverError> {
    let mut scalar = 1.0_f64;
    let mut non_consts: Vec<ExprId> = Vec::new();
    for c in children {
        if let Some(k) = as_const(arena, *c) {
            scalar *= k;
        } else {
            non_consts.push(*c);
        }
    }
    if non_consts.is_empty() {
        return Ok(LoweredExpr::Linear(linear_constant(scalar)));
    }
    if non_consts.len() == 1 {
        let l = lower(arena, non_consts[0], ctx)?;
        return Ok(lowered_scale(l, scalar));
    }
    if non_consts.len() == 2 {
        let a = lower(arena, non_consts[0], ctx)?;
        let b = lower(arena, non_consts[1], ctx)?;
        // Linear * Linear -> quadratic, if either side has variable terms.
        if let (LoweredExpr::Linear(la), LoweredExpr::Linear(lb)) = (&a, &b) {
            let q = multiply_linears(la, lb, scalar);
            return Ok(LoweredExpr::Quadratic(q));
        }
        // Mixed with Quadratic or Var -> materialize aux vars and recompose.
        let va = as_aux_var(a, ctx)?;
        let vb = as_aux_var(b, ctx)?;
        let mut q = QuadExpr::new();
        q.add_qterm(scalar, va, vb);
        return Ok(LoweredExpr::Quadratic(q));
    }
    // 3+ non-constant factors: degree > 2. Fold left, materializing aux vars.
    let mut acc_var = {
        let a = lower(arena, non_consts[0], ctx)?;
        let b = lower(arena, non_consts[1], ctx)?;
        let va = as_aux_var(a, ctx)?;
        let vb = as_aux_var(b, ctx)?;
        let mut q = QuadExpr::new();
        q.add_qterm(1.0, va, vb);
        as_aux_var(LoweredExpr::Quadratic(q), ctx)?
    };
    for c in &non_consts[2..] {
        let next = lower(arena, *c, ctx)?;
        let vn = as_aux_var(next, ctx)?;
        let mut q = QuadExpr::new();
        q.add_qterm(1.0, acc_var, vn);
        acc_var = as_aux_var(LoweredExpr::Quadratic(q), ctx)?;
    }
    Ok(lowered_scale(LoweredExpr::Var(acc_var), scalar))
}

fn multiply_linears(a: &LinExpr, b: &LinExpr, scalar: f64) -> QuadExpr {
    let a_off = a.get_offset();
    let b_off = b.get_offset();
    let mut q = QuadExpr::new();
    q.add_constant(scalar * a_off * b_off);
    for (va, ca) in a.iter_terms() {
        q.add_term(scalar * ca * b_off, *va);
    }
    for (vb, cb) in b.iter_terms() {
        q.add_term(scalar * a_off * cb, *vb);
    }
    for (va, ca) in a.iter_terms() {
        for (vb, cb) in b.iter_terms() {
            q.add_qterm(scalar * ca * cb, *va, *vb);
        }
    }
    q
}

fn lower_pow(
    arena: &ExprArena,
    base: ExprId,
    exp: ExprId,
    ctx: &mut LoweringCtx<'_>,
) -> Result<LoweredExpr, SolverError> {
    // Constant exponent -> either expand to a Mul chain (small ints) or use
    // Gurobi's add_genconstr_pow. Non-constant exponent -> exp(b*log(a)).
    if let Some(alpha) = as_const(arena, exp) {
        if alpha == 0.0 {
            return Ok(LoweredExpr::Linear(linear_constant(1.0)));
        }
        if (alpha - 1.0).abs() < f64::EPSILON {
            return lower(arena, base, ctx);
        }
        if (alpha - alpha.round()).abs() < f64::EPSILON && alpha > 0.0 && alpha <= 4.0 {
            // Pre-checks guarantee alpha in {1.0, 2.0, 3.0, 4.0}, alpha == 1 was
            // already handled above, so bucket the remaining three values.
            let n: u32 = match alpha.round() {
                v if v < 2.5 => 2,
                v if v < 3.5 => 3,
                _ => 4,
            };
            let base_lowered = lower(arena, base, ctx)?;
            let v = as_aux_var(base_lowered, ctx)?;
            let mut q = QuadExpr::new();
            q.add_qterm(1.0, v, v);
            let mut acc = LoweredExpr::Quadratic(q);
            for _ in 2..n {
                let aux = as_aux_var(acc, ctx)?;
                let mut next = QuadExpr::new();
                next.add_qterm(1.0, aux, v);
                acc = LoweredExpr::Quadratic(next);
            }
            return Ok(acc);
        }
        // General constant exponent via Gurobi's genconstr_pow
        let base_lowered = lower(arena, base, ctx)?;
        let x = as_aux_var(base_lowered, ctx)?;
        let y = ctx.new_aux("pow", f64::NEG_INFINITY, f64::INFINITY).map_err(map_grb)?;
        let name = ctx.next_name("pow");
        ctx.model.add_genconstr_pow(&name, x, y, alpha, "").map_err(map_grb)?;
        return Ok(LoweredExpr::Var(y));
    }
    // exp(b*log(a))
    let base_lowered = lower(arena, base, ctx)?;
    let x = as_aux_var(base_lowered, ctx)?;
    let log_y = ctx.new_aux("log", f64::NEG_INFINITY, f64::INFINITY).map_err(map_grb)?;
    let log_name = ctx.next_name("log");
    ctx.model.add_genconstr_natural_log(&log_name, x, log_y, "").map_err(map_grb)?;
    let b_lowered = lower(arena, exp, ctx)?;
    let b_var = as_aux_var(b_lowered, ctx)?;
    let prod = ctx.new_aux("mul", f64::NEG_INFINITY, f64::INFINITY).map_err(map_grb)?;
    let mut q = QuadExpr::new();
    q.add_qterm(1.0, b_var, log_y);
    let prod_eq = ctx.next_name("mul_eq");
    ctx.model.add_qconstr(&prod_eq, c!(prod == q)).map_err(map_grb)?;
    let result = ctx.new_aux("exp", 0.0, f64::INFINITY).map_err(map_grb)?;
    let exp_name = ctx.next_name("exp");
    ctx.model.add_genconstr_natural_exp(&exp_name, prod, result, "").map_err(map_grb)?;
    Ok(LoweredExpr::Var(result))
}

/// Lower `num / den` as `num * recip`, introducing a reciprocal variable
/// `recip` pinned by the bilinear equality `den * recip == 1`.
///
/// This avoids a `pow(den, -1)` lowering. Gurobi's `genconstr_pow` has a pole
/// at 0 and so requires its base to stay in a strictly positive domain,
/// which rules out negative or zero-straddling denominators. The bilinear pin
/// carries no such restriction: it is a plain non-convex quadratic
/// (handled via `NonConvex = 2`) valid for `den` of either sign,
/// and is infeasible only at `den == 0`.
fn lower_div(
    arena: &ExprArena,
    num: ExprId,
    den: ExprId,
    ctx: &mut LoweringCtx<'_>,
) -> Result<LoweredExpr, SolverError> {
    // `div_into` folds every nonzero constant denominator into the linear path
    // at construction, so the only constant `den` that reaches here is a literal
    // zero.
    if as_const(arena, den) == Some(0.0) {
        return Err(SolverError::Backend("division by zero: constant denominator is 0".into()));
    }

    // recip = 1 / den, pinned by `den * recip == 1`.
    let den_lowered = lower(arena, den, ctx)?;
    let x = as_aux_var(den_lowered, ctx)?;
    let recip = ctx.new_aux("recip", f64::NEG_INFINITY, f64::INFINITY).map_err(map_grb)?;
    let mut pin = QuadExpr::new();
    pin.add_qterm(1.0, x, recip);
    let name = ctx.next_name("recip_eq");
    ctx.model.add_qconstr(&name, c!(pin == 1.0)).map_err(map_grb)?;

    // Constant numerator stays linear in `recip`, otherwise form `num * recip`.
    if let Some(k) = as_const(arena, num) {
        return Ok(lowered_scale(LoweredExpr::Var(recip), k));
    }
    let num_lowered = lower(arena, num, ctx)?;
    let vn = as_aux_var(num_lowered, ctx)?;
    let mut q = QuadExpr::new();
    q.add_qterm(1.0, vn, recip);
    Ok(LoweredExpr::Quadratic(q))
}

enum UnaryFn {
    Sin,
    Cos,
    Exp,
    Log,
    Abs,
}

fn lower_unary(
    arena: &ExprArena,
    inner: ExprId,
    ctx: &mut LoweringCtx<'_>,
    f: UnaryFn,
) -> Result<LoweredExpr, SolverError> {
    let lowered = lower(arena, inner, ctx)?;
    let x = as_aux_var(lowered, ctx)?;
    let (lb, ub, tag) = match f {
        UnaryFn::Sin | UnaryFn::Cos => (-1.0, 1.0, "trig"),
        UnaryFn::Exp => (0.0, f64::INFINITY, "exp"),
        UnaryFn::Log => (f64::NEG_INFINITY, f64::INFINITY, "log"),
        UnaryFn::Abs => (0.0, f64::INFINITY, "abs"),
    };
    let y = ctx.new_aux(tag, lb, ub).map_err(map_grb)?;
    let name = ctx.next_name(tag);
    match f {
        UnaryFn::Sin => ctx.model.add_genconstr_sin(&name, x, y, "").map_err(map_grb)?,
        UnaryFn::Cos => ctx.model.add_genconstr_cos(&name, x, y, "").map_err(map_grb)?,
        UnaryFn::Exp => ctx.model.add_genconstr_natural_exp(&name, x, y, "").map_err(map_grb)?,
        UnaryFn::Log => ctx.model.add_genconstr_natural_log(&name, x, y, "").map_err(map_grb)?,
        UnaryFn::Abs => ctx.model.add_genconstr_abs(&name, y, x).map_err(map_grb)?,
    };
    Ok(LoweredExpr::Var(y))
}

/// Helpers used by translate.rs to materialize a lowered constraint or
/// objective expression.
impl LoweredExpr {
    pub(crate) fn into_expr_for_objective(self) -> grb::expr::Expr {
        match self {
            LoweredExpr::Linear(e) => grb::expr::Expr::from(e),
            LoweredExpr::Quadratic(e) => grb::expr::Expr::from(e),
            LoweredExpr::Var(v) => grb::expr::Expr::from(v),
        }
    }
}
