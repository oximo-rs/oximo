use thiserror::Error;

use crate::arena::{ExprArena, ExprId, ExprNode, ParamId, VarId};

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("variable {0:?} has no value bound in the evaluation context")]
    UnboundVar(VarId),
    #[error("parameter {0:?} has no value bound in the evaluation context")]
    UnboundParam(ParamId),
}

/// Source of variable and parameter values during expression evaluation.
pub trait EvalContext {
    fn var(&self, v: VarId) -> Option<f64>;
    fn param(&self, p: ParamId) -> Option<f64>;
}

impl EvalContext for &[f64] {
    fn var(&self, v: VarId) -> Option<f64> {
        self.get(v.index()).copied()
    }
    fn param(&self, _p: ParamId) -> Option<f64> {
        None
    }
}

/// Evaluate `id` to an `f64`, pulling variable / parameter values from `ctx`.
///
/// # Errors
///
/// Returns an [`EvalError`] if a needed variable or parameter is missing from the context.
pub fn evaluate<C: EvalContext>(arena: &ExprArena, id: ExprId, ctx: &C) -> Result<f64, EvalError> {
    Ok(match arena.get(id) {
        ExprNode::Const(c) => *c,
        ExprNode::Var(v) => ctx.var(*v).ok_or(EvalError::UnboundVar(*v))?,
        ExprNode::Param(p) => ctx
            .param(*p)
            .or_else(|| arena.try_param_value(*p))
            .ok_or(EvalError::UnboundParam(*p))?,
        ExprNode::Add(children) => children
            .iter()
            .try_fold(0.0, |acc, c| Ok::<_, EvalError>(acc + evaluate(arena, *c, ctx)?))?,
        ExprNode::Mul(children) => children
            .iter()
            .try_fold(1.0, |acc, c| Ok::<_, EvalError>(acc * evaluate(arena, *c, ctx)?))?,
        ExprNode::Neg(inner) => -evaluate(arena, *inner, ctx)?,
        ExprNode::Pow(base, exp) => evaluate(arena, *base, ctx)?.powf(evaluate(arena, *exp, ctx)?),
        ExprNode::Div(num, den) => evaluate(arena, *num, ctx)? / evaluate(arena, *den, ctx)?,
        ExprNode::Sin(inner) => evaluate(arena, *inner, ctx)?.sin(),
        ExprNode::Cos(inner) => evaluate(arena, *inner, ctx)?.cos(),
        ExprNode::Exp(inner) => evaluate(arena, *inner, ctx)?.exp(),
        ExprNode::Log(inner) => evaluate(arena, *inner, ctx)?.ln(),
        ExprNode::Abs(inner) => evaluate(arena, *inner, ctx)?.abs(),
        ExprNode::Linear { coeffs, constant } => {
            let mut acc = *constant;
            for (v, c) in coeffs {
                acc += c * ctx.var(*v).ok_or(EvalError::UnboundVar(*v))?;
            }
            acc
        }
    })
}
