//! Translate an `ExprNode` subtree into NL opcodes via the [`Writer`].
//!
//! Mapping (D. M. Gay, operator Tables 4 unary, 6 binary, 8 n-ary, 11 n-ary operators).
//!
//! | `ExprNode`        | NL opcode                             |
//! |-------------------|---------------------------------------|
//! | `Const(c)`        | `n<c>` (binary picks `s`/`l`/`n`)     |
//! | `Var(v)`          | `v<permuted_idx>`                     |
//! | `Neg(x)`          | `o16`                                 |
//! | `Add(2)`          | `o0` (binary plus)                    |
//! | `Add(>=3)`        | `o54 <N>` (n-ary sumlist)             |
//! | `Mul(2)`          | `o2` (binary times)                   |
//! | `Mul(>=3)`        | left-folded `o2` chain                |
//! | `Pow(b, e)`       | `o5`                                  |
//! | `Div(n, d)`       | `o3`                                  |
//! | `Sin/Cos/Exp/Log` | `o41`, `o46`, `o44`, `o43`            |
//! | `Abs`             | `o15`                                 |
//! | `Linear`          | expanded to `o54 <N+1>` of `o2 n_c v` |

use std::io::Write;

use oximo_expr::{ExprArena, ExprId, ExprNode, SignedExpr, VarId};
use rustc_hash::FxHashMap;

use super::writer::Writer;
use crate::error::IoError;

/// Emit a sum of nonlinear residual summands (each optionally negated). Mirrors
/// [`emit_add`]'s opcode choices: `o0` for two summands, `o54` for three or
/// more, so the byte output matches a materialized `Add`/`Neg` tree.
pub(crate) fn emit_residual<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    residual: &[SignedExpr],
) -> Result<(), IoError> {
    match residual.len() {
        0 => w.num(0.0)?,
        1 => emit_signed(w, arena, var_index, residual[0])?,
        2 => {
            w.op(0)?;
            emit_signed(w, arena, var_index, residual[0])?;
            emit_signed(w, arena, var_index, residual[1])?;
        }
        n => {
            w.op(54)?;
            w.int(i64::try_from(n).expect("arity"))?;
            w.eor()?;
            for s in residual {
                emit_signed(w, arena, var_index, *s)?;
            }
        }
    }
    Ok(())
}

fn emit_signed<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    s: SignedExpr,
) -> Result<(), IoError> {
    if s.neg {
        w.op(16)?;
    }
    emit_expr(w, arena, var_index, s.id)
}

pub(crate) fn emit_expr<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    id: ExprId,
) -> Result<(), IoError> {
    match arena.get(id) {
        ExprNode::Const(c) => w.num(*c)?,
        ExprNode::Var(v) => {
            let idx = var_index
                .get(v)
                .copied()
                .ok_or_else(|| IoError::UnknownVar(format!("#{}", v.index())))?;
            w.var(idx)?;
        }
        ExprNode::Param(p) => w.num(arena.param_value(*p))?,
        ExprNode::Neg(x) => {
            w.op(16)?;
            emit_expr(w, arena, var_index, *x)?;
        }
        ExprNode::Add(children) => emit_add(w, arena, var_index, children)?,
        ExprNode::Mul(children) => emit_mul(w, arena, var_index, children)?,
        ExprNode::Pow(b, e) => {
            w.op(5)?;
            emit_expr(w, arena, var_index, *b)?;
            emit_expr(w, arena, var_index, *e)?;
        }
        ExprNode::Div(num, den) => {
            w.op(3)?;
            emit_expr(w, arena, var_index, *num)?;
            emit_expr(w, arena, var_index, *den)?;
        }
        ExprNode::Sin(x) => {
            w.op(41)?;
            emit_expr(w, arena, var_index, *x)?;
        }
        ExprNode::Cos(x) => {
            w.op(46)?;
            emit_expr(w, arena, var_index, *x)?;
        }
        ExprNode::Exp(x) => {
            w.op(44)?;
            emit_expr(w, arena, var_index, *x)?;
        }
        ExprNode::Log(x) => {
            w.op(43)?;
            emit_expr(w, arena, var_index, *x)?;
        }
        ExprNode::Abs(x) => {
            w.op(15)?;
            emit_expr(w, arena, var_index, *x)?;
        }
        ExprNode::Linear { coeffs, constant } => {
            emit_linear_inline(w, var_index, coeffs, *constant)?;
        }
    }
    Ok(())
}

fn emit_add<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    children: &[ExprId],
) -> Result<(), IoError> {
    match children.len() {
        0 => w.num(0.0)?,
        1 => emit_expr(w, arena, var_index, children[0])?,
        2 => {
            w.op(0)?;
            emit_expr(w, arena, var_index, children[0])?;
            emit_expr(w, arena, var_index, children[1])?;
        }
        n => {
            w.op(54)?;
            w.int(i64::try_from(n).expect("arity"))?;
            w.eor()?;
            for c in children {
                emit_expr(w, arena, var_index, *c)?;
            }
        }
    }
    Ok(())
}

fn emit_mul<W: Write>(
    w: &mut Writer<'_, W>,
    arena: &ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    children: &[ExprId],
) -> Result<(), IoError> {
    match children.len() {
        0 => w.num(1.0)?,
        1 => emit_expr(w, arena, var_index, children[0])?,
        2 => {
            w.op(2)?;
            emit_expr(w, arena, var_index, children[0])?;
            emit_expr(w, arena, var_index, children[1])?;
        }
        n => {
            // Left-folded binary chain: (((a*b)*c)*d).
            for _ in 0..(n - 1) {
                w.op(2)?;
            }
            for c in children {
                emit_expr(w, arena, var_index, *c)?;
            }
        }
    }
    Ok(())
}

/// Expand a `Linear { coeffs, constant }` node into an `o54` sumlist when it
/// appears as a sub-expression inside a nonlinear residual.
fn emit_linear_inline<W: Write>(
    w: &mut Writer<'_, W>,
    var_index: &FxHashMap<VarId, u32>,
    coeffs: &[(VarId, f64)],
    constant: f64,
) -> Result<(), IoError> {
    let nz: Vec<(VarId, f64)> = coeffs.iter().copied().filter(|(_, c)| *c != 0.0).collect();
    let has_const = constant != 0.0;
    let n = nz.len() + usize::from(has_const);
    if n == 0 {
        return w.num(0.0);
    }
    if n == 1 {
        if has_const {
            return w.num(constant);
        }
        let (v, c) = nz[0];
        return emit_term(w, var_index, v, c);
    }
    if n == 2 {
        w.op(0)?;
        for (v, c) in &nz {
            emit_term(w, var_index, *v, *c)?;
        }
        if has_const {
            w.num(constant)?;
        }
        return Ok(());
    }
    w.op(54)?;
    w.int(i64::try_from(n).expect("arity"))?;
    w.eor()?;
    for (v, c) in &nz {
        emit_term(w, var_index, *v, *c)?;
    }
    if has_const {
        w.num(constant)?;
    }
    Ok(())
}

fn emit_term<W: Write>(
    w: &mut Writer<'_, W>,
    var_index: &FxHashMap<VarId, u32>,
    v: VarId,
    c: f64,
) -> Result<(), IoError> {
    let idx =
        var_index.get(&v).copied().ok_or_else(|| IoError::UnknownVar(format!("#{}", v.index())))?;
    if (c - 1.0).abs() == 0.0 {
        w.var(idx)?;
    } else {
        w.op(2)?;
        w.num(c)?;
        w.var(idx)?;
    }
    Ok(())
}
