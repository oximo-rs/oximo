//! Translate an `ExprNode` subtree into NL opcodes via the [`Writer`].
//!
//! Mapping (D. M. Gay, operator Tables 4 unary, 6 binary, 8 n-ary, 11 n-ary operators).
//!
//! | `ExprNode`        | NL opcode                                   |
//! |-------------------|---------------------------------------------|
//! | `Const(c)`        | `n<c>` (binary picks `s`/`l`/`n`)           |
//! | `Var(v)`          | `v<permuted_idx>`                           |
//! | `Unary(Neg, x)`   | `o16`                                       |
//! | `Add(2)`          | `o0` (binary plus)                          |
//! | `Add(>=3)`        | `o54 <N>` (n-ary sumlist)                   |
//! | `Mul(2)`          | `o2` (binary times)                         |
//! | `Mul(>=3)`        | left-folded `o2` chain                      |
//! | `Pow(b, e)`       | `o5`                                        |
//! | `Div(n, d)`       | `o3`                                        |
//! | `Unary`           | documented unary opcodes (see `emit_unary`) |
//! | `Atan2(y, x)`     | `o48`                                       |
//! | `Min/Max`         | `o11`/`o12` with arity and children        |
//! | `Linear`          | expanded to `o54 <N+1>` of `o2 n_c v`       |

use std::io::Write;

use oximo_expr::{
    ExprArena, ExprId, ExprNode, LinearTerms, SignedExpr, UnaryOp, VarId, extract_linear,
};
use rustc_hash::FxHashMap;

use super::writer::Writer;
use crate::error::IoError;

/// Per-export scratch. Reuse up to eight affine extractions across rows and
/// the objective, without retaining every coefficient vector in a wide model.
#[derive(Default)]
pub(crate) struct EmitCache<'a> {
    linear: Vec<(ExprId, LinearTerms<'a>)>,
}

impl<'a> EmitCache<'a> {
    fn extract(&mut self, arena: &'a ExprArena, id: ExprId) -> Option<&LinearTerms<'a>> {
        if let Some(index) = self.linear.iter().position(|(root, _)| *root == id) {
            return Some(&self.linear[index].1);
        }
        let terms = extract_linear(arena, id)?;
        if self.linear.len() == 8 {
            self.linear.remove(0);
        }
        self.linear.push((id, terms));
        self.linear.last().map(|(_, terms)| terms)
    }
}

/// Emit a sum of nonlinear residual summands (each optionally negated). Mirrors
/// `Add`'s opcode choices: `o0` for two summands, `o54` for three or more.
/// Affine subtrees inside each summand are normalized during emission.
pub(crate) fn emit_residual<'a, W: Write>(
    w: &mut Writer<'_, W>,
    arena: &'a ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    residual: &[SignedExpr],
    cache: &mut EmitCache<'a>,
) -> Result<(), IoError> {
    match residual.len() {
        0 => w.num(0.0)?,
        1 => emit_signed(w, arena, var_index, residual[0], cache)?,
        2 => {
            w.op(0)?;
            emit_signed(w, arena, var_index, residual[0], cache)?;
            emit_signed(w, arena, var_index, residual[1], cache)?;
        }
        n => {
            w.op(54)?;
            w.int(i64::try_from(n).expect("arity"))?;
            w.eor()?;
            for s in residual {
                emit_signed(w, arena, var_index, *s, cache)?;
            }
        }
    }
    Ok(())
}

fn emit_signed<'a, W: Write>(
    w: &mut Writer<'_, W>,
    arena: &'a ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    s: SignedExpr,
    cache: &mut EmitCache<'a>,
) -> Result<(), IoError> {
    if s.neg {
        w.op(16)?;
    }
    emit_expr(w, arena, var_index, s.id, cache)
}

pub(crate) fn emit_expr<'a, W: Write>(
    w: &mut Writer<'_, W>,
    arena: &'a ExprArena,
    var_index: &FxHashMap<VarId, u32>,
    id: ExprId,
    cache: &mut EmitCache<'a>,
) -> Result<(), IoError> {
    let mut pending = vec![id];
    let mut affine = FxHashMap::default();
    while let Some(id) = pending.pop() {
        // Normalize maximal affine arithmetic inside nonlinear expressions.
        if matches!(
            arena.get(id),
            ExprNode::Add(_)
                | ExprNode::Mul(_)
                | ExprNode::Div(_, _)
                | ExprNode::Unary(UnaryOp::Neg, _)
        ) && affine.get(&id).is_none_or(Option::is_some)
        {
            if let Some(terms) = cache.extract(arena, id)
                && terms.constant.is_finite()
                && terms.coeffs.iter().all(|(_, coefficient)| coefficient.is_finite())
            {
                emit_linear_inline(w, var_index, &terms.coeffs, terms.constant)?;
                continue;
            }
            // Successful extraction doesn't need a separate classification pass.
            affine_degree(arena, id, &mut affine);
            affine.insert(id, None);
        }
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
            ExprNode::Unary(op, x) => {
                emit_unary(w, *op)?;
                pending.push(*x);
            }
            ExprNode::Add(children) => {
                match children.len() {
                    0 => w.num(0.0)?,
                    1 => {}
                    2 => w.op(0)?,
                    n => {
                        w.op(54)?;
                        w.int(i64::try_from(n).expect("arity"))?;
                        w.eor()?;
                    }
                }
                pending.extend(children.iter().rev().copied());
            }
            ExprNode::Mul(children) => {
                if children.is_empty() {
                    w.num(1.0)?;
                } else {
                    // Prefix encoding of the same left-folded binary product.
                    for _ in 1..children.len() {
                        w.op(2)?;
                    }
                    pending.extend(children.iter().rev().copied());
                }
            }
            ExprNode::Pow(b, e) => {
                w.op(5)?;
                pending.extend([*e, *b]);
            }
            ExprNode::Div(num, den) => {
                w.op(3)?;
                pending.extend([*den, *num]);
            }
            ExprNode::Atan2(y, x) => {
                w.op(48)?;
                pending.extend([*x, *y]);
            }
            ExprNode::Min(children) | ExprNode::Max(children) => {
                w.op(if matches!(arena.get(id), ExprNode::Min(_)) { 11 } else { 12 })?;
                w.int(i64::try_from(children.len()).expect("arity"))?;
                w.eor()?;
                pending.extend(children.iter().rev().copied());
            }
            ExprNode::Linear { coeffs, constant } => {
                emit_linear_inline(w, var_index, coeffs, *constant)?;
            }
        }
    }
    Ok(())
}

/// Conservative affine recognition for emission, memoized across arithmetic
/// regions. `Some(false)` is constant, `Some(true)` contains affine variables,
/// and `None` requires the original nonlinear encoding.
fn affine_degree(
    arena: &ExprArena,
    root: ExprId,
    cache: &mut FxHashMap<ExprId, Option<bool>>,
) -> Option<bool> {
    let mut pending = vec![(root, false)];
    while let Some((id, finish)) = pending.pop() {
        if cache.contains_key(&id) {
            continue;
        }
        let value = match arena.get(id) {
            ExprNode::Const(_) | ExprNode::Param(_) => Some(false),
            ExprNode::Var(_) => Some(true),
            ExprNode::Linear { coeffs, .. } => Some(!coeffs.is_empty()),
            ExprNode::Add(children) | ExprNode::Mul(children) if !finish => {
                pending.push((id, true));
                pending.extend(children.iter().rev().map(|&child| (child, false)));
                continue;
            }
            ExprNode::Add(children) => children.iter().try_fold(false, |contains_vars, child| {
                cache[child].map(|child_vars| contains_vars || child_vars)
            }),
            ExprNode::Mul(children) => {
                let mut contains_vars = false;
                let mut value = Some(false);
                for child in children {
                    match cache[child] {
                        Some(false) => {}
                        Some(true) if !contains_vars => contains_vars = true,
                        _ => {
                            value = None;
                            break;
                        }
                    }
                }
                value.map(|_| contains_vars)
            }
            ExprNode::Unary(UnaryOp::Neg, child) if finish => cache[child],
            ExprNode::Div(num, den) if finish => {
                if cache[den] == Some(false) {
                    cache[num]
                } else {
                    None
                }
            }
            ExprNode::Unary(UnaryOp::Neg, child) => {
                pending.extend([(id, true), (*child, false)]);
                continue;
            }
            ExprNode::Div(num, den) => {
                pending.extend([(id, true), (*den, false), (*num, false)]);
                continue;
            }
            _ => None,
        };
        cache.insert(id, value);
    }
    cache[&root]
}

fn emit_unary<W: Write>(w: &mut Writer<'_, W>, op: UnaryOp) -> Result<(), IoError> {
    if op == UnaryOp::Exp2 {
        w.op(5)?;
        w.num(2.0)?;
        return Ok(());
    }
    let opcode = match op {
        UnaryOp::Neg => 16,
        UnaryOp::Abs => 15,
        UnaryOp::Sqrt => 39,
        UnaryOp::Exp => 44,
        UnaryOp::Log => 43,
        UnaryOp::Log10 => 42,
        UnaryOp::Sin => 41,
        UnaryOp::Cos => 46,
        UnaryOp::Tan => 38,
        UnaryOp::Asin => 51,
        UnaryOp::Acos => 53,
        UnaryOp::Atan => 49,
        UnaryOp::Sinh => 40,
        UnaryOp::Cosh => 45,
        UnaryOp::Tanh => 37,
        UnaryOp::Asinh => 50,
        UnaryOp::Acosh => 52,
        UnaryOp::Atanh => 47,
        UnaryOp::Cbrt | UnaryOp::Expm1 | UnaryOp::Log2 | UnaryOp::Log1p => {
            return Err(IoError::UnsupportedNonlinearOperator { operator: op.name() });
        }
        UnaryOp::Exp2 => unreachable!(),
    };
    w.op(opcode)?;
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
    let nonzero = || coeffs.iter().copied().filter(|(_, c)| *c != 0.0);
    let has_const = constant != 0.0;
    let n = nonzero().count() + usize::from(has_const);
    if n == 0 {
        return w.num(0.0);
    }
    if n == 1 {
        if has_const {
            return w.num(constant);
        }
        let (v, c) = nonzero().next().expect("one nonzero coefficient");
        return emit_term(w, var_index, v, c);
    }
    if n == 2 {
        w.op(0)?;
        for (v, c) in nonzero() {
            emit_term(w, var_index, v, c)?;
        }
        if has_const {
            w.num(constant)?;
        }
        return Ok(());
    }
    w.op(54)?;
    w.int(i64::try_from(n).expect("arity"))?;
    w.eor()?;
    for (v, c) in nonzero() {
        emit_term(w, var_index, v, c)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonfinite_extracted_terms_keep_tree_encoding() {
        let mut arena = ExprArena::new();
        let var = arena.var(VarId(0));
        let large = arena.constant(f64::MAX);
        let scaled = arena.push(ExprNode::Mul(vec![large, var].into()));
        let constant_overflow = arena.push(ExprNode::Add(vec![large, large].into()));
        let coefficient_overflow = arena.push(ExprNode::Add(vec![scaled, scaled].into()));
        let var_index = FxHashMap::from_iter([(VarId(0), 0)]);
        let opts = super::super::options::WriteOptions::ascii_lean();
        for (root, child) in [
            (constant_overflow, format!("n{}\n", f64::MAX)),
            (coefficient_overflow, format!("o2\nn{}\nv0\n", f64::MAX)),
        ] {
            let mut bytes = Vec::new();
            emit_expr(
                &mut Writer::new(&mut bytes, &opts),
                &arena,
                &var_index,
                root,
                &mut EmitCache::default(),
            )
            .expect("finite source terms must export despite extraction overflow");
            assert_eq!(String::from_utf8(bytes).unwrap(), format!("o0\n{child}{child}"));
        }
    }

    #[test]
    fn export_cache_is_bounded_and_new_exports_observe_rebinding() {
        let mut arena = ExprArena::new();
        let pid = arena.new_param(2.0);
        let parameter = arena.push(ExprNode::Param(pid));
        let var = arena.var(VarId(0));
        let roots: Vec<_> = (0..9)
            .map(|i| {
                let constant = arena.constant(f64::from(i));
                arena.push(ExprNode::Add(vec![var, parameter, constant].into()))
            })
            .collect();
        let mut cache = EmitCache::default();
        for (i, &root) in roots.iter().enumerate() {
            let terms = cache.extract(&arena, root).unwrap();
            assert_eq!(terms.coeffs.as_ref(), [(VarId(0), 1.0)]);
            assert_eq!(terms.constant, 2.0 + f64::from(u32::try_from(i).unwrap()));
            assert!(cache.linear.len() <= 8);
        }
        assert_eq!(cache.linear.len(), 8);
        assert!(cache.linear.iter().all(|(root, _)| *root != roots[0]));
        let stored = cache.extract(&arena, roots[8]).unwrap().coeffs.as_ptr();
        assert_eq!(cache.extract(&arena, roots[8]).unwrap().coeffs.as_ptr(), stored);
        drop(cache);
        arena.set_param_value(pid, 3.0);
        let mut next_export = EmitCache::default();
        assert_eq!(next_export.extract(&arena, roots[8]).unwrap().constant, 11.0);
    }
}
