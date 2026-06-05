//! Per-row classification.
//!
//! For each constraint and the objective we run `split_linear` on the body.
//! The result is a `Row { linear, residual }` where:
//!
//! - `linear` carries the linear coefficients (goes into `J`/`G` segments
//!   and into the RHS constant shift).
//! - `residual` is `Some(ExprId)` when nonlinear summands remain (emitted in
//!   the corresponding `C`/`O` segment), `None` when the whole body is
//!   purely linear.
//!
//! Each row also caches the union of variables appearing anywhere in the body
//! (linear OR nonlinear), used to size the Jacobian/gradient sparsity. The
//! variable sets `nl_vars_c`, `nl_vars_o` only count vars appearing inside the
//! nonlinear residual.

use oximo_core::{Constraint, Objective, Variable};
use oximo_expr::{ExprArena, ExprId, ExprNode, LinearTerms, SignedExpr, VarId, split_linear};
use rustc_hash::FxHashSet;

use crate::error::IoError;

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) linear: LinearTerms,
    /// Nonlinear summands of the body, empty when the row is purely linear.
    pub(crate) residual: Vec<SignedExpr>,
}

impl Row {
    pub(crate) fn is_nonlinear(&self) -> bool {
        !self.residual.is_empty()
    }
}

#[derive(Debug)]
pub(crate) struct Analysis {
    pub(crate) cons: Vec<Row>,
    pub(crate) obj: Row,
    pub(crate) cons_vars: Vec<Vec<VarId>>,
    pub(crate) obj_vars: Vec<VarId>,
    pub(crate) nl_vars_c: FxHashSet<VarId>,
    pub(crate) nl_vars_o: FxHashSet<VarId>,
}

impl Analysis {
    pub(crate) fn build(
        arena: &ExprArena,
        _vars: &[Variable],
        constraints: &[Constraint],
        objective: &Objective,
    ) -> Result<Self, IoError> {
        let mut nl_vars_c: FxHashSet<VarId> = FxHashSet::default();
        let mut nl_vars_o: FxHashSet<VarId> = FxHashSet::default();
        let mut cons: Vec<Row> = Vec::with_capacity(constraints.len());
        let mut cons_vars: Vec<Vec<VarId>> = Vec::with_capacity(constraints.len());

        for c in constraints {
            let (linear, residual) = split_linear(arena, c.lhs);
            let mut all = FxHashSet::default();
            for (v, _) in &linear.coeffs {
                all.insert(*v);
            }
            if !residual.is_empty() {
                let mut nl_set: FxHashSet<VarId> = FxHashSet::default();
                for r in &residual {
                    validate(arena, r.id)?;
                    collect_vars(arena, r.id, &mut nl_set)?;
                }
                for v in &nl_set {
                    nl_vars_c.insert(*v);
                    all.insert(*v);
                }
            }
            cons.push(Row { linear, residual });
            cons_vars.push(sorted(all));
        }

        let (obj_linear, obj_residual) = split_linear(arena, objective.expr);
        let mut obj_all = FxHashSet::default();
        for (v, _) in &obj_linear.coeffs {
            obj_all.insert(*v);
        }
        if !obj_residual.is_empty() {
            let mut nl_set: FxHashSet<VarId> = FxHashSet::default();
            for r in &obj_residual {
                validate(arena, r.id)?;
                collect_vars(arena, r.id, &mut nl_set)?;
            }
            for v in &nl_set {
                nl_vars_o.insert(*v);
                obj_all.insert(*v);
            }
        }
        let obj = Row { linear: obj_linear, residual: obj_residual };

        Ok(Self { cons, obj, cons_vars, obj_vars: sorted(obj_all), nl_vars_c, nl_vars_o })
    }
}

fn sorted(set: FxHashSet<VarId>) -> Vec<VarId> {
    let mut v: Vec<VarId> = set.into_iter().collect();
    v.sort_by_key(|v| v.0);
    v
}

fn validate(arena: &ExprArena, id: ExprId) -> Result<(), IoError> {
    match arena.get(id) {
        ExprNode::Const(c) => {
            if !c.is_finite() {
                return Err(IoError::InvalidNumber);
            }
            Ok(())
        }
        ExprNode::Var(_) => Ok(()),
        ExprNode::Param(_) => Err(IoError::UnsupportedNode("Param")),
        ExprNode::Neg(x)
        | ExprNode::Sin(x)
        | ExprNode::Cos(x)
        | ExprNode::Exp(x)
        | ExprNode::Log(x)
        | ExprNode::Abs(x) => validate(arena, *x),
        ExprNode::Pow(b, e) => {
            validate(arena, *b)?;
            validate(arena, *e)
        }
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            for c in children {
                validate(arena, *c)?;
            }
            Ok(())
        }
        ExprNode::Div(num, den) => {
            validate(arena, *num)?;
            validate(arena, *den)
        }
        ExprNode::Linear { coeffs: _, constant } => {
            if !constant.is_finite() {
                return Err(IoError::InvalidNumber);
            }
            Ok(())
        }
    }
}

fn collect_vars(arena: &ExprArena, id: ExprId, out: &mut FxHashSet<VarId>) -> Result<(), IoError> {
    match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Param(_) => Ok(()),
        ExprNode::Var(v) => {
            out.insert(*v);
            Ok(())
        }
        ExprNode::Neg(x)
        | ExprNode::Sin(x)
        | ExprNode::Cos(x)
        | ExprNode::Exp(x)
        | ExprNode::Log(x)
        | ExprNode::Abs(x) => collect_vars(arena, *x, out),
        ExprNode::Pow(b, e) => {
            collect_vars(arena, *b, out)?;
            collect_vars(arena, *e, out)
        }
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            for c in children {
                collect_vars(arena, *c, out)?;
            }
            Ok(())
        }
        ExprNode::Div(num, den) => {
            collect_vars(arena, *num, out)?;
            collect_vars(arena, *den, out)
        }
        ExprNode::Linear { coeffs, .. } => {
            for (v, _) in coeffs {
                out.insert(*v);
            }
            Ok(())
        }
    }
}
