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

use oximo_core::{Constraint, Domain, Objective, Variable};
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
        vars: &[Variable],
        constraints: &[Constraint],
        objective: &Objective,
        nonfinite_strings: bool,
    ) -> Result<Self, IoError> {
        for v in vars {
            match v.domain {
                Domain::Real | Domain::Integer | Domain::Binary => {}
                Domain::SemiContinuous { .. } => {
                    return Err(IoError::UnsupportedDomain("SemiContinuous"));
                }
                Domain::SemiInteger { .. } => {
                    return Err(IoError::UnsupportedDomain("SemiInteger"));
                }
            }
        }

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
                    validate(arena, r.id, nonfinite_strings)?;
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
                validate(arena, r.id, nonfinite_strings)?;
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

/// Walk a nonlinear residual, rejecting nodes the writer cannot emit. Non-finite
/// constants are an error only when `nonfinite_strings` is off. When on, the
/// writer emits them as `Infinity`/`NaN`, so they are allowed through this function
/// to keep `WriteOptions::nonfinite_strings` effective for expression constants.
fn validate(arena: &ExprArena, id: ExprId, nonfinite_strings: bool) -> Result<(), IoError> {
    match arena.get(id) {
        ExprNode::Const(c) => {
            if !nonfinite_strings && !c.is_finite() {
                return Err(IoError::InvalidNumber {
                    value: *c,
                    location: "an expression constant".into(),
                });
            }
            Ok(())
        }
        ExprNode::Var(_) => Ok(()),
        ExprNode::Param(p) => {
            let value = arena.param_value(*p);
            if !nonfinite_strings && !value.is_finite() {
                return Err(IoError::InvalidNumber { value, location: "a parameter".into() });
            }
            Ok(())
        }
        ExprNode::Neg(x)
        | ExprNode::Sin(x)
        | ExprNode::Cos(x)
        | ExprNode::Exp(x)
        | ExprNode::Log(x)
        | ExprNode::Abs(x) => validate(arena, *x, nonfinite_strings),
        ExprNode::Pow(b, e) => {
            validate(arena, *b, nonfinite_strings)?;
            validate(arena, *e, nonfinite_strings)
        }
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            for c in children {
                validate(arena, *c, nonfinite_strings)?;
            }
            Ok(())
        }
        ExprNode::Div(num, den) => {
            validate(arena, *num, nonfinite_strings)?;
            validate(arena, *den, nonfinite_strings)
        }
        ExprNode::Linear { coeffs: _, constant } => {
            if !nonfinite_strings && !constant.is_finite() {
                return Err(IoError::InvalidNumber {
                    value: *constant,
                    location: "a linear expression constant".into(),
                });
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
