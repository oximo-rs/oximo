//! Conversion from oximo's expression arena to POUNCE's FBBT tapes.
//!
//! The builder owns the presolve/FBBT driver in `pounce-rs`.
//! oximo only needs to provide the structural expression
//! capability through `Problem`.
//!
//! This capability is currently used by the stable builder path.

// TODO: Add support for Nightly/TNLP once v0.12 releases.

use std::collections::HashMap;

use oximo_core::{ExprArenaSnapshot, ExprId, ExprNode, Model};
use pounce_rs::{FbbtOp, FbbtTape};

/// Build one tape per algebraic constraint.
/// Unsupported operators are kept as opaque slots so the builder
/// can safely fall back to callback evaluation.
pub(crate) fn constraint_tapes(model: &Model) -> Vec<Option<FbbtTape>> {
    let arena = model.arena();
    model
        .constraints()
        .algebraic()
        .iter()
        .map(|constraint| Some(tape_for(&arena, model, constraint.lhs)))
        .collect()
}

fn tape_for(arena: &ExprArenaSnapshot<'_>, model: &Model, root: ExprId) -> FbbtTape {
    let mut ops = Vec::new();
    let mut slots = HashMap::new();
    emit(arena, model, root, &mut ops, &mut slots);
    FbbtTape { ops }
}

fn emit(
    arena: &ExprArenaSnapshot<'_>,
    model: &Model,
    id: ExprId,
    ops: &mut Vec<FbbtOp>,
    slots: &mut HashMap<ExprId, usize>,
) -> usize {
    if let Some(&slot) = slots.get(&id) {
        return slot;
    }
    let slot = match arena.get(id).clone() {
        ExprNode::Const(value) => push(ops, FbbtOp::Const(value)),
        ExprNode::Param(param) => push(ops, FbbtOp::Const(model.param_value(param))),
        ExprNode::Var(var) => push(ops, FbbtOp::Var(var.index())),
        ExprNode::Neg(child) => {
            let a = emit(arena, model, child, ops, slots);
            push(ops, FbbtOp::Neg(a))
        }
        ExprNode::Sin(child) => {
            let a = emit(arena, model, child, ops, slots);
            push(ops, FbbtOp::Sin(a))
        }
        ExprNode::Cos(child) => {
            let a = emit(arena, model, child, ops, slots);
            push(ops, FbbtOp::Cos(a))
        }
        ExprNode::Exp(child) => {
            let a = emit(arena, model, child, ops, slots);
            push(ops, FbbtOp::Exp(a))
        }
        ExprNode::Log(child) => {
            let a = emit(arena, model, child, ops, slots);
            push(ops, FbbtOp::Ln(a))
        }
        ExprNode::Abs(child) => {
            let a = emit(arena, model, child, ops, slots);
            push(ops, FbbtOp::Abs(a))
        }
        ExprNode::Div(lhs, rhs) => {
            let a = emit(arena, model, lhs, ops, slots);
            let b = emit(arena, model, rhs, ops, slots);
            push(ops, FbbtOp::Div(a, b))
        }
        ExprNode::Pow(base, exponent) => {
            let a = emit(arena, model, base, ops, slots);
            match constant_value(arena, model, exponent) {
                Some(0.5) => push(ops, FbbtOp::Sqrt(a)),
                Some(value)
                    if value.is_finite()
                        && value >= 0.0
                        && value.fract() == 0.0
                        && value <= f64::from(u32::MAX) =>
                {
                    push(ops, FbbtOp::PowInt(a, integer_exponent(value)))
                }
                _ => push(ops, FbbtOp::Opaque),
            }
        }
        ExprNode::Add(children) => fold(arena, model, children.as_slice(), ops, slots, 0.0, true),
        ExprNode::Mul(children) => fold(arena, model, children.as_slice(), ops, slots, 1.0, false),
        ExprNode::Linear { coeffs, constant } => {
            let mut acc = push(ops, FbbtOp::Const(constant));
            for (var, coefficient) in coeffs {
                let variable = push(ops, FbbtOp::Var(var.index()));
                let factor = push(ops, FbbtOp::Const(coefficient));
                let term = push(ops, FbbtOp::Mul(variable, factor));
                acc = push(ops, FbbtOp::Add(acc, term));
            }
            acc
        }
    };
    slots.insert(id, slot);
    slot
}

fn fold(
    arena: &ExprArenaSnapshot<'_>,
    model: &Model,
    children: &[ExprId],
    ops: &mut Vec<FbbtOp>,
    slots: &mut HashMap<ExprId, usize>,
    identity: f64,
    add: bool,
) -> usize {
    let mut acc = push(ops, FbbtOp::Const(identity));
    for &child in children {
        let next = emit(arena, model, child, ops, slots);
        acc =
            if add { push(ops, FbbtOp::Add(acc, next)) } else { push(ops, FbbtOp::Mul(acc, next)) };
    }
    acc
}

fn constant_value(arena: &ExprArenaSnapshot<'_>, model: &Model, id: ExprId) -> Option<f64> {
    match arena.get(id) {
        ExprNode::Const(value) => Some(*value),
        ExprNode::Param(param) => Some(model.param_value(*param)),
        _ => None,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the caller proves the finite value is an integer in the u32 range"
)]
fn integer_exponent(value: f64) -> u32 {
    value as u32
}

#[inline]
fn push(ops: &mut Vec<FbbtOp>, op: FbbtOp) -> usize {
    let index = ops.len();
    ops.push(op);
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::{Model, constraint, objective, param, variable};

    #[test]
    fn emits_valid_tape_for_quadratic_constraint() {
        let model = Model::new("fbbt");
        variable!(model, -2.0 <= x <= 2.0);
        objective!(model, Min, x);
        constraint!(model, c, x * x <= 1.0);
        let tape = constraint_tapes(&model).remove(0).unwrap();
        assert_eq!(tape.first_invalid_slot(), None);
        assert!(tape.ops.iter().any(|op| matches!(op, FbbtOp::Mul(..) | FbbtOp::PowInt(..))));
    }

    #[test]
    fn emits_supported_nodes_and_opaque_powers() {
        let model = Model::new("fbbt_nodes");
        param!(model, exponent = 3.0);
        variable!(model, -2.0 <= x <= 2.0);
        variable!(model, -2.0 <= y <= 2.0);

        let constant = model.__constant(4.0);
        let arena = model.arena();
        let constant_tape = tape_for(&arena, &model, constant.id);
        assert!(matches!(constant_tape.ops.as_slice(), [FbbtOp::Const(4.0)]));

        let shared = x.sin();
        constraint!(model, unary, -shared + x.cos() + x.exp() + x.log() + x.abs() <= 100.0);
        constraint!(model, division, x.sin() / (y + 1.0) <= 100.0);
        constraint!(model, square_root, x.powf(0.5) <= 100.0);
        constraint!(model, integer_power, x.powi(3) <= 100.0);
        constraint!(model, parameter_power, x.pow(exponent) <= 100.0);
        constraint!(model, variable_power, x.pow(y) <= 100.0);
        constraint!(model, linear, 2.0 * x + 3.0 * y + 1.0 <= 100.0);
        constraint!(model, constant, model.__constant(1.0) <= 2.0);

        let tapes = constraint_tapes(&model);
        assert_eq!(tapes.len(), 8);
        let ops = tapes.into_iter().flatten().flat_map(|tape| tape.ops).collect::<Vec<_>>();

        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Neg(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Sin(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Cos(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Exp(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Ln(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Abs(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Div(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Sqrt(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::PowInt(_, 3))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Opaque)));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Add(..))));
        assert!(ops.iter().any(|op| matches!(op, FbbtOp::Mul(..))));
    }

    #[test]
    fn memoizes_reused_expression_slots() {
        let model = Model::new("fbbt_memo");
        variable!(model, -2.0 <= x <= 2.0);
        let shared = x.sin();
        constraint!(model, reused, shared + shared <= 1.0);

        let tape = constraint_tapes(&model).remove(0).unwrap();
        let sin_slot = tape.ops.iter().position(|op| matches!(op, FbbtOp::Sin(..))).unwrap();
        assert_eq!(tape.ops.iter().filter(|op| matches!(op, FbbtOp::Sin(..))).count(), 1);
        assert_eq!(
            tape.ops
                .iter()
                .filter(|op| matches!(op, FbbtOp::Add(left, right) if *left == sin_slot || *right == sin_slot))
                .count(),
            2
        );
    }
}
