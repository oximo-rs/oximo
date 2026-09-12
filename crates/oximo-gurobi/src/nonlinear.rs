//! Lower nonlinear oximo expressions onto Gurobi 13 expression trees.

use std::collections::HashMap;

use gurobi_rs::Opcode;
use gurobi_rs::expr::{LinExpr, QuadExpr};
use gurobi_rs::prelude::*;
use oximo_expr::{ExprArena, ExprId, ExprNode, UnaryOp, VarId};
use oximo_solver::SolverError;

/// A value that can stay on Gurobi's direct linear/quadratic fast path, or a
/// variable containing a native nonlinear expression-tree result.
#[derive(Clone)]
pub(crate) enum LoweredExpr {
    Linear(LinExpr),
    Quadratic(QuadExpr),
    Var(Var),
}

/// A definition generated while lowering one original nonlinear constraint.
/// IIS membership of any of these definitions is attributed to that original
/// constraint by the translation layer.
pub(crate) enum GeneratedConstraint {
    Linear(Constr),
    Quadratic(QConstr),
    General(GenConstr),
}

pub(crate) struct LoweringCtx<'a> {
    pub model: &'a mut Model,
    pub gurobi_vars: &'a [Var],
    pub aux_counter: u32,
    pub generated: Vec<GeneratedConstraint>,
    quadratic_cache: HashMap<ExprId, Option<LoweredExpr>>,
    materialized: HashMap<ExprId, Var>,
    reference_counts: Vec<usize>,
}

impl<'a> LoweringCtx<'a> {
    pub(crate) fn new(model: &'a mut Model, gurobi_vars: &'a [Var], aux_counter: u32) -> Self {
        LoweringCtx {
            model,
            gurobi_vars,
            aux_counter,
            generated: Vec::new(),
            quadratic_cache: HashMap::new(),
            materialized: HashMap::new(),
            reference_counts: Vec::new(),
        }
    }

    fn next_name(&mut self, tag: &str) -> String {
        let n = self.aux_counter;
        self.aux_counter = self.aux_counter.saturating_add(1);
        format!("aux_{tag}_{n}")
    }

    fn new_aux(&mut self, tag: &str, lb: f64, ub: f64) -> Result<Var, gurobi_rs::Error> {
        let name = self.next_name(tag);
        self.model.add_var(&name, Continuous, 0.0, lb, ub, [])
    }

    fn variable(&self, v: VarId) -> Result<Var, SolverError> {
        self.gurobi_vars
            .get(v.index())
            .copied()
            .ok_or_else(|| SolverError::Backend(format!("variable index {} is out of range", v.0)))
    }
}

fn map_gurobi(e: gurobi_rs::Error) -> SolverError {
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
    let (coeffs, offset) = e.into_parts();
    q.add_constant(offset);
    for (v, c) in coeffs {
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
    let (coeffs, offset) = linexpr.into_parts();
    a.add_constant(offset);
    for (v, c) in coeffs {
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

fn linear_is_constant(e: &LinExpr) -> bool {
    e.iter_terms().next().is_none()
}

fn multiply_linears(a: &LinExpr, b: &LinExpr) -> QuadExpr {
    let a_off = a.get_offset();
    let b_off = b.get_offset();
    let mut q = QuadExpr::new();
    q.add_constant(a_off * b_off);
    for (va, ca) in a.iter_terms() {
        q.add_term(ca * b_off, *va);
    }
    for (vb, cb) in b.iter_terms() {
        q.add_term(a_off * cb, *vb);
    }
    for (va, ca) in a.iter_terms() {
        for (vb, cb) in b.iter_terms() {
            q.add_qterm(ca * cb, *va, *vb);
        }
    }
    q
}

fn into_linexpr(l: LoweredExpr) -> LinExpr {
    match l {
        LoweredExpr::Linear(e) => e,
        LoweredExpr::Var(v) => linear_from_var(v),
        LoweredExpr::Quadratic(_) => panic!("internal: expected a linear expression"),
    }
}

fn try_add(a: LoweredExpr, b: LoweredExpr) -> LoweredExpr {
    use LoweredExpr::{Linear, Quadratic, Var};
    match (a, b) {
        (Linear(x), Linear(y)) => Linear(add_linears(x, y)),
        (Quadratic(x), Quadratic(y)) => Quadratic(add_quads(x, y)),
        (Quadratic(x), other) | (other, Quadratic(x)) => {
            Quadratic(add_quads(x, quad_from_linear(into_linexpr(other))))
        }
        (Var(v), Linear(y)) | (Linear(y), Var(v)) => Linear(add_linears(y, linear_from_var(v))),
        (Var(v), Var(w)) => Linear(add_linears(linear_from_var(v), linear_from_var(w))),
    }
}

fn try_scale(l: LoweredExpr, k: f64) -> LoweredExpr {
    match l {
        LoweredExpr::Linear(e) => LoweredExpr::Linear(scale_linear(e, k)),
        LoweredExpr::Quadratic(e) => LoweredExpr::Quadratic(scale_quad(e, k)),
        LoweredExpr::Var(v) => LoweredExpr::Linear(scale_linear(linear_from_var(v), k)),
    }
}

fn try_mul(values: Vec<LoweredExpr>) -> Option<LoweredExpr> {
    let mut acc = LoweredExpr::Linear(linear_constant(1.0));
    for value in values {
        acc = match (acc, value) {
            (LoweredExpr::Linear(a), LoweredExpr::Linear(b)) => {
                if linear_is_constant(&a) {
                    LoweredExpr::Linear(scale_linear(b, a.get_offset()))
                } else if linear_is_constant(&b) {
                    LoweredExpr::Linear(scale_linear(a, b.get_offset()))
                } else {
                    LoweredExpr::Quadratic(multiply_linears(&a, &b))
                }
            }
            (LoweredExpr::Quadratic(a), LoweredExpr::Linear(b)) => {
                if linear_is_constant(&b) {
                    LoweredExpr::Quadratic(scale_quad(a, b.get_offset()))
                } else {
                    return None;
                }
            }
            (LoweredExpr::Linear(a), LoweredExpr::Quadratic(b)) => {
                if linear_is_constant(&a) {
                    LoweredExpr::Quadratic(scale_quad(b, a.get_offset()))
                } else {
                    return None;
                }
            }
            (LoweredExpr::Quadratic(_), LoweredExpr::Quadratic(_)) => return None,
            (LoweredExpr::Var(_), _) | (_, LoweredExpr::Var(_)) => {
                unreachable!("quadratic probe normalizes variables")
            }
        };
    }
    Some(acc)
}

/// Probe whether an expression is representable directly as linear or
/// quadratic.
fn try_quadratic(
    arena: &ExprArena,
    id: ExprId,
    ctx: &mut LoweringCtx<'_>,
) -> Result<Option<LoweredExpr>, SolverError> {
    if let Some(value) = ctx.quadratic_cache.get(&id) {
        return Ok(value.clone());
    }
    let value = match arena.get(id) {
        ExprNode::Const(c) => LoweredExpr::Linear(linear_constant(*c)),
        ExprNode::Param(p) => LoweredExpr::Linear(linear_constant(arena.param_value(*p))),
        ExprNode::Var(v) => LoweredExpr::Linear(linear_from_var(ctx.variable(*v)?)),
        ExprNode::Linear { coeffs, constant } => {
            let mut e = linear_constant(*constant);
            for &(v, c) in coeffs {
                e.add_term(c, ctx.variable(v)?);
            }
            LoweredExpr::Linear(e)
        }
        ExprNode::Unary(UnaryOp::Neg, inner) => {
            let Some(inner) = try_quadratic(arena, *inner, ctx)? else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            try_scale(inner, -1.0)
        }
        ExprNode::Add(children) => {
            let mut acc = LoweredExpr::Linear(linear_constant(0.0));
            for child in children {
                let Some(value) = try_quadratic(arena, *child, ctx)? else {
                    return Ok(cache_quadratic_none(ctx, id));
                };
                acc = try_add(acc, value);
            }
            acc
        }
        ExprNode::Mul(children) => {
            let mut values = Vec::with_capacity(children.len());
            for child in children {
                let Some(value) = try_quadratic(arena, *child, ctx)? else {
                    return Ok(cache_quadratic_none(ctx, id));
                };
                values.push(value);
            }
            let Some(value) = try_mul(values) else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            value
        }
        ExprNode::Pow(base, exp) => {
            let Some(alpha) = as_const(arena, *exp) else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            let Some(base) = try_quadratic(arena, *base, ctx)? else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            if alpha == 0.0 {
                LoweredExpr::Linear(linear_constant(1.0))
            } else if (alpha - 1.0).abs() < f64::EPSILON {
                base
            } else if (alpha - 2.0).abs() < f64::EPSILON {
                let LoweredExpr::Linear(base) = base else {
                    return Ok(cache_quadratic_none(ctx, id));
                };
                LoweredExpr::Quadratic(multiply_linears(&base, &base))
            } else {
                return Ok(cache_quadratic_none(ctx, id));
            }
        }
        ExprNode::Div(num, den) => {
            let Some(den) = as_const(arena, *den) else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            if den == 0.0 {
                return Err(SolverError::Backend(
                    "division by zero: constant denominator is 0".into(),
                ));
            }
            let Some(num) = try_quadratic(arena, *num, ctx)? else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            try_scale(num, 1.0 / den)
        }
        ExprNode::Unary(_, _) | ExprNode::Atan2(_, _) | ExprNode::Min(_) | ExprNode::Max(_) => {
            return Ok(cache_quadratic_none(ctx, id));
        }
    };
    ctx.quadratic_cache.insert(id, Some(value.clone()));
    Ok(Some(value))
}

fn cache_quadratic_none(ctx: &mut LoweringCtx<'_>, id: ExprId) -> Option<LoweredExpr> {
    ctx.quadratic_cache.insert(id, None);
    None
}

fn for_each_child(node: &ExprNode, f: &mut impl FnMut(ExprId)) {
    match node {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => {
            for child in children {
                f(*child);
            }
        }
        ExprNode::Unary(_, inner) => f(*inner),
        ExprNode::Pow(base, exp) | ExprNode::Div(base, exp) | ExprNode::Atan2(base, exp) => {
            f(*base);
            f(*exp);
        }
        ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. } => {}
    }
}

fn reference_counts(arena: &ExprArena, root: ExprId) -> Vec<usize> {
    let mut seen = vec![false; arena.len()];
    let mut order = Vec::new();
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if expanded {
            order.push(id);
            continue;
        }
        if seen[id.index()] {
            continue;
        }
        seen[id.index()] = true;
        stack.push((id, true));
        for_each_child(arena.get(id), &mut |child| stack.push((child, false)));
    }

    let mut counts = vec![0_usize; arena.len()];
    counts[root.index()] = 1;
    for id in order.into_iter().rev() {
        let count = counts[id.index()];
        for_each_child(arena.get(id), &mut |child| {
            counts[child.index()] = counts[child.index()].saturating_add(count);
        });
    }
    counts
}

struct TreeBuilder {
    opcode: Vec<i32>,
    data: Vec<f64>,
    parent: Vec<i32>,
    materializing: Option<ExprId>,
}

impl TreeBuilder {
    fn new(materializing: Option<ExprId>) -> Self {
        Self { opcode: Vec::new(), data: Vec::new(), parent: Vec::new(), materializing }
    }

    fn push(
        &mut self,
        opcode: Opcode,
        data: f64,
        parent: Option<usize>,
    ) -> Result<usize, SolverError> {
        let index = self.opcode.len();
        let parent = parent.map_or(Ok(-1), i32::try_from).map_err(|_| {
            SolverError::Backend("nonlinear expression tree parent index exceeds i32".into())
        })?;
        self.opcode.push(opcode as i32);
        self.data.push(data);
        self.parent.push(parent);
        Ok(index)
    }
}

fn as_const(arena: &ExprArena, id: ExprId) -> Option<f64> {
    match arena.get(id) {
        ExprNode::Const(c) => Some(*c),
        ExprNode::Param(p) => Some(arena.param_value(*p)),
        _ => None,
    }
}

fn append_sum(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    ids: &[ExprId],
    parent: Option<usize>,
    tree: &mut TreeBuilder,
) -> Result<usize, SolverError> {
    match ids {
        [] => tree.push(Opcode::Constant, 0.0, parent),
        [id] => append_tree(ctx, arena, *id, parent, tree),
        _ => {
            let mut operators = Vec::with_capacity(ids.len() - 1);
            let mut current_parent = tree.push(Opcode::Plus, 0.0, parent)?;
            operators.push(current_parent);
            for _ in 1..ids.len() - 1 {
                current_parent = tree.push(Opcode::Plus, 0.0, Some(current_parent))?;
                operators.push(current_parent);
            }

            append_tree(ctx, arena, ids[0], Some(current_parent), tree)?;
            if ids.len() > 2 {
                append_tree(ctx, arena, ids[1], Some(current_parent), tree)?;
                for (id, op) in ids[2..ids.len() - 1].iter().zip(operators.iter().rev().skip(1)) {
                    append_tree(ctx, arena, *id, Some(*op), tree)?;
                }
            }
            append_tree(ctx, arena, ids[ids.len() - 1], Some(operators[0]), tree)
        }
    }
}

fn append_product(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    ids: &[ExprId],
    parent: Option<usize>,
    tree: &mut TreeBuilder,
) -> Result<usize, SolverError> {
    match ids {
        [] => tree.push(Opcode::Constant, 1.0, parent),
        [id] => append_tree(ctx, arena, *id, parent, tree),
        _ => {
            let mut operators = Vec::with_capacity(ids.len() - 1);
            let mut current_parent = tree.push(Opcode::Multiply, 0.0, parent)?;
            operators.push(current_parent);
            for _ in 1..ids.len() - 1 {
                current_parent = tree.push(Opcode::Multiply, 0.0, Some(current_parent))?;
                operators.push(current_parent);
            }

            append_tree(ctx, arena, ids[0], Some(current_parent), tree)?;
            if ids.len() > 2 {
                append_tree(ctx, arena, ids[1], Some(current_parent), tree)?;
                for (id, op) in ids[2..ids.len() - 1].iter().zip(operators.iter().rev().skip(1)) {
                    append_tree(ctx, arena, *id, Some(*op), tree)?;
                }
            }
            append_tree(ctx, arena, ids[ids.len() - 1], Some(operators[0]), tree)
        }
    }
}

enum LinearPart {
    Constant(f64),
    Term(Var, f64),
}

fn append_linear_parts(
    ctx: &mut LoweringCtx<'_>,
    parts: &[LinearPart],
    parent: Option<usize>,
    tree: &mut TreeBuilder,
) -> Result<usize, SolverError> {
    fn append_part(
        ctx: &mut LoweringCtx<'_>,
        part: &LinearPart,
        parent: Option<usize>,
        tree: &mut TreeBuilder,
    ) -> Result<usize, SolverError> {
        match part {
            LinearPart::Constant(c) => tree.push(Opcode::Constant, *c, parent),
            LinearPart::Term(var, coeff) => {
                let op = tree.push(Opcode::Multiply, 0.0, parent)?;
                tree.push(Opcode::Constant, *coeff, Some(op))?;
                let index = ctx.model.var_index(var).map_err(map_gurobi)?;
                tree.push(Opcode::Variable, f64::from(index), Some(op))
            }
        }
    }

    match parts {
        [] => tree.push(Opcode::Constant, 0.0, parent),
        [part] => append_part(ctx, part, parent, tree),
        _ => {
            let mut operators = Vec::with_capacity(parts.len() - 1);
            let mut current_parent = tree.push(Opcode::Plus, 0.0, parent)?;
            operators.push(current_parent);
            for _ in 1..parts.len() - 1 {
                current_parent = tree.push(Opcode::Plus, 0.0, Some(current_parent))?;
                operators.push(current_parent);
            }

            append_part(ctx, &parts[0], Some(current_parent), tree)?;
            if parts.len() > 2 {
                append_part(ctx, &parts[1], Some(current_parent), tree)?;
                for (part, op) in
                    parts[2..parts.len() - 1].iter().zip(operators.iter().rev().skip(1))
                {
                    append_part(ctx, part, Some(*op), tree)?;
                }
            }
            append_part(ctx, &parts[parts.len() - 1], Some(operators[0]), tree)
        }
    }
}

fn materialize_lowered(ctx: &mut LoweringCtx<'_>, value: LoweredExpr) -> Result<Var, SolverError> {
    match value {
        LoweredExpr::Var(v) => Ok(v),
        LoweredExpr::Linear(e) => {
            let aux =
                ctx.new_aux("affine", f64::NEG_INFINITY, f64::INFINITY).map_err(map_gurobi)?;
            let name = ctx.next_name("affine_eq");
            let row = ctx.model.add_constr(&name, c!(aux == e)).map_err(map_gurobi)?;
            ctx.generated.push(GeneratedConstraint::Linear(row));
            Ok(aux)
        }
        LoweredExpr::Quadratic(e) => {
            let aux =
                ctx.new_aux("quadratic", f64::NEG_INFINITY, f64::INFINITY).map_err(map_gurobi)?;
            let name = ctx.next_name("quadratic_eq");
            let row = ctx.model.add_qconstr(&name, c!(aux == e)).map_err(map_gurobi)?;
            ctx.generated.push(GeneratedConstraint::Quadratic(row));
            Ok(aux)
        }
    }
}

fn materialize_abs(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    inner: ExprId,
) -> Result<Var, SolverError> {
    let argument = if let Some(value) = try_quadratic(arena, inner, ctx)? {
        materialize_lowered(ctx, value)?
    } else {
        native_expr(ctx, arena, inner)?
    };
    let result = ctx.new_aux("abs", 0.0, f64::INFINITY).map_err(map_gurobi)?;
    let name = ctx.next_name("abs_def");
    let generated = ctx.model.add_genconstr_abs(&name, result, argument).map_err(map_gurobi)?;
    ctx.generated.push(GeneratedConstraint::General(generated));
    Ok(result)
}

pub(crate) fn validate_supported(
    arena: &ExprArena,
    roots: impl IntoIterator<Item = ExprId>,
) -> Result<(), SolverError> {
    let mut seen = vec![false; arena.len()];
    let mut stack: Vec<_> = roots.into_iter().collect();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.index()], true) {
            continue;
        }
        match arena.get(id) {
            ExprNode::Unary(op, child) => {
                if !matches!(
                    op,
                    UnaryOp::Neg
                        | UnaryOp::Abs
                        | UnaryOp::Sqrt
                        | UnaryOp::Exp
                        | UnaryOp::Exp2
                        | UnaryOp::Log
                        | UnaryOp::Log2
                        | UnaryOp::Log10
                        | UnaryOp::Sin
                        | UnaryOp::Cos
                        | UnaryOp::Tan
                        | UnaryOp::Tanh
                ) {
                    return Err(SolverError::UnsupportedNonlinearOperator {
                        backend: "Gurobi",
                        operator: op.name(),
                    });
                }
                stack.push(*child);
            }
            ExprNode::Add(children)
            | ExprNode::Mul(children)
            | ExprNode::Min(children)
            | ExprNode::Max(children) => stack.extend(children.iter().copied()),
            ExprNode::Pow(a, b) | ExprNode::Div(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            ExprNode::Atan2(_, _) => {
                return Err(SolverError::UnsupportedNonlinearOperator {
                    backend: "Gurobi",
                    operator: "atan2",
                });
            }
            ExprNode::Const(_)
            | ExprNode::Var(_)
            | ExprNode::Param(_)
            | ExprNode::Linear { .. } => {}
        }
    }
    Ok(())
}

fn materialize_extrema(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    children: &[ExprId],
    is_min: bool,
) -> Result<Var, SolverError> {
    let mut arguments = Vec::with_capacity(children.len());
    for child in children {
        let argument = if let Some(value) = try_quadratic(arena, *child, ctx)? {
            materialize_lowered(ctx, value)?
        } else {
            native_expr(ctx, arena, *child)?
        };
        arguments.push(argument);
    }
    let tag = if is_min { "min" } else { "max" };
    let result = ctx.new_aux(tag, f64::NEG_INFINITY, f64::INFINITY).map_err(map_gurobi)?;
    let name = ctx.next_name(&format!("{tag}_def"));
    let generated = if is_min {
        ctx.model.add_genconstr_min(&name, result, arguments, Some(gurobi_rs::INFINITY))
    } else {
        ctx.model.add_genconstr_max(&name, result, arguments, Some(-gurobi_rs::INFINITY))
    }
    .map_err(map_gurobi)?;
    ctx.generated.push(GeneratedConstraint::General(generated));
    Ok(result)
}

fn materialize_shared(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    id: ExprId,
) -> Result<Var, SolverError> {
    if let Some(&result) = ctx.materialized.get(&id) {
        return Ok(result);
    }
    let result = if let Some(value) = try_quadratic(arena, id, ctx)? {
        materialize_lowered(ctx, value)?
    } else {
        match arena.get(id) {
            ExprNode::Unary(UnaryOp::Abs, inner) => materialize_abs(ctx, arena, *inner)?,
            ExprNode::Min(children) => materialize_extrema(ctx, arena, children, true)?,
            ExprNode::Max(children) => materialize_extrema(ctx, arena, children, false)?,
            _ => native_expr(ctx, arena, id)?,
        }
    };
    ctx.materialized.insert(id, result);
    Ok(result)
}

#[expect(clippy::too_many_lines)]
fn append_tree(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    id: ExprId,
    parent: Option<usize>,
    tree: &mut TreeBuilder,
) -> Result<usize, SolverError> {
    if tree.materializing != Some(id) {
        if let Some(&result) = ctx.materialized.get(&id) {
            let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
            return tree.push(Opcode::Variable, f64::from(index), parent);
        }
        let shared = ctx.reference_counts.get(id.index()).copied().unwrap_or_default() > 1;
        let leaf =
            matches!(arena.get(id), ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_));
        if shared && !leaf {
            let result = materialize_shared(ctx, arena, id)?;
            let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
            return tree.push(Opcode::Variable, f64::from(index), parent);
        }
    }
    match arena.get(id) {
        ExprNode::Const(c) => tree.push(Opcode::Constant, *c, parent),
        ExprNode::Param(p) => tree.push(Opcode::Constant, arena.param_value(*p), parent),
        ExprNode::Var(v) => {
            let var = ctx.variable(*v)?;
            let index = ctx.model.var_index(&var).map_err(map_gurobi)?;
            tree.push(Opcode::Variable, f64::from(index), parent)
        }
        ExprNode::Linear { coeffs, constant } => {
            let mut terms = Vec::with_capacity(coeffs.len() + usize::from(*constant != 0.0));
            if *constant != 0.0 {
                terms.push(LinearPart::Constant(*constant));
            }
            for &(var, coeff) in coeffs {
                let v = ctx.variable(var)?;
                terms.push(LinearPart::Term(v, coeff));
            }
            append_linear_parts(ctx, &terms, parent, tree)
        }
        ExprNode::Unary(UnaryOp::Neg, inner) => {
            let op = tree.push(Opcode::Uminus, 0.0, parent)?;
            append_tree(ctx, arena, *inner, Some(op), tree)
        }
        ExprNode::Add(children) => append_sum(ctx, arena, children, parent, tree),
        ExprNode::Mul(children) => append_product(ctx, arena, children, parent, tree),
        ExprNode::Pow(base, exp) => {
            let op = tree.push(Opcode::Pow, 0.0, parent)?;
            append_tree(ctx, arena, *base, Some(op), tree)?;
            append_tree(ctx, arena, *exp, Some(op), tree)
        }
        ExprNode::Div(num, den) => {
            if let Some(0.0) = as_const(arena, *den) {
                return Err(SolverError::Backend(
                    "division by zero: constant denominator is 0".into(),
                ));
            }
            let op = tree.push(Opcode::Divide, 0.0, parent)?;
            append_tree(ctx, arena, *num, Some(op), tree)?;
            append_tree(ctx, arena, *den, Some(op), tree)
        }
        ExprNode::Unary(op, inner)
            if matches!(
                op,
                UnaryOp::Sqrt
                    | UnaryOp::Sin
                    | UnaryOp::Cos
                    | UnaryOp::Tan
                    | UnaryOp::Exp
                    | UnaryOp::Log
                    | UnaryOp::Log2
                    | UnaryOp::Log10
                    | UnaryOp::Tanh
            ) =>
        {
            let opcode = match op {
                UnaryOp::Sqrt => Opcode::Sqrt,
                UnaryOp::Sin => Opcode::Sin,
                UnaryOp::Cos => Opcode::Cos,
                UnaryOp::Tan => Opcode::Tan,
                UnaryOp::Exp => Opcode::Exp,
                UnaryOp::Log => Opcode::Log,
                UnaryOp::Log2 => Opcode::Log2,
                UnaryOp::Log10 => Opcode::Log10,
                UnaryOp::Tanh => Opcode::Tanh,
                _ => unreachable!(),
            };
            let op = tree.push(opcode, 0.0, parent)?;
            append_tree(ctx, arena, *inner, Some(op), tree)
        }
        ExprNode::Unary(UnaryOp::Exp2, inner) => {
            let op = tree.push(Opcode::Pow, 0.0, parent)?;
            tree.push(Opcode::Constant, 2.0, Some(op))?;
            append_tree(ctx, arena, *inner, Some(op), tree)
        }
        ExprNode::Unary(UnaryOp::Abs, inner) => {
            let result = materialize_abs(ctx, arena, *inner)?;
            let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
            tree.push(Opcode::Variable, f64::from(index), parent)
        }
        ExprNode::Min(children) | ExprNode::Max(children) => {
            let result = materialize_extrema(
                ctx,
                arena,
                children,
                matches!(arena.get(id), ExprNode::Min(_)),
            )?;
            let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
            tree.push(Opcode::Variable, f64::from(index), parent)
        }
        ExprNode::Unary(op, _) => Err(SolverError::UnsupportedNonlinearOperator {
            backend: "Gurobi",
            operator: op.name(),
        }),
        ExprNode::Atan2(_, _) => {
            Err(SolverError::UnsupportedNonlinearOperator { backend: "Gurobi", operator: "atan2" })
        }
    }
}

fn native_expr(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    id: ExprId,
) -> Result<Var, SolverError> {
    let result = ctx.new_aux("nl", f64::NEG_INFINITY, f64::INFINITY).map_err(map_gurobi)?;
    let mut tree = TreeBuilder::new(Some(id));
    append_tree(ctx, arena, id, None, &mut tree)?;
    let name = ctx.next_name("nl_def");
    let generated = ctx
        .model
        .add_genconstr_nl(&name, result, &tree.opcode, &tree.data, &tree.parent)
        .map_err(map_gurobi)?;
    ctx.generated.push(GeneratedConstraint::General(generated));
    ctx.materialized.insert(id, result);
    Ok(result)
}

pub(crate) fn lower(
    arena: &ExprArena,
    id: ExprId,
    ctx: &mut LoweringCtx<'_>,
) -> Result<LoweredExpr, SolverError> {
    if let Some(value) = try_quadratic(arena, id, ctx)? {
        return Ok(value);
    }
    ctx.reference_counts = reference_counts(arena, id);
    Ok(LoweredExpr::Var(native_expr(ctx, arena, id)?))
}

impl LoweredExpr {
    pub(crate) fn into_expr_for_objective(self) -> gurobi_rs::expr::Expr {
        match self {
            Self::Linear(e) => gurobi_rs::expr::Expr::from(e),
            Self::Quadratic(e) => gurobi_rs::expr::Expr::from(e),
            Self::Var(v) => gurobi_rs::expr::Expr::from(v),
        }
    }
}
