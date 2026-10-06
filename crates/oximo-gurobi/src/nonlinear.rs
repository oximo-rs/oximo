//! Lower nonlinear oximo expressions onto Gurobi 13 expression trees.

use std::collections::HashMap;

use gurobi_rs::Opcode;
use gurobi_rs::expr::{LinExpr, QuadExpr};
use gurobi_rs::prelude::*;
use oximo_expr::{ExprArena, ExprId, ExprNode, UnaryOp, VarId, extract_linear};
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
    if let Some(terms) = extract_linear(arena, id) {
        let mut value = linear_constant(terms.constant);
        for &(var, coeff) in terms.coeffs.iter() {
            value.add_term(coeff, ctx.variable(var)?);
        }
        let value = LoweredExpr::Linear(value);
        ctx.quadratic_cache.insert(id, Some(value.clone()));
        return Ok(Some(value));
    }

    enum Task<'a> {
        Visit(ExprId),
        Next { parent: ExprId, child: ExprId, rest: &'a [ExprId] },
    }
    // Visit one child at a time so a failed polynomial child stops its parent.
    let mut affine = HashMap::new();
    let mut pending = vec![Task::Visit(id)];
    while let Some(task) = pending.pop() {
        let (current, children) = match task {
            Task::Next { parent, child, rest } => {
                if ctx.quadratic_cache[&child].is_none() {
                    cache_quadratic_none(ctx, parent);
                    continue;
                }
                (parent, rest)
            }
            Task::Visit(current) => {
                if ctx.quadratic_cache.contains_key(&current) {
                    continue;
                }
                if current != id
                    && affine_region(arena, current, &mut affine).is_some()
                    && let Some(terms) = extract_linear(arena, current)
                {
                    let mut value = linear_constant(terms.constant);
                    for &(var, coeff) in terms.coeffs.iter() {
                        value.add_term(coeff, ctx.variable(var)?);
                    }
                    ctx.quadratic_cache.insert(current, Some(LoweredExpr::Linear(value)));
                    continue;
                }
                let children = match arena.get(current) {
                    ExprNode::Unary(UnaryOp::Neg, child) => std::slice::from_ref(child),
                    ExprNode::Add(children) | ExprNode::Mul(children) => children.as_slice(),
                    ExprNode::Pow(base, exp) if as_const(arena, *exp).is_some() => {
                        std::slice::from_ref(base)
                    }
                    ExprNode::Div(num, den) if as_const(arena, *den).is_some() => {
                        if as_const(arena, *den) == Some(0.0) {
                            return Err(SolverError::Backend(
                                "division by zero: constant denominator is 0".into(),
                            ));
                        }
                        std::slice::from_ref(num)
                    }
                    _ => &[],
                };
                (current, children)
            }
        };
        if let Some((&child, rest)) = children.split_first() {
            pending.push(Task::Next { parent: current, child, rest });
            pending.push(Task::Visit(child));
        } else {
            probe_quadratic_node(arena, current, ctx)?;
        }
    }
    Ok(ctx.quadratic_cache[&id].clone())
}

/// Return `Some(false)` if the region is constant, `Some(true)` if it is affine
/// and contains variables, or `None` if extraction cannot use an affine fast path.
/// Memoize results and inspect descendants only for operators supported by linear extraction.
#[expect(clippy::float_cmp, reason = "validated exponents are exact integers")]
fn affine_region(
    arena: &ExprArena,
    root: ExprId,
    cache: &mut HashMap<ExprId, Option<bool>>,
) -> Option<bool> {
    if let Some(value) = cache.get(&root) {
        return *value;
    }
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
            ExprNode::Add(children) => children
                .iter()
                .try_fold(false, |vars, child| cache[child].map(|child_vars| vars || child_vars)),
            ExprNode::Mul(children) => {
                let mut vars = false;
                children
                    .iter()
                    .try_for_each(|child| match cache[child] {
                        Some(false) => Some(()),
                        Some(true) if !vars => {
                            vars = true;
                            Some(())
                        }
                        _ => None,
                    })
                    .map(|()| vars)
            }
            ExprNode::Unary(UnaryOp::Neg, child) if finish => cache[child],
            ExprNode::Unary(UnaryOp::Neg, child) => {
                pending.extend([(id, true), (*child, false)]);
                continue;
            }
            ExprNode::Pow(base, exponent) => {
                let ExprNode::Const(e) = arena.get(*exponent) else {
                    cache.insert(id, None);
                    continue;
                };
                if !e.is_finite() || *e < 0.0 || (*e - e.round()).abs() >= f64::EPSILON {
                    None
                } else if e.round() == 0.0 {
                    Some(false)
                } else if !finish {
                    pending.extend([(id, true), (*base, false)]);
                    continue;
                } else {
                    cache[base].filter(|vars| !vars || e.round() == 1.0)
                }
            }
            _ => None,
        };
        cache.insert(id, value);
    }
    cache[&root]
}

fn quadratic_child(ctx: &LoweringCtx<'_>, id: ExprId) -> Option<LoweredExpr> {
    ctx.quadratic_cache[&id].clone()
}

fn probe_quadratic_node(
    arena: &ExprArena,
    id: ExprId,
    ctx: &mut LoweringCtx<'_>,
) -> Result<Option<LoweredExpr>, SolverError> {
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
            let Some(inner) = quadratic_child(ctx, *inner) else {
                return Ok(cache_quadratic_none(ctx, id));
            };
            try_scale(inner, -1.0)
        }
        ExprNode::Add(children) => {
            let mut acc = LoweredExpr::Linear(linear_constant(0.0));
            for child in children {
                let Some(value) = quadratic_child(ctx, *child) else {
                    return Ok(cache_quadratic_none(ctx, id));
                };
                acc = try_add(acc, value);
            }
            acc
        }
        ExprNode::Mul(children) => {
            let mut values = Vec::with_capacity(children.len());
            for child in children {
                let Some(value) = quadratic_child(ctx, *child) else {
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
            let Some(base) = quadratic_child(ctx, *base) else {
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
            let Some(num) = quadratic_child(ctx, *num) else {
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

// Emit the same left-associated binary operator spine as the recursive
// translator, then schedule its operands in reverse for preorder traversal.
fn append_fold(
    ids: &[ExprId],
    opcode: Opcode,
    identity: f64,
    parent: Option<usize>,
    tree: &mut TreeBuilder,
    pending: &mut Vec<(ExprId, Option<usize>)>,
) -> Result<(), SolverError> {
    match ids {
        [] => {
            tree.push(Opcode::Constant, identity, parent)?;
        }
        [id] => pending.push((*id, parent)),
        _ => {
            let mut operators = Vec::with_capacity(ids.len() - 1);
            let mut current_parent = tree.push(opcode, 0.0, parent)?;
            operators.push(current_parent);
            for _ in 1..ids.len() - 1 {
                current_parent = tree.push(opcode, 0.0, Some(current_parent))?;
                operators.push(current_parent);
            }
            for (i, &id) in ids.iter().enumerate().rev() {
                let op = operators[(ids.len() - 1 - i).min(ids.len() - 2)];
                pending.push((id, Some(op)));
            }
        }
    }
    Ok(())
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

fn materialize_abs(ctx: &mut LoweringCtx<'_>, inner: ExprId) -> Result<Var, SolverError> {
    let argument = materialized_argument(ctx, inner)?;
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
    children: &[ExprId],
    is_min: bool,
) -> Result<Var, SolverError> {
    let mut arguments = Vec::with_capacity(children.len());
    for child in children {
        let argument = materialized_argument(ctx, *child)?;
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

fn materialized_argument(ctx: &LoweringCtx<'_>, id: ExprId) -> Result<Var, SolverError> {
    ctx.materialized.get(&id).copied().ok_or_else(|| {
        SolverError::Backend("internal: nonlinear argument was not materialized".into())
    })
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
            ExprNode::Unary(UnaryOp::Abs, inner) => materialize_abs(ctx, *inner)?,
            ExprNode::Min(children) => materialize_extrema(ctx, children, true)?,
            ExprNode::Max(children) => materialize_extrema(ctx, children, false)?,
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
) -> Result<(), SolverError> {
    let mut pending = vec![(id, parent)];
    while let Some((id, parent)) = pending.pop() {
        if tree.materializing != Some(id)
            && let Some(&result) = ctx.materialized.get(&id)
        {
            let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
            tree.push(Opcode::Variable, f64::from(index), parent)?;
            continue;
        }
        match arena.get(id) {
            ExprNode::Const(c) => {
                tree.push(Opcode::Constant, *c, parent)?;
            }
            ExprNode::Param(p) => {
                tree.push(Opcode::Constant, arena.param_value(*p), parent)?;
            }
            ExprNode::Var(v) => {
                let var = ctx.variable(*v)?;
                let index = ctx.model.var_index(&var).map_err(map_gurobi)?;
                tree.push(Opcode::Variable, f64::from(index), parent)?;
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
                append_linear_parts(ctx, &terms, parent, tree)?;
            }
            ExprNode::Unary(UnaryOp::Neg, inner) => {
                let op = tree.push(Opcode::Uminus, 0.0, parent)?;
                pending.push((*inner, Some(op)));
            }
            ExprNode::Add(children) => {
                append_fold(children, Opcode::Plus, 0.0, parent, tree, &mut pending)?;
            }
            ExprNode::Mul(children) => {
                append_fold(children, Opcode::Multiply, 1.0, parent, tree, &mut pending)?;
            }
            ExprNode::Pow(base, exp) => {
                let op = tree.push(Opcode::Pow, 0.0, parent)?;
                pending.push((*exp, Some(op)));
                pending.push((*base, Some(op)));
            }
            ExprNode::Div(num, den) => {
                if let Some(0.0) = as_const(arena, *den) {
                    return Err(SolverError::Backend(
                        "division by zero: constant denominator is 0".into(),
                    ));
                }
                let op = tree.push(Opcode::Divide, 0.0, parent)?;
                pending.push((*den, Some(op)));
                pending.push((*num, Some(op)));
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
                pending.push((*inner, Some(op)));
            }
            ExprNode::Unary(UnaryOp::Exp2, inner) => {
                let op = tree.push(Opcode::Pow, 0.0, parent)?;
                tree.push(Opcode::Constant, 2.0, Some(op))?;
                pending.push((*inner, Some(op)));
            }
            ExprNode::Unary(UnaryOp::Abs, inner) => {
                let result = materialize_abs(ctx, *inner)?;
                let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
                tree.push(Opcode::Variable, f64::from(index), parent)?;
            }
            ExprNode::Min(children) | ExprNode::Max(children) => {
                let result =
                    materialize_extrema(ctx, children, matches!(arena.get(id), ExprNode::Min(_)))?;
                let index = ctx.model.var_index(&result).map_err(map_gurobi)?;
                tree.push(Opcode::Variable, f64::from(index), parent)?;
            }
            ExprNode::Unary(op, _) => {
                return Err(SolverError::UnsupportedNonlinearOperator {
                    backend: "Gurobi",
                    operator: op.name(),
                });
            }
            ExprNode::Atan2(_, _) => {
                return Err(SolverError::UnsupportedNonlinearOperator {
                    backend: "Gurobi",
                    operator: "atan2",
                });
            }
        }
    }
    Ok(())
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

/// Schedule auxiliary definitions before emitting trees that reference them.
fn materialize_dependencies(
    ctx: &mut LoweringCtx<'_>,
    arena: &ExprArena,
    root: ExprId,
) -> Result<Var, SolverError> {
    let mut seen = vec![false; arena.len()];
    let mut pending = vec![(root, true, false)];
    while let Some((id, required, finish)) = pending.pop() {
        if finish {
            materialize_shared(ctx, arena, id)?;
            continue;
        }
        if ctx.materialized.contains_key(&id) {
            continue;
        }
        let node = arena.get(id);
        let general =
            matches!(node, ExprNode::Unary(UnaryOp::Abs, _) | ExprNode::Min(_) | ExprNode::Max(_));
        let leaf = matches!(node, ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_));
        let shared = ctx.reference_counts[id.index()] > 1 && !leaf;
        let required = required || general || shared;
        // An earlier inline visit does not satisfy a later abs/min/max argument.
        if std::mem::replace(&mut seen[id.index()], true) && !required {
            continue;
        }
        if required {
            if let Some(value) = try_quadratic(arena, id, ctx)? {
                let result = materialize_lowered(ctx, value)?;
                ctx.materialized.insert(id, result);
                continue;
            }
            pending.push((id, true, true));
        }
        // Abs/min/max require variable arguments even for unshared polynomial
        // children. Other operators inline unshared children in their tree.
        for_each_child(node, &mut |child| pending.push((child, general, false)));
    }
    materialized_argument(ctx, root)
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
    Ok(LoweredExpr::Var(materialize_dependencies(ctx, arena, id)?))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_arguments_can_be_materialized_after_inline_leaf_visits() {
        let env = Env::new("").unwrap();
        let mut model = Model::with_env("argument_order", &env).unwrap();
        model.set_param(gurobi_rs::param::OutputFlag, 0).unwrap();
        let var = model.add_var("x", Continuous, 0.0, 1.0, 1.0, []).unwrap();
        let vars = [var];
        let mut arena = ExprArena::new();
        let x = arena.var(VarId(0));
        let pid = arena.new_param(-2.0);
        let p = arena.param(pid);
        let constant = arena.constant(-2.0);
        let zero = arena.constant(0.0);
        let mut counter = 0;
        for argument in [x, p, constant] {
            for kind in 0..3 {
                for reversed in [false, true] {
                    let smooth = arena.push(ExprNode::Unary(UnaryOp::Sin, argument));
                    let general = arena.push(match kind {
                        0 => ExprNode::Unary(UnaryOp::Abs, argument),
                        1 => ExprNode::Min(vec![argument, zero].into()),
                        _ => ExprNode::Max(vec![argument, zero].into()),
                    });
                    let children = if reversed { [smooth, general] } else { [general, smooth] };
                    let root = arena.push(ExprNode::Add(children.to_vec().into()));
                    let mut ctx = LoweringCtx::new(&mut model, &vars, counter);
                    assert!(matches!(lower(&arena, root, &mut ctx).unwrap(), LoweredExpr::Var(_)));
                    assert!(ctx.materialized.contains_key(&argument));
                    counter = ctx.aux_counter;
                }
            }
        }
        model.update().unwrap();
    }

    #[test]
    fn mixed_polynomial_probes_retain_only_maximal_affine_regions() {
        use oximo_expr::{Expr, ExprArenaCell};
        let env = Env::new("").unwrap();
        let mut model = Model::with_env("affine_regions", &env).unwrap();
        model.set_param(gurobi_rs::param::OutputFlag, 0).unwrap();
        let cell = ExprArenaCell::new(ExprArena::new());
        let vars: Vec<_> = (0..4096).map(|i| Expr::from_var(&cell, VarId(i))).collect();
        let native_vars: Vec<_> = (0..4096)
            .map(|i| model.add_var(&format!("x{i}"), Continuous, 0.0, 0.0, 0.0, []).unwrap())
            .collect();
        let sum = vars.iter().copied().reduce(|a, b| a + b).unwrap();
        let sine = vars[0].sin();
        let quadratic = vars[0] * vars[0];
        let roots =
            [(sine + sum).id(), (sum + sine).id(), (quadratic + sum).id(), (sum + quadratic).id()];
        let arena = cell.borrow();
        for (index, root) in roots.into_iter().enumerate() {
            let mut ctx = LoweringCtx::new(&mut model, &native_vars, 0);
            let value = try_quadratic(&arena, root, &mut ctx).unwrap();
            if index < 2 {
                assert!(value.is_none());
            } else {
                let Some(LoweredExpr::Quadratic(value)) = value else {
                    panic!("quadratic plus affine sum must stay quadratic");
                };
                let (quad, linear) = value.into_parts();
                assert_eq!(quad.len(), 1);
                assert_eq!(linear.iter_terms().count(), 4096);
                assert!(
                    linear
                        .iter_terms()
                        .all(|(_, coefficient)| coefficient.to_bits() == 1.0_f64.to_bits())
                );
            }
            let retained: usize = ctx
                .quadratic_cache
                .values()
                .map(|value| match value {
                    Some(LoweredExpr::Linear(value)) => value.iter_terms().count(),
                    _ => 0,
                })
                .sum();
            assert!(retained <= 4097, "cached {retained} affine coefficient entries");
            if index == 0 {
                assert!(
                    !ctx.quadratic_cache.contains_key(&sum.id()),
                    "failed first child must stop probing siblings"
                );
            } else {
                assert!(ctx.quadratic_cache.contains_key(&sum.id()));
            }
        }
    }

    #[test]
    fn binary_fold_preserves_grouping_and_parent_order() {
        // Large cancelling terms distinguish a left fold from regrouping.
        let values = [1e16, -1e16, 1.0, 2.0, 3.0];
        let ids: Vec<_> = (0..5).map(ExprId).collect();
        let mut tree = TreeBuilder::new(None);
        let mut pending = Vec::new();
        append_fold(&ids, Opcode::Plus, 0.0, None, &mut tree, &mut pending).unwrap();
        while let Some((id, parent)) = pending.pop() {
            tree.push(Opcode::Constant, values[id.index()], parent).unwrap();
        }
        assert_eq!(tree.parent, [-1, 0, 1, 2, 3, 3, 2, 1, 0]);
        let mut evaluated = tree.data.clone();
        for i in (1..evaluated.len()).rev() {
            let parent = usize::try_from(tree.parent[i]).unwrap();
            evaluated[parent] += evaluated[i];
        }
        assert_eq!(evaluated[0], 6.0);

        for (opcode, identity) in [(Opcode::Plus, 0.0), (Opcode::Multiply, 1.0)] {
            let mut tree = TreeBuilder::new(None);
            append_fold(&[], opcode, identity, None, &mut tree, &mut pending).unwrap();
            assert_eq!(tree.data, [identity]);
            assert_eq!(tree.parent, [-1]);
            let mut tree = TreeBuilder::new(None);
            append_fold(&ids[..1], opcode, identity, None, &mut tree, &mut pending).unwrap();
            assert_eq!(tree.opcode, [] as [i32; 0]);
            assert_eq!(pending.pop(), Some((ids[0], None)));
        }
    }

    #[test]
    fn deep_probes_and_auxiliary_dependencies_are_stack_safe() {
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let env = Env::new("").unwrap();
                let mut model = Model::with_env("deep_lowering", &env).unwrap();
                let var = model.add_var("x", Continuous, 0.0, 0.0, 0.0, []).unwrap();
                let vars = [var];
                let mut ctx = LoweringCtx::new(&mut model, &vars, 0);
                let mut arena = ExprArena::new();
                let x = arena.var(VarId(0));
                let one = arena.constant(1.0);
                let two = arena.constant(2.0);
                let three = arena.constant(3.0);

                // A polynomial path with no linear-extraction shortcut.
                let mut quadratic = arena.push(ExprNode::Pow(x, two));
                for _ in 0..20_000 {
                    quadratic = arena.push(ExprNode::Div(quadratic, one));
                }
                let LoweredExpr::Quadratic(value) = lower(&arena, quadratic, &mut ctx).unwrap()
                else {
                    panic!("deep division must stay quadratic");
                };
                let (terms, _) = value.into_parts();
                assert_eq!(terms.len(), 1);
                for ((a, b), coeff) in terms {
                    assert_eq!((a, b), (var, var));
                    assert_eq!(coeff, 1.0);
                }
                assert!(ctx.generated.is_empty());

                // A failed polynomial probe must also unwind iteratively.
                let mut negated = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
                for _ in 0..20_000 {
                    negated = arena.push(ExprNode::Unary(UnaryOp::Neg, negated));
                }
                lower(&arena, negated, &mut ctx).unwrap();
                assert_eq!(ctx.generated.len(), 1);

                let mut power = x;
                for _ in 0..10_000 {
                    power = arena.push(ExprNode::Pow(power, three));
                }
                lower(&arena, power, &mut ctx).unwrap();
                assert_eq!(ctx.generated.len(), 2);

                // Alternate native general constraints and nonlinear trees.
                let mut general = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
                for _ in 0..1_000 {
                    general = arena.push(ExprNode::Unary(UnaryOp::Abs, general));
                    general = arena.push(ExprNode::Min(vec![general, x].into()));
                    general = arena.push(ExprNode::Max(vec![general, x].into()));
                }
                lower(&arena, general, &mut ctx).unwrap();

                // Shared nonlinear descendants are defined once, then reused.
                let mut shared = arena.push(ExprNode::Unary(UnaryOp::Sin, x));
                for _ in 0..1_000 {
                    let sum = arena.push(ExprNode::Add(vec![shared, shared].into()));
                    shared = arena.push(ExprNode::Unary(UnaryOp::Sin, sum));
                }
                let before = ctx.generated.len();
                let first = lower(&arena, shared, &mut ctx).unwrap();
                assert_eq!(ctx.generated.len() - before, 2_000);
                let before = ctx.generated.len();
                let second = lower(&arena, shared, &mut ctx).unwrap();
                assert_eq!(ctx.generated.len(), before);
                match (first, second) {
                    (LoweredExpr::Var(a), LoweredExpr::Var(b)) => assert_eq!(a, b),
                    _ => panic!("shared nonlinear expression must use an auxiliary variable"),
                }
                ctx.model.update().unwrap();
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
