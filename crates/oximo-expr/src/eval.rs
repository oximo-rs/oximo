use std::cell::RefCell;

use thiserror::Error;

use crate::arena::{ExprArena, ExprId, ExprNode, ParamId, ParameterSnapshot, VarId};

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

    /// Resolve a parameter, falling back to the arena by default.
    /// Use [`WithParameters`] to evaluate strictly against a parameter snapshot.
    fn resolve_param(&self, id: ParamId, arena: &ExprArena) -> Option<f64> {
        self.param(id).or_else(|| arena.try_param_value(id))
    }
}

/// An evaluation context that takes variables from another context and parameters
/// exclusively from a snapshot. Missing parameters never fall back to the arena
/// or the wrapped context.
///
/// ```
/// use oximo_expr::{ExprArena, WithParameters, evaluate};
/// let mut arena = ExprArena::new();
/// let p = arena.new_param(2.0);
/// let expr = arena.param(p);
/// let parameters = arena.parameter_snapshot();
/// arena.set_param_value(p, 4.0);
/// let variables: &[f64] = &[];
/// let context = WithParameters::new(&variables, &parameters);
/// assert_eq!(evaluate(&arena, expr, &context).unwrap(), 2.0);
/// ```
#[derive(Debug)]
pub struct WithParameters<'a, C: ?Sized> {
    context: &'a C,
    parameters: &'a ParameterSnapshot,
}

impl<'a, C: EvalContext + ?Sized> WithParameters<'a, C> {
    pub fn new(context: &'a C, parameters: &'a ParameterSnapshot) -> Self {
        Self { context, parameters }
    }
}

impl<C: EvalContext + ?Sized> EvalContext for WithParameters<'_, C> {
    fn var(&self, id: VarId) -> Option<f64> {
        self.context.var(id)
    }

    fn param(&self, id: ParamId) -> Option<f64> {
        self.parameters.value(id)
    }

    fn resolve_param(&self, id: ParamId, _: &ExprArena) -> Option<f64> {
        self.param(id)
    }
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
/// The traversal is iterative and visits each reachable node once, so deep
/// chains cannot overflow the call stack and heavily shared DAGs are
/// evaluated in time proportional to the number of distinct nodes.
///
/// # Errors
///
/// Returns an [`EvalError`] if a needed variable or parameter is missing from the context.
#[inline]
pub fn evaluate<C: EvalContext>(arena: &ExprArena, id: ExprId, ctx: &C) -> Result<f64, EvalError> {
    match arena.get(id) {
        ExprNode::Const(c) => Ok(*c),
        ExprNode::Var(v) => ctx.var(*v).ok_or(EvalError::UnboundVar(*v)),
        ExprNode::Param(p) => ctx.resolve_param(*p, arena).ok_or(EvalError::UnboundParam(*p)),
        ExprNode::Linear { coeffs, constant } => eval_linear(coeffs, *constant, ctx),
        _ => eval_compound(arena, id, ctx),
    }
}

#[inline]
fn eval_linear<C: EvalContext>(
    coeffs: &[(VarId, f64)],
    mut value: f64,
    ctx: &C,
) -> Result<f64, EvalError> {
    for &(v, c) in coeffs {
        value += c * ctx.var(v).ok_or(EvalError::UnboundVar(v))?;
    }
    Ok(value)
}

#[inline]
fn eval_leaf<C: EvalContext>(
    arena: &ExprArena,
    node: &ExprNode,
    ctx: &C,
) -> Option<Result<f64, EvalError>> {
    Some(match node {
        ExprNode::Const(c) => Ok(*c),
        ExprNode::Var(v) => ctx.var(*v).ok_or(EvalError::UnboundVar(*v)),
        ExprNode::Param(p) => ctx.resolve_param(*p, arena).ok_or(EvalError::UnboundParam(*p)),
        ExprNode::Linear { coeffs, constant } => eval_linear(coeffs, *constant, ctx),
        _ => return None,
    })
}

fn eval_compound<C: EvalContext>(
    arena: &ExprArena,
    mut id: ExprId,
    ctx: &C,
) -> Result<f64, EvalError> {
    // A negation spine has no branching or sharing to memoize.
    let mut negate = false;
    while let ExprNode::Unary(crate::UnaryOp::Neg, inner) = arena.get(id) {
        negate = !negate;
        id = *inner;
    }
    let value = eval_shallow(arena, id, ctx).unwrap_or_else(|| eval_general(arena, id, ctx))?;
    Ok(if negate { -value } else { value })
}

fn eval_shallow<C: EvalContext>(
    arena: &ExprArena,
    id: ExprId,
    ctx: &C,
) -> Option<Result<f64, EvalError>> {
    let node = arena.get(id);
    if let Some(value) = eval_leaf(arena, node, ctx) {
        return Some(value);
    }
    if let ExprNode::Unary(op, inner) = node {
        return eval_leaf(arena, arena.get(*inner), ctx).map(|v| v.map(|v| op.apply(v)));
    }
    let (children, mut value) = match node {
        ExprNode::Add(c) => (c, 0.0),
        ExprNode::Mul(c) => (c, 1.0),
        ExprNode::Min(c) => (c, f64::INFINITY),
        ExprNode::Max(c) => (c, f64::NEG_INFINITY),
        _ => return None,
    };
    // Ascending IDs prove these leaves are distinct without a hash set.
    // We do the shape check before querying the context, so fallback never
    // repeats context calls or changes which missing value is reported first.
    let mut previous = None;
    if !children.iter().all(|&child| {
        let distinct = previous.is_none_or(|p| child.index() > p);
        previous = Some(child.index());
        distinct
            && matches!(
                arena.get(child),
                ExprNode::Const(_)
                    | ExprNode::Var(_)
                    | ExprNode::Param(_)
                    | ExprNode::Linear { .. }
            )
    }) {
        return None;
    }
    for &child in children {
        let x = match eval_leaf(arena, arena.get(child), ctx)? {
            Ok(x) => x,
            Err(error) => return Some(Err(error)),
        };
        value = match node {
            ExprNode::Add(_) => value + x,
            ExprNode::Mul(_) => value * x,
            ExprNode::Min(_) => value.min(x),
            ExprNode::Max(_) => value.max(x),
            _ => unreachable!(),
        };
    }
    Some(Ok(value))
}

fn eval_general<C: EvalContext>(arena: &ExprArena, id: ExprId, ctx: &C) -> Result<f64, EvalError> {
    if let Some(result) = eval_array(arena, id, ctx) {
        return result;
    }
    eval_hashed(arena, id, ctx)
}

#[derive(Default)]
struct EvalScratch {
    values: Vec<f64>,
    seen: Vec<u32>,
    stack: Vec<(ExprId, usize)>,
    generation: u32,
}

thread_local! {
    static SCRATCH: RefCell<EvalScratch> = const { RefCell::new(EvalScratch {
        values: Vec::new(), seen: Vec::new(), stack: Vec::new(), generation: 0,
    }) };
}

const ARRAY_PATH_MAX_NODES: usize = 1 << 16;

/// Stack-safe, sharing-aware evaluation using arena-indexed scratch buffers.
///
/// Returns `None` when the caller should use [`eval_hashed`] instead (arena
/// too large or scratch re-entered).
fn eval_array<C: EvalContext>(
    arena: &ExprArena,
    id: ExprId,
    ctx: &C,
) -> Option<Result<f64, EvalError>> {
    let n = arena.len();
    if n > ARRAY_PATH_MAX_NODES || id.index() >= n {
        return None;
    }
    SCRATCH.with(|scratch| {
        // A nested evaluation uses hashed storage.
        let mut scratch = scratch.try_borrow_mut().ok()?;
        let EvalScratch { values, seen, stack, generation } = &mut *scratch;
        values.resize(values.len().max(n), 0.0);
        seen.resize(seen.len().max(n), 0);
        stack.clear();
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            seen.fill(0);
            *generation = 1;
        }
        stack.push((id, 0));
        seen[id.index()] = *generation;
        while let Some((cur, next)) = stack.last_mut() {
            let cur = *cur;
            if let Some(child) = eval_child(arena, cur, *next) {
                *next += 1;
                if child.index() >= n {
                    return None;
                }
                if seen[child.index()] != *generation {
                    seen[child.index()] = *generation;
                    stack.push((child, 0));
                }
            } else {
                match eval_node_array(arena, ctx, values, cur) {
                    Ok(value) => values[cur.index()] = value,
                    Err(error) => return Some(Err(error)),
                }
                stack.pop();
            }
        }
        Some(Ok(values[id.index()]))
    })
}

/// Value of a single node from already-computed child values.
fn eval_node_array<C: EvalContext>(
    arena: &ExprArena,
    ctx: &C,
    values: &[f64],
    cur: ExprId,
) -> Result<f64, EvalError> {
    Ok(match arena.get(cur) {
        ExprNode::Const(c) => *c,
        ExprNode::Var(v) => ctx.var(*v).ok_or(EvalError::UnboundVar(*v))?,
        ExprNode::Param(p) => ctx.resolve_param(*p, arena).ok_or(EvalError::UnboundParam(*p))?,
        ExprNode::Add(children) => {
            let mut acc = 0.0;
            for c in children {
                acc += values[c.index()];
            }
            acc
        }
        ExprNode::Mul(children) => {
            let mut acc = 1.0;
            for c in children {
                acc *= values[c.index()];
            }
            acc
        }
        ExprNode::Unary(op, inner) => op.apply(values[inner.index()]),
        ExprNode::Pow(base, exp) => values[base.index()].powf(values[exp.index()]),
        ExprNode::Div(num, den) => values[num.index()] / values[den.index()],
        ExprNode::Atan2(y, x) => values[y.index()].atan2(values[x.index()]),
        ExprNode::Min(children) => {
            let mut acc = f64::INFINITY;
            for c in children {
                acc = acc.min(values[c.index()]);
            }
            acc
        }
        ExprNode::Max(children) => {
            let mut acc = f64::NEG_INFINITY;
            for c in children {
                acc = acc.max(values[c.index()]);
            }
            acc
        }
        ExprNode::Linear { coeffs, constant } => {
            let mut acc = *constant;
            for (v, c) in coeffs {
                acc += c * ctx.var(*v).ok_or(EvalError::UnboundVar(*v))?;
            }
            acc
        }
    })
}

/// General fallback: same post-order, but `seen`/`values` live in hash maps
/// so only reachable nodes are touched no matter how large the arena is.
fn eval_hashed<C: EvalContext>(arena: &ExprArena, id: ExprId, ctx: &C) -> Result<f64, EvalError> {
    let mut seen = rustc_hash::FxHashSet::default();
    let mut order = Vec::new();
    seen.insert(id);
    let mut stack: Vec<(ExprId, usize)> = vec![(id, 0)];
    while let Some((cur, next)) = stack.last_mut() {
        let cur = *cur;
        let Some(child) = eval_child(arena, cur, *next) else {
            order.push(cur);
            stack.pop();
            continue;
        };
        *next += 1;
        if seen.insert(child) {
            stack.push((child, 0));
        }
    }

    let mut values =
        rustc_hash::FxHashMap::with_capacity_and_hasher(order.len(), rustc_hash::FxBuildHasher);
    for cur in order {
        let value = match arena.get(cur) {
            ExprNode::Const(c) => *c,
            ExprNode::Var(v) => ctx.var(*v).ok_or(EvalError::UnboundVar(*v))?,
            ExprNode::Param(p) => {
                ctx.resolve_param(*p, arena).ok_or(EvalError::UnboundParam(*p))?
            }
            ExprNode::Add(children) => {
                let mut acc = 0.0;
                for c in children {
                    acc += values[c];
                }
                acc
            }
            ExprNode::Mul(children) => {
                let mut acc = 1.0;
                for c in children {
                    acc *= values[c];
                }
                acc
            }
            ExprNode::Unary(op, inner) => op.apply(values[inner]),
            ExprNode::Pow(base, exp) => values[base].powf(values[exp]),
            ExprNode::Div(num, den) => values[num] / values[den],
            ExprNode::Atan2(y, x) => values[y].atan2(values[x]),
            ExprNode::Min(children) => {
                let mut acc = f64::INFINITY;
                for c in children {
                    acc = acc.min(values[c]);
                }
                acc
            }
            ExprNode::Max(children) => {
                let mut acc = f64::NEG_INFINITY;
                for c in children {
                    acc = acc.max(values[c]);
                }
                acc
            }
            ExprNode::Linear { coeffs, constant } => {
                let mut acc = *constant;
                for (v, c) in coeffs {
                    acc += c * ctx.var(*v).ok_or(EvalError::UnboundVar(*v))?;
                }
                acc
            }
        };
        values.insert(cur, value);
    }
    Ok(values[&id])
}

/// The `next`-th lowering dependency of `id` (the `next`-th child), or `None`
/// when all children have been yielded.
fn eval_child(arena: &ExprArena, id: ExprId, next: usize) -> Option<ExprId> {
    match arena.get(id) {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => children.get(next).copied(),
        ExprNode::Unary(_, inner) => (next == 0).then_some(*inner),
        ExprNode::Pow(base, exp) | ExprNode::Div(base, exp) | ExprNode::Atan2(base, exp) => {
            match next {
                0 => Some(*base),
                1 => Some(*exp),
                _ => None,
            }
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {
            None
        }
    }
}
