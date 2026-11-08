//! Flat instruction tapes compiled from [`ExprArena`] subtrees.
//!
//! A [`Tape`] is the bridge between oximo's runtime expression data and
//! `std::autodiff`, which differentiates compiled Rust functions. Every
//! expression is lowered to instructions executed by the single interpreter
//! `eval_tape`, and the `enzyme` feature differentiates that interpreter
//! once at compile time.
//!
//! The tape is in SSA form.
use oximo_expr::{ExprArena, ExprId, ExprNode, UnaryOp};
use rustc_hash::FxHashMap;

// Opcodes. `a`/`b` are operand register indices unless noted.
pub(crate) const OP_CONST: u32 = 0; // consts[a]
pub(crate) const OP_VAR: u32 = 1; // x[a]
pub(crate) const OP_PARAM: u32 = 2; // params[a]
pub(crate) const OP_MULT: u32 = 3; // mults[a]
pub(crate) const OP_ADD: u32 = 4;
pub(crate) const OP_MUL: u32 = 5;
pub(crate) const OP_DIV: u32 = 6;
pub(crate) const OP_NEG: u32 = 7;
pub(crate) const OP_POWC: u32 = 8; // regs[a] ^ consts[b]
pub(crate) const OP_POW: u32 = 9; // regs[a] ^ regs[b]
pub(crate) const OP_SIN: u32 = 10;
pub(crate) const OP_COS: u32 = 11;
pub(crate) const OP_EXP: u32 = 12;
pub(crate) const OP_LOG: u32 = 13;
pub(crate) const OP_ABS: u32 = 14;
pub(crate) const OP_LINEAR: u32 = 15; // sum of lin_coeffs[a..b] * x[lin_vars[a..b]]
pub(crate) const OP_SQRT: u32 = 16;
pub(crate) const OP_CBRT: u32 = 17;
pub(crate) const OP_EXP2: u32 = 18;
pub(crate) const OP_EXPM1: u32 = 19;
pub(crate) const OP_LOG2: u32 = 20;
pub(crate) const OP_LOG10: u32 = 21;
pub(crate) const OP_LOG1P: u32 = 22;
pub(crate) const OP_TAN: u32 = 23;
pub(crate) const OP_ASIN: u32 = 24;
pub(crate) const OP_ACOS: u32 = 25;
pub(crate) const OP_ATAN: u32 = 26;
pub(crate) const OP_SINH: u32 = 27;
pub(crate) const OP_COSH: u32 = 28;
pub(crate) const OP_TANH: u32 = 29;
pub(crate) const OP_ASINH: u32 = 30;
pub(crate) const OP_ACOSH: u32 = 31;
pub(crate) const OP_ATANH: u32 = 32;
pub(crate) const OP_ATAN2: u32 = 33;
pub(crate) const OP_MIN: u32 = 34;
pub(crate) const OP_MAX: u32 = 35;
pub(crate) const OP_SCALED_SQUARE: u32 = 36; // (coefficient * x) * x; coefficient=regs[a], x=regs[b]

/// An expression (or weighted sum of expressions) compiled to a flat
/// instruction tape, evaluable at any point without touching the arena.
///
/// Parameter and multiplier values are not baked in. [`Tape::value`] takes
/// them as slices, so `set_param` or new Lagrange multipliers never force a
/// recompile.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    ops: Vec<u32>,
    a: Vec<u32>,
    b: Vec<u32>,
    consts: Vec<f64>,
    lin_vars: Vec<u32>,
    lin_coeffs: Vec<f64>,
    n_mults: usize,
}

impl Tape {
    /// Compile the subtree rooted at `root`.
    ///
    /// Lowering is iterative and visits each distinct node once, so
    /// arbitrarily deep expressions cannot overflow the call stack and
    /// heavily shared DAGs lower in time proportional to the number of
    /// distinct nodes.
    pub fn compile(arena: &ExprArena, root: ExprId) -> Self {
        let mut builder = Builder::default();
        let reg = builder.lower(arena, root);
        builder.finish_root(reg);
        builder.tape
    }

    /// Compile `sum_k mults[k] * exprs[k]` with the weights resolved at
    /// evaluation time. The Lagrangian tape used for Hessian computation.
    /// Shared subexpressions across the `exprs` are lowered once.
    pub fn compile_weighted(arena: &ExprArena, exprs: &[ExprId]) -> Self {
        let mut builder = Builder::default();
        let mut acc: Option<u32> = None;
        for (k, &expr) in exprs.iter().enumerate() {
            let value = builder.lower(arena, expr);
            let weight = builder.push(OP_MULT, to_u32(k), 0);
            let term = builder.push(OP_MUL, weight, value);
            acc = Some(match acc {
                None => term,
                Some(prev) => builder.push(OP_ADD, prev, term),
            });
        }
        let root = match acc {
            Some(reg) => reg,
            None => builder.push_const(0.0),
        };
        builder.finish_root(root);
        builder.tape.n_mults = exprs.len();
        builder.tape
    }

    /// Evaluate with a reusable [`TapeScratch`].
    /// The scratch is grown to [`Tape::n_regs`] when necessary and
    /// reused across calls.
    pub fn value_with_scratch(
        &self,
        x: &[f64],
        params: &[f64],
        mults: &[f64],
        scratch: &mut TapeScratch,
    ) -> f64 {
        scratch.ensure(self.n_regs());
        self.value(x, params, mults, scratch.regs_mut())
    }

    /// Number of registers the interpreter needs, equal to the instruction
    /// count. A compiled tape always has at least one instruction (`compile`
    /// lowers a root, `compile_weighted` emits a `0.0` constant for an empty
    /// sum), so this is only `0` for a `Tape::default()` that was never
    /// compiled.
    pub fn n_regs(&self) -> usize {
        self.ops.len()
    }

    /// Number of multiplier slots expected in the `mults` slice.
    pub fn n_mults(&self) -> usize {
        self.n_mults
    }

    /// Evaluate the tape at `x`. `regs` is caller-provided scratch of length
    /// [`Tape::n_regs`], `mults` must have length [`Tape::n_mults`].
    ///
    /// # Panics
    ///
    /// Panics if `regs` is shorter than [`Tape::n_regs`], or if `mults` is
    /// shorter than [`Tape::n_mults`] (indexed by `OP_MULT`). Also panics on an
    /// empty (never-compiled) tape, which has no result register.
    pub fn value(&self, x: &[f64], params: &[f64], mults: &[f64], regs: &mut [f64]) -> f64 {
        assert!(regs.len() >= self.n_regs(), "register scratch too short");
        debug_assert!(!self.ops.is_empty(), "cannot evaluate an empty tape");
        let mut out = [0.0];
        eval_tape(
            &self.ops,
            &self.a,
            &self.b,
            &self.consts,
            &self.lin_vars,
            &self.lin_coeffs,
            x,
            params,
            mults,
            regs,
            &mut out,
        );
        out[0]
    }

    /// The raw tape slices, in [`eval_tape`] argument order. Used by the
    /// `enzyme` module to drive the differentiated interpreter.
    #[cfg_attr(not(feature = "enzyme"), expect(dead_code))]
    #[expect(clippy::type_complexity)]
    pub(crate) fn parts(&self) -> (&[u32], &[u32], &[u32], &[f64], &[u32], &[f64]) {
        (&self.ops, &self.a, &self.b, &self.consts, &self.lin_vars, &self.lin_coeffs)
    }
}

/// The tape interpreter, the one function `std::autodiff` differentiates.
///
/// Three properties are load-bearing for Enzyme's type analysis:
/// - every match arm stores into `regs[i]` directly. Collecting arm results
///   into one value first creates an LLVM `phi` mixing typed and untyped
///   constants,
/// - the result leaves through the `out` parameter instead of a return value,
///   an `Active` return is plumbed through an `enzyme_primal_return` marker
///   global that breaks forward-over-reverse across crate boundaries,
/// - `OP_LINEAR` ranges are non-empty (builder invariant) and the accumulator
///   starts from the first product, not a `0.0` literal, and there is no
///   `n == 0` fallback (builder emits at least one instruction), both would
///   store-sink into untyped-`0.0` phis.
#[expect(clippy::too_many_arguments)]
#[inline]
pub(crate) fn eval_tape(
    ops: &[u32],
    a: &[u32],
    b: &[u32],
    consts: &[f64],
    lin_vars: &[u32],
    lin_coeffs: &[f64],
    x: &[f64],
    params: &[f64],
    mults: &[f64],
    regs: &mut [f64],
    out: &mut [f64],
) {
    let n = ops.len();
    assert!(a.len() == n && b.len() == n && regs.len() >= n, "tape dimensions");
    for i in 0..n {
        let ai = a[i] as usize;
        let bi = b[i] as usize;
        let op = ops[i];
        // Keep the common opcodes in a compact dispatch table.
        // The extended nonlinear vocabulary is cold for existing tapes.
        if op <= OP_LINEAR {
            match op {
                OP_CONST => regs[i] = consts[ai],
                OP_VAR => regs[i] = x[ai],
                OP_PARAM => regs[i] = params[ai],
                OP_MULT => regs[i] = mults[ai],
                OP_ADD => regs[i] = regs[ai] + regs[bi],
                OP_MUL => regs[i] = regs[ai] * regs[bi],
                OP_DIV => regs[i] = regs[ai] / regs[bi],
                OP_NEG => regs[i] = -regs[ai],
                OP_POWC => regs[i] = regs[ai].powf(consts[bi]),
                OP_POW => regs[i] = regs[ai].powf(regs[bi]),
                OP_SIN => regs[i] = regs[ai].sin(),
                OP_COS => regs[i] = regs[ai].cos(),
                OP_EXP => regs[i] = regs[ai].exp(),
                OP_LOG => regs[i] = regs[ai].ln(),
                OP_ABS => regs[i] = regs[ai].abs(),
                OP_LINEAR => {
                    regs[i] = lin_coeffs[ai] * x[lin_vars[ai] as usize];
                    for k in (ai + 1)..bi {
                        regs[i] += lin_coeffs[k] * x[lin_vars[k] as usize];
                    }
                }
                _ => regs[i] = f64::NAN,
            }
        } else {
            match op {
                OP_SQRT => regs[i] = regs[ai].sqrt(),
                OP_CBRT => regs[i] = regs[ai].cbrt(),
                OP_EXP2 => regs[i] = regs[ai].exp2(),
                OP_EXPM1 => regs[i] = regs[ai].exp_m1(),
                OP_LOG2 => regs[i] = regs[ai].log2(),
                OP_LOG10 => regs[i] = regs[ai].log10(),
                OP_LOG1P => regs[i] = regs[ai].ln_1p(),
                OP_TAN => regs[i] = regs[ai].tan(),
                OP_ASIN => regs[i] = regs[ai].asin(),
                OP_ACOS => regs[i] = regs[ai].acos(),
                OP_ATAN => regs[i] = regs[ai].atan(),
                OP_SINH => regs[i] = regs[ai].sinh(),
                OP_COSH => regs[i] = regs[ai].cosh(),
                OP_TANH => regs[i] = regs[ai].tanh(),
                OP_ASINH => regs[i] = regs[ai].asinh(),
                OP_ACOSH => regs[i] = regs[ai].acosh(),
                OP_ATANH => regs[i] = regs[ai].atanh(),
                OP_ATAN2 => regs[i] = regs[ai].atan2(regs[bi]),
                OP_MIN => regs[i] = regs[ai].min(regs[bi]),
                OP_MAX => regs[i] = regs[ai].max(regs[bi]),
                OP_SCALED_SQUARE => {
                    regs[i] = (regs[ai] * regs[bi]) * regs[bi];
                }
                // Unreachable for a well-formed tape (the builder emits only
                // the opcodes above).
                _ => regs[i] = f64::NAN,
            }
        }
    }
    out[0] = regs[n - 1];
}

fn to_u32(v: usize) -> u32 {
    u32::try_from(v).expect("index exceeds u32::MAX")
}

/// Snapshot the arena's current parameter values into a dense vector, the
/// `params` input of `eval_tape`. Public so value-only consumers can drive [`Tape::value`].
///
/// # Panics
///
/// Panics if the arena holds more than `u32::MAX` parameters.
pub fn params_snapshot(arena: &ExprArena) -> Vec<f64> {
    (0..arena.num_params()).map(|i| arena.param_value(oximo_expr::ParamId(to_u32(i)))).collect()
}

#[derive(Default)]
struct Builder {
    tape: Tape,
    memo: FxHashMap<ExprId, u32>,
}

struct LoweringFrame {
    id: ExprId,
    next: usize,
    acc: Option<u32>,
    // OP_CONST denotes a non-reduction parent.
    nary_op: u32,
}

impl LoweringFrame {
    fn new(arena: &ExprArena, id: ExprId) -> Self {
        let nary_op = match arena.get(id) {
            ExprNode::Add(_) => OP_ADD,
            ExprNode::Mul(_) => OP_MUL,
            ExprNode::Min(_) => OP_MIN,
            ExprNode::Max(_) => OP_MAX,
            _ => OP_CONST,
        };
        Self { id, next: 0, acc: None, nary_op }
    }
}

impl Builder {
    fn push(&mut self, op: u32, a: u32, b: u32) -> u32 {
        let reg = to_u32(self.tape.ops.len());
        self.tape.ops.push(op);
        self.tape.a.push(a);
        self.tape.b.push(b);
        reg
    }

    /// Append a value to the constant pool and return its index, an operand
    /// for `OP_CONST`/`OP_POWC` (not a register).
    fn add_const(&mut self, v: f64) -> u32 {
        let idx = to_u32(self.tape.consts.len());
        self.tape.consts.push(v);
        idx
    }

    fn push_const(&mut self, v: f64) -> u32 {
        let idx = self.add_const(v);
        self.push(OP_CONST, idx, 0)
    }

    // TODO: Can we improve this?

    /// The interpreter returns the last register, so if the root register is
    /// not last (memo hit on a shared subexpression), append `root * 1.0`.
    fn finish_root(&mut self, root: u32) {
        if root as usize != self.tape.ops.len() - 1 {
            let one = self.push_const(1.0);
            self.push(OP_MUL, root, one);
        }
    }

    /// Lower `id` and every unmemoized dependency with an explicit stack,
    /// so arbitrarily deep expressions are safe and each distinct node is
    /// emitted once.
    fn lower(&mut self, arena: &ExprArena, id: ExprId) -> u32 {
        if let Some(&reg) = self.memo.get(&id) {
            return reg;
        }
        if is_lowering_leaf(arena.get(id)) {
            let reg = self.emit_node(arena, id);
            self.memo.insert(id, reg);
            return reg;
        }
        if matches!(arena.get(id), ExprNode::Unary(_, _)) {
            let mut chain = smallvec::SmallVec::<[(ExprId, UnaryOp); 16]>::new();
            let mut current = id;
            let base = loop {
                if let Some(&reg) = self.memo.get(&current) {
                    break reg;
                }
                if let ExprNode::Unary(op, inner) = arena.get(current) {
                    chain.push((current, *op));
                    current = *inner;
                } else {
                    break self.lower(arena, current);
                }
            };
            let mut reg = base;
            for (node, op) in chain.into_iter().rev() {
                reg = self.push(unary_opcode(op), reg, 0);
                self.memo.insert(node, reg);
            }
            return reg;
        }
        // Avoid traversal setup for the common one- and two-level cases.
        let dependencies = [
            lowering_child(arena, id, 0),
            lowering_child(arena, id, 1),
            lowering_child(arena, id, 2),
        ];
        if dependencies[0].is_some()
            && dependencies[2].is_none()
            && dependencies[..2].iter().flatten().all(|&child| is_lowering_leaf(arena.get(child)))
        {
            for &child in dependencies[..2].iter().flatten() {
                self.lower_leaf(arena, child);
            }
            let reg = self.emit_node(arena, id);
            self.memo.insert(id, reg);
            return reg;
        }
        // Complete one dependency at a time.
        // Completed nodes are already in memo.
        let mut stack = smallvec::SmallVec::<[LoweringFrame; 16]>::new();
        stack.push(LoweringFrame::new(arena, id));
        while let Some(frame) = stack.last_mut() {
            let cur = frame.id;
            if frame.nary_op != OP_CONST {
                let (children, _, identity) = nary_parts(arena.get(cur)).expect("n-ary frame");
                let mut descend = None;
                while let Some(&child) = children.get(frame.next) {
                    let child_reg = if let Some(&reg) = self.memo.get(&child) {
                        reg
                    } else if is_lowering_leaf(arena.get(child)) {
                        self.lower_leaf(arena, child)
                    } else {
                        descend = Some(child);
                        break;
                    };
                    frame.next += 1;
                    frame.acc = Some(match frame.acc {
                        Some(previous) => self.push(frame.nary_op, previous, child_reg),
                        None => child_reg,
                    });
                }
                if let Some(child) = descend {
                    stack.push(LoweringFrame::new(arena, child));
                    continue;
                }
                let reg = frame.acc.unwrap_or_else(|| self.push_const(identity));
                self.memo.insert(cur, reg);
                stack.pop();
                continue;
            }
            if let Some(child) = lowering_child(arena, cur, frame.next) {
                frame.next += 1;
                if !self.memo.contains_key(&child) {
                    if matches!(
                        arena.get(child),
                        ExprNode::Const(_)
                            | ExprNode::Var(_)
                            | ExprNode::Param(_)
                            | ExprNode::Linear { .. }
                    ) {
                        let reg = self.emit_node(arena, child);
                        self.memo.insert(child, reg);
                    } else {
                        stack.push(LoweringFrame::new(arena, child));
                    }
                }
            } else {
                let reg = self.emit_node(arena, cur);
                self.memo.insert(cur, reg);
                stack.pop();
            }
        }
        self.memo[&id]
    }

    fn lower_leaf(&mut self, arena: &ExprArena, id: ExprId) -> u32 {
        if let Some(&reg) = self.memo.get(&id) {
            return reg;
        }
        let reg = self.emit_node(arena, id);
        self.memo.insert(id, reg);
        reg
    }

    /// Lower several roots into one shared instruction stream, returning one
    /// register per root. Shared subexpressions across the roots are emitted
    /// once. Used by [`CompiledBatch`].
    fn lower_many(&mut self, arena: &ExprArena, roots: &[ExprId]) -> Vec<u32> {
        roots.iter().map(|&root| self.lower(arena, root)).collect()
    }

    /// Emit the instructions for a single node. All of its lowering
    /// dependencies must already be in `memo`.
    fn emit_node(&mut self, arena: &ExprArena, id: ExprId) -> u32 {
        match arena.get(id) {
            ExprNode::Const(c) => self.push_const(*c),
            ExprNode::Var(v) => self.push(OP_VAR, v.0, 0),
            ExprNode::Param(p) => self.push(OP_PARAM, p.0, 0),
            ExprNode::Add(children) => self.emit_nary(children, OP_ADD, 0.0),
            ExprNode::Mul(children) => self.emit_nary(children, OP_MUL, 1.0),
            ExprNode::Unary(op, inner) => {
                let r = self.memo[inner];
                self.push(unary_opcode(*op), r, 0)
            }
            ExprNode::Pow(base, exp) => {
                let base_reg = self.memo[base];
                if let ExprNode::Const(e) = arena.get(*exp) {
                    let idx = self.add_const(*e);
                    self.push(OP_POWC, base_reg, idx)
                } else {
                    let exp_reg = self.memo[exp];
                    self.push(OP_POW, base_reg, exp_reg)
                }
            }
            ExprNode::Div(num, den) => {
                let n = self.memo[num];
                let d = self.memo[den];
                self.push(OP_DIV, n, d)
            }
            ExprNode::Atan2(y, x) => {
                let y = self.memo[y];
                let x = self.memo[x];
                self.push(OP_ATAN2, y, x)
            }
            ExprNode::Min(children) => self.emit_nary(children, OP_MIN, f64::INFINITY),
            ExprNode::Max(children) => self.emit_nary(children, OP_MAX, f64::NEG_INFINITY),
            // OP_LINEAR requires a non-empty range (see `eval_tape`), so a
            // coefficient-free Linear node lowers to its constant.
            ExprNode::Linear { coeffs, constant } if coeffs.is_empty() => {
                self.push_const(*constant)
            }
            ExprNode::Linear { coeffs, constant } => {
                let start = to_u32(self.tape.lin_vars.len());
                for (v, c) in coeffs {
                    self.tape.lin_vars.push(v.0);
                    self.tape.lin_coeffs.push(*c);
                }
                let end = to_u32(self.tape.lin_vars.len());
                let sum = self.push(OP_LINEAR, start, end);
                if *constant == 0.0 {
                    sum
                } else {
                    let c = self.push_const(*constant);
                    self.push(OP_ADD, sum, c)
                }
            }
            ExprNode::Quadratic(q) => {
                let mut sum = self.push_const(q.constant());
                for &(v, c) in q.linear() {
                    let x = self.push(OP_VAR, v.0, 0);
                    let c = self.push_const(c);
                    let term = self.push(OP_MUL, c, x);
                    sum = self.push(OP_ADD, sum, term);
                }

                for &(a, b, h) in q.quadratic() {
                    let x = self.push(OP_VAR, a.0, 0);
                    let h = self.push_const(h);
                    let term = if a == b {
                        self.push(OP_SCALED_SQUARE, h, x)
                    } else {
                        let y = self.push(OP_VAR, b.0, 0);
                        let term = self.push(OP_MUL, h, x);
                        self.push(OP_MUL, term, y)
                    };
                    sum = self.push(OP_ADD, sum, term);
                }

                sum
            }
        }
    }

    fn emit_nary(&mut self, children: &[ExprId], op: u32, identity: f64) -> u32 {
        let Some((&first, rest)) = children.split_first() else {
            return self.push_const(identity);
        };
        let mut acc = self.memo[&first];
        for &child in rest {
            let r = self.memo[&child];
            acc = self.push(op, acc, r);
        }
        acc
    }
}

/// Reusable register scratch for [`Tape`] evaluation.
///
/// Grows to the largest tape evaluated through it and is reused across
/// calls, so hot evaluation loops never reallocate.
#[derive(Clone, Debug, Default)]
pub struct TapeScratch {
    regs: Vec<f64>,
}

impl TapeScratch {
    /// Empty scratch; grows on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Scratch pre-sized for `tape`.
    #[must_use]
    pub fn for_tape(tape: &Tape) -> Self {
        Self { regs: vec![0.0; tape.n_regs()] }
    }

    /// Ensure room for `n_regs` registers, preserving existing contents.
    pub fn ensure(&mut self, n_regs: usize) {
        if self.regs.len() < n_regs {
            self.regs.resize(n_regs, 0.0);
        }
    }

    /// Number of registers currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regs.len()
    }

    /// Whether no registers are held yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regs.is_empty()
    }

    fn regs_mut(&mut self) -> &mut [f64] {
        &mut self.regs
    }
}

/// A single expression compiled to a [`Tape`] with its own reusable scratch.
///
/// Compile once, evaluate many times without touching the arena or
/// reallocating.
#[derive(Clone, Debug)]
pub struct CompiledExpr {
    tape: Tape,
    scratch: TapeScratch,
}

impl CompiledExpr {
    /// Compile `root` and allocate a scratch buffer for it.
    #[must_use]
    pub fn compile(arena: &ExprArena, root: ExprId) -> Self {
        let tape = Tape::compile(arena, root);
        let scratch = TapeScratch::for_tape(&tape);
        Self { tape, scratch }
    }

    /// Evaluate at `x` with `params` (use [`params_snapshot`] for the arena's
    /// live parameter values and `&[]` when the expression has no parameters).
    pub fn eval(&mut self, x: &[f64], params: &[f64]) -> f64 {
        self.eval_with_mults(x, params, &[])
    }

    /// Evaluate with explicit multiplier weights (empty for plain expressions).
    pub fn eval_with_mults(&mut self, x: &[f64], params: &[f64], mults: &[f64]) -> f64 {
        let n = self.tape.n_regs();
        self.scratch.ensure(n);
        self.tape.value(x, params, mults, self.scratch.regs_mut())
    }

    /// The underlying tape.
    #[must_use]
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// Register count (scratch size).
    #[must_use]
    pub fn n_regs(&self) -> usize {
        self.tape.n_regs()
    }
}

/// Several expressions compiled into one shared instruction stream.
///
/// Shared subexpressions across the batch are lowered once and a single
/// interpreter pass evaluates every output, with one reusable scratch buffer
/// sized to the whole tape.
#[derive(Clone, Debug, Default)]
pub struct CompiledBatch {
    tape: Tape,
    roots: Vec<u32>,
    scratch: TapeScratch,
}

impl CompiledBatch {
    /// Compile every expression in `exprs`, deduplicating shared
    /// subexpressions across the batch.
    #[must_use]
    pub fn compile(arena: &ExprArena, exprs: &[ExprId]) -> Self {
        if exprs.is_empty() {
            return Self::default();
        }
        let mut builder = Builder::default();
        let roots = builder.lower_many(arena, exprs);
        let tape = std::mem::take(&mut builder.tape);
        let scratch = TapeScratch { regs: vec![0.0; tape.ops.len()] };
        Self { tape, roots, scratch }
    }

    /// Number of expressions in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Whether the batch holds no expressions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Register count (scratch size) of the shared tape.
    #[must_use]
    pub fn n_regs(&self) -> usize {
        self.tape.n_regs()
    }

    /// The shared tape.
    #[must_use]
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// Evaluate every expression at `x` into `out` (must have length
    /// [`CompiledBatch::len`]) with a single interpreter pass.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != self.len()`.
    pub fn eval(&mut self, x: &[f64], params: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), self.roots.len(), "batch output dimension");
        if self.roots.is_empty() {
            return;
        }
        self.scratch.ensure(self.tape.n_regs());
        let regs = self.scratch.regs_mut();
        let mut tail = [0.0];
        eval_tape(
            &self.tape.ops,
            &self.tape.a,
            &self.tape.b,
            &self.tape.consts,
            &self.tape.lin_vars,
            &self.tape.lin_coeffs,
            x,
            params,
            &[],
            regs,
            &mut tail,
        );
        for (slot, &root) in out.iter_mut().zip(&self.roots) {
            *slot = regs[root as usize];
        }
    }

    /// Evaluate the batch at each point in `points`, appending one output row
    /// per point into `out` (cleared first).
    pub fn eval_points(&mut self, points: &[Vec<f64>], params: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.reserve(points.len() * self.roots.len());
        let mut row = vec![0.0; self.roots.len()];
        for point in points {
            self.eval(point, params, &mut row);
            out.extend_from_slice(&row);
        }
    }
}

/// The `next`-th lowering dependency of `id`, or `None` when all have been
/// yielded. A constant `Pow` exponent folds into `OP_POWC` and is not a
/// dependency.
fn lowering_child(arena: &ExprArena, id: ExprId, next: usize) -> Option<ExprId> {
    match arena.get(id) {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => children.get(next).copied(),
        ExprNode::Unary(_, inner) => (next == 0).then_some(*inner),
        ExprNode::Pow(base, exp) => {
            if matches!(arena.get(*exp), ExprNode::Const(_)) {
                (next == 0).then_some(*base)
            } else {
                match next {
                    0 => Some(*base),
                    1 => Some(*exp),
                    _ => None,
                }
            }
        }
        ExprNode::Div(num, den) | ExprNode::Atan2(num, den) => match next {
            0 => Some(*num),
            1 => Some(*den),
            _ => None,
        },
        ExprNode::Const(_)
        | ExprNode::Var(_)
        | ExprNode::Param(_)
        | ExprNode::Linear { .. }
        | ExprNode::Quadratic(_) => None,
    }
}

fn is_lowering_leaf(node: &ExprNode) -> bool {
    matches!(
        node,
        ExprNode::Const(_)
            | ExprNode::Var(_)
            | ExprNode::Param(_)
            | ExprNode::Linear { .. }
            | ExprNode::Quadratic(_)
    )
}

fn nary_parts(node: &ExprNode) -> Option<(&[ExprId], u32, f64)> {
    match node {
        ExprNode::Add(children) => Some((children, OP_ADD, 0.0)),
        ExprNode::Mul(children) => Some((children, OP_MUL, 1.0)),
        ExprNode::Min(children) => Some((children, OP_MIN, f64::INFINITY)),
        ExprNode::Max(children) => Some((children, OP_MAX, f64::NEG_INFINITY)),
        _ => None,
    }
}

fn unary_opcode(op: UnaryOp) -> u32 {
    match op {
        UnaryOp::Neg => OP_NEG,
        UnaryOp::Abs => OP_ABS,
        UnaryOp::Sqrt => OP_SQRT,
        UnaryOp::Cbrt => OP_CBRT,
        UnaryOp::Exp => OP_EXP,
        UnaryOp::Exp2 => OP_EXP2,
        UnaryOp::Expm1 => OP_EXPM1,
        UnaryOp::Log => OP_LOG,
        UnaryOp::Log2 => OP_LOG2,
        UnaryOp::Log10 => OP_LOG10,
        UnaryOp::Log1p => OP_LOG1P,
        UnaryOp::Sin => OP_SIN,
        UnaryOp::Cos => OP_COS,
        UnaryOp::Tan => OP_TAN,
        UnaryOp::Asin => OP_ASIN,
        UnaryOp::Acos => OP_ACOS,
        UnaryOp::Atan => OP_ATAN,
        UnaryOp::Sinh => OP_SINH,
        UnaryOp::Cosh => OP_COSH,
        UnaryOp::Tanh => OP_TANH,
        UnaryOp::Asinh => OP_ASINH,
        UnaryOp::Acosh => OP_ACOSH,
        UnaryOp::Atanh => OP_ATANH,
    }
}
