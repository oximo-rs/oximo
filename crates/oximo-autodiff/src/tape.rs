//! Flat instruction tapes compiled from [`ExprArena`] subtrees.
//!
//! A [`Tape`] is the bridge between oximo's runtime expression data and
//! `std::autodiff`, which differentiates compiled Rust functions. Every
//! expression is lowered to instructions executed by the single interpreter
//! `eval_tape`, and the `enzyme` feature differentiates that interpreter
//! once at compile time.
//!
//! The tape is in SSA form.
use oximo_expr::{ExprArena, ExprId, ExprNode};
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
    #[cfg_attr(not(feature = "enzyme"), allow(dead_code))]
    #[allow(clippy::type_complexity)]
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
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
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
    for i in 0..n {
        let ai = a[i] as usize;
        let bi = b[i] as usize;
        match ops[i] {
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
            // Unreachable for a well-formed tape (the builder emits only the
            // opcodes above).
            _ => regs[i] = f64::NAN,
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

    fn lower(&mut self, arena: &ExprArena, id: ExprId) -> u32 {
        if let Some(&reg) = self.memo.get(&id) {
            return reg;
        }
        let reg = match arena.get(id) {
            ExprNode::Const(c) => self.push_const(*c),
            ExprNode::Var(v) => self.push(OP_VAR, v.0, 0),
            ExprNode::Param(p) => self.push(OP_PARAM, p.0, 0),
            ExprNode::Add(children) => self.lower_nary(arena, children, OP_ADD, 0.0),
            ExprNode::Mul(children) => self.lower_nary(arena, children, OP_MUL, 1.0),
            ExprNode::Neg(inner) => {
                let r = self.lower(arena, *inner);
                self.push(OP_NEG, r, 0)
            }
            ExprNode::Pow(base, exp) => {
                let base_reg = self.lower(arena, *base);
                if let ExprNode::Const(e) = arena.get(*exp) {
                    let idx = self.add_const(*e);
                    self.push(OP_POWC, base_reg, idx)
                } else {
                    let exp_reg = self.lower(arena, *exp);
                    self.push(OP_POW, base_reg, exp_reg)
                }
            }
            ExprNode::Div(num, den) => {
                let n = self.lower(arena, *num);
                let d = self.lower(arena, *den);
                self.push(OP_DIV, n, d)
            }
            ExprNode::Sin(inner) => {
                let r = self.lower(arena, *inner);
                self.push(OP_SIN, r, 0)
            }
            ExprNode::Cos(inner) => {
                let r = self.lower(arena, *inner);
                self.push(OP_COS, r, 0)
            }
            ExprNode::Exp(inner) => {
                let r = self.lower(arena, *inner);
                self.push(OP_EXP, r, 0)
            }
            ExprNode::Log(inner) => {
                let r = self.lower(arena, *inner);
                self.push(OP_LOG, r, 0)
            }
            ExprNode::Abs(inner) => {
                let r = self.lower(arena, *inner);
                self.push(OP_ABS, r, 0)
            }
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
        };
        self.memo.insert(id, reg);
        reg
    }

    fn lower_nary(
        &mut self,
        arena: &ExprArena,
        children: &[ExprId],
        op: u32,
        identity: f64,
    ) -> u32 {
        let Some((&first, rest)) = children.split_first() else {
            return self.push_const(identity);
        };
        let mut acc = self.lower(arena, first);
        for &child in rest {
            let r = self.lower(arena, child);
            acc = self.push(op, acc, r);
        }
        acc
    }
}
