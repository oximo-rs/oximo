# oximo-expr

Arena-allocated expression tree for [oximo](https://github.com/oximo-rs/oximo).

All expressions in oximo are nodes in a single `ExprArena` owned by the `Model` through an `ExprArenaCell`. User code holds lightweight `Expr` handles, an `(ExprId, &ExprArenaCell)` pair. Copying an `Expr` copies an ID, not a subtree. Operator overloads collapse linear combinations into a single `Linear` node so LP/MILP construction never traverses an `Add(Mul(Const, Var), ...)` tree.

`ExprArenaCell` provides synchronized interior mutability, so expression handles can be shared by indexed-family callbacks. Large indexed builds use isolated worker-local arena forks and merge their nodes in set order before serial model registration. Ordinary scalar expression construction keeps the same arena-backed API.

`ExprArenaCell::borrow()` returns an `ExprArenaSnapshot`, a cheap immutable
copy-on-write snapshot.

This crate is the fundamental layer. End users depend on `oximo-core`, which re-exports all types from here so a separate `oximo-expr` import is not needed.

## Key types

| Type            | Description                                             |
|-----------------|---------------------------------------------------------|
| `ExprArena`     | Contiguous backing store of `ExprNode` values           |
| `ExprArenaCell` | Synchronized owner used by expression handles           |
| `ExprId`        | Newtype `u32` index into an arena                       |
| `ExprNode`      | Enum of all node kinds (see below)                      |
| `UnaryOp`       | Public enum of unary scalar operations                  |
| `Expr<'a>`      | Lightweight handle: `id` + borrow of the arena. `Copy`. |
| `VarId`         | Opaque variable index                                   |
| `ParamId`       | Opaque parameter index                                  |
| `LinearTerms`   | Extracted `Vec<(VarId, f64)>` + constant                |

### `ExprNode` variants

```rust,ignore
Const(f64)
Var(VarId)
Param(ParamId)
Add(Children) // generic n-ary add
Mul(Children) // generic n-ary mul
Unary(UnaryOp, ExprId)
Pow(ExprId, ExprId)
Div(ExprId, ExprId) // numerator / denominator
Atan2(ExprId, ExprId) // y, x order
Min(Children) / Max(Children)
Linear { coeffs: Vec<(VarId, f64)>, constant: f64 } // LP fast-path
```

`Linear` is produced automatically by operator overloads when all operands are linear. LP/MILP backends detect it and skip tree traversal entirely.

## Operator overloads

`Expr` implements `Add`, `Sub`, `Mul`, `Div`, `Neg` against other `Expr` values and against `f64`. All operations that stay linear produce a `Linear` node. For example:

```rust,ignore
// All of these produce a single Linear node, not an Add/Mul tree:
let e = 2.0 * x + 3.0 * y - 1.0;
let e = x + y;
let e = -x;
let e = x / 2.0; // constant denominator: stays linear (x*0.5)
```

## Nonlinear methods on `Expr`

```rust,ignore
expr.pow(exponent) // Expr ^ Expr
expr.powi(n: i32)  // integer exponent shorthand
expr.powf(n: f64)  // float exponent shorthand
expr.neg()         // explicit UnaryOp::Neg (the `-expr` operator keeps affine lowering)
expr.sin()
expr.cos()
expr.tan()
expr.exp()
expr.exp2()
expr.expm1() // alias: exp_m1()
expr.log()   // alias: ln()
expr.log1p() // alias: ln_1p()
expr.log2()
expr.log10()
expr.abs()
expr.sqrt()
expr.cbrt()
expr.asin() / expr.acos() / expr.atan()
expr.sinh() / expr.cosh() / expr.tanh()
expr.asinh() / expr.acosh() / expr.atanh()
expr.atan2(x) // self is y
expr.min(other) / expr.max(other) // flattened n-ary nodes
expr/expr
```

## Utilities

### Summing expressions

`Expr` implements `std::iter::Sum`, so any iterator of `Expr` (or `&Expr`) can
be collapsed with the standard `.sum()`. The arena is taken from the first
element, empty iterators panic.

```rust,ignore
let total: Expr = vars.iter().copied().sum();
```

For coefficient-weighted sums (`sum_{i} c_i * e_i`) use the `dot` helper:

```rust,ignore
use oximo_expr::dot;
let total = dot(&vars, &coeffs);
```

For sums indexed by an `oximo-core` `Set`, prefer the `sum!` macro (re-exported by
`oximo-core` / `oximo::prelude`).

### `extract_linear`

```rust,ignore
use oximo_expr::extract_linear;
let terms: Option<LinearTerms> = extract_linear(&arena, expr_id);
```

Returns `Some(LinearTerms)` if the subtree is affine, `None` if it contains nonlinear nodes. Used by backends and `oximo-io` to validate and serialize models.

### `evaluate`

```rust,ignore
use oximo_expr::{evaluate, EvalContext};
let mut ctx = EvalContext::new();
ctx.set_var(var_id, 3.0);
let val: f64 = evaluate(&arena, expr_id, &ctx)?;
```

Numerically evaluates an expression subtree given variable assignments.

### `simplify`

```rust,ignore
use oximo_expr::simplify;
let simplified_id = simplify(&mut arena, expr_id);
```

Constant-folds and simplifies the subtree in place.

### `Visitor` / `walk`

```rust,ignore
use oximo_expr::{Visitor, walk};
struct MyVisitor;
impl Visitor for MyVisitor {
    fn visit(&mut self, arena: &ExprArena, id: ExprId) { /* ... */ }
}
walk(&arena, root_id, &mut MyVisitor);
```

Depth-first traversal for custom analysis passes.

## Usage

End users do not need to depend on this crate directly. Depend on `oximo-core` instead, it re-exports all types from `oximo-expr`.

To use this crate directly (e.g. for a custom backend):

```toml
[dependencies]
oximo-expr = "0.6.0"
```

## License

MIT OR Apache-2.0
