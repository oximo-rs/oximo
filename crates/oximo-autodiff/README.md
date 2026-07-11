# oximo-autodiff

Automatic differentiation for [oximo](https://github.com/oximo-rs/oximo) models: gradients, sparse Jacobians, and sparse Hessians of the Lagrangian, computed exactly with [`std::autodiff`](https://doc.rust-lang.org/nightly/std/autodiff/) (Enzyme).

Model expressions are runtime data, while Enzyme differentiates compiled Rust.
This crate bridges the two by compiling each expression into a flat instruction tape and differentiating the tape interpreter once at compile time: reverse mode for gradients, forward-over-reverse for Hessian-vector products.

Linear and quadratic expressions never hit the AD engine, since they use the closed-form `extract_linear`/`extract_quadratic` fast paths.

## Two layers: stable and nightly

The crate is split into a value layer that builds on stable Rust and a derivative layer gated behind the nightly-only `enzyme` feature.

**`Tape` (stable).** `Tape::compile` lowers an expression
once into a flat SSA instruction tape, deduplicating shared subexpressions. `Tape::value` then evaluates it at any point in a tight loop over owned buffers, without borrowing the model. It is a compile-once, evaluate-many value oracle. Re-walking the expression arena at every point (as the general `oximo_expr::evaluate` does) re-discovers the sharing, re-dispatches per node, and re-borrows the model on each call; the tape pays that structural cost once at compile time and never again. It needs neither nightly nor the `enzyme` feature.

**The derivative engine (`enzyme`, nightly).** Differentiates that same tape interpreter with `std::autodiff` for exact gradients, Jacobians, and Hessians. The stable value path is the primal it differentiates, so finite-difference and exact backends agree on exactly what "the function" is.

## Toolchain requirements

The derivative engine is gated behind the non-default `enzyme` feature and needs:

- a nightly toolchain with the `enzyme` component
  (See [Enzyme installation instructions](https://rustc-dev-guide.rust-lang.org/autodiff/installation.html)),
- `RUSTFLAGS="-Zautodiff=Enable"`,
- a fat-LTO profile, the workspace provides `--profile enzyme`.

```bash
RUSTFLAGS="-Zautodiff=Enable" cargo +nightly test -p oximo-autodiff --features enzyme --profile enzyme
```

## Who this is for

Most users never depend on this crate directly. It is the derivative engine that oximo's nonlinear backends use internally to feed exact gradients, Jacobians, and Hessians to the solver.

Reach for `oximo-autodiff` directly only when writing a new NLP backend, when you need raw derivatives of a model's expressions, or when you need a fast compiled evaluator (`Tape`) for their values.

## How a backend uses it

Build a model with the `oximo` modeling macros, then construct one
`NlpEvaluator`. It owns the compiled tapes, sparsity patterns, and scratch buffers, so evaluation never touches the model again. A backend calls it from its solver callbacks.

```rust,ignore
use oximo_core::Model;
use oximo_core::prelude::*;
use oximo_autodiff::NlpEvaluator;

let m = Model::new("nlp");
variable!(m, -10.0 <= x <= 10.0);
variable!(m, -10.0 <= y <= 10.0);
objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));
constraint!(m, disk, x.powi(2) + y.powi(2) <= 4.0);

let eval = NlpEvaluator::new(&m)?;
let point = [0.5, 0.5];

// Objective value and its dense gradient
let f = eval.eval_objective(&point);
let mut grad = vec![0.0; eval.num_variables()];
eval.eval_objective_gradient(&point, &mut grad);

// Constraint values
let mut g = vec![0.0; eval.num_constraints()];
eval.eval_constraint(&point, &mut g);

// Sparse constraint Jacobian
let jac_pattern = eval.jacobian_structure(); // &[(usize, usize)]
let mut jac = vec![0.0; jac_pattern.len()];
eval.eval_constraint_jacobian(&point, &mut jac);

// Sparse lower-triangle Hessian of the Lagrangian
let hess_pattern = eval.hessian_lagrangian_structure();
let mut hess = vec![0.0; hess_pattern.len()];
let sigma = 1.0;
let lambda = vec![0.0; eval.num_constraints()];
eval.eval_hessian_lagrangian(&point, sigma, &lambda, &mut hess);
```

For a parameter sweep, a resident backend can keep one evaluator and update it in place with `eval.try_refresh(&m)` (reuses the tapes when the structure is unchanged, returns `false` to signal a rebuild).

The dense gradient of a single expression at a point is also available directly, without a full evaluator:

```rust,ignore
// grad indexed by variable
let grad = oximo_autodiff::gradient_at(&m, expr, &x_bar)?;
```

## License

MIT OR Apache-2.0
