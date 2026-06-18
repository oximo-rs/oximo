# oximo-highs

HiGHS LP/MILP backend for [oximo](https://github.com/oximo-rs/oximo).

Wraps the [`highs`](https://crates.io/crates/highs) crate (HiGHS bundled, no separate install required). Supports `LP`, `MILP`, and convex continuous `QP` model kinds.

## Usage

Most users should depend on the umbrella `oximo` crate with the `highs` feature (enabled by default):

```toml
[dependencies]
oximo = "0.2"
```

To use this crate directly:

```toml
[dependencies]
oximo-highs = "0.2"
oximo-core   = "0.2"
oximo-solver = "0.2"
```

## Quick example

```rust,no_run
use oximo_core::prelude::*;
use oximo_highs::{Highs, HighsOptions};
use oximo_solver::{Solver, SolverStatus};

let m = Model::new("toy");
variable!(m, x >= 0.0);
variable!(m, 0.0 <= y <= 4.0);
constraint!(m, c1, x + 2.0 * y <= 14.0);
constraint!(m, c2, 3.0 * x - y >= 0.0);
objective!(m, Max, 3.0 * x + 4.0 * y);

let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
assert_eq!(result.status, SolverStatus::Optimal);
println!("obj = {}", result.objective().unwrap()); // 34.0
println!("x   = {}", result.value_of(x).unwrap()); // 6.0
println!("y   = {}", result.value_of(y).unwrap()); // 4.0
```

## Quadratic programs (QP)

A quadratic objective (e.g. `x.powi(2)`, `x * y`) with linear constraints is
detected as `QP` and solved by uploading the Hessian `Q` via
`Highs_passHessian`. oximo derives `Q` from the objective with
`oximo_expr::extract_quadratic`; HiGHS then minimizes `c'x + 0.5·x'Qx`.

```rust,no_run
use oximo_core::prelude::*;
use oximo_highs::{Highs, HighsOptions};
use oximo_solver::{Solver, SolverStatus};

let m = Model::new("qp");
variable!(m, -10.0 <= x <= 10.0);
variable!(m, -10.0 <= y <= 10.0);
constraint!(m, c, x + y == 1.0);
objective!(m, Min, x.powi(2) + y.powi(2)); // min x^2 + y^2

let result = Highs.solve(&m, &HighsOptions::default()).unwrap();
assert_eq!(result.status, SolverStatus::Optimal); // x = y = 0.5, obj = 0.5
```

> **Convexity.** HiGHS supports only convex QPs.
> For minimization, `Q` must be positive semidefinite (PSD),
> and for maximization, `Q` must be negative semidefinite (NSD).
> HiGHS does not check this condition, so supplying an indefinite
> or incorrectly signed Hessian may lead to incorrect or non-optimal solutions.

HiGHS does **not** support MIQP (integer + quadratic, returned as
`UnsupportedKind`) or quadratic *constraints* (returned as `Nonlinear`).

## Options

`HighsOptions` is a typed builder. All methods return `Self` for chaining.

| Method                       | Type       | HiGHS option                    | Default |
|------------------------------|------------|---------------------------------|---------|
| `.time_limit(Duration)`      | universal  | `time_limit`                    | none    |
| `.threads(usize)`            | universal  | `threads`                       | none    |
| `.verbose(bool)`             | universal  | `output_flag`, `log_to_console` | none    |
| `.mip_gap(f64)`              | HiGHS-only | `mip_rel_gap`                   | none    |
| `.presolve(HighsPresolve)`   | HiGHS-only | `presolve`                      | none    |
| `.method(HighsMethod)`       | HiGHS-only | `solver`                        | none    |
| `.parallel(bool)`            | HiGHS-only | `parallel`                      | none    |

`HighsMethod` variants: `Choose` (let HiGHS decide), `Simplex`, `Ipm` (interior-point), `PdLp` (first-order).

`HighsPresolve` variants: `Off`, `On`, `Auto` (`"choose"`).

```rust
use std::time::Duration;
use oximo_highs::{HighsOptions, HighsMethod, HighsPresolve};
use oximo_solver::UniversalOptionsExt;

let opts = HighsOptions::default()
    .time_limit(Duration::from_secs(60))
    .threads(4)
    .mip_gap(0.01)
    .presolve(HighsPresolve::On)
    .method(HighsMethod::Ipm)
    .parallel(true);
```

## Result

`SolverResult` fields populated on `Optimal` or `Feasible`:

- `solutions` - primal points (`Vec<SolutionPoint>`). This backend returns a single point holding the `primal` values keyed by `VarId` and the `objective` (adjusted for any constant term). Access via `result.objective()` / `result.value_of(var)`
- `dual` - constraint duals, keyed by `ConstraintId`, access via `result.dual_of(c)`.
- `reduced_costs` - variable reduced costs, keyed by `VarId`
- `solve_time` - wall time measured around the HiGHS solve call

## License

MIT OR Apache-2.0
