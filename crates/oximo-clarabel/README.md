# oximo-clarabel

[Clarabel](https://clarabel.org) backend for [oximo](https://github.com/oximo-rs/oximo).

Wraps the [`clarabel`](https://crates.io/crates/clarabel) crate: an open-source,
pure-Rust interior-point solver for convex conic programs. Supports `LP`, convex `QP`
(quadratic objective, linear constraints), and `SOCP` model kinds. It has no
integer support, so mixed-integer models are rejected.

## Usage

Enable the `clarabel` feature on the umbrella `oximo` crate:

```toml
[dependencies]
oximo = { version = "0.3", features = ["clarabel"] }
```

To use this crate directly:

```toml
[dependencies]
oximo-clarabel = "0.3"
oximo-core     = "0.3"
oximo-solver   = "0.3"
```

The optional `faer` feature adds the [faer](https://crates.io/crates/faer)
sparse LDL factorization as an alternative KKT backend.

## Quick example

```rust,no_run
use oximo_core::prelude::*;
use oximo_clarabel::{Clarabel, ClarabelOptions};
use oximo_solver::{Solver, TerminationStatus};

let m = Model::new("socp");
variable!(m, x);
variable!(m, y);
variable!(m, t >= 0.0);
m.fix(t, 1.0);
soc_constraint!(m, disk, [x, y] <= t); // ||(x, y)||_2 <= t
objective!(m, Min, x + y);

let result = Clarabel.solve(&m, &ClarabelOptions::default()).unwrap();
assert_eq!(result.termination, TerminationStatus::Optimal);
println!("obj = {}", result.objective().unwrap()); // -sqrt(2)
```

Run the bundled SOCP example:

```sh
cargo run -p oximo --example gradostat_multiperiod_socp --features clarabel
```

## Supported model kinds

| Kind                                           | Supported                                                                                                                 |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `LP`                                           | Yes                                                                                                                       |
| `QP` (quadratic objective, linear constraints) | Yes, the objective Hessian must be positive semidefinite (Clarabel does not check and a non-convex `P` fails numerically) |
| `SOCP`                                         | Yes                                                                                                                       |
| `MILP`/`MIQP`/`MISOCP`                         | No, Clarabel has no integer support (returned as `UnsupportedKind`)                                                       |
| `QCP`/`MIQCP`                                  | No, convex-QCP-to-SOC reformulation is not implemented. Write the constraint in SOC form instead.                         |

A quadratic constraint that is already SOC-shaped (`sum p_i x_i^2 <= n t^2`,
`t >= 0`) is detected as `SOCP` and solved, a general convex quadratic
constraint is not.

## Persistent handle (repeated solves)

`Clarabel.solve` builds a fresh solver for each call. When you re-solve one model
many times (parameter sweeps, sensitivity studies, column generation, rolling
horizons, etc) build a resident handle with `Clarabel.persistent()` and call
`solve` on it instead. `ClarabelPersistent` is a plain `Solver`. When the next
model has the same dimensions, cone layout, and `P`/`A` sparsity pattern, only
the numeric data (`P`, `q`, `A`, `b`) is overwritten in place via Clarabel's
`update_data`, reusing the KKT symbolic factorization. Any structural change
(added/removed rows or columns, a changed sparsity pattern, a new cone, or a
flipped constraint sense that moves a row between cones) triggers a transparent
rebuild, so results always match a cold solve.

## Convex quadratic objectives (QP)

A quadratic objective (e.g. `x.powi(2)`, `x * y`) with linear constraints is
detected as `QP`. oximo derives the Hessian from the objective and Clarabel
minimizes `c'x + 0.5·x'Px`.

> **Convexity.** Clarabel solves only convex QPs. For minimization `P` must be
> positive semidefinite; for maximization, negative semidefinite. Clarabel does
> not check this, so an indefinite or incorrectly signed Hessian may fail
> numerically or return a non-optimal point.

## Options

`ClarabelOptions` is a typed builder. All methods return `Self` for chaining,
and every field is `Option`. `None` leaves Clarabel's own default in place.

### Universal options (via `UniversalOptionsExt`)

| Method                  | Clarabel setting                           | Default |
|-------------------------|--------------------------------------------|---------|
| `.time_limit(Duration)` | `time_limit`                               | none    |
| `.threads(usize)`       | `max_threads` (KKT solvers only, see note) | none    |
| `.verbose(bool)`        | `verbose`                                  | none    |

`.threads` maps to `max_threads`, which only affects multithreaded KKT solvers.
The default `qdldl` is single-threaded.

### Clarabel options (selected)

| Method                                      | Clarabel setting                | Default |
|---------------------------------------------|---------------------------------|---------|
| `.direct_solve_method(ClarabelDirectSolve)` | `direct_solve_method`           | `Auto`  |
| `.max_iter(u32)`                            | `max_iter`                      | `200`   |
| `.max_step_fraction(f64)`                   | `max_step_fraction`             | `0.99`  |
| `.tol_gap_abs(f64)`                         | `tol_gap_abs`                   | `1e-8`  |
| `.tol_gap_rel(f64)`                         | `tol_gap_rel`                   | `1e-8`  |
| `.tol_feas(f64)`                            | `tol_feas`                      | `1e-8`  |
| `.tol_infeas_abs(f64)`                      | `tol_infeas_abs`                | `1e-8`  |
| `.tol_infeas_rel(f64)`                      | `tol_infeas_rel`                | `1e-8`  |
| `.tol_ktratio(f64)`                         | `tol_ktratio`                   | `1e-6`  |
| `.equilibrate_enable(bool)`                 | `equilibrate_enable`            | `true`  |
| `.static_regularization_enable(bool)`       | `static_regularization_enable`  | `true`  |
| `.dynamic_regularization_enable(bool)`      | `dynamic_regularization_enable` | `true`  |
| `.iterative_refinement_enable(bool)`        | `iterative_refinement_enable`   | `true`  |
| `.presolve_enable(bool)`                    | `presolve_enable`               | `true`  |
| `.input_sparse_dropzeros(bool)`             | `input_sparse_dropzeros`        | `false` |

There is also a matching set of low-accuracy fallback tolerances
(`reduced_tol_*`) and further equilibration, line-search, and regularization
options. The full list is in [src/options.rs](src/options.rs).

```rust,ignore
use oximo_clarabel::{ClarabelOptions, ClarabelDirectSolve};
use oximo_solver::UniversalOptionsExt;

let opts = ClarabelOptions::default()
    .max_iter(100)
    .tol_gap_rel(1e-9)
    .direct_solve_method(ClarabelDirectSolve::Qdldl)
    .verbose(true);
```

## Result

`SolverResult` fields, populated whenever a usable point is available
(`primal_status` is `FeasiblePoint` or `OptimalPoint`):

- `solutions` - primal points (`Vec<SolutionPoint>`). This backend returns a single point holding the `primal` values keyed by `VarId` and the `objective` (adjusted for the objective sign and any constant term). Access via `result.objective()` / `result.value_of(var)`
- `dual` - constraint duals, keyed by `ConstraintId`, access via `result.dual_of(c)`. Reported for equality and single-sided/range linear constraints in the LP convention (`gradient = A' y` of the problem as posed)
- `soc_dual` - for each explicit `soc_constraint!`, its norm-form bound multiplier (`z0`), keyed by `SocConstraintId`, access via `result.soc_dual_of(soc)`
- `reduced_costs` - not populated by this backend (always empty)
- `termination` - why the solve stopped, mapped from Clarabel's `SolverStatus`: `Optimal` (`Solved`), `Infeasible` (`PrimalInfeasible`), `Unbounded` (`DualInfeasible`), `IterationLimit` (`MaxIterations`), `TimeLimit` (`MaxTime`), `Interrupted` (`AlmostSolved`), `NumericError` (`NumericalError` / `InsufficientProgress`), `NotSolved` (`Unsolved`)
- `primal_status` - whether a usable point came back (`NoSolution` / `FeasiblePoint` / `OptimalPoint`), `result.has_solution()` is the shortcut
- `best_bound` / `gap` - always `None` (conic solver, no MIP bound)
- `solve_time` - wall time measured around the Clarabel solve call
- `iterations` - interior-point iteration count

## License

MIT OR Apache-2.0
