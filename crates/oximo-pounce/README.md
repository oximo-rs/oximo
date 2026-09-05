# oximo-pounce

A [POUNCE](https://github.com/jkitchin/pounce) backend for oximo. POUNCE is a
pure-Rust optimization suite. This backend uses its convex LP/QP/conic engines
and its IPOPT-based general NLP engines.

## Derivatives

- **Default (stable):** every objective/constraint is classified once.
  - A model that is entirely linear/quadratic (LP/QP/QCP) solves on POUNCE's
    low-level `TNLP` surface with exact analytic gradients, Jacobian rows,
    and the exact constant Hessian of the Lagrangian including duals,
    reduced costs, iteration counts, and full primal-dual warm starts
    via the persistent handle.
  - A model with a nonlinear function solves through POUNCE's `builder` surface,
    where it uses finite differences and limited-memory L-BFGS Hessian.
    The backend supplies values from compiled tapes and still fills the exact
    closed-form parts through the builder's `gradient`/`jacobian` hooks.
    The builder doesn't expose solver internals, so on this path reduced costs and
    iteration counts are unavailable and warm starts are primal-only.
- **`enzyme` feature (nightly):** exact gradient, sparse Jacobian, and sparse
  Hessian of the Lagrangian for everything, including nonlinear functions,
  from `oximo-autodiff` using `TNLP`.

```rust,ignore
use oximo_pounce::Pounce;
use oximo_solver::Solver;

let res = Pounce.solve(&model, &Default::default())?;
```

## Options

`PounceOptions` has dedicated setters for the common controls (`tol`, `max_iter`,
`print_level`, `mu_strategy`, `solver_selection`) plus typed builder methods for POUNCE's
[documented option reference](https://kitchingroup.cheme.cmu.edu/pounce/options.html)
(barrier-µ strategy, quality-function oracle, L1 penalty-barrier, NLP
presolve/FBBT/auxiliary preprocessing, FERAL backend tuning, the
convergence/restoration controls, convex-IPM controls, and active-set QP tuning).
Each method is named exactly like the POUNCE option:

```rust,ignore
use oximo_pounce::{MuStrategy, PounceOptions, Pounce, PounceSolverSelection};
use oximo_solver::Solver;

let opts = PounceOptions::default()
    .tol(1e-8)
    .mu_strategy(MuStrategy::Adaptive)
    .mu_oracle("probing")
    .solver_selection(PounceSolverSelection::Auto)
    .presolve(true)
    .linear_solver("feral");

// Escape hatch:
let opts = opts.set("acceptable_tol", 1e-5);

let res = Pounce.solve(&model, &opts)?;
```

A few NLP options are managed by this backend and should not be set by hand:
`print_level` (via `verbose`/the `print_level` setter), `max_wall_time` (via `time_limit`),
`warm_start_init_point` (via the persistent handle), and `hessian_approximation`
(set to `limited-memory` only when the model has a nonlinear function and the
`enzyme` feature is off).

## Solver type/routing

`PounceSolverSelection::Auto` is the default. It normalizes the objective
sense, certifies a quadratic Hessian as positive semidefinite using POUNCE's
FERAL inertia interface, and routes only a proved-convex model:

- LP and convex QP use the convex IPM.
- SOCP with a convex objective uses the conic IPM. Both explicit oximo cones
  and recognized quadratic SOC forms are translated.
- QCP, indefinite QP, and general NLP use the TNLP/builder path.

An inconclusive convexity test always falls back to NLP. If an automatic LP or
detected-SOCP convex solve ends in a numerical failure, it receives one NLP
attempt.
Forced routes never change engines: `LpIpm`, `QpIpm`, `QpActiveSet`, and
`Socp` either run that engine or return a compatibility error. `Nlp` always
selects the general path.


## Licensing

`pounce-rs` is licensed under EPL-2.0.
oximo itself is MIT OR Apache-2.0.
