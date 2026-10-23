# oximo-scip

`oximo-scip` is the SCIP backend for [oximo](https://github.com/oximo-rs/oximo).
It uses [russcip 0.11](https://crates.io/crates/russcip/0.11.0).

## Installation

Bundled SCIP binaries are enabled by default:

```toml
[dependencies]
oximo = { version = "0.7", features = ["scip"] }
```

For an existing SCIP installation, disable default features and use `system`:

```toml
[dependencies]
oximo-scip = { version = "0.7", default-features = false, features = ["system"] }
```

The system build discovers SCIP through `SCIPOPTDIR`, `CONDA_PREFIX`, or the
standard installation directories and generates bindings with libclang. Bundled
and system modes are mutually exclusive.

`SCIPOPTDIR` must point to a SCIP development tree with the upstream layout
`include/scip/{scip.h,scipdefplugins.h,def.h}` and `lib/libscip*`; a runtime-only
SCIP package (or a distribution-specific header layout) is not sufficient.

## Solving

`Scip` accepts LP, MILP, QP, MIQP, QCP, MIQCP, SOCP, MISOCP, NLP, and MINLP
models. Linear and quadratic expressions are emitted through SCIP's native row
builders. General nonlinear constraints use russcip's expression API. Supported
operations are arithmetic, division, constant exponentiation, negation, abs,
sqrt, exp, log, sin, and cos. SCIP's documented `sqrt(x) = x^0.5` expression is
used for square roots. Nonlinear objectives use the documented auxiliary-variable
epigraph/hypograph construction.

```rust
use oximo_core::prelude::*;
use oximo_scip::{Scip, ScipOptions};
use oximo_solver::Solver;

let model = Model::new("box");
variable!(model, 0.0 <= x <= 10.0);
constraint!(model, limit, x.exp() <= 20.0);
objective!(model, Max, x);

let result = Scip::new().solve(&model, &ScipOptions::default())?;
assert!(result.has_solution());
# Ok::<(), oximo_solver::SolverError>(())
```

Native SOS1 and SOS2 constraints, binary-triggered affine indicator constraints,
semi-continuous and semi-integer variables, and explicit second-order cones are
supported.

`ScipOptions` embeds oximo's universal time, thread, and verbosity options. It
also provides typed builders for common SCIP limits and tolerances plus a
`param(name, value)` escape hatch for safe russcip parameter types. Use
`Scip::write_model` to export `cip`, `lp`, or `mps` files.

`ScipSolveOutput` adds SCIP's statistics JSON and a separate map for columns
created by a pricer. `SolverResult` contains original-variable solution points,
solution pools, best bounds, gaps, termination/primal status, timing, node and
LP-iteration counts, and certified LP duals/reduced costs when SCIP exposes a
complete original-model certificate.

## Persistent solves

```rust,no_run
use oximo_core::prelude::*;
use oximo_scip::{Scip, ScipOptions};
use oximo_solver::{PersistentSolver, Solver};

let model = Model::new("box");
variable!(model, 0.0 <= x <= 10.0);
objective!(model, Max, x);
let mut solver = Scip::new().persistent();
let first = solver.solve(&model, &ScipOptions::default()).unwrap();
let second = solver.solve(&model, &ScipOptions::default()).unwrap();
```

An unchanged model is retained. Append-only variables, rows, SOS sets, and
indicator constraints are appended to the resident original problem. Changes
to sparse linear row coefficients update in place when the row bounds remain equal
and the objective is not quadratic. Dense edits and QP row edits rebuild.
Other row edits, variable bounds, domains, objectives, or options rebuild
automatically.
Unaffected formulas retain their preparation. A previous incumbent and declared
initial values are submitted as partial starts.

## SCIP plugins

`Scip::with_plugins` receives a `ScipPluginRegistry` on every native model build.
The registry exposes safe, oximo-ID-aware traits for event handlers, separators,
pricers, branching rules, heuristics, constraint handlers, and node selectors.
Callbacks can inspect original values, submit solutions and local/global cuts,
read LP/Farkas prices, add generated columns, and identify generated variables.
Callback panics are caught and returned as solver errors after SCIP unwinds.
Concurrent SCIP solving is rejected when custom Rust plugins are installed.

The registry also has a `native()` hook for safe russcip model configuration;
native wrappers must not be retained by callback state.

## License

MIT OR Apache-2.0
