<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/oximo-rs/oximo/main/media/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oximo-rs/oximo/main/media/logo-dark.svg">
  <img alt="oximo logo" src="https://raw.githubusercontent.com/oximo-rs/oximo/main/media/logo-dark.svg">
</picture>

<a href="https://oximo.dev">
    <img src="https://img.shields.io/badge/oximo-website-orange" alt="Website">
</a>
<a href="https://github.com/oximo-rs/oximo/tree/main/crates/oximo/examples">
    <img src="https://img.shields.io/badge/oximo-examples-orange" alt="Examples">
</a>
<a href="https://crates.io/crates/oximo">
    <img src="https://img.shields.io/crates/v/oximo?logo=rust&color=E05D44" alt="crates version">
</a>
<a href="https://github.com/oximo-rs/oximo/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/oximo-rs/oximo/ci.yml?branch=main&label=oximo%20CI&logo=github" alt="CI">
</a>
<a href="https://docs.rs/oximo">
    <img src="https://docs.rs/oximo/badge.svg" alt="oximo Docs.rs Build Status">
</a>
<a href="https://codecov.io/gh/oximo-rs/oximo">
    <img src="https://codecov.io/gh/oximo-rs/oximo/graph/badge.svg?token=ff0RIQTCA4" alt="code coverage">
</a>

oximo is a Rust algebraic modeling library for mathematical optimization. See the
[webdocs](https://oximo.dev), [API documentation](https://docs.rs/oximo), and
[examples](https://github.com/oximo-rs/oximo/tree/main/crates/oximo/examples) for more.

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Highs;

let demand = [4.0, 6.0, 5.0];

let m = Model::new("production");
variable!(m, production[p in 0..2, t in 0..3] >= 0.0);
constraint!(m, meet[t in 0..3],
    sum!(production[p, t] for p in 0..2) >= demand[t]);
objective!(m, Min, sum!(production[p, t] for p in 0..2, t in 0..3));

let mut solver = Highs;
let result = solver.solve(&m, &HighsOptions::default())?;
println!("objective = {:?}", result.objective());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Problem types

The modeling layer supports a range of algebraic optimization problems. The available problem types depend on the solver backend:

- Linear programming (LP)
- Quadratic programming and quadratically constrained programming (QP/QCP)
- Nonlinear programming (NLP)
- Mixed-integer linear programming (MILP)
- Mixed-integer nonlinear programming (MINLP)
- Mixed-integer quadratic and quadratically constrained programming (MIQP/MIQCP)
- Second-order cone programming (SOCP/MISOCP)
- Semidefinite programming (SDP)

## Features

| Feature                   | What it adds                                                 | Default |
| ------------------------- | ------------------------------------------------------------ | ------- |
| `io`                      | NL, MPS, and LP file readers and writers                     | yes     |
| `highs`                   | HiGHS solver (bundled, requires a C/C++ compiler)            | no      |
| `gurobi`                  | Gurobi v13+ solver (requires licensed install)               | no      |
| `mosek`                   | MOSEK 11.2 solver                                            | no      |
| `gams`                    | GAMS bridge                                                  | no      |
| `baron`                   | BARON - global non-convex solver (requires licensed install) | no      |
| `clarabel`                | Clarabel conic solver (pure Rust, no install)                | no      |
| `clarabel-faer`           | Clarabel with the faer sparse linear-algebra backend         | no      |
| `clarabel-sdp`            | Clarabel SDP support (requires a BLAS/LAPACK provider)       | no      |
| `clarabel-sdp-openblas`   | Clarabel SDP with OpenBLAS                                   | no      |
| `clarabel-sdp-mkl`        | Clarabel SDP with Intel MKL                                  | no      |
| `clarabel-sdp-accelerate` | Clarabel SDP with Apple Accelerate                           | no      |
| `pounce`                  | POUNCE - pure-Rust IPOPT port (no install)                   | no      |
| `pounce-enzyme`           | POUNCE with exact Enzyme derivatives (nightly)               | no      |
| `scip`                    | SCIP (bundled)                                               | no      |
| `scip-system`             | SCIP via a system installation (set `SCIPOPTDIR`)            | no      |

## Workspace layout

| Crate            | Role                                                      |
| ---------------- | --------------------------------------------------------- |
| `oximo`          | Umbrella crate                                            |
| `oximo-expr`     | Arena-allocated expression tree                           |
| `oximo-core`     | `Model`, `Variable`, `Constraint`, `Objective`, `Set`     |
| `oximo-macros`   | `variable!`, `constraint!`, `objective!` and other macros |
| `oximo-autodiff` | Gradients, sparse Jacobians/Hessians via Enzyme           |
| `oximo-solver`   | `Solver` trait, `SolverResult`, `SolverOptions`           |
| `oximo-io`       | MPS, LP and NL readers and writers                        |
| `oximo-highs`    | HiGHS backend                                             |
| `oximo-gurobi`   | Gurobi 13  backend                                        |
| `oximo-mosek`    | MOSEK 11.2 backend                                        |
| `oximo-gams`     | GAMS writer and backend                                   |
| `oximo-baron`    | BARON writer and backend                                  |
| `oximo-clarabel` | Clarabel backend                                          |
| `oximo-pounce`   | POUNCE (pure-Rust IPOPT) backend                          |
| `oximo-scip`     | SCIP backend via russcip                                  |

## License

MIT OR Apache-2.0
