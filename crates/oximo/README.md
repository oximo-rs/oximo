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
<a href="https://docs.rs/oximo">
    <img src="https://docs.rs/oximo/badge.svg" alt="oximo Docs.rs Build Status">
</a>

# oximo

oximo is a Rust algebraic modeling library for mathematical optimization. It
provides a compact modeling API for variables, indexed domains, algebraic
constraints, objectives, nonlinear expressions, and multiple solver backends.

- [Webdocs](https://oximo.dev)
- [API documentation](https://docs.rs/oximo)
- [Examples](https://github.com/oximo-rs/oximo/tree/main/crates/oximo/examples)
- [Repository](https://github.com/oximo-rs/oximo)

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

Expressions use standard Rust operators. Variable domains and bounds are
declared directly in the model:

```rust,ignore
variable!(m, x >= 0.0);       // continuous
variable!(m, y, Bin);         // binary
variable!(m, z >= 0.0, Int);  // integer

constraint!(m, capacity, 2.0 * x + y <= 10.0);
objective!(m, Max, 3.0 * x + 4.0 * y);
```

Quadratic, nonlinear, and second-order-cone expressions are also available:

```rust,ignore
objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));
constraint!(m, disk, x.powi(2) + y.powi(2) <= 1.0);
soc_constraint!(m, cone, [x, y] <= t);
```

## Problem types

oximo supports the following algebraic problem classes. Which classes can be
solved depends on the selected backend:

- Linear programming (LP)
- Quadratic programming and quadratically constrained programming (QP/QCP)
- Nonlinear programming (NLP)
- Mixed-integer linear programming (MILP)
- Mixed-integer quadratic and quadratically constrained programming (MIQP/MIQCP)
- Mixed-integer nonlinear programming (MINLP)
- Second-order cone programming (SOCP/MISOCP)

## Solver features

| Feature         | Backend / capability                                         | Default |
| --------------- | ------------------------------------------------------------ | ------- |
| `highs`         | HiGHS - LP/MILP/QP (bundled, requires a C/C++ compiler)      | no      |
| `io`            | NL, MPS, and LP readers and writers                          | yes     |
| `gurobi`        | Gurobi v13+ (requires licensed install)                      | no      |
| `mosek`         | MOSEK 11.2 - convex LP/MIP/QP/QCP/SOCP                       | no      |
| `gams`          | GAMS bridge - capability depends on the selected sub-solver  | no      |
| `baron`         | BARON - global non-convex solver (requires licensed install) | no      |
| `clarabel`      | Clarabel - LP/QP/SOCP (pure Rust, no install)                | no      |
| `clarabel-faer` | Clarabel with the faer sparse linear-algebra backend         | no      |
| `pounce`        | POUNCE - pure-Rust LP/QP/QCP/NLP backend                     | no      |
| `pounce-enzyme` | POUNCE with exact Enzyme derivatives (nightly)               | no      |

For example, use HiGHS for a bundled LP/MILP/QP solver:

```toml
[dependencies]
oximo = { version = "0.7", features = ["highs"] }
```

Licensed or external backends may require an installed solver, environment
variables, and a valid license. See the backend documentation and the
[webdocs](https://oximo.dev) for setup details.

## Results

`SolverResult` provides the termination status, objective value, variable values, duals, reduced costs, and (where supported) bounds, gaps, and solution pools.

## License

MIT OR Apache-2.0
