<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/oximo-rs/oximo/main/media/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oximo-rs/oximo/main/media/logo-dark.svg">
  <img alt="oximo logo" src="https://raw.githubusercontent.com/oximo-rs/oximo/main/media/logo-dark.svg">
</picture>

<a href="https://github.com/oximo-rs/oximo/tree/main/crates/oximo/examples">
    <img src="https://img.shields.io/badge/oximo-examples-orange" alt = "Examples">
</a>
<a href="https://crates.io/crates/oximo">
    <img src="https://img.shields.io/crates/v/oximo?logo=rust&color=E05D44" alt="crates version" />
</a>
<a href="https://github.com/oximo-rs/oximo/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/oximo-rs/oximo/ci.yml?branch=main&label=oximo%20CI&logo=github" alt="CI" />
</a>

oximo is a Rust algebraic modeling library for mathematical optimization. Build LP, MILP, QP/MIQP, NLP, and MINLP models with a concise macro API, then solve them with bundled or commercial solvers.

```rust,no_run
use oximo::prelude::*;
use oximo::solvers::Highs;

let m = Model::new("transport");
variable!(m, x >= 0.0);
variable!(m, 0.0 <= y <= 4.0);

constraint!(m, c1, x + 2.0 * y <= 14.0);
constraint!(m, c2, 3.0 * x >= y);
constraint!(m, c3, x <= y + 2.0);
objective!(m, Max, 3.0 * x + 4.0 * y);

let result = Highs.solve(&m, &HighsOptions::default())?;
println!("obj = {:?}", result.objective()); // 34.0
println!("x   = {:?}", result.value_of(x)); // 6.0
println!("y   = {:?}", result.value_of(y)); // 4.0
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Building models

### Variables

```rust,ignore
let m = Model::new("my_model");

variable!(m, x >= 0.0);                 // continuous, x >= 0
variable!(m, 0.0 <= y <= 10.0);         // continuous, 0 <= y <= 10
variable!(m, z);                        // free (unbounded by default)
variable!(m, b, Bin);                   // binary {0, 1}   (also Binary)
variable!(m, n >= 0.0, Int);            // general integer (also Integer)
variable!(m, s <= 10.0, SemiCont(2.0)); // semicontinuous: 0 or in [2, 10] (SemiInt too)
```

Bounds, domain, warm start, and fixing can also be given as keyword args after the
name:

```rust,ignore
variable!(m, x, lb = 0.0, ub = 1.0);       // same as `0.0 <= x <= 1.0`
variable!(m, n, lb = 0.0, domain = Int);   // keyword domain
variable!(m, w, lb = 0.0, ub = 10.0, Int); // mixed with positional domain token
variable!(m, p, lb = 0.0, initial = 3.0);  // warm start (scalar only)
variable!(m, q, fix = 5.0);                // fixed to 5.0 (scalar only)
```

### Constraints and objectives

Expressions are built with standard Rust operators. The macros let you write the
relational operators `==`, `<=`, `>=` directly. Scalar multiplication, addition,
subtraction, and nonlinear operations (see [Nonlinear Expressions](#nonlinear-expressions)) all work out of the box:

```rust,ignore
constraint!(m, cap, 2.0 * x + 3.0 * y <= 100.0);
constraint!(m, demand, x >= 5.0);
constraint!(m, balance, x - y == 0.0);
constraint!(m, band, 1.0 <= x + y <= 10.0); // two-sided range -> band_lo + band_hi

objective!(m, Min, 3.0 * x + 5.0 * y);
// or
objective!(m, Max, x + 2.0 * y);            // also Minimize/min, Maximize/max
```

### Index sets

`Set` is the modeling-layer container for an ordered, finite index set. Build
one over integers, strings, or arbitrary tuples. You can combine sets with the
Cartesian product operator `&a * &b`, and filter sparsely.

```rust,ignore
use oximo::prelude::*;

let items = Set::range(0..5);
let n_items = Set::range(0..weights.len());
let plants = Set::strings(["seattle", "san-diego"]);

// Cartesian product -> tuple keys, flattens automatically across nesting.
let routes = &plants * &Set::strings(["nyc", "chi", "topeka"]);
assert_eq!(routes.len(), 6);

// Sparse subsets via filter without self-loops
let arcs = (&plants * &plants).filter(|k| {
    let p = k.as_tuple().unwrap();
    p[0] != p[1]
});
```

### Indexed variables

`variable!(m, x[k in set])` registers one scalar per key with auto-named entries
like `x[seattle,nyc]`. Bounds apply uniformly by default.

```rust,ignore
let m = Model::new("transport");
variable!(m, x[r in routes] >= 0.0);

// Scalar lookup: any type that converts to IndexKey works.
let e1 = x[("seattle", "nyc")];
let e2 = x[("san-diego", "chi")];

// Per-key upper bound (capacity per arc) -> index-dependent bound.
variable!(m, 0.0 <= y[(p, q) in routes] <= capacity_for(&p, &q));
```

### Summing over sets

`sum!(body for k in set)` reads as `sum_{k in set} body`.

```rust,ignore
// Single sum: sum_{i in items} weights[i] * x[i]
constraint!(m, cap, sum!(weights[i] * x[i] for i in items) <= capacity);

// Double sum, flat: sum_{(p,q) in P*M} c[p,q] * x[p,q]
let total_cost = sum!(c[p, q] * x[p, q] for p in plants, q in markets);

// Filtered sum.
let active = sum!(x[i] for i in 0..n if online[i]);
```

### Rule-style constraints

The indexed form of `constraint!` emits one constraint per key, auto-named like
`supply[seattle]`. A trailing `if` filters the keys, and `name = expr` gives a
computed run-time name.

```rust,ignore
// Scalar set: one constraint per period.
let periods = Set::range(0..T);
constraint!(m, setup[t in periods], x[t] <= capacity * s[t]);

// Tuple set + inner sum builds the LHS expression (key types inferred).
constraint!(m, supply[p in plants], sum!(x[p, q] for q in markets) <= supply_of(&p));

// Filtered family: only the keys passing the guard are built.
constraint!(m, diag[(i, j) in arcs if i == j], x[i, j] <= 1.0);

// Computed run-time name.
constraint!(m, name = format!("bal_{p}"), inflow[p] - outflow[p] == 0.0);
```

### Nonlinear expressions

`Pow`, `Sin`, `Cos`, `Exp`, `Log`, `Abs`, and bilinear products are first-class. The
model's kind (`LP`/`MILP`/`QP`/`MIQP`/`NLP`/`MINLP`) is inferred from the
expressions.

```rust,ignore
// Rosenbrock NLP
objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

// Quadratic constraint
constraint!(m, disk, x.powi(2) + y.powi(2) <= 1.0);

// Transcendental utility (MINLP when any variable is integer/binary)
objective!(m, Max, sum!(u[i] * (1.0 + w[i] * x[i]).log() for i in items));
```

## Solving

All backends implement the `Solver` trait:

```rust,ignore
pub trait Solver {
    fn solve(&mut self, model: &Model, opts: &Self::Options) -> Result<SolverResult, SolverError>;
}
```

## Features

| Feature  | What it adds                                                          | Default |
|----------|-----------------------------------------------------------------------|---------|
| `highs`  | HiGHS - LP/MILP/QP solver (bundled, no install)                       | yes     |
| `io`     | MPS and LP file writers                                               | yes     |
| `gurobi` | Gurobi - LP/MILP/QP/MIQP/NLP/MINLP solver (requires licensed install) | no      |
| `gams`   | GAMS bridge - LP/MILP/QP/MIQP/NLP/MINLP depending on solver           | no      |
| `baron`  | BARON  - LP/MILP/QP/MIQP/NLP/MINLP solver (requires licensed install) | no      |

### HiGHS (default)

No install required, HiGHS is compiled from source via the `highs` crate.

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Highs;

let result = Highs.solve(&m, &HighsOptions::default()
    .time_limit(Duration::from_secs(60))
    .threads(4)
    .mip_gap(0.01)
    .method(HighsMethod::Ipm))?;
```

### Gurobi

Requires a licensed Gurobi install and `GUROBI_HOME` set. See [`crates/oximo-gurobi/README.md`](crates/oximo-gurobi/README.md).

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Gurobi;

let result = Gurobi.solve(&m, &GurobiOptions::default()
    .time_limit(Duration::from_secs(120))
    .mip_focus(1)
    .seed(101))?;
```

### GAMS

Requires GAMS on `PATH`. Supports solving models via GAMS solvers (CPLEX, BARON, etc.). See [`crates/oximo-gams/README.md`](crates/oximo-gams/README.md).

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Gams;

let result = Gams.solve(&m, &GamsOptions::default())?;
```

### BARON

Requires a licensed BARON install on `PATH`. Global solver for nonconvex LP/MILP/QP/MIQP/NLP/MINLP. See [`crates/oximo-baron/README.md`](crates/oximo-baron/README.md).

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Baron;

let result = Baron::new().solve(&m, &BaronOptions::default())?;
```

## Reading results

```rust,ignore
let result = Highs.solve(&m, &HighsOptions::default())?;

match result.status {
    SolverStatus::Optimal => println!("optimal: {}", result.objective().unwrap()),
    SolverStatus::Infeasible => println!("infeasible"),
    SolverStatus::TimeLimit => println!("time limit, best = {:?}", result.objective()),
    _ => {}
}

// Variable values (best solution)
let x_val = result.value_of(x); // Option<f64>

// Constraint duals
let dual = result.dual_of(constraint_id); // Option<f64>

// Reduced costs, keyed by VarId
let rc = result.reduced_costs.get(&x.var_id().unwrap());

// Solution pools (e.g. Gurobi, BARON with .num_sol(n)): all points, best first
for i in 0..result.result_count() {
    let point = result.solution(i).unwrap();
    println!("objective {:?}", point.objective);
}
```

## Model export

With the `io` feature (default), you can export models to MPS, LP and NL format for inspection or use with external solvers.

## Requirements

- Gurobi feature: Gurobi, `GUROBI_HOME` set, valid license
- GAMS feature: GAMS on `PATH`, valid license
- BARON feature: BARON on `PATH`, valid license

## License

MIT OR Apache-2.0
