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

oximo is a Rust algebraic modeling library for mathematical optimization.

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
constraint!(m, band, 1.0 <= x + y <= 10.0); // two-sided range -> one constraint

objective!(m, Min, 3.0 * x + 5.0 * y);
// or
objective!(m, Max, x + 2.0 * y); // also Minimize/min, Maximize/max
```

### Indexed variables

`variable!(m, x[k in set])` registers one scalar per key with auto-named entries
like `x[seattle,nyc]`. Bounds apply uniformly by default; a multi-index family
ranges over a Cartesian product.

```rust,ignore
let m = Model::new("transport");
variable!(m, x[r in routes] >= 0.0);        // one var per route
variable!(m, y[k in items] >= 0.0, Int);    // integer family
variable!(m, z[a in rows, b in cols], Bin); // multi-index (Cartesian product)

// Scalar lookup: any type that converts to IndexKey works.
let e1 = x[("seattle", "nyc")];
let e2 = z[a, b];

// Per-key bounds may reference the index
variable!(m, 0.0 <= w[(p, q) in routes] <= capacity_for(&p, &q));
variable!(m, v[k in items], lb = 0.0, ub = cap[k]);

// Filtered family: keep only matching keys (no trivial elements built).
variable!(m, d[(i, j) in rc if i == j] >= 0.0);
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

### Index sets

A `Set` is the modeling-layer container for an ordered, finite index set over
integers, strings, or tuples. Most domains need no explicit `Set`: an integer
range is already a domain (`x[i in 0..5]`, `sum!(.. for i in 0..n)`). Reach for
`Set` when keys are strings, tuples, sparse, or a subset reused across statements.

The `set!` macro binds a named set. A plain right side is normalized to an owned
set, a `pat in domain[ if cond]` comprehension builds (and optionally filters)
one.

```rust,ignore
use oximo::prelude::*;

let plants = Set::strings(["seattle", "san-diego"]);

set!(items = 0..5);             // range normalized to Set<usize>
set!(routes = plants * plants); // Cartesian product

// Comprehension: product domain + by-value `if`. These two are equivalent.
set!(arcs = (p, q) in &plants * &plants if p != q); // single tuple pattern
set!(arcs = i in plants, j in plants if i != j);    // multi-bind product

// The typed filter is also a Set method (the receiver pins the key type):
let diag = (&plants * &plants).filter_typed(|(p, q)| p == q);

// Sparse/string leaf sets:
let sparse = Set::from_ints([0, 2, 4, 8]);
```

### Nonlinear expressions

`Pow`, `Sin`, `Cos`, `Exp`, `Log`, `Abs`, and bilinear products are first-class. The
model's kind is inferred from the expressions.

```rust,ignore
// Rosenbrock NLP
objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

// Quadratic constraint (model kind: QCP)
constraint!(m, disk, x.powi(2) + y.powi(2) <= 1.0);

// Second-order cone ||(x, y)|| <= t (model kind: SOCP)
soc_constraint!(m, cone, [x, y] <= t);

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

| Feature         | What it adds                                                 | Default |
|-----------------|--------------------------------------------------------------|---------|
| `highs`         | HiGHS - LP/MILP/QP solver (bundled, no install)              | yes     |
| `io`            | MPS and LP file writers                                      | yes     |
| `gurobi`        | Gurobi solver (requires licensed install)                    | no      |
| `gams`          | GAMS bridge - solve type depends on the selected sub-solver  | no      |
| `baron`         | BARON - global LP...MINLP solver (requires licensed install) | no      |
| `clarabel`      | Clarabel - LP/QP/SOCP conic solver (pure Rust, no install)   | no      |
| `clarabel-faer` | Clarabel with the faer sparse linear-algebra backend         | no      |
| `pounce`        | POUNCE - pure-Rust IPOPT for LP/QP/QCP/NLP (no install)      | no      |
| `pounce-enzyme` | POUNCE with exact Enzyme derivatives (nightly)               | no      |

## Reading results

For a quick, model-aware summary, print `result.report(&m)`. For programmatic access:

```rust,ignore
let result = Highs.solve(&m, &HighsOptions::default())?;

match result.termination {
    TerminationStatus::Optimal => {
        // `objective()` is `Option` (a model may have no objective), so print it
        // only when present.
        if let Some(obj) = result.objective() {
            println!("optimal: {obj}");
        }
    }
    TerminationStatus::Infeasible => println!("infeasible"),
    TerminationStatus::TimeLimit if result.has_solution() => {
        println!("time limit, best = {:?}", result.objective());
    }
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

## Workspace layout

| Crate            | Role                                                      |
|------------------|-----------------------------------------------------------|
| `oximo`          | Umbrella crate                                            |
| `oximo-expr`     | Arena-allocated expression tree                           |
| `oximo-core`     | `Model`, `Variable`, `Constraint`, `Objective`, `Set`     |
| `oximo-macros`   | `variable!`, `constraint!`, `objective!` and other macros |
| `oximo-autodiff` | Gradients, sparse Jacobians/Hessians via Enzyme           |
| `oximo-solver`   | `Solver` trait, `SolverResult`, `SolverOptions`           |
| `oximo-io`       | MPS, LP and NL writers                                    |
| `oximo-highs`    | HiGHS backend                                             |
| `oximo-gurobi`   | Gurobi backend                                            |
| `oximo-gams`     | GAMS writer and backend                                   |
| `oximo-baron`    | BARON writer and backend                                  |
| `oximo-clarabel` | Clarabel backend                                          |
| `oximo-pounce`   | POUNCE (pure-Rust IPOPT) backend                          |

## Requirements

- Gurobi feature: Gurobi, `GUROBI_HOME` set, valid license
- GAMS feature: GAMS on `PATH`, valid license
- BARON feature: BARON on `PATH`, valid license

## License

MIT OR Apache-2.0
