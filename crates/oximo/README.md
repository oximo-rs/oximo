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

oximo is a Rust algebraic modeling library for mathematical optimization. Build LP, MILP, QP/MIQP, NLP, and MINLP models with a clean builder API, then solve them with bundled or commercial solvers.

```rust,no_run
use oximo::prelude::*;
use oximo::solvers::Highs;

let m = Model::new("transport");
let x = m.var("x").lb(0.0).build();
let y = m.var("y").lb(0.0).ub(4.0).build();

m.constraint("c1", (x + 2.0 * y).le(14.0));
m.constraint("c2", (3.0 * x).ge(y));
m.constraint("c3", x.le(y + 2.0));
m.maximize(3.0 * x + 4.0 * y);

let result = Highs.solve(&m, &HighsOptions::default())?;
println!("obj = {:?}", result.objective);   // 34.0
println!("x   = {:?}", result.value_of(x)); // 6.0
println!("y   = {:?}", result.value_of(y)); // 4.0
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Building models

### Variables

```rust,ignore
let m = Model::new("my_model");

let x = m.var("x").lb(0.0).build();           // continuous, x >= 0
let y = m.var("y").lb(0.0).ub(10.0).build();  // continuous, 0 <= y <= 10
let z = m.var("z").build();                   // free (unbounded by default)
let b = m.var("b").binary().build();          // binary {0, 1}
let n = m.var("n").lb(0.0).integer().build(); // general integer
```

### Constraints and objectives

Expressions are built with standard Rust operators. Scalar multiplication, addition, subtraction, and nonlinear operations (see [Nonlinear Expressions](#nonlinear-expressions)) all work out of the box:

```rust,ignore
m.constraint("cap", (2.0 * x + 3.0 * y).le(100.0));
m.constraint("demand", x.ge(5.0));
m.constraint("balance", (x - y).eq(0.0));

m.minimize(3.0 * x + 5.0 * y);
// or
m.maximize(x + 2.0 * y);
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

`Model::indexed_var(name, &set)` registers one scalar per key with auto-named
entries like `x[seattle,nyc]`. Bounds apply uniformly by default, you can use
`lb_by` / `ub_by` for per-key bounds.

```rust,ignore
let m = Model::new("transport");
let x = m.indexed_var("x", &routes).lb(0.0).build();

// Scalar lookup: any type that converts to IndexKey works.
let e1 = x[("seattle", "nyc")];
let e2 = x[("san-diego", "chi")];

// Per-key upper bound (e.g. capacity per arc).
let _y = m.indexed_var("y", &routes)
    .lb(0.0)
    .ub_by(|(p, q): (String, String)| capacity_for(&p, &q))
    .build();
```

### Summing over sets

`sum_over(&set, |k| expr)` reads as `sum_{k in set} expr(k)`. The closure
receives the index as a typed value via `FromIndexKey`. Built-in impls cover
`i64`, `i32`, `usize`, `String`, raw `IndexKey`, and tuples up to arity 4.
State the shape in the closure-arg annotation.

```rust,ignore
// Single sum: sum_{i in items} weights[i] * x[i]
let total_weight = sum_over(&items, |i: usize| weights[i] * x[i]);
m.constraint("cap", total_weight.le(capacity));

// Double sum, flat: sum_{(p,q) in P*M} c[p,q] * x[p,q]
let total_cost = sum_over(&(&plants * &markets), |(p, q): (String, String)| {
    c[(&p, &q)] * x[(p, q)]
});

// Coefficient-weighted sum on paired Vecs: sum_{i} w_i * x_i
let weight_sum = dot(&xs, &weights);

// Freeform iterator -> use Iterator::sum.
let active = (0..n).filter(|&i| online[i]).map(|i| x[i]).sum::<Expr>();
```

### Rule-style constraints

`Model::add_constraints_over` is the constraint equivalent of `sum_over`, a
closure receives the index as a typed value and returns one constraint per
set element.

```rust,ignore
// Scalar set: one constraint per period.
let periods = Set::range(0..T);
m.add_constraints_over("setup", &periods, |t: usize| {
    (x[t] - capacity * s[t]).le(0.0)
});

// Tuple set: destructure inline. Inner `sum_over` builds the LHS expression.
m.add_constraints_over("supply", &plants, |p: String| {
    sum_over(&markets, |q: String| x[(&p, q)]).le(supply_of(&p))
});

// Want the raw key? Annotate as IndexKey (clones once per iteration).
m.add_constraints_over("c", &set, |k: IndexKey| x[&k].le(1.0));
```

### Nonlinear expressions

`Pow`, `Sin`, `Cos`, `Exp`, `Log`, `Abs`, and bilinear products are first-class. The
model's kind (`LP`/`MILP`/`QP`/`MIQP`/`NLP`/`MINLP`) is inferred from the
expressions.

```rust,ignore
// Rosenbrock NLP
m.minimize((1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

// Quadratic constraint
m.constraint("disk", (x.powi(2) + y.powi(2)).le(1.0));

// Transcendental utility (MINLP when any variable is integer/binary)
m.maximize(sum_over(&items, |i: usize| u[i] * (1.0 + w[i] * x[i]).log()));
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
    SolverStatus::Optimal => println!("optimal: {}", result.objective.unwrap()),
    SolverStatus::Infeasible => println!("infeasible"),
    SolverStatus::TimeLimit => println!("time limit, best = {:?}", result.objective),
    _ => {}
}

// Variable values
let x_val = result.value_of(x); // Option<f64>

// Constraint duals (LP only)
let dual = result.dual.get(&constraint_id);

// Reduced costs
let rc = result.reduced_costs.get(&x.id);
```

## Model export

With the `io` feature (default), you can export models to MPS, LP and NL format for inspection or use with external solvers.

## Requirements

- Gurobi feature: Gurobi, `GUROBI_HOME` set, valid license
- GAMS feature: GAMS on `PATH`, valid license
- BARON feature: BARON on `PATH`, valid license

## License

MIT OR Apache-2.0
