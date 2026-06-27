# oximo-core

Core modeling types for [oximo](https://github.com/oximo-rs/oximo): `Model`, `Variable`, `Set`, `Constraint`, `Objective`, `Parameter`, `IndexedVar`, `Domain`, and `ModelKind`.

Re-exports `oximo-expr` types (`Expr`, `ExprArena`, `ExprId`, `ExprNode`, `ParamId`, `VarId`) so downstream code does not need a separate `oximo-expr` import. End users typically depend on the umbrella `oximo` crate rather than this one directly.

## Usage

```toml
[dependencies]
oximo-core = "0.3"
```

Or via the umbrella crate (recommended for end users):

```toml
[dependencies]
oximo = "0.3"
```

## Quick example

```rust
use oximo_core::prelude::*;

let m = Model::new("transport");

// Scalar variables
variable!(m, x >= 0.0);
variable!(m, 0.0 <= y <= 10.0);

// Constraints (incl. a two-sided range -> band_lo + band_hi)
constraint!(m, c1, x + 2.0 * y <= 14.0);
constraint!(m, c2, 3.0 * x - y >= 0.0);
constraint!(m, band, 1.0 <= x + y <= 12.0);

// Objective
objective!(m, Max, 3.0 * x + 4.0 * y);

println!("kind = {:?}", m.kind()); // LP
```

## Modeling API

The modeling surface is a set of macros: `variable!`, `constraint!`, `objective!`,
`sum!`, `set!`, and `param!`. Each expands to the underlying typed model operations,
so there is no runtime cost and full compile-time type/borrow checking is preserved.

`Model` uses interior mutability (`RefCell`), so a macro can take `&m`, register
variables/constraints, and the `variable!`-introduced bindings (`x`, `y`, ...) are
locals you can use immediately.

```rust,ignore
let m = Model::new("my_model");
variable!(m, x >= 0.0);        // binds a local `x: Expr<'_>`
constraint!(m, cap, x <= 5.0); // uses x while holding &m
```

Names are unique per registry. Registering a duplicate variable or constraint name **panics**.

### Accessors

```rust,ignore
m.num_variables()      // usize
m.num_constraints()    // usize
m.variables()          // Ref<'_, Vec<Variable>>
m.constraints()        // Ref<'_, Vec<Constraint>>
m.arena()              // Ref<'_, ExprArena>
m.kind()               // ModelKind, cached, invalidated on change
m.try_objective()      // Result<Objective, Error>
m.variable_id("x")     // Option<VarId>
m.constraint_id("cap") // Option<ConstraintId>
```

### Fixing and unfixing variables

```rust,ignore
m.fix_var(var_id, 3.0);         // lb = ub = 3.0
m.unfix_var(var_id, 0.0, 10.0); // restore bounds
```

## Variables

### Scalar variables

```rust,ignore
variable!(m, x);                        // free (-inf, +inf)
variable!(m, x >= 0.0);                 // lower bound only
variable!(m, 0.0 <= x <= 10.0);         // both bounds
variable!(m, b, Bin);                   // binary {0, 1}  (also Binary)
variable!(m, 0.0 <= n <= 100.0, Int);   // general integer  (also Integer)
variable!(m, s <= 10.0, SemiCont(2.0)); // semicontinuous: 0 or in [2, 10]
variable!(m, t <= 5.0, SemiInt(1.0));   // semi-integer: 0 or integer in [1, 5]

// Keyword args:
variable!(m, u, lb = 0.0, ub = 10.0);    // same as `0.0 <= u <= 10.0`
variable!(m, v, lb = 0.0, domain = Int); // keyword domain (or a positional `Int`)
variable!(m, w, initial = 3.0);          // warm start  (scalar only)
variable!(m, p, fix = 5.0);              // fixed to 5.0 (scalar only)
```

### Indexed variables

Creates one scalar variable per key in a `Set` (or range), named `base[key]`,
and binds an `IndexedVar`.

```rust,ignore
let i = Set::range(0..5);
variable!(m, 0.0 <= x[k in i] <= 10.0);     // uniform bounds
variable!(m, y[k in i] >= 0.0, Int);        // integer family
variable!(m, z[a in rows, b in cols], Bin); // multi-index (Cartesian product)

// Access by key (panics on missing key):
let expr = x[2];  // single key (usize / "name" / (a, b))
let e2 = z[a, b]; // inside the macros: multi-index sugar == z[(&a, &b)]

// Bounds may reference the index -> lowered to per-key bounds:
variable!(m, lower[k] <= w[k in i] <= upper[k]);

// Filtered family: keep only matching keys (no trivial elements built).
variable!(m, d[(i, j) in rc if i == j] >= 0.0);
```

## Domain

| Variant                                | Description                   |
|----------------------------------------|-------------------------------|
| `Domain::Real`                         | Any real number (default)     |
| `Domain::Integer`                      | Any integer                   |
| `Domain::Binary`                       | 0 or 1                        |
| `Domain::SemiContinuous { threshold }` | 0 or any value >= threshold   |
| `Domain::SemiInteger { threshold }`    | 0 or any integer >= threshold |

## Sets

`Set` is an ordered finite index set. Three variants:

```rust,ignore
let i = Set::range(0..5);              // Range: i64 keys 0..5
let j = Set::strings(["a", "b", "c"]); // Strings
let k = Set::product(&i, &j);          // Tuples: (0,"a"), (0,"b"), ...
let k = &i * &j;                       // Same via Mul operator

// From sparse ints:
let s = Set::from_ints([0, 2, 4, 8]);

// Filter:
let evens = i.filter(|k| k.as_i64().unwrap() % 2 == 0);
```

## Constraints

`==`, `<=`, and `>=` are written directly, the macro intercepts the tokens, so
these are real constraint operators.

```rust,ignore
constraint!(m, name, lhs <= rhs);                  // named, also >= and ==
constraint!(m, lhs >= rhs);                        // anonymous (auto-named _c0, _c1, ...)
constraint!(m, band, 1.0 <= e <= 3.0);             // two-sided range -> band_lo + band_hi
constraint!(m, name = format!("c_{k}"), e == rhs); // computed run-time name
```

### Indexed family over a set

```rust,ignore
// One constraint per key, auto-named supply[seattle], ...
constraint!(m, supply[p in plants], sum!(x[p, q] for q in markets) <= cap[p]);

// Multi-index family (multi-index access sugar: x[i, j]).
constraint!(m, flow[i in 0..n, j in 0..m], x[i, j] >= 0.0);

// Filtered family: only keys passing the guard.
constraint!(m, diag[(i, j) in rc if i == j], x[i, j] <= 1.0);
```

### Summation

`sum!(body for k in domain)` reads as `sum_{k in domain} body`. Nest with extra
clauses and filter with a trailing `if`:

```rust,ignore
constraint!(m, cap, sum!(weights[i] * x[i] for i in items) <= capacity);
objective!(m, Min, sum!(c[i, j] * x[i, j] for i in rows, j in cols));
let evens = sum!(x[i] for i in items if i % 2 == 0); // filtered
```

## Objectives

```rust,ignore
objective!(m, Min, cost_expr);
objective!(m, Max, revenue_expr);
```

## Parameters

```rust,ignore
param!(m, rate = 0.05);     // binds a re-bindable `rate: Expr<'_>`
rate.set_param_value(0.07); // change between solves without rebuilding
```

## Model kind

Inferred automatically from variables and expressions, cached and invalidated on change:

| Kind    | Conditions                                          |
|---------|-----------------------------------------------------|
| `LP`    | All continuous, all linear                          |
| `MILP`  | Any integer/binary, all linear                      |
| `QP`    | All continuous, `Mul` with >=2 non-const children   |
| `MIQP`  | Any integer/binary + quadratic                      |
| `NLP`   | All continuous, `Pow`/`Sin`/`Cos`/`Exp`/`Log`/`Abs` |
| `MINLP` | Any integer/binary + nonlinear                      |

## License

MIT OR Apache-2.0
