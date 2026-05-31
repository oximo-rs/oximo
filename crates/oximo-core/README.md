# oximo-core

Core modeling types for [oximo](https://github.com/germanheim/oximo): `Model`, `Variable`, `Set`, `Constraint`, `Objective`, `Parameter`, `IndexedVar`, `Domain`, and `ModelKind`.

Re-exports `oximo-expr` types (`Expr`, `ExprArena`, `ExprId`, `ExprNode`, `ParamId`, `VarId`) so downstream code does not need a separate `oximo-expr` import.

End users typically depend on the umbrella `oximo` crate rather than this one directly.

## Usage

```toml
[dependencies]
oximo-core = "0.1"
```

Or via the umbrella crate (recommended for end users):

```toml
[dependencies]
oximo = "0.1"
```

## Quick example

```rust
use oximo_core::prelude::*;

let m = Model::new("transport");

// Scalar variables
let x = m.var("x").lb(0.0).build();
let y = m.var("y").lb(0.0).ub(10.0).build();

// Constraints
m.constraint("c1", (x + 2.0 * y).le(14.0));
m.constraint("c2", (3.0 * x - y).ge(0.0));

// Objective
m.maximize(3.0 * x + 4.0 * y);

println!("kind = {:?}", m.kind()); // LP
```

## Model

`Model` uses interior mutability (`RefCell`) so the builder API takes `&self`. This lets you hold a `&Model` reference, build variables and constraints, and immediately use the returned `Expr` handles, no `&mut` threading required.

```rust,ignore
let m = Model::new("my_model");
let x = m.var("x").lb(0.0).build(); // returns Expr<'_>
m.constraint("cap", x.le(5.0));     // uses x while holding &m
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

### Scalar variable builder

```rust,ignore
let x = m.var("x")
    .lb(0.0)              // lower bound (default: -inf)
    .ub(10.0)             // upper bound (default: +inf)
    .domain(Domain::Real) // explicit domain (default: Real)
    .build();             // returns Expr<'_>

// Shorthand domain setters:
let b = m.var("b").binary().build();          // Domain::Binary, bounds [0, 1]
let n = m.var("n").integer().lb(0.0).build(); // Domain::Integer
```

### Indexed variable builder

Creates one scalar variable per key in a `Set`, named `base[key]`.

```rust,ignore
let i = Set::range(0..5);
let x = m.indexed_var("x", &i)
    .lb(0.0)
    .integer()
    .build(); // IndexedVar<'_>

// Access by key (panics on missing key):
let expr = x[2]; // or x["name"], x[(a, b)]

// Per-key bounds:
let x = m.indexed_var("x", &i)
    .lb_by(|k: usize| lower_bounds[k])
    .ub_by(|k: usize| upper_bounds[k])
    .build();
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

### Single constraint

```rust,ignore
let c_id = m.constraint("name", expr.le(rhs)); // <=
let c_id = m.constraint("name", expr.ge(rhs)); // >=
let c_id = m.constraint("name", expr.eq(rhs)); // ==
```

### Bulk, rule over a set

```rust,ignore
m.add_constraints_over("supply", &plants, |p: String| {
    supply[&p].le(capacity[&p])
});

// Tuple sets, destructure inline:
m.add_constraints_over("flow", &(&plants * &markets), |(p, m): (String, String)| {
    x[(&p, &m)].le(capacity[&p])
});
```

## Objectives

```rust,ignore
m.minimize(cost_expr);
m.maximize(revenue_expr);
```

## Model kind

Inferred automatically from variables and expressions, cached and invalidated on change:

| Kind    | Conditions                                       |
|---------|--------------------------------------------------|
| `LP`    | All continuous, all linear                       |
| `MILP`  | Any integer/binary, all linear                   |
| `QP`    | All continuous, `Mul` with ≥2 non-const children |
| `MIQP`  | Any integer/binary + quadratic                   |
| `NLP`   | All continuous, `Pow`/`Sin`/`Cos`/`Exp`/`Log`    |
| `MINLP` | Any integer/binary + nonlinear                   |

## License

MIT OR Apache-2.0
