# oximo-core

Core modeling types for [oximo](https://github.com/oximo-rs/oximo): `Model`, `Variable`, `Set`, `Constraint`, `Objective`, `Parameter`, `IndexedVar`, `Domain`, and `ModelKind`.

Re-exports `oximo-expr` types (`Expr`, `ExprArena`, `ExprId`, `ExprNode`, `UnaryOp`, `Children`, `ParamId`, `VarId`) so downstream code does not need a separate `oximo-expr` import. End users typically depend on the umbrella `oximo` crate rather than this one directly.

## Usage

```toml
[dependencies]
oximo-core = "0.6.0"
```

Or via the umbrella crate (recommended for end users):

```toml
[dependencies]
oximo = "0.6.0"
```

## Quick example

```rust
use oximo_core::prelude::*;

let m = Model::new("transport");

// Scalar variables
variable!(m, x >= 0.0);
variable!(m, 0.0 <= y <= 10.0);

// Constraints (incl. a two-sided range, kept as one constraint)
constraint!(m, c1, x + 2.0 * y <= 14.0);
constraint!(m, c2, 3.0 * x - y >= 0.0);
constraint!(m, band, 1.0 <= x + y <= 12.0);

// Objective (or `objective!(m, Feasibility)` for a pure feasibility problem)
objective!(m, Max, 3.0 * x + 4.0 * y);

println!("kind = {:?}", m.kind()); // LP
```

## Modeling API

The modeling surface is a set of macros: `variable!`, `constraint!`, `objective!`,
`sum!`, `min!`, `max!`, `set!`, and `param!`. Each expands to the underlying typed model operations,
so there is no runtime cost and full compile-time type/borrow checking is preserved.

`Model` uses interior mutability, so a macro can take `&m`, register
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
m.num_variables()   // usize
m.num_constraints() // algebraic + SOC + SOS constraints
m.variables()       // Ref<'_, Vec<Variable>>
m.constraints()     // ModelConstraints<'_>
m.constraints().algebraic()          // typed algebraic registry
m.constraints().second_order_cones() // typed explicit-SOC registry
m.constraints().special_ordered_sets() // typed SOS registry
m.arena()          // ExprArenaSnapshot<'_>
m.kind()           // ModelKind, cached, invalidated on change
m.try_objective()  // Result<Objective, Error>
m.variable_id("x") // Option<VarId>
m.constraint_id("cap")      // Option<ConstraintId> (algebraic)
m.soc_constraint_id("cone") // Option<SocConstraintId>
m.sos_constraint_id("choice") // Option<SosConstraintId>
```

`ModelConstraints::iter()` visits every declared constraint. Its `algebraic()`
and `second_order_cones()` slices are convenient when both kinds are needed in
one scope. Typed views keep backend passes homogeneous while preserving the
same storage and typed IDs.
SOC-shaped quadratic constraints declared with `constraint!` remain algebraic
entries; only constraints declared with `soc_constraint!` or
`add_soc_constraint` appear in `second_order_cones()`.

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

Large indexed variable, parameter, algebraic, range, SOC, and SOS families are
prepared in parallel when the family is large enough, then registered serially
in set iteration order.

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
constraint!(m, band, 1.0 <= e <= 3.0);             // two-sided range -> one constraint (expr bounds -> band_lo/band_hi)
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

Capture the macro result to query rows without reconstructing their names:

```rust
use oximo_core::prelude::*;

let m = Model::new("constraint_handles");
variable!(m, x[i in 0..4] >= 0.0);
let capacity: ConstraintId = constraint!(m, capacity, x[0] <= 10.0);
let cover: IndexedConstraint<usize> =
    constraint!(m, cover[i in 0..4 if i % 2 == 0], x[i] >= 1.0);
assert!(cover.get(1).is_none()); // filtered out
for (i, cid) in cover.iter() {
    assert_eq!(cover.get(i), Some(cid));
}

let bands: IndexedRangeConstraint<usize> =
    constraint!(m, bands[i in 0..4], 1.0 <= x[i] <= 4.0);
let RangeConstraintIds::Interval(cid) = bands.get(0).unwrap() else {
    panic!("constant bounds and a linear body produce an interval row");
};
```

`get(key)` returns copied IDs; `iter()` yields typed `(K, ID)` pairs in domain
order for dense, sparse, string, tuple, and filtered domains.

Two-sided ranges return `RangeConstraintIds::Interval(cid)` or
`RangeConstraintIds::Split { lower, upper }`.

### Summation

`sum!(body for k in domain)` reads as `sum_{k in domain} body`. Nest with extra
clauses and filter with a trailing `if`:

```rust,ignore
constraint!(m, cap, sum!(weights[i] * x[i] for i in items) <= capacity);
objective!(m, Min, sum!(c[i, j] * x[i, j] for i in rows, j in cols));
let evens = sum!(x[i] for i in items if i % 2 == 0); // filtered
```

### Special ordered sets

SOS1 and SOS2 constraints apply ordering weights to bare variables. SOS1 allows
at most one nonzero member. SOS2 allows at most two adjacent nonzero members.
Weights must be finite and unique within each set.

```rust,ignore
variable!(m, 0.0 <= choice_a <= 1.0);
variable!(m, 0.0 <= choice_b <= 1.0);
variable!(m, 0.0 <= choice_c <= 1.0);

let one_choice = sos_constraint!(m, one_choice, SOS1, [choice_a, choice_b, choice_c]);
sos_constraint!(m, adjacent_choice, SOS2, [choice_a, choice_b, choice_c]);
sos_constraint!(m, weighted, SOS2, [
    (choice_a, 10.0), (choice_b, 20.0), (choice_c, 30.0),
]);
```

The indexed form creates one set for every key in the binder. The binder is
required because the macro cannot infer an index domain from an `IndexedVar`:
`choice[i in 0..n]` produces `choice[0]`, ..., `choice[n - 1]`. To create one
SOS over a dynamically assembled collection, use
`m.add_sos_constraint_auto_weights("choice", SosType::Sos1, members)`.

The single-constraint form returns a model-bound `SosConstraintHandle`.
Backends without native SOS support return an error.
Reformulate one constraint through its handle, or every active SOS in a model explicitly:

```rust,ignore
let transformed = one_choice.to_reformulated_model(SosReformulationOptions::default())?;
solver.solve(&transformed, &options)?;

let transformed = m.to_reformulated_sos_model(
    SosReformulationOptions::default().with_fallback_big_m(1.0e6),
)?;

// Reformulate every active SOS in place when the native form is no longer needed.
let artifacts = m.reformulate_sos(
    SosReformulationOptions::default().with_fallback_big_m(1.0e6),
)?;
solver.solve(&m, &options)?;
```

The `to_reformulated_*` methods produce an independent model.
`reformulate*` methods modify the source model.

All forms preserve original IDs, append binary variables and linear rows,
and retain the source SOS entries as inactive provenance. Finite variable bounds
are used as member-specific Big-M values. An infinite bound is an error unless
the caller explicitly supplies a positive finite fallback. A fallback that is too small can truncate
the feasible region.

Prefer solving the original model with a backend that supports native SOS.
Reformulate when the selected backend lacks that capability or when you intentionally
need a MILP representation. In-place reformulation first validates every active set,
then appends all generated artifacts without copying the source model.

Because the generated rows embed the current member bounds, those bounds cannot
be changed on a reformulated model. Change bounds before reformulating, or
produce a fresh reformulated copy after changing the source model.

### Second-order cone constraints

`soc_constraint!` registers `||terms||_2 <= bound`. Every term and the bound
must be affine. The model classifies as `SOCP`/`MISOCP`.

```rust,ignore
soc_constraint!(m, cone, [x, y] <= t);                       // named -> SocConstraintId
soc_constraint!(m, [x - y, 2.0 * y] <= t + 1.0);             // anonymous (auto-named _soc0, ...)
soc_constraint!(m, name = format!("c_{k}"), [x] <= t);       // computed run-time name
soc_constraint!(m, risk[i in assets], [s[i] * w[i]] <= cap); // family: risk[key] per key
```

The method form `m.add_soc_constraint("cone", [x, y], t)` is equivalent.

## Objectives

```rust,ignore
objective!(m, Min, cost_expr);
objective!(m, Max, revenue_expr);
```

Nonlinear expressions use `Expr` methods such as `sqrt`, `exp2`, `log1p`,
`atan2`, `min`, and `max`. `min!`/`max!` provide the same indexed-domain
syntax as `sum!` and preserve deterministic child order.

## Parameters

```rust,ignore
param!(m, rate = 0.05);     // binds a re-bindable `rate: Expr<'_>`
rate.set_param_value(0.07); // change between solves without rebuilding
```

### Indexed parameters

Mirror indexed variables: one re-bindable scalar parameter per key, bound as an
`IndexedParam`. The right-hand side is evaluated per key and may reference the
index.

```rust,ignore
let items = Set::range(0..3);
param!(m, cost[i in items] = base_cost[i]); // one parameter per key
param!(m, w[(i, j) in rc] = weight(i, j));  // multi-index
param!(m, c[p in plants] = price[p]);       // string-keyed (sparse)

let unit = cost[1];             // index for a param `Expr`
cost[1].set_param_value(9.0);   // re-bind one entry via its handle
m.set_param_idx(&cost, 1, 9.0); // ...or by key on the model
m.param_value_idx(&cost, 1);    // -> Some(9.0)
```

## Model kind

Inferred automatically from variables and expressions, cached and invalidated
on change. The decision ladder runs top-down. Any integer/binary variable
picks the `MI*` variant of the row that matches:

| Kind (continuous/integer) | Conditions                                                       |
|---------------------------|------------------------------------------------------------------|
| `NLP`/`MINLP`             | Any nonlinear expression (degree > 2, transcendentals, division) |
| `QCP`/`MIQCP`             | Any quadratic constraint not recognized as a second-order cone   |
| `SOCP`/`MISOCP`           | Second-order cones present (explicit or detected)                |
| `QP`/`MIQP`               | Quadratic objective, linear constraints                          |
| `LP`/`MILP`               | Everything linear                                                |

## License

MIT OR Apache-2.0
