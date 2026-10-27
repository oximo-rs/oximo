# oximo-gdp

Generalized disjunctive programming (GDP) modeling and solver-independent
reformulation for oximo. Model Boolean decisions, conditional constraints, and
nested disjunctions, then transform them into an algebraic optimization model.

Enable the optional `gdp` feature on `oximo` and import `oximo::prelude::*`.
The examples below use the standalone crate's `oximo_gdp::prelude::*`.
See the [GDP guide](https://oximo.dev/gdp/) for an introduction to GDP and
the modeling syntax inside oximo.

## Reformulations available

Currently, `oximo-gdp` supports two reformulation methods:

| Method            | How it works                                             | What to consider                                                                                       |
| ----------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `BigM::default()` | Relaxes inactive rows using M values.                    | Compact formulation. Needs finite M estimates or justified supplied values.                            |
| `Hull::default()` | Introduces variable copies scaled by branch activations. | Can strengthen the relaxation, with more variables and rows. Copied coordinates require finite bounds. |

Choose a solver that supports the resulting algebraic model: an MILP solver for
affine rows, or an MINLP solver for nonlinear rows.

`GdpReformulationOptions::default()` selects Big-M. Pass either method directly
or use `GdpReformulationOptions` to configure individual disjunctions.

### Big-M

Big-M adds a relaxation term to each conditional row. For `x <= b`, it emits a
row of the form `x <= b + M * (1 - indicator)`. When `indicator = 1`, the source row
applies; when `indicator = 0`, a sufficiently large M relaxes it over the global domain.

Automatic estimates use variable bounds and unconditional affine constraints.
Hierarchical tightening is enabled by default, so nested rows can use tighter
bounds derived from their ancestors while retaining relaxation terms for inactive
ancestors. Equalities and ranges have separate lower and upper M values.

The main options are:

| Option                                                | Use                                                                                       |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `options.with_big_m(row, BigMValues::upper(m))`       | Supply an explicit M for one row side. Also available: `lower` and `symmetric`.           |
| `options.with_range_big_m(range, values)`             | Supply values for a range, including split rows.                                          |
| `BigM::default().with_fallback_big_m(m)`              | Supply a fallback when an automatic estimate is unavailable.                              |
| `BigM::default().with_hierarchical_tightening(false)` | Use the single-indicator formulation for nested rows.                                     |
| `options.with_bound_tightening(false)`                | Skip affine propagation and use declared bounds. This also disables ancestor propagation. |

Explicit M values take precedence over automatic estimates; fallbacks are used
only when estimation fails. Supplied values must be finite, nonnegative, and
large enough to preserve every feasible inactive state. There is no large default M.

Nonlinear expressions must be defined over their global domain, including when
the branch is inactive. An explicit M does not repair an unsafe function domain.

### Hull

Hull disaggregates variables into continuous copies for each alternative and
links them back to the source coordinates. Original integer and Boolean domains
remain in force. Nested disjunctions are processed inside-out, preserving their
structure and the links between child and parent copies.

Before choosing Hull, check the following requirements:

- Coordinates that need copies must have finite lower and upper bounds. Default
  affine propagation can derive bounds from unconditional constraints.
- Hull supports `ExactlyOne` by default. `AtLeastOne` requires explicit pattern
  enumeration, described below. Standalone conditional blocks use an on/off Hull.
- Nonlinear rows may need a reference coordinate that makes the generated
  perspectives safe. References also shift affine copies and rows.

#### Nonlinear rows and references

Nonlinear Hull uses regularized perspectives with epsilon `1e-6`. Set it with
`Hull::default().with_epsilon(value)`, where the value is finite and strictly
between zero and one. The formulation recovers the source behavior at binary
activations. Changing epsilon affects the nonlinear relaxation.

Reference coordinates default to zero. Restricted-domain functions such as
`ln(x)` may require an explicit reference:

```rust
use oximo_gdp::prelude::*;

let model = Model::new("logarithmic Hull");
variable!(model, 1.0 <= flow <= 4.0);
boolean_variable!(model, active);
disjunct_constraint!(model, active, flow.ln() <= 3.0_f64.ln());

let options = GdpReformulationOptions::new(Hull::default())
    .with_hull_reference_coordinate(flow, 1.0);
model.reformulate_gdp(options)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

A copy represents `activation * (source - reference)`. Pass a bare variable
handle to the reference setter. For indexed variables, use
`with_hull_reference_coordinates(x.iter().map(|(_, v)| (v, reference)))`, or supply a different reference for each entry.

A reference need not satisfy the branch constraints. It must make both reference
evaluations and the entire generated variable box safe. Being inside the source
domain is insufficient: for `ln(x)` with `x` in `[1, 4]`, reference `1` works,
while references `2` and `4` make the generated argument box cross zero.

#### Inclusive disjunctions

To preserve simultaneous selection in `AtLeastOne`, enable enumeration:

```rust
use oximo_gdp::prelude::*;

let hull = Hull::default().with_inclusive_policy(
    HullInclusivePolicy::Enumerate { max_patterns: 1023 },
);
```

Each inclusive group with `n` branches creates `2^n - 1` nonempty patterns.
Every selected branch's constraints apply. The pattern limit is per group;
nested expansion can still produce many copies and rows. Use the total size and
planning limits below when expansion is a concern.

## Using a reformulation

A disjunct groups constraints under a Boolean decision. A disjunction selects
exactly one branch by default:

```rust
use oximo_gdp::prelude::*;

let model = Model::new("unit selection");
variable!(model, 0.0 <= flow <= 10.0);
variable!(model, 0.0 <= cost <= 2.0);
let small = disjunct!(model, small, |d| {
    constraint!(d, capacity, flow <= 3.0);
    constraint!(d, fixed_cost, cost == 0.0);
});
let large = disjunct!(model, large, |d| {
    constraint!(d, operating_range, 5.0 <= flow <= 8.0);
    constraint!(d, fixed_cost, cost == 2.0);
});
disjunction!(model, unit, [small, large]);
objective!(model, Max, flow - cost);

let report = model.reformulate_gdp(Hull::default())?;
assert_eq!(report.rows.len(), 4);
assert!(!model.has_unreformulated_gdp());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `boolean_variable!` and `disjunct_constraint!` for individual conditional rows,
and `logical_constraint!` for Boolean logic and cardinality rules. Add
`parent = ...` to a disjunction for nesting and `AtLeastOne` for inclusive selection.
Declare variables and objectives on the global model. **Conditional variables,
objectives, cones, native indicators, and SOS constraints are not supported.**

### When to apply it

| API                                                                 | Behavior                                                                       |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `model.reformulate_gdp(method_or_options)`                          | Transform pending GDP components in place; original handles remain valid.      |
| `model.to_reformulated_gdp_model(method_or_options)`                | Transform an independent copy and retain the source for other data or methods. |
| `solver.with_gdp(method_or_options).solve(&model, &solver_options)` | Reformulate in place, solve, and project the result to source coordinates.     |

Reformulate before exporting or using an ordinary solver call. Validation errors
leave the model unchanged. Successful reformulation locks bounds and parameter
values used by generated rows, including dependencies of supporting global rows.
Retain an unreformulated source when comparing methods or changing dependent data.

Repeated calls transform only pending components. Changing the method on an
already transformed group requires a fresh source model. A backend failure after
successful solve-time reformulation leaves that transformation applied.

### Mixing methods

Choose a default method and override individual disjunctions:

```rust
# use oximo_gdp::prelude::*;
# let model = Model::new("mixed methods");
# variable!(model, 0.0 <= x <= 4.0);
# boolean_variable!(model, active);
# boolean_variable!(model, other);
# disjunct_constraint!(model, active, x <= 2.0);
# let choice = disjunction!(model, [active, other]);
let options = GdpReformulationOptions::new(BigM::default())
    .with_method(choice, Hull::default());
model.reformulate_gdp(options)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Each nested disjunction uses its own override or the default method. Standalone conditional rows use the default.
Within mixed subtrees, Big-M uses global bounds for safe M estimation.
Independent subtrees using only Big-M retain ancestor tightening.

### Planning and size limits

You can inspect a validated size summary before changing the model, and cap expansion:

```rust
use oximo_gdp::prelude::*;

let model = Model::new("planned Hull");
variable!(model, 0.0 <= x <= 4.0);
boolean_variable!(model, active);
disjunct_constraint!(model, active, x <= 2.0);

let options = GdpReformulationOptions::new(Hull::default())
    .with_size_limits(GdpSizeLimits {
        variables: Some(10_000),
        constraints: Some(50_000),
        expression_nodes: Some(100_000),
    })
    .with_planning_limits(GdpPlanningLimits {
        row_expansions: Some(200_000),
        expression_nodes: Some(150_000),
    });

let summary = model.plan_gdp(options.clone())?;
assert_eq!(summary.hull_copies, 2);

let report = model.reformulate_gdp(options)?;
assert_eq!(report.hull_variables.len(), summary.hull_copies);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`summary.generated` and `summary.total` give exact counts of variables, constraints,
and committed expression nodes. Totals include existing objects and native
constraints. `GdpSizeLimits` caps that final size across all groups.
`GdpPlanningLimits` caps row expansions and private arena nodes during planning,
including nested expansion and temporary rewrites. `None` leaves a resource
unlimited.

Planning performs full validation and builds temporary artifacts, so it costs approximately as much as reformulation.
It leaves the source and its dependency locks unchanged.

### Reading results and reports

Read Boolean selections with `result.boolean_value_of(handle)?`. The result is
`Option<bool>`: values within `1e-5` of zero or one become `false` or `true`, and
missing values return `None`. Non-finite values and fractional values outside the
tolerance return `BooleanValueError`. Use `result.value_of(handle.binary())?`
to inspect fractional selections.

The returned `GdpReformulationReport` maps source components to generated rows
and variables. Transformation history is available through
`model.gdp_reformulations()`.

Each report includes complete `generated_variables` and `generated_rows` index
ranges. The row range includes Hull copy bounds and aggregation rows even when
they have no individual source-row mapping. Boolean binary counterparts are source
variables and are excluded from the generated-variable range.

Print the report directly for a compact summary, or include model names, variable
bounds, and generated algebraic rows with `report.display(&model)`:

```rust,ignore
println!("{report}");
println!("{}", report.display(&model));
```

For programmatic variable inspection, use `report.generated_variables(&model)`.

Read a Hull activation with `result.hull_activation_value_of(handle)?`, obtaining
the handle from `report.hull_activation(copy_index)` or `report.hull_activations()`
in copy order.

Results returned by `GdpSolver::solve` are projected to source coordinates:
generated rows, copies, and multipliers are dropped, while Hull activation
variables and Boolean binaries remain queryable. Reformulating with
`model.reformulate_gdp` and calling the backend directly returns the
reformulated model's coordinates instead.

## Implementation details

### Hierarchy and coordinates

Hull follows the inside-out hierarchical construction of
[Perez and Grossmann (2024)](https://doi.org/10.1007/s11081-023-09831-x).
Child auxiliaries remain local to their enclosing branch. With zero references,
aggregation links satisfy `sum(child copies) = parent copy`. Original physical
coordinates retain their references. Descendant selectors and child auxiliaries
use reference zero when copied by inclusive ancestors, so inactive nested groups
recombine to zero. Reports record the effective reference for each copy's scope.

Nonlinear rows use the [Furman–Sawaya–Grossmann (2020)](https://link.springer.com/article/10.1007/s10589-020-00176-0)
regularized perspective. For activation `y`,
shifted copy `z`, and reference `r`, the denominator is
`d = epsilon + (1 - epsilon) * y`. The transformed upper row is
`d * f(z / d + r) - epsilon * f(r) * (1 - y) <= upper * y`.
For convex disjunct sets, the continuous relaxation approximates the closed
perspective as epsilon approaches zero.

In pure Big-M subtrees, ancestor constraints can split M into child and ancestor
relaxation terms. Reports retain global M values and their origins, with actual
split terms in `hierarchical_m`. Pure and mixed planning paths share row domain
analysis, coefficient normalization, M precedence, outward rounding, and precision
checks, while retaining their own bound contexts and ancestor treatment.

### Numerical validation

Hull validates reference evaluations and the independent box of generated bounds,
including points outside the scaled bound rows. `UnsafeHullDomain` distinguishes
reference failures from generated-box failures and reports the argument intervals
in operand order, source row, and disjunction or standalone Boolean scope.

Precision checks reject shifts that lose significant information in source bounds
or transformed rows. Own and ancestor selector terms are combined with finite row
bounds using compensated arithmetic, including affine parts of nonlinear rows.
Active-row recovery accounts for both summation and multiplication roundoff using
fused multiply-add residuals. Zero effective bounds use a roundoff allowance from
retained source coefficients that does not grow with reference shifts; nonzero
bounds retain relative checks. `HullRowPrecisionLoss` reports the absolute recovery
`error` and allowed `tolerance` for the failed coefficient or bound check.

## License

MIT OR Apache-2.0
