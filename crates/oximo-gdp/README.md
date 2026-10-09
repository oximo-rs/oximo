# oximo-gdp

Generalized disjunctive programming (GDP) modeling and solver-independent
Big-M reformulation, applied explicitly or at solve time. Enable the optional
`gdp` feature on `oximo` and import `oximo::prelude::*`.

## Quick start

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

let report = model.reformulate_gdp(BigM::default())?;
assert_eq!(report.rows.len(), 4);
assert!(!model.has_unreformulated_gdp());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Read the [GDP guide](https://oximo.dev/dev/gdp/) for an in-depth explanation about GDP.

Reformulate before exporting. For solve-time selection, use
`solver.with_gdp(BigM::default()).solve(&model, &solver_options)`.
This reformulates in place and preserves the original handles.
Choose a solver that supports the resulting algebraic model.

Read a selection with `result.boolean_value_of(large)?`, which returns
`Option<bool>`. Values within `1e-5` of zero or one become `false` or `true`;
fractional values outside that tolerance return `BooleanValueError`. A missing
solution or selection value returns `None`. The same query works on an individual
`SolutionPoint`.

## Modeling and reformulation

- Use `boolean_variable!` and `disjunct_constraint!` to attach conditions to
  individual rows. `.binary()` gives the ordinary binary expression for
  objectives and numeric result queries, including LP relaxations.
- Use `logical_constraint!` for Boolean logic and cardinality rules. Disjunctions
  support inclusive selection with `AtLeastOne` and nesting with `parent = ...`.
- `reformulate_gdp` validates the complete plan before changing the model.
  `to_reformulated_gdp_model` transforms an independent copy and preserves the
  source. Reports retain source-to-generated mappings and the M values used.
  Report clones share storage; edits detach their data. Use `report.into_owned()`
  to move artifact vectors out of a report.

Big-M is currently the only reformulation method. It estimates M values from
global bounds and unconditional affine constraints. Nested rows also use affine
ancestor constraints to split M into child and ancestor relaxation terms.

Nonlinear expressions must be defined over the global domain, including when a
branch is inactive. Conditional cones, native indicators, SOS constraints,
variables, and objectives are currently not supported. Reformulations lock bounds
and parameter values used by generated rows, so change dependent data on the source
before reformulating, or retain the source through the copy API.

## License

MIT OR Apache-2.0
