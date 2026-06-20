# oximo-solver

Solver trait, result types, status codes, and shared option building blocks for [oximo](https://github.com/oximo-rs/oximo).

This crate defines the contract that backend crates implement. End users interact with concrete backends (`Highs`, `Gurobi`, `Gams`) exposed by the umbrella `oximo` crate, they do not depend on this crate directly unless they are writing a new backend.

## Usage

```toml
[dependencies]
oximo-solver = "0.3"
oximo-core   = "0.3"
```

## `Solver` trait

```rust,ignore
pub trait Solver {
    type Options;

    fn name(&self) -> &str;
    fn supports(&self, kind: ModelKind) -> bool;
    fn solve(&mut self, model: &Model, opts: &Self::Options) -> Result<SolverResult, SolverError>;
}
```

Each backend defines its own `Options` type. Users get compile-time validation and LSP autocomplete on the options that actually apply to that backend.

## `SolverResult`

Populated by the backend on `Optimal` or `Feasible`. Sparse maps (`FxHashMap`) mean backends that don't return duals or reduced costs simply leave those fields empty.

| Field           | Type                           | Description                                       |
|-----------------|--------------------------------|---------------------------------------------------|
| `status`        | `SolverStatus`                 | Outcome of the solve                              |
| `solutions`     | `Vec<SolutionPoint>`           | Primal points, best first (empty if no solution)  |
| `dual`          | `FxHashMap<ConstraintId, f64>` | Constraint duals at the best point                |
| `reduced_costs` | `FxHashMap<VarId, f64>`        | Variable reduced costs at the best point          |
| `solve_time`    | `Duration`                     | Wall time around the solve call                   |
| `iterations`    | `u64`                          | Simplex iteration count (if reported)             |
| `raw_log`       | `Option<String>`               | Solver stdout/stderr                              |

Each `SolutionPoint` holds the `primal` variable values (`FxHashMap<VarId, f64>`) and that point's `objective` (`Option<f64>`). Index `0` is the best/incumbent. Backends with solution pools return the extra points after it.

### Accessors

```rust,ignore
result.objective()           // Option<f64>, best solution's objective
result.value_of(expr)        // Option<f64>, primal value for a Var expr (best solution)
result.value(var_id)         // Option<f64>, primal value by VarId (best solution)
result.dual_of(c_id)         // Option<f64>, dual for a constraint
result.best()                // Option<&SolutionPoint>, same as .solution(0)
result.solution(i)           // Option<&SolutionPoint>, i-th pooled point
result.result_count()        // usize, number of returned points
result.status.has_solution() // true if Optimal or Feasible
result.report(&model)        // Display: model-aware summary (status, vars, duals)

// Indexed variables
result.value_of_idx(&flow, "nyc")                  // Option<f64>, value at a specific index
result.values_of(&flow)                            // Iterator<(&IndexKey, f64)>, all entries with a primal value
result.values_of(&flow).filter(|(_, v)| *v != 0.0) // nonzero only (sparse solutions)
```

## `SolverStatus`

| Variant         | Meaning                                                          |
|-----------------|------------------------------------------------------------------|
| `Optimal`       | Proven optimal                                                   |
| `Feasible`      | Feasible but not proven optimal (e.g. time limit with incumbent) |
| `Infeasible`    | No feasible solution exists                                      |
| `Unbounded`     | Objective is unbounded                                           |
| `TimeLimit`     | Time limit reached with no feasible solution                     |
| `NumericError`  | Solver reported numerical difficulties                           |
| `NotSolved`     | Default, solve not yet called                                    |
| `Other(String)` | Backend-specific status not covered above                        |

## `SolverError`

| Variant                      | Cause                                                     |
|------------------------------|-----------------------------------------------------------|
| `UnsupportedKind(ModelKind)` | Backend does not support this model kind                  |
| `NoObjective`                | Model has no objective set                                |
| `Nonlinear`                  | Backend cannot handle nonlinear expressions               |
| `Backend(String)`            | Backend-reported error (e.g. license failure, bad option) |
| `Core(Error)`                | Error from `oximo-core`                                   |

## Universal options

All backend options structs embed `UniversalOptions` and implement `HasUniversal`, which enables the `UniversalOptionsExt` blanket impl:

```rust,ignore
use oximo_solver::UniversalOptionsExt;
use std::time::Duration;

let opts = MyBackendOptions::default()
    .time_limit(Duration::from_secs(120))
    .threads(4)
    .verbose(true);
```

| Method                  | Field        | Type                |
|-------------------------|--------------|---------------------|
| `.time_limit(Duration)` | `time_limit` | `Option<Duration>`  |
| `.threads(u32)`         | `threads`    | `Option<u32>`       |
| `.verbose(bool)`        | `verbose`    | `Option<bool>`      |

### Implementing `HasUniversal` for a new backend

```rust
use oximo_solver::{HasUniversal, UniversalOptions};

#[derive(Default)]
pub struct MyOptions {
    universal: UniversalOptions,
    // backend-specific fields ...
}

impl HasUniversal for MyOptions {
    fn universal(&self) -> &UniversalOptions { &self.universal }
    fn universal_mut(&mut self) -> &mut UniversalOptions { &mut self.universal }
}
```

## Writing a new backend

Mirror the layout of an existing backend crate (`oximo-highs`, `oximo-gurobi`, `oximo-gams`):

1. `lib.rs`: public struct + `impl Solver`. `supports()` declares which `ModelKind`s are handled. `solve()` delegates to `translate::solve`.
2. `options.rs`: converts `MyOptions` into the backend's native option calls.
3. `translate.rs`: `Model` -> backend conversion and result extraction.

Add an optional dep + feature in `oximo/Cargo.toml` and re-export the type under `oximo::solvers`.

## License

MIT OR Apache-2.0
