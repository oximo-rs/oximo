# oximo-gams

GAMS backend for [oximo](https://github.com/oximo-rs/oximo).

Writes an oximo `Model` to a temporary `.gms` file, invokes the GAMS executable via `std::process::Command`, and parses the solution from a PUT-generated text file. Supports `LP`, `MILP`, `QP`, `MIQP`, `NLP`, and `MINLP` model kinds.

The sub-solver is determined by the GAMS installation (default) or set explicitly via `GamsOptions::solver`. Any solver available in your GAMS distribution can be targeted, see [Sub-solver selection](#sub-solver-selection) below.

## Requirements

**A licensed GAMS installation must be on `PATH`.**

1. Download and install [GAMS](https://www.gams.com/download/) for your platform.
2. Obtain and activate a GAMS license.
3. Verify by running `gams` in a terminal.

If `gams` is not on `PATH`, pass an explicit path via `GamsOptions::gams_path` or `Gams::with_exec`:

```rust
use oximo_gams::Gams;
let solver = Gams::with_exec("/opt/gams53/gams");
```

## Usage

Enable the `gams` feature on the umbrella `oximo` crate:

```toml
[dependencies]
oximo = { version = "0.5", features = ["gams"] }
```

To use this crate directly:

```toml
[dependencies]
oximo-gams   = "0.4"
oximo-core   = "0.4"
oximo-solver = "0.4"
```

## Quick example

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Gams;

let m = Model::new("transport");
variable!(m, x >= 0.0);
variable!(m, 0.0 <= y <= 4.0);

constraint!(m, c1, x + 2.0 * y <= 14.0);
constraint!(m, c2, 3.0 * x - y >= 0.0);
constraint!(m, c3, x - y <= 2.0);
objective!(m, Max, 3.0 * x + 4.0 * y);

let result = Gams::new().solve(&m, &GamsOptions::default())?;
println!("termination = {:?}", result.termination);
println!("primal      = {:?}", result.primal_status);
println!("obj         = {:?}", result.objective());
```

Run the bundled example:

```sh
cargo run -p oximo --example reaction_path --features gams
```

## Options

`GamsOptions` is a plain struct with builder methods. Universal options come from `UniversalOptionsExt`.

### Universal options (via `UniversalOptionsExt`)

| Method                  | GAMS statement                                               | Default |
|-------------------------|--------------------------------------------------------------|---------|
| `.time_limit(Duration)` | `option ResLim = <seconds>;`                                 | none    |
| `.threads(usize)`       | `option threads = <n>;`                                      | none    |
| `.verbose(bool)`        | Forwards GAMS stdout/stderr to `raw_log` (suppresses `lo=0`) | none    |

### GAMS-specific options

| Method              | Description                                                    | Default              |
|---------------------|----------------------------------------------------------------|----------------------|
| `.mip_gap(f64)`     | `option OptCR = <gap>;` relative MIP optimality gap            | none                 |
| `.solver(config)`   | Sub-solver selection, see [below](#sub-solver-selection)       | none                 |
| `.gams_path(path)`  | Override path to the `gams` executable                         | `"gams"` from `PATH` |

```rust
use std::time::Duration;
use oximo_gams::GamsOptions;
use oximo_solver::UniversalOptionsExt;

let opts = GamsOptions::default()
    .time_limit(Duration::from_secs(300))
    .threads(4)
    .mip_gap(0.005)
    .verbose(true);
```

## Sub-solver selection

Pass a `GamsSolverConfig` to `.solver(...)` to select a GAMS sub-solver. This emits
`option {LP|MIP|NLP|MINLP|QCP|MIQCP} = <NAME>;` in the generated `.gms` file, scoped to
the solve type resolved from `Model::kind()` (`QP` -> `QCP`, `MIQP` -> `MIQCP`).

```rust
use oximo_gams::{GamsOptions, GamsSolver, GamsSolverConfig};

// Named selection, no typed option file
let opts = GamsOptions::default().solver(GamsSolver::Cplex);

// Or via Custom for any solver name not in the enum
let opts = GamsOptions::default().solver(GamsSolver::Custom("MOSEK".into()));
```

### Per-solver typed options

Wrap the sub-solver's options struct in `GamsSolverConfig` to have oximo write a
`<solver>.opt` file. GAMS picks it up via `model.optfile = 1`. Each struct exposes
one builder setter per documented GAMS option.

```rust
use oximo_gams::{GamsBaronOptions, GamsSolverConfig, GamsOptions};
use std::time::Duration;
use oximo_solver::UniversalOptionsExt;

let opts = GamsOptions::default()
    .time_limit(Duration::from_secs(120))
    .solver(GamsSolverConfig::Baron(
        GamsBaronOptions::default().eps_r(1e-4).threads(4),
    ));
```

For any option without a generated setter, push a verbatim line onto the public
`raw` field:

```rust
use oximo_gams::{GamsCplexOptions, GamsSolverConfig};

let cfg = GamsSolverConfig::Cplex(GamsCplexOptions {
    raw: vec!["solnpool out.gdx".into(), "solnpoolpop 2".into()],
    ..Default::default()
});
```

The `Gams<Name>Options` structs and `GamsSolverConfig` are generated at build
time from the checked-in [`option-snapshots/`](option-snapshots), one struct per
GAMS solver oximo supports from a modelling standpoint: ALPHAECP, ANTIGONE, BARON,
CBC, CONOPT, CONOPT3, COPT, CPLEX, DECIS, DICOPT, GUROBI, HIGHS, IPOPT, KNITRO,
LINDO, MINOS, MOSEK, ODHCPLEX, SBB, SCIP, SHOT, SNOPT, SOPLEX and XPRESS. Each
solver's options are documented at `https://www.gams.com/latest/docs/S_<NAME>.html`.

For any other solver, select it by name with
`GamsSolverConfig::Named(GamsSolver::Custom("NAME".into()))`, or write verbatim
option lines with `GamsSolverConfig::Raw(GamsSolver::Custom("NAME".into()), vec![..])`.

## Result

`SolverResult` fields, populated whenever a usable point is available (`primal_status` is `FeasiblePoint` or `OptimalPoint`):

- `solutions` - primal points (`Vec<SolutionPoint>`), best first. Each point holds its `primal` values keyed by `VarId` and its `objective`. The vector holds the incumbent, plus any alternative points read from a sub-solver solution pool (e.g. CPLEX `solnpool`). Access the best point via `result.objective()` / `result.value_of(var)` and the rest via `result.solution(i)`
- `dual` - constraint marginals (GAMS `.m`), keyed by `ConstraintId`, access via `result.dual_of(c)`
- `reduced_costs` - variable marginals, keyed by `VarId`
- `termination` - why the solve stopped, driven by the GAMS solve status (`solvestat`), with the model status (`modelstat`) resolving the outcome on normal completion (`Optimal`, `Infeasible`, `Unbounded`, `TimeLimit`, `IterationLimit`, ...)
- `primal_status` - whether a usable point came back (`NoSolution` / `FeasiblePoint` / `OptimalPoint`), `result.has_solution()` is the shortcut
- `best_bound` / `gap` - left unset (`None`), not exposed in the GAMS PUT solution file
- `solve_time` - wall time around the GAMS process invocation
- `iterations` - iterations used, read from the GAMS model attribute `oximo_m.iterusd`
- `solver_name` - `GAMS/<sub-solver>` when one is selected via `GamsOptions::solver` (e.g. `GAMS/CPLEX`, `GAMS/BARON`), otherwise just `GAMS`
- `raw_log` - GAMS stdout/stderr, populated when `verbose(true)` or when GAMS exits non-zero

`dual` and `reduced_costs` are filled with whatever marginals GAMS reports, for every model kind: globally valid duals for LP, locally valid duals at the returned point for QP/NLP, and duals of the integer-fixed problem for MIP/MIQP/MINLP (most GAMS solver links re-solve with integers fixed, e.g. CPLEX `solvefinal`). Entries the solver did not compute (`NA`/`UNDF`) are skipped, so a solver configured to skip the fixed re-solve simply leaves the maps empty.

## License

MIT OR Apache-2.0

> **Note:** GAMS itself is commercial software. A valid GAMS license is required at runtime. This crate is not affiliated with GAMS Development Corporation.
