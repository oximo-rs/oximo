# oximo-baron

BARON (Branch-And-Reduce Optimization Navigator) backend for [oximo](https://github.com/oximo-rs/oximo).

Writes an oximo `Model` to a temporary `.bar` file, invokes the BARON executable via `std::process::Command`, and parses the solution from BARON's times (`tim.lst`) and results (`res.lst`) files. Supports `LP`, `MILP`, `QP`, `MIQP`, `NLP`, and `MINLP` model kinds.

[BARON](https://minlp.com/baron-solver) is a global optimizer for nonconvex MINLP.

## Requirements

1. Install BARON for your platform.
2. Obtain and activate a BARON license.
3. Verify by running `baron` in a terminal.

If `baron` is not on `PATH`, pass an explicit path via `BaronOptions::baron_path` or `Baron::with_exec`:

```rust
use oximo_baron::Baron;
let solver = Baron::with_exec("/opt/baron/baron");
```

## Usage

Enable the `baron` feature on the umbrella `oximo` crate:

```toml
[dependencies]
oximo = { version = "0.4", features = ["baron"] }
```

To use this crate directly:

```toml
[dependencies]
oximo-baron  = "0.4"
oximo-core   = "0.4"
oximo-solver = "0.4"
```

## Quick example

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Baron;

let m = Model::new("box");
variable!(m, 0.1 <= x <= 10.0);
variable!(m, 0.1 <= y <= 10.0);

constraint!(m, c, x + y <= 8.0);
objective!(m, Max, (1.0 + x).log() + 2.0 * y);

let result = Baron::new().solve(&m, &BaronOptions::default())?;
println!("termination = {:?}", result.termination);
println!("primal      = {:?}", result.primal_status);
println!("obj         = {:?}", result.objective());
```

Run the bundled example:

```sh
cargo run -p oximo --example baron_robot --features baron
```

## Options

`BaronOptions` is a plain struct with builder methods. Universal options come from `UniversalOptionsExt`. Every other knob maps to a BARON keyword written into the `OPTIONS{ ... }` block of the generated `.bar` file.

### Universal options (via `UniversalOptionsExt`)

| Method                  | BARON option         | Default |
|-------------------------|----------------------|---------|
| `.time_limit(Duration)` | `MaxTime: <seconds>` | none    |
| `.threads(u32)`         | `threads: <n>`       | none    |
| `.verbose(bool)`        | `PrLevel: 1/0`       | none    |

### BARON-specific options

Each builder method is the snake_case form of a BARON keyword. Highlights:

| Method                | BARON option | Description                                  |
|-----------------------|--------------|----------------------------------------------|
| `.eps_r(f64)`         | `EpsR`       | Relative termination tolerance               |
| `.eps_a(f64)`         | `EpsA`       | Absolute termination tolerance               |
| `.max_iter(i64)`      | `MaxIter`    | Node limit (`-1` = unlimited)                |
| `.num_sol(i64)`       | `NumSol`     | Number of feasible solutions to find         |
| `.first_feas(bool)`   | `FirstFeas`  | Stop at first feasible solution              |
| `.num_loc(i64)`       | `NumLoc`     | Multistart local searches (`-1` = automatic) |
| `.want_dual(bool)`    | `WantDual`   | Return duals at the best primal point        |
| `.lp_sol(i64)`        | `LPSol`      | LP subsolver selection                       |
| `.nlp_sol(i64)`       | `NLPSol`     | NLP subsolver selection                      |
| `.pr_level(i64)`      | `PrLevel`    | Print level (`0` silent)                     |
| `.baron_path(path)`   | -            | Override path to the `baron` executable      |

See the BARON manual for the full set (branching, range-reduction, and feasibility-tolerance options are all exposed). For any keyword without a dedicated builder, use the escape hatch:

```rust
// Note: The following options ARE exposed in the BaronOptions builder
use oximo_baron::BaronOptions;
let opts = BaronOptions::default().raw("MaxTime", "300").raw("LBTTDo", "1");
```

```rust
use std::time::Duration;
use oximo_baron::BaronOptions;
use oximo_solver::UniversalOptionsExt;

let opts = BaronOptions::default()
    .time_limit(Duration::from_secs(300))
    .threads(4)
    .eps_r(1e-4)
    .verbose(true);
```

`ResName`, `TimName`, `results`, and `times` are managed by the backend (it needs those files to parse the solution) and cannot be overridden. Every other documented BARON option has a dedicated builder. `MaxTime`/`threads`/`PrLevel` come from the universal options. When no `time_limit` is set, the backend emits `MaxTime: -1` (no limit) so BARON does not silently fall back to its own default.

## Limitations

BARON's `.bar` format has no trigonometric intrinsics, so a model whose objective or constraints contain `sin`/`cos` is rejected with a clear `SolverError::Backend`. A variable-exponent power `x^y` is rewritten to the equivalent `exp(y*log(x))`, and an absolute value `|x|` to `(x^2)^(1/2)`, both as suggested by the BARON manual. Symbolic parameters and semicontinuous/semi-integer variables are not representable in `.bar` and are likewise rejected.

## Result

`SolverResult` fields, populated whenever a usable point is available (`primal_status` is `FeasiblePoint` or `OptimalPoint`):

- `solutions`: primal points (`Vec<SolutionPoint>`), best first. Each point holds its `primal` values keyed by `VarId` and its `objective`. With `.num_sol(n)` BARON enumerates up to `n` distinct solutions into the pool, otherwise the vector holds the single incumbent. Access the best point via `result.objective()` / `result.value_of(var)` and the rest via `result.solution(i)`
- `dual`: constraint marginals at the best point, keyed by `ConstraintId`, access via `result.dual_of(c)`
- `reduced_costs`: variable marginals at the best point, keyed by `VarId`
- `termination`: why the solve stopped, read from BARON's `res.lst` termination banner
- `primal_status`: whether a usable point came back (`NoSolution` / `FeasiblePoint` / `OptimalPoint`), `result.has_solution()` is the shortcut
- `best_bound`: BARON's relaxation/dual bound (the bound opposite the incumbent)
- `gap`: relative optimality gap from BARON's lower/upper bounds, when both are finite
- `solve_time`: wall time around the BARON process invocation
- `iterations`: branch-and-reduce iteration count from `tim.lst`
- `raw_log`: BARON stdout/stderr, captured when BARON exits non-zero in quiet mode. With `verbose(true)` the output is streamed live to the terminal and not captured (`raw_log` is `None`)

## Acknowledgements

We would like to thank The Optimization Firm for providing a BARON license for development and testing of this crate.

## License

MIT OR Apache-2.0

> **Note:** BARON itself is commercial software. A valid BARON license is required at runtime. This crate is not affiliated with The Optimization Firm.
