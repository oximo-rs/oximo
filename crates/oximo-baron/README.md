# oximo-baron

BARON (Branch-And-Reduce Optimization Navigator) backend for [oximo](https://github.com/germanheim/oximo).

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
oximo = { version = "0.1", features = ["baron"] }
```

To use this crate directly:

```toml
[dependencies]
oximo-baron  = "0.1"
oximo-core   = "0.1"
oximo-solver = "0.1"
```

## Quick example

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Baron;

let m = Model::new("box");
let x = m.var("x").lb(0.1).ub(10.0).build();
let y = m.var("y").lb(0.1).ub(10.0).build();

m.constraint("c", (x + y).le(8.0));
let one = Expr::constant(x.arena, 1.0);
m.maximize((one + x).log() + 2.0 * y);

let result = Baron::new().solve(&m, &BaronOptions::default())?;
println!("status = {:?}", result.status);
println!("obj    = {:?}", result.objective);
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

BARON's `.bar` format has no trigonometric intrinsics, so a model whose objective or constraints contain `sin`/`cos` is rejected with a clear `SolverError::Backend`. A variable-exponent power `x^y` is rewritten to the equivalent `exp(y*log(x))`, as suggested by the BARON manual. Symbolic parameters and semicontinuous/semi-integer variables are not representable in `.bar` and are likewise rejected.

## Result

`SolverResult` fields populated on `Optimal` or `Feasible`:

- `objective`: objective value (the incumbent bound from `tim.lst`)
- `primal`: variable values keyed by `VarId`, access via `result.value_of(var)`
- `status`: mapped from BARON solver/model status codes
- `solve_time`: wall time around the BARON process invocation
- `iterations`: branch-and-reduce iteration count from `tim.lst`
- `raw_log`: BARON stdout/stderr, populated when `verbose(true)` or when BARON exits non-zero

`dual` and `reduced_costs` are not populated by this backend (duals are not generally meaningful for global nonconvex/integer optimization).

## Acknowledgements

We would like to thank The Optimization Firm for providing a BARON license for development and testing of this crate.

## License

MIT OR Apache-2.0

> **Note:** BARON itself is commercial software. A valid BARON license is required at runtime. This crate is not affiliated with The Optimization Firm.
