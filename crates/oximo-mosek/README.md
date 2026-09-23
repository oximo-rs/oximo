# oximo-mosek

[MOSEK](https://www.mosek.com/) 11.2 backend for [oximo](https://crates.io/crates/oximo).

This backend supports LP, MILP, convex QP/MIQP, convex QCP/MIQCP, and
SOCP/MISOCP models. MOSEK validates convexity.

Install [MOSEK 11.2](https://www.mosek.com/downloads/), configure a valid license, and point `MOSEK_BINDIR_112` at
the platform `bin` directory when it is not in MOSEK's default location.

## Supported model kinds

| Kind             | Support                                                                       |
| ---------------- | ----------------------------------------------------------------------------- |
| `LP`, `MILP`     | Yes                                                                           |
| `QP`, `MIQP`     | Yes when the objective is convex for minimization or concave for maximization |
| `QCP`, `MIQCP`   | Yes when the quadratic rows and objective satisfy MOSEK's convexity rules     |
| `SOCP`, `MISOCP` | Yes, for explicit oximo cones and detected diagonal quadratic cones           |

Quadratic expressions use MOSEK's lower-triangular `0.5 x'Qx` convention.
Indefinite and otherwise nonconvex data is sent to MOSEK and its native
validation message is returned as a backend error.

```toml
[dependencies]
oximo = { version = "0.7.0", features = ["mosek"] }

# Optional when using MOSEK enumeration constants in option builders:
mosek = "11.2"
```

```rust,ignore
use oximo::prelude::*;
use oximo::solvers::Mosek;

let model = Model::new("example");
variable!(model, x >= 0.0);
objective!(model, Min, x);

let options = MosekOptions::default()
    .mio_tol_rel_gap(1e-4)
    .presolve_use(mosek::Presolvemode::ON)
    .optimizer(mosek::Optimizertype::FREE)
    .remote_optserver_host("server.example.com");
let result = Mosek.solve(&model, &options)?;
# Ok::<(), SolverError>(())
```

Universal `time_limit`, `threads`, and `verbose` options are applied first.
MOSEK-specific parameter builders are applied afterward in call order, so the
last backend-specific write wins. Results include a primal point and objective,
continuous duals and reduced costs, explicit SOC bound duals, MIP bound and
relative gap, iteration counts, solve time, and the solver name when available.
`raw_log` and solution pools are not populated.

`Mosek` also implements `PersistentSolver`. Its persistent handle updates linear
objective coefficients, objective constants, and variable bounds in place for
LP/MILP models, allowing MOSEK to reuse warm-start information. Other model
changes, and all quadratic or conic models for now, rebuild transparently.

Every MOSEK 11.2 `Dparam`, `Iparam`, and `Sparam` has a snake-case builder.
Double builders accept `f64`, integer builders accept `i32`, and string builders
accept `impl Into<String>`. MOSEK enumeration constants can be passed directly
to integer builders. Repeated calls are preserved and applied in order.

## License

MIT OR Apache-2.0
