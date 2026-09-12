# oximo-io

Model I/O for [oximo](https://github.com/oximo-rs/oximo): MPS, LP, and NL writers/readers.

Converts an oximo [`oximo_core::Model`] to standard text formats for exchanging models with external solvers and tools.

## Usage

Enabled by default via the `io` feature on the umbrella `oximo` crate:

```toml
[dependencies]
oximo = "0.6.0" # io is on by default
```

To opt out:

```toml
[dependencies]
oximo = { version = "0.6.0", default-features = false, features = ["highs"] }
```

To use this crate directly:

```toml
[dependencies]
oximo-io   = "0.6.0"
oximo-core = "0.6.0"
```

## Quick example

```rust,ignore
use oximo::prelude::*;
use oximo::io::{to_mps_string, to_lp_string};

let m = Model::new("knapsack");
variable!(m, x >= 0.0);
variable!(m, 0.0 <= y <= 4.0);

constraint!(m, c1, x + 2.0 * y <= 14.0);
constraint!(m, c2, 3.0 * x - y >= 0.0);
objective!(m, Max, 3.0 * x + 4.0 * y);

let mps = to_mps_string(&m)?;
let lp  = to_lp_string(&m)?;
println!("{mps}");
```

## Formats

### MPS

Whitespace-delimited MPS, compatible with conventional fixed-column files whose names do not contain spaces.
Widely supported by commercial and open-source solvers.

| Feature            | Behavior                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| Objective sense    | Written with `OBJSENSE`                                                                                    |
| Linear models      | `ROWS`, `COLUMNS`, `RHS`, `RANGES`, bounds, integer markers, binary and semi domains                       |
| Quadratic import   | `QUADOBJ`, `QMATRIX`, `QCMATRIX`, and `QSECTION`. Gurobi, CPLEX or MOSEK constraint scaling is selectable. |
| Quadratic export   | `QUADOBJ`/`QCMATRIX` for Gurobi and CPLEX, `QSECTION` for MOSEK                                            |
| SOS sections       | Native SOS1/SOS2 sections are imported and exported. MOSEK MPS export rejects SOS                          |
| Unsupported import | Indicator sections return `IoError::UnsupportedMps`                                                        |
| Constant terms     | Objective constants use `RHS OBJ`, constraint constants are folded into `RHS`                              |

```rust,ignore
use oximo_io::{
    MpsQuadraticFormat, MpsReadOptions, MpsWriteOptions, read_mps_file, read_mps_with,
    to_mps_string, to_mps_string_with, write_mps,
};
use std::fs::File;
use std::io::BufWriter;

// To string
let s = to_mps_string(&model)?;

// To file
let mut f = BufWriter::new(File::create("model.mps")?);
write_mps(&model, &mut f)?;

// Read a file with the default Gurobi quadratic-constraint convention.
let imported = read_mps_file("model.mps")?;

// Serialize and import with matching CPLEX quadratic conventions.
let cplex_write_options = MpsWriteOptions { quadratic_format: MpsQuadraticFormat::Cplex };
let cplex_mps = to_mps_string_with(&model, &cplex_write_options)?;
let cplex_read_options = MpsReadOptions { quadratic_format: MpsQuadraticFormat::Cplex };
let imported_cplex = read_mps_with(cplex_mps.as_bytes(), &cplex_read_options)?;

// Export quadratic sections in a solver-compatible dialect.
let write_options = MpsWriteOptions { quadratic_format: MpsQuadraticFormat::Mosek };
let quadratic_mps = to_mps_string_with(&model, &write_options)?;
```

### LP (CPLEX LP format)

Human-readable CPLEX LP format. Sections emitted: header comment, `Minimize`/`Maximize`, `Subject To`, `Bounds` (non-default only), `General`, `Binaries`, `Semi-Continuous`, `SOS`, `End`.

| Feature            | Behavior                                                           |
| ------------------ | ------------------------------------------------------------------ |
| Objective sense    | `Minimize` / `Maximize` keyword, no negation needed                |
| Quadratic terms    | CPLEX bracket notation: objective `[Q]/2`, constraints `[q]`       |
| Integer variables  | `General` section (integer/semi-integer), `Binaries` section       |
| Semicont variables | `Semi-Continuous` section, threshold emitted as the lower bound    |
| Bounds             | Free variables declared with `free`; default lb=0, ub=+inf omitted |
| Objective constant | Written as a final numeric term if non-zero                        |

```rust,ignore
use oximo_io::{write_lp, to_lp_string};
use std::fs::File;
use std::io::BufWriter;

// To string
let s = to_lp_string(&model)?;

// To file
let mut f = BufWriter::new(File::create("model.lp")?);
write_lp(&model, &mut f)?;

let imported = oximo_io::read_lp(s.as_bytes())?;
```

### NL

The standard format for sharing nonlinear and mixed-integer models. Unlike MPS/LP, it carries full nonlinear expressions, emitted as prefix (Polish) opcode trees.

| Feature              | Behavior                                                                    |
| -------------------- | --------------------------------------------------------------------------- |
| Nonlinear bodies     | Linear part goes to `J`/`G`; nonlinear residual to `C`/`O` opcode trees     |
| Supported operators  | `+ - * /`, all documented AMPL unary/trig/hyperbolic opcodes, min/max, etc. |
| Output encoding      | ASCII (default) or binary, via `WriteOptions::format`                       |
| Precision / comments | `precision` and `comments` knobs tune the ASCII output                      |
| Variable ordering    | Standard ASL order: nonlinear-first (by appearance), then linear            |
| Name sidecars        | `write_nl_files` also writes `.row` / `.col` name files                     |
| Optional segments    | `F`/`S`/`V`/`d`/`r` segments supplied via `WriteOptions`                    |

```rust,ignore
use oximo::prelude::*;
use oximo::io::{to_nl_string, write_nl_with, write_nl_files, WriteOptions};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

// Rosenbrock: min (1 - x)^2 + 100 (y - x^2)^2
let m = Model::new("rosen");
variable!(m, -5.0 <= x <= 5.0);
variable!(m, -5.0 <= y <= 5.0);
objective!(m, Min, (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2));

// To string (ASCII only)
let nl = to_nl_string(&m)?;

// To <stub>.nl plus sibling .row / .col name files
let opts = WriteOptions { aux_files: true, ..Default::default() };
write_nl_files(&m, Path::new("rosen"), &opts)?;

// Binary output needs to be written to a byte sink
let mut f = BufWriter::new(File::create("rosen.nl")?);
write_nl_with(&m, &mut f, &WriteOptions::binary())?;
```

NL files can also be imported with `read_nl` (a stream) or `read_nl_file` (a
path). The latter automatically uses sibling `.row` and `.col` sidecars when
present, otherwise deterministic `c0`/`x0` names are generated. Both ASCII and
the binary encoding emitted by this crate are accepted. Imported functions,
defined variables, logical constraints, and complementarity are reported as
unsupported.

The reader preserves interval rows and initial values. Hollerith strings are
malformed NL input and return `IoError::InvalidNl`. Parameter nodes are not
reader input, so the writer-side `Param` behavior remains documented as
`IoError::UnsupportedNode`. Defined-variable sections return
`IoError::UnsupportedNl`. Imported functions and logical/network sections are
also rejected with that variant because they are not representable by the core
model.

## Errors

All functions return `Result<_, IoError>`:

| Variant                       | Cause                                                                      |
| ----------------------------- | -------------------------------------------------------------------------- |
| `IoError::NoObjective`        | Model has no objective set                                                 |
| `IoError::Nonlinear`          | Unsupported nonlinear node in an MPS/LP model (LP accepts degree <= 2)     |
| `IoError::UnsupportedNode(n)` | Node not representable in the target format, e.g. `Param` in NL            |
| `IoError::InvalidNumber`      | Non-finite (NaN/Inf) constant while `nonfinite_strings` is off             |
| `IoError::BinaryToString`     | `to_nl_string` used with binary output; use `write_nl_with` to a byte sink |
| `IoError::Io(e)`              | Underlying `std::io::Error` from the writer                                |
| `IoError::InvalidNl`          | Malformed or truncated NL input                                            |
| `IoError::UnsupportedNl`      | Semantics not representable by the core model                              |
| `IoError::InvalidLp`          | Invalid LP input, with line and column                                     |
| `IoError::UnsupportedLp`      | LP semantics not representable by oximo's core model                       |
| `IoError::InvalidMps`         | Invalid MPS input, with line and column                                    |
| `IoError::UnsupportedMps`     | MPS semantics not representable by oximo's core model                      |

## License

MIT OR Apache-2.0
