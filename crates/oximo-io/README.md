# oximo-io

Model I/O for [oximo](https://github.com/oximo-rs/oximo): MPS, LP, and NLP writers.

Converts an oximo [`oximo_core::Model`] to standard text formats for exchanging models with external solvers and tools.

## Usage

Enabled by default via the `io` feature on the umbrella `oximo` crate:

```toml
[dependencies]
oximo = "0.4" # io is on by default
```

To opt out:

```toml
[dependencies]
oximo = { version = "0.4", default-features = false, features = ["highs"] }
```

To use this crate directly:

```toml
[dependencies]
oximo-io   = "0.4"
oximo-core = "0.4"
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

Fixed-format MPS (fixed-column, 10-char field width). Widely supported by commercial and open-source solvers.

| Feature           | Behavior                                                                                                                       |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Objective row     | Named `OBJ`, maximization models are negated with a `* sense: maximize` comment so re-importers can recover the original sense |
| Integer variables | Wrapped in `INTORG`/`INTEND` markers                                                                                           |
| Semicont variables| `LO` (threshold) + upper-bound semi marker: `SC` for semi-continuous, `SI` for semi-integer                                    |
| Bounds            | `FR` (free), `MI`+`UP` (lower=-inf), `LO`/`UP` as needed. Default lb=0 omitted                                                 |
| Constant terms    | Objective constant written to `RHS OBJ`, constraint constants folded into `RHS`                                                |

```rust,ignore
use oximo_io::{write_mps, to_mps_string};
use std::fs::File;
use std::io::BufWriter;

// To string
let s = to_mps_string(&model)?;

// To file
let mut f = BufWriter::new(File::create("model.mps")?);
write_mps(&model, &mut f)?;
```

### LP (CPLEX LP format)

Human-readable CPLEX LP format. Sections emitted: header comment, `Minimize`/`Maximize`, `Subject To`, `Bounds` (non-default only), `General`, `Binaries`, `Semi-Continuous`, `End`.

| Feature            | Behavior                                                           |
|--------------------|--------------------------------------------------------------------|
| Objective sense    | `Minimize` / `Maximize` keyword, no negation needed                |
| Integer variables  | `General` section (integer/semi-integer), `Binaries` section       |
| Semicont variables | `Semi-Continuous` section, threshold emitted as the lower bound    |
| Bounds             | Free variables declared with `free`; default lb=0, ub=+inf omitted |
| Objective constant | Written as a comment if non-zero                                   |

```rust,ignore
use oximo_io::{write_lp, to_lp_string};
use std::fs::File;
use std::io::BufWriter;

// To string
let s = to_lp_string(&model)?;

// To file
let mut f = BufWriter::new(File::create("model.lp")?);
write_lp(&model, &mut f)?;
```

### NL

The standard format for sharing nonlinear and mixed-integer models. Unlike MPS/LP, it carries full nonlinear expressions, emitted as prefix (Polish) opcode trees.

| Feature              | Behavior                                                                |
|----------------------|-------------------------------------------------------------------------|
| Nonlinear bodies     | Linear part goes to `J`/`G`; nonlinear residual to `C`/`O` opcode trees |
| Supported operators  | `+ - * /`, negation, `pow`, `abs`, `sin`, `cos`, `exp`, `log` (natural) |
| Output encoding      | ASCII (default) or binary, via `WriteOptions::format`                   |
| Precision / comments | `precision` and `comments` knobs tune the ASCII output                  |
| Variable ordering    | Standard ASL order: nonlinear-first (by appearance), then linear        |
| Name sidecars        | `write_nl_files` also writes `.row` / `.col` name files                 |
| Optional segments    | `F`/`S`/`V`/`d`/`r` segments supplied via `WriteOptions`                |

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

Not yet supported: range constraints, Hollerith (string) constants, and `Param` nodes.

## Errors

All functions return `Result<_, IoError>`:

| Variant                       | Cause                                                                      |
|-------------------------------|----------------------------------------------------------------------------|
| `IoError::NoObjective`        | Model has no objective set                                                 |
| `IoError::Nonlinear`          | Nonlinear node in an MPS/LP model (NL supports nonlinear bodies)           |
| `IoError::UnsupportedNode(n)` | Node not representable in the target format, e.g. `Param` in NL            |
| `IoError::InvalidNumber`      | Non-finite (NaN/Inf) constant while `nonfinite_strings` is off             |
| `IoError::BinaryToString`     | `to_nl_string` used with binary output; use `write_nl_with` to a byte sink |
| `IoError::Io(e)`              | Underlying `std::io::Error` from the writer                                |

## License

MIT OR Apache-2.0
