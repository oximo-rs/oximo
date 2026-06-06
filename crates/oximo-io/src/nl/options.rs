//! Caller-facing options for the `.nl` writer.
//!
//! Mirrors the configuration surface of the official AMPL `nl-writer2`:
//! ASCII vs binary output, precision control, comment toggle, NaN/Inf
//! handling, and pluggable sources for the segments oximo's `Model` cannot
//! supply directly (imported functions `F`, suffixes `S`, defined variables
//! `V`, dual seeds `d`, complementarity entries in `r`).

/// Encoding of the body that follows the (always-ASCII) 10-line header.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum NlFormat {
    /// ASCII `g` header, every token on its own line.
    #[default]
    Ascii,
    /// Binary `b` header, opcodes as single bytes, ints as 4-byte LE, doubles
    /// as 8-byte LE, integer-valued numeric constants emitted as `s` (short)
    /// or `l` (long) when they fit.
    Binary,
}

/// Options for the writer.
///
/// `Default::default()` matches the writer's original behaviour: ASCII output
/// with trailing `\t# ...` comments, shortest-round-trip numeric formatting,
/// and an error on NaN/Inf constants.
#[derive(Clone, Debug)]
pub struct WriteOptions {
    pub format: NlFormat,
    /// `None` -> shortest round-trip (default Rust `{}` for `f64`).
    /// `Some(n)` -> `n` significant digits in scientific notation (`{:.*e}`,
    /// e.g. `3.142e0`).
    pub precision: Option<u32>,
    /// Emit trailing `\t# ...` comments in the header and segment introducers
    /// (ASCII format only). Ignored in binary mode.
    pub comments: bool,
    /// When `true`, NaN/Inf numeric constants are emitted as `Infinity` /
    /// `-Infinity` / `NaN` strings (matching AMPL `NL_LIB_GFMT`). When `false`
    /// (default) the writer fails with `IoError::InvalidNumber`.
    pub nonfinite_strings: bool,
    /// When `true`, a `write_nl_files` call also emits the sibling
    /// `<stub>.row` (constraint names) and `<stub>.col` (variable names) files
    /// next to the `.nl`, and populates the header's `max_name_len` fields.
    pub aux_files: bool,
    /// Imported function declarations (F segments). Empty by default.
    pub functions: Vec<ImportedFunction>,
    /// Suffix entries (S segments). Empty by default.
    pub suffixes: Vec<SuffixData>,
    /// Defined variables (V segments). Empty by default.
    pub defined_vars: Vec<DefinedVar>,
    /// Dual seeds (d segment), sparse `(constraint_nl_index, value)` pairs.
    pub dual_init: Vec<(u32, f64)>,
    /// Complementarity entries for the `r` segment. Maps a constraint index
    /// to its complementarity info. Empty by default.
    pub complementarity: Vec<(usize, Complementarity)>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            format: NlFormat::Ascii,
            precision: None,
            comments: true,
            nonfinite_strings: false,
            aux_files: false,
            functions: Vec::new(),
            suffixes: Vec::new(),
            defined_vars: Vec::new(),
            dual_init: Vec::new(),
            complementarity: Vec::new(),
        }
    }
}

impl WriteOptions {
    /// ASCII with comments. Identity with `Default::default()`.
    pub fn ascii() -> Self {
        Self::default()
    }

    /// ASCII, no trailing comments. Slimmer files for production use.
    pub fn ascii_lean() -> Self {
        Self { comments: false, ..Self::default() }
    }

    /// Binary `b` header. Comments are unused in binary mode.
    pub fn binary() -> Self {
        Self { format: NlFormat::Binary, comments: false, ..Self::default() }
    }
}

/// One entry in the `F` segment: an externally-defined function the model
/// references via opcode `f<i> <n_args>` inside expression graphs.
#[derive(Clone, Debug)]
pub struct ImportedFunction {
    pub name: String,
    /// `0` -> no string arguments.
    /// `1` -> string arguments allowed.
    pub allow_string_args: u8,
    /// `>= 0` -> exact arity.
    /// `< 0` -> at least `-(n + 1)` arguments.
    pub n_args: i32,
}

/// One `S` segment: suffix values attached to one of four entity kinds.
#[derive(Clone, Debug)]
pub struct SuffixData {
    pub name: String,
    pub kind: SuffixKind,
    pub flavour: SuffixFlavour,
    /// Sparse `(offset, value)` pairs. `offset` is the NL index of the
    /// entity (0-based). Only nonzero values need to be listed.
    pub values: Vec<(u32, f64)>,
}

/// Which kind of entity a suffix attaches to. Encoded as the low two bits of
/// the suffix kind word (D. M. Gay, Table 14).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SuffixKind {
    Variable = 0,
    Constraint = 1,
    Objective = 2,
    Problem = 3,
}

/// Whether suffix values are integer-valued (`Int`) or real-valued (`Real`).
/// Sets the `4` bit of the suffix kind word.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SuffixFlavour {
    Int,
    Real,
}

/// One `V` segment: a defined variable holding a named common subexpression.
#[derive(Clone, Debug)]
pub struct DefinedVar {
    /// The NL variable index this defined var occupies. Must be `>= n_var`
    /// (real decision variables come first).
    pub nl_index: u32,
    /// Linear part `sum(coef_i * v_i)` (referencing other NL indices).
    pub linear: Vec<(u32, f64)>,
    /// Which constraint or objective this defined var is private to:
    /// `0` -> shared (appears in V block at the top, before C/L/O),
    /// `m+1` -> only in constraint `m`, `n_con + n_lcon + m` -> only in
    /// objective `m`.
    pub appearance: u32,
    /// Polish-prefix expression text for the nonlinear part (e.g. `"o5\nv0\nn2\n"`).
    /// `""` means a purely-linear defined var (the expression part collapses
    /// to a single `n0` constant when emitted).
    pub nonlinear_polish: String,
}

/// Complementarity declaration for a constraint (D. M. Gay, Table 17, line type 5:
/// `5 k i` where `k` says which bounds on `v_{i-1}` are finite).
#[derive(Copy, Clone, Debug)]
pub struct Complementarity {
    /// Bits: `1` = finite lower bound, `2` = finite upper bound, `3` = both.
    pub k: u8,
    /// 1-based variable index `i` (the constraint complements `v_{i-1}`).
    /// The writer's `WriteOptions::complementarity` field carries the
    /// constraint index in original (unpermuted) order and the `Complementarity`
    /// here uses NL variable indices already.
    pub i: u32,
}
