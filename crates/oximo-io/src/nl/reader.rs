//! Reader for the portable (ASCII) subset of the AMPL NL format.
//!
//! The reader deliberately builds the `Model` only after all records have been
//! validated.  This keeps malformed files from leaving a partially populated
//! model behind and also lets the `J`/`G` linear records be merged with their
//! nonlinear `C`/`O` trees.

#![allow(clippy::many_single_char_names)]

const MAX_EXPRESSION_DEPTH: usize = 256;

use std::fs::File;
use std::io::Read;
use std::path::Path;

use oximo_core::{Domain, Expr, Model, ObjectiveSense};

use crate::error::IoError;

use super::options::{SuffixData, SuffixFlavour, SuffixKind, WriteOptions};

#[derive(Debug, Clone, Copy)]
struct Header {
    n_var: usize,
    n_con: usize,
    n_obj: usize,
    _n_ranges: usize,
    _nl_con: usize,
    _nl_obj: usize,
    nl_c: usize,
    nl_o: usize,
    nl_both: usize,
    n_bin: usize,
    n_int: usize,
    nl_int_b: usize,
    nl_int_c: usize,
    nl_int_o: usize,
}

#[derive(Debug, Default)]
struct Parsed {
    header: Option<Header>,
    name: Option<String>,
    bounds: Vec<(f64, f64)>,
    starts: Vec<(usize, f64)>,
    rows: Vec<(u8, Vec<f64>)>,
    c_expr: Vec<Option<Node>>,
    objective: Option<(ObjectiveSense, Node)>,
    jac: Vec<Vec<(usize, f64)>>,
    grad: Vec<(usize, f64)>,
    suffixes: Vec<SuffixData>,
    dual_init: Vec<(u32, f64)>,
}

#[derive(Debug, Clone)]
enum Node {
    Const(f64),
    Var(usize),
    Unary(u32, Box<Node>),
    Binary(u32, Box<Node>, Box<Node>),
    Sum(Vec<Node>),
}

/// Result of reading an NL stream when caller-supplied writer metadata should
/// be retained for a later rewrite.
#[derive(Debug)]
pub struct NlReadData {
    pub model: Model,
    pub suffixes: Vec<SuffixData>,
    pub dual_init: Vec<(u32, f64)>,
}

impl NlReadData {
    /// Build writer options that re-emit retained NL metadata.
    #[must_use]
    pub fn write_options(&self) -> WriteOptions {
        WriteOptions {
            suffixes: self.suffixes.clone(),
            dual_init: self.dual_init.clone(),
            ..WriteOptions::default()
        }
    }

    /// Consume the read data and build writer options that re-emit retained NL
    /// metadata.
    #[must_use]
    pub fn into_write_options(self) -> WriteOptions {
        WriteOptions {
            suffixes: self.suffixes,
            dual_init: self.dual_init,
            ..WriteOptions::default()
        }
    }
}

/// Read an NL stream.
///
/// # Errors
///
/// Returns [`IoError`] when the stream is malformed or uses unsupported NL
/// semantics.
pub fn read_nl<R: Read>(input: R) -> Result<Model, IoError> {
    Ok(read_nl_data(input)?.model)
}

/// Read an NL stream, retaining suffix and dual-seed metadata for rewrite
/// workflows.
///
/// # Errors
///
/// Returns [`IoError`] when the stream is malformed or uses unsupported NL
/// semantics.
pub fn read_nl_data<R: Read>(mut input: R) -> Result<NlReadData, IoError> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    parse_bytes(&bytes, None, None)
}

/// Read an NL file and optional sibling `.col`/`.row` sidecars.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed input, or unsupported NL
/// sections.
pub fn read_nl_file(path: impl AsRef<Path>) -> Result<Model, IoError> {
    Ok(read_nl_file_data(path)?.model)
}

/// Read an NL file and optional sibling `.col`/`.row` sidecars, retaining
/// suffix and dual-seed metadata for rewrite workflows.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed input, or unsupported NL
/// sections.
pub fn read_nl_file_data(path: impl AsRef<Path>) -> Result<NlReadData, IoError> {
    let path = path.as_ref();
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let stem = path.with_extension("");
    let cols = read_names(&stem.with_extension("col"), "column")?;
    let rows = read_names(&stem.with_extension("row"), "row")?;
    parse_bytes(&bytes, Some(&stem), Some((&cols, &rows)))
}

fn read_names(path: &Path, what: &str) -> Result<Vec<String>, IoError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)?;
    let names = text.lines().map(|line| line.trim_end_matches('\r').to_owned()).collect::<Vec<_>>();
    for name in &names {
        if name.is_empty() {
            return Err(invalid("sidecar", format!("empty {what} name")));
        }
        if name.chars().any(char::is_control) {
            return Err(invalid("sidecar", format!("{what} name contains a control character")));
        }
    }
    Ok(names)
}

fn parse_bytes(
    bytes: &[u8],
    stem: Option<&Path>,
    names: Option<(&[String], &[String])>,
) -> Result<NlReadData, IoError> {
    let first_end = bytes
        .iter()
        .position(|b| *b == b'\n')
        .ok_or_else(|| invalid("header", "missing first line"))?;
    let first =
        std::str::from_utf8(&bytes[..first_end]).map_err(|_| invalid("header", "not UTF-8"))?;
    let binary = first.trim_start().starts_with('b');
    if !binary && !first.trim_start().starts_with('g') {
        return Err(invalid("header", "unsupported NL header"));
    }
    let mut header_end = 0usize;
    let mut header_lines = Vec::with_capacity(10);
    for _ in 0..10 {
        let end = bytes[header_end..]
            .iter()
            .position(|b| *b == b'\n')
            .ok_or_else(|| invalid("header", "truncated header"))?
            + header_end;
        header_lines.push(
            String::from_utf8(bytes[header_end..end].to_vec())
                .map_err(|_| invalid("header", "header is not UTF-8"))?,
        );
        header_end = end + 1;
    }
    let (header, name) = parse_header(&header_lines)?;
    let mut p = Parsed { header: Some(header), name, ..Parsed::default() };
    if binary {
        parse_binary_body(&bytes[header_end..], &mut p)?;
    } else {
        let text =
            std::str::from_utf8(bytes).map_err(|_| invalid("body", "ASCII NL is not UTF-8"))?;
        let mut lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
        parse_body(&mut lines, &mut p)?;
    }
    let suffixes = std::mem::take(&mut p.suffixes);
    let dual_init = std::mem::take(&mut p.dual_init);
    let model = build_model(p, stem, names)?;
    Ok(NlReadData { model, suffixes, dual_init })
}

fn parse_header(lines: &[String]) -> Result<(Header, Option<String>), IoError> {
    if lines.len() < 10 {
        return Err(invalid("header", "expected ten lines"));
    }
    let (first, comment) = split_comment(&lines[0]);
    let mut f = first.split_whitespace();
    let magic = f.next().unwrap_or_default();
    if magic != "g3" && magic != "b3" {
        return Err(invalid("header", format!("unsupported header {magic}")));
    }
    let name =
        comment.and_then(|c| c.strip_prefix("problem ").map(str::trim).map(ToOwned::to_owned));
    let a = nums(&lines[1], 5)?;
    let b = nums(&lines[2], 2)?;
    let d = nums(&lines[4], 3)?;
    let i = nums(&lines[6], 5)?;
    let h = Header {
        n_var: a[0],
        n_con: a[1],
        n_obj: a[2],
        _n_ranges: a[3],
        _nl_con: b[0],
        _nl_obj: b[1],
        nl_c: d[0],
        nl_o: d[1],
        nl_both: d[2],
        n_bin: i[0],
        n_int: i[1],
        nl_int_b: i[2],
        nl_int_c: i[3],
        nl_int_o: i[4],
    };
    if h.n_obj > 1 || h.n_var == usize::MAX {
        return Err(invalid("header", "invalid counts"));
    }
    Ok((h, name))
}

fn nums(line: &str, n: usize) -> Result<Vec<usize>, IoError> {
    let (body, _) = split_comment(line);
    let vals = body
        .split_whitespace()
        .map(|s| s.parse::<usize>().map_err(|_| invalid("header", format!("invalid integer {s}"))))
        .collect::<Result<Vec<_>, _>>()?;
    if vals.len() < n {
        return Err(invalid("header", "too few fields"));
    }
    Ok(vals)
}

fn parse_body(lines: &mut [String], p: &mut Parsed) -> Result<(), IoError> {
    let h = p.header.unwrap();
    let mut i = 10usize;
    while i < lines.len() {
        let line = lines[i].trim().to_owned();
        if line.is_empty() {
            i += 1;
        } else {
            let tag = line.as_bytes()[0] as char;
            parse_ascii_segment(tag, &line, lines, &mut i, p, h)?;
        }
    }
    if p.bounds.len() != h.n_var {
        return Err(invalid("b", "missing variable bounds"));
    }
    if p.rows.len() != h.n_con {
        return Err(invalid("r", "missing constraint bounds"));
    }
    Ok(())
}

fn parse_ascii_segment(
    tag: char,
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    match tag {
        'C' => parse_ascii_c(line, lines, pos, parsed, header),
        'O' => parse_ascii_o(line, lines, pos, parsed),
        'r' => parse_ascii_r(lines, pos, parsed, header),
        'b' => parse_ascii_b(lines, pos, parsed, header),
        'x' => parse_ascii_x(line, lines, pos, parsed),
        'J' | 'G' => parse_ascii_jg(tag, line, lines, pos, parsed, header),
        'k' => {
            *pos += 1 + parse_suffix(line, 'k')?;
            Ok(())
        }
        'S' => parse_ascii_suffix(line, lines, pos, parsed, header),
        'd' => parse_ascii_dual(line, lines, pos, parsed, header),
        'F' => Err(IoError::UnsupportedNl {
            section: "F".into(),
            feature: "imported functions".into(),
        }),
        'L' | 'N' => Err(IoError::UnsupportedNl {
            section: tag.to_string(),
            feature: "logical/network constraints".into(),
        }),
        'V' => {
            Err(IoError::UnsupportedNl { section: "V".into(), feature: "defined variables".into() })
        }
        _ => Err(invalid("body", format!("unknown segment {tag}"))),
    }
}

fn parse_ascii_c(
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    let idx = parse_suffix(line, 'C')?;
    *pos += 1;
    let node = parse_expr(lines, pos, 0)?;
    if idx >= header.n_con {
        return Err(invalid("C", "row index out of range"));
    }
    if parsed.c_expr.len() < header.n_con {
        parsed.c_expr.resize(header.n_con, None);
    }
    parsed.c_expr[idx] = Some(node);
    Ok(())
}

fn parse_ascii_o(
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
) -> Result<(), IoError> {
    let fields = line[1..].split_whitespace().collect::<Vec<_>>();
    if fields.len() < 2 {
        return Err(invalid("O", "missing objective sense"));
    }
    let sense = match fields[1] {
        "0" => ObjectiveSense::Minimize,
        "1" => ObjectiveSense::Maximize,
        _ => return Err(invalid("O", "invalid objective sense")),
    };
    *pos += 1;
    parsed.objective = Some((sense, parse_expr(lines, pos, 0)?));
    Ok(())
}

fn parse_ascii_r(
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    *pos += 1;
    for _ in 0..header.n_con {
        let vals = record(lines, pos, "r")?;
        if vals.is_empty() {
            return Err(invalid("r", "empty row"));
        }
        let row_type =
            u8::try_from(as_index(vals[0])?).map_err(|_| invalid("r", "row type out of range"))?;
        parsed.rows.push((row_type, vals[1..].to_vec()));
    }
    Ok(())
}

fn parse_ascii_b(
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    *pos += 1;
    for _ in 0..header.n_var {
        parsed.bounds.push(decode_bound(&record(lines, pos, "b")?)?);
    }
    Ok(())
}

fn parse_ascii_x(
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
) -> Result<(), IoError> {
    let n = parse_suffix(line, 'x')?;
    *pos += 1;
    for _ in 0..n {
        let values = record(lines, pos, "x")?;
        if values.len() != 2 {
            return Err(invalid("x", "expected index and value"));
        }
        parsed.starts.push((as_index(values[0])?, values[1]));
    }
    Ok(())
}

fn parse_ascii_jg(
    tag: char,
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    let fields = line[1..]
        .split_whitespace()
        .map(|x| x.parse::<usize>().map_err(|_| invalid("J/G", "invalid header")))
        .collect::<Result<Vec<_>, _>>()?;
    if fields.len() < 2 {
        return Err(invalid("J/G", "missing count"));
    }
    let row = fields[0];
    if tag == 'J' && row >= header.n_con {
        return Err(invalid("J", "row index out of range"));
    }
    let n = fields[1];
    *pos += 1;
    let mut entries = Vec::with_capacity(bounded_capacity(n, lines.len().saturating_sub(*pos)));
    for _ in 0..n {
        let values = record(lines, pos, "J/G")?;
        if values.len() != 2 {
            return Err(invalid("J/G", "expected index and coefficient"));
        }
        entries.push((as_index(values[0])?, values[1]));
    }
    if tag == 'G' {
        parsed.grad = entries;
    } else {
        if parsed.jac.len() <= row {
            parsed.jac.resize(row + 1, Vec::new());
        }
        parsed.jac[row] = entries;
    }
    Ok(())
}

fn parse_ascii_suffix(
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    let fields = line[1..].split_whitespace().collect::<Vec<_>>();
    if fields.len() < 3 {
        return Err(invalid("S", "missing suffix metadata"));
    }
    let (kind, flavour) = decode_suffix_kind(fields[0])?;
    let n = fields[1].parse::<usize>().map_err(|_| invalid("S", "invalid suffix count"))?;
    let name = fields[2].to_owned();
    *pos += 1;
    let mut values = Vec::with_capacity(bounded_capacity(n, lines.len().saturating_sub(*pos)));
    for _ in 0..n {
        let record = record(lines, pos, "S")?;
        if record.len() != 2 {
            return Err(invalid("S", "expected suffix index and value"));
        }
        let index = as_index(record[0])?;
        validate_suffix_index(kind, index, header)?;
        if record[1].is_nan() {
            return Err(invalid("S", "NaN suffix value"));
        }
        values.push((as_u32_index("S", index)?, record[1]));
    }
    parsed.suffixes.push(SuffixData { name, kind, flavour, values });
    Ok(())
}

fn parse_ascii_dual(
    line: &str,
    lines: &[String],
    pos: &mut usize,
    parsed: &mut Parsed,
    header: Header,
) -> Result<(), IoError> {
    let n = parse_suffix(line, 'd')?;
    *pos += 1;
    for _ in 0..n {
        let record = record(lines, pos, "d")?;
        if record.len() != 2 {
            return Err(invalid("d", "expected constraint index and value"));
        }
        let index = as_index(record[0])?;
        if index >= header.n_con {
            return Err(invalid("d", "constraint index out of range"));
        }
        if record[1].is_nan() {
            return Err(invalid("d", "NaN dual value"));
        }
        parsed.dual_init.push((as_u32_index("d", index)?, record[1]));
    }
    Ok(())
}

struct Bin<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Bin<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn byte(&mut self) -> Result<u8, IoError> {
        let b = self.bytes.get(self.pos).copied().ok_or_else(|| invalid("binary", "truncated"))?;
        self.pos += 1;
        Ok(b)
    }
    fn i32(&mut self) -> Result<i32, IoError> {
        let end = self.pos.checked_add(4).ok_or_else(|| invalid("binary", "overflow"))?;
        let x =
            self.bytes.get(self.pos..end).ok_or_else(|| invalid("binary", "truncated integer"))?;
        self.pos = end;
        Ok(i32::from_le_bytes(x.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, IoError> {
        let end = self.pos.checked_add(8).ok_or_else(|| invalid("binary", "overflow"))?;
        let x =
            self.bytes.get(self.pos..end).ok_or_else(|| invalid("binary", "truncated number"))?;
        self.pos = end;
        Ok(f64::from_le_bytes(x.try_into().unwrap()))
    }
}

fn parse_binary_body(bytes: &[u8], p: &mut Parsed) -> Result<(), IoError> {
    let h = p.header.unwrap();
    let mut b = Bin::new(bytes);
    while b.pos < bytes.len() {
        parse_binary_segment(b.byte()? as char, &mut b, p, h)?;
    }
    if p.bounds.len() != h.n_var {
        return Err(invalid("b", "missing variable bounds"));
    }
    if p.rows.len() != h.n_con {
        return Err(invalid("r", "missing row bounds"));
    }
    Ok(())
}

fn parse_binary_segment(
    tag: char,
    b: &mut Bin<'_>,
    p: &mut Parsed,
    h: Header,
) -> Result<(), IoError> {
    match tag {
        'C' => parse_binary_c(b, p, h),
        'O' => parse_binary_o(b, p),
        'r' => parse_binary_r(b, p, h),
        'b' => parse_binary_b(b, p, h),
        'x' => parse_binary_x(b, p),
        'k' => skip_binary_ints(b, "k"),
        'd' => parse_binary_duals(b, p, h),
        'S' => parse_binary_suffix(b, p, h),
        'J' => parse_binary_j(b, p, h),
        'G' => parse_binary_g(b, p),
        'F' | 'V' | 'L' | 'N' => Err(IoError::UnsupportedNl {
            section: tag.to_string(),
            feature: match tag {
                'F' => "imported functions",
                'V' => "defined variables",
                'L' | 'N' => "logical/network constraints",
                _ => unreachable!(),
            }
            .into(),
        }),
        x => Err(invalid("binary", format!("unknown segment {x}"))),
    }
}

fn parse_binary_c(b: &mut Bin<'_>, p: &mut Parsed, h: Header) -> Result<(), IoError> {
    let idx = nonneg(b.i32()?, "C index")?;
    let node = parse_binary_expr(b, 0)?;
    if idx >= h.n_con {
        return Err(invalid("C", "row index out of range"));
    }
    if p.c_expr.len() < h.n_con {
        p.c_expr.resize(h.n_con, None);
    }
    p.c_expr[idx] = Some(node);
    Ok(())
}

fn parse_binary_o(b: &mut Bin<'_>, p: &mut Parsed) -> Result<(), IoError> {
    let _idx = nonneg(b.i32()?, "O index")?;
    let sense = match b.i32()? {
        0 => ObjectiveSense::Minimize,
        1 => ObjectiveSense::Maximize,
        _ => return Err(invalid("O", "invalid sense")),
    };
    p.objective = Some((sense, parse_binary_expr(b, 0)?));
    Ok(())
}

fn parse_binary_r(b: &mut Bin<'_>, p: &mut Parsed, h: Header) -> Result<(), IoError> {
    for _ in 0..h.n_con {
        let ty = b.i32()?;
        let vals = match ty {
            0 => vec![b.f64()?, b.f64()?],
            1 | 2 | 4 => vec![b.f64()?],
            3 => vec![],
            5 => {
                return Err(IoError::UnsupportedNl {
                    section: "r".into(),
                    feature: "complementarity".into(),
                });
            }
            _ => return Err(invalid("r", "invalid row type")),
        };
        p.rows.push((u8::try_from(ty).unwrap(), vals));
    }
    Ok(())
}

fn parse_binary_b(b: &mut Bin<'_>, p: &mut Parsed, h: Header) -> Result<(), IoError> {
    for _ in 0..h.n_var {
        let ty = b.i32()?;
        let bounds = match ty {
            0 => (b.f64()?, b.f64()?),
            1 => (f64::NEG_INFINITY, b.f64()?),
            2 => (b.f64()?, f64::INFINITY),
            3 => (f64::NEG_INFINITY, f64::INFINITY),
            4 => {
                let x = b.f64()?;
                (x, x)
            }
            _ => return Err(invalid("b", "invalid bound type")),
        };
        p.bounds.push(bounds);
    }
    Ok(())
}

fn parse_binary_x(b: &mut Bin<'_>, p: &mut Parsed) -> Result<(), IoError> {
    let n = nonneg(b.i32()?, "x count")?;
    for _ in 0..n {
        p.starts.push((nonneg(b.i32()?, "x index")?, b.f64()?));
    }
    Ok(())
}

fn skip_binary_ints(b: &mut Bin<'_>, section: &str) -> Result<(), IoError> {
    let n = nonneg(b.i32()?, format!("{section} count").as_str())?;
    for _ in 0..n {
        let _ = b.i32()?;
    }
    Ok(())
}

fn parse_binary_duals(b: &mut Bin<'_>, p: &mut Parsed, h: Header) -> Result<(), IoError> {
    let n = nonneg(b.i32()?, "d count")?;
    for _ in 0..n {
        let index = nonneg(b.i32()?, "d index")?;
        if index >= h.n_con {
            return Err(invalid("d", "constraint index out of range"));
        }
        let value = b.f64()?;
        if value.is_nan() {
            return Err(invalid("d", "NaN dual value"));
        }
        p.dual_init.push((as_u32_index("d", index)?, value));
    }
    Ok(())
}

fn parse_binary_suffix(b: &mut Bin<'_>, p: &mut Parsed, h: Header) -> Result<(), IoError> {
    let (kind, flavour) = decode_suffix_kind_word(b.i32()?)?;
    let n = nonneg(b.i32()?, "S count")?;
    let name_len = nonneg(b.i32()?, "S name length")?;
    let end =
        b.pos.checked_add(name_len).ok_or_else(|| invalid("binary", "suffix name overflow"))?;
    let name_bytes =
        b.bytes.get(b.pos..end).ok_or_else(|| invalid("binary", "truncated suffix name"))?;
    let name =
        std::str::from_utf8(name_bytes).map_err(|_| invalid("S", "suffix name is not UTF-8"))?;
    if name.is_empty() {
        return Err(invalid("S", "empty suffix name"));
    }
    b.pos = end;
    let mut values = Vec::with_capacity(bounded_capacity(n, b.bytes.len().saturating_sub(b.pos)));
    for _ in 0..n {
        let index = nonneg(b.i32()?, "S index")?;
        validate_suffix_index(kind, index, h)?;
        let value = b.f64()?;
        if value.is_nan() {
            return Err(invalid("S", "NaN suffix value"));
        }
        values.push((as_u32_index("S", index)?, value));
    }
    p.suffixes.push(SuffixData { name: name.to_owned(), kind, flavour, values });
    Ok(())
}

fn decode_suffix_kind(raw: &str) -> Result<(SuffixKind, SuffixFlavour), IoError> {
    let word = raw.parse::<i32>().map_err(|_| invalid("S", "invalid suffix kind"))?;
    decode_suffix_kind_word(word)
}

fn decode_suffix_kind_word(word: i32) -> Result<(SuffixKind, SuffixFlavour), IoError> {
    if word < 0 || (word & !7) != 0 {
        return Err(invalid("S", "invalid suffix kind"));
    }
    let kind = match word & 3 {
        0 => SuffixKind::Variable,
        1 => SuffixKind::Constraint,
        2 => SuffixKind::Objective,
        3 => SuffixKind::Problem,
        _ => unreachable!(),
    };
    let flavour = if word & 4 == 0 { SuffixFlavour::Int } else { SuffixFlavour::Real };
    Ok((kind, flavour))
}

fn validate_suffix_index(kind: SuffixKind, index: usize, h: Header) -> Result<(), IoError> {
    let limit = match kind {
        SuffixKind::Variable => h.n_var,
        SuffixKind::Constraint => h.n_con,
        SuffixKind::Objective => h.n_obj,
        SuffixKind::Problem => 1,
    };
    if index >= limit {
        return Err(invalid("S", "suffix index out of range"));
    }
    Ok(())
}

fn as_u32_index(section: &str, index: usize) -> Result<u32, IoError> {
    u32::try_from(index).map_err(|_| invalid(section, "index out of range"))
}

fn parse_binary_j(b: &mut Bin<'_>, p: &mut Parsed, h: Header) -> Result<(), IoError> {
    let row = nonneg(b.i32()?, "J row")?;
    if row >= h.n_con {
        return Err(invalid("J", "row index out of range"));
    }
    let n = nonneg(b.i32()?, "J count")?;
    let mut entries = Vec::with_capacity(bounded_capacity(n, b.bytes.len().saturating_sub(b.pos)));
    for _ in 0..n {
        entries.push((nonneg(b.i32()?, "J index")?, b.f64()?));
    }
    if p.jac.len() <= row {
        p.jac.resize(row + 1, Vec::new());
    }
    p.jac[row] = entries;
    Ok(())
}

fn parse_binary_g(b: &mut Bin<'_>, p: &mut Parsed) -> Result<(), IoError> {
    let _row = nonneg(b.i32()?, "G row")?;
    let n = nonneg(b.i32()?, "G count")?;
    let mut entries = Vec::with_capacity(bounded_capacity(n, b.bytes.len().saturating_sub(b.pos)));
    for _ in 0..n {
        entries.push((nonneg(b.i32()?, "G index")?, b.f64()?));
    }
    p.grad = entries;
    Ok(())
}

fn parse_binary_expr(b: &mut Bin<'_>, depth: usize) -> Result<Node, IoError> {
    match b.byte()? as char {
        'v' => Ok(Node::Var(nonneg(b.i32()?, "variable index")?)),
        'n' => Ok(Node::Const(b.f64()?)),
        's' => {
            let end = b.pos.checked_add(2).ok_or_else(|| invalid("binary", "overflow"))?;
            let bytes =
                b.bytes.get(b.pos..end).ok_or_else(|| invalid("binary", "truncated short"))?;
            b.pos = end;
            Ok(Node::Const(f64::from(i16::from_le_bytes(bytes.try_into().unwrap()))))
        }
        'l' => Ok(Node::Const(f64::from(b.i32()?))),
        'o' => {
            let c = u32::try_from(nonneg(b.i32()?, "opcode")?)
                .map_err(|_| invalid("binary", "opcode out of range"))?;
            let ar = match c {
                15 | 16 | 41 | 43 | 44 | 46 => 1,
                0 | 1 | 2 | 3 | 5 => 2,
                54 => nonneg(b.i32()?, "sum arity")?,
                _ => {
                    return Err(IoError::UnsupportedNl {
                        section: "expression".into(),
                        feature: format!("opcode {c}"),
                    });
                }
            };
            if c == 54 {
                let mut xs =
                    Vec::with_capacity(bounded_capacity(ar, b.bytes.len().saturating_sub(b.pos)));
                let child_depth = next_expression_depth(depth)?;
                for _ in 0..ar {
                    xs.push(parse_binary_expr(b, child_depth)?);
                }
                Ok(Node::Sum(xs))
            } else {
                let child_depth = next_expression_depth(depth)?;
                let a = parse_binary_expr(b, child_depth)?;
                if ar == 1 {
                    Ok(Node::Unary(c, Box::new(a)))
                } else {
                    Ok(Node::Binary(c, Box::new(a), Box::new(parse_binary_expr(b, child_depth)?)))
                }
            }
        }
        x => Err(invalid("binary expression", format!("unknown token {x}"))),
    }
}

fn parse_expr(lines: &[String], i: &mut usize, depth: usize) -> Result<Node, IoError> {
    if *i >= lines.len() {
        return Err(invalid("expression", "truncated"));
    }
    let s = lines[*i].trim();
    *i += 1;
    if let Some(v) = s.strip_prefix('v') {
        return Ok(Node::Var(v.parse().map_err(|_| invalid("expression", "bad variable index"))?));
    }
    if let Some(v) = s.strip_prefix('n') {
        return Ok(Node::Const(parse_float(v)?));
    }
    let code: u32 = s
        .strip_prefix('o')
        .ok_or_else(|| invalid("expression", format!("unexpected token {s}")))?
        .parse()
        .map_err(|_| invalid("expression", "bad opcode"))?;
    let arity = match code {
        15 | 16 | 41 | 43 | 44 | 46 => 1,
        0 | 1 | 2 | 3 | 5 => 2,
        54 => {
            if *i >= lines.len() {
                return Err(invalid("expression", "missing sum arity"));
            }
            let n = lines[*i]
                .trim()
                .parse::<usize>()
                .map_err(|_| invalid("expression", "bad sum arity"))?;
            *i += 1;
            n
        }
        _ => {
            return Err(IoError::UnsupportedNl {
                section: "expression".into(),
                feature: format!("opcode {code}"),
            });
        }
    };
    if code == 54 {
        let mut xs = Vec::with_capacity(bounded_capacity(arity, lines.len().saturating_sub(*i)));
        let child_depth = next_expression_depth(depth)?;
        for _ in 0..arity {
            xs.push(parse_expr(lines, i, child_depth)?);
        }
        return Ok(Node::Sum(xs));
    }
    let child_depth = next_expression_depth(depth)?;
    let a = parse_expr(lines, i, child_depth)?;
    if arity == 1 {
        Ok(Node::Unary(code, Box::new(a)))
    } else {
        let b = parse_expr(lines, i, child_depth)?;
        Ok(Node::Binary(code, Box::new(a), Box::new(b)))
    }
}

fn build_model(
    p: Parsed,
    stem: Option<&Path>,
    names: Option<(&[String], &[String])>,
) -> Result<Model, IoError> {
    let h = p.header.unwrap();
    for &(lb, ub) in &p.bounds {
        if lb.is_nan() || ub.is_nan() {
            return Err(invalid("b", "NaN bound"));
        }
    }
    for &(_, value) in &p.starts {
        if value.is_nan() {
            return Err(invalid("x", "NaN initial value"));
        }
    }
    for (_, values) in &p.rows {
        if values.iter().any(|value| value.is_nan()) {
            return Err(invalid("r", "NaN row bound"));
        }
    }
    for entries in &p.jac {
        if entries.iter().any(|(_, value)| value.is_nan()) {
            return Err(invalid("J", "NaN coefficient"));
        }
    }
    if p.grad.iter().any(|(_, value)| value.is_nan()) {
        return Err(invalid("G", "NaN coefficient"));
    }
    let (cols, rows) = names.unwrap_or((&[], &[]));
    if (!cols.is_empty() && cols.len() != h.n_var) || (!rows.is_empty() && rows.len() < h.n_con) {
        return Err(invalid("sidecar", "name count does not match header"));
    }
    ensure_unique(cols, "column")?;
    ensure_unique(rows, "row")?;
    let model_name = p
        .name
        .or_else(|| {
            stem.and_then(|x| x.file_stem()).and_then(|x| x.to_str()).map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "nl_model".into());
    let m = Model::new(model_name);
    let domains = domains(&h);
    let starts = start_values(&p.starts, h.n_var);
    let mut vars = Vec::with_capacity(h.n_var);
    for (j, domain) in domains.iter().enumerate().take(h.n_var) {
        let name = cols.get(j).cloned().unwrap_or_else(|| format!("x{j}"));
        let (lb, ub) = p.bounds[j];
        let mut b = m.__var(name).bounds(lb, ub).domain(*domain);
        if let Some(Some(v)) = starts.get(j) {
            b = b.initial(*v);
        }
        vars.push(b.build());
    }
    let mut exprs = Vec::with_capacity(h.n_con);
    let c_nodes = p.c_expr.iter().cloned().chain(std::iter::repeat(None));
    for (j, base) in c_nodes.take(h.n_con).enumerate() {
        let base = base.unwrap_or(Node::Const(0.0));
        let mut e = lower(&m, &vars, base, 0)?;
        if let Some(entries) = p.jac.get(j) {
            for &(v, c) in entries {
                if v >= vars.len() {
                    return Err(invalid("J", "variable index out of range"));
                }
                e = e + c * vars[v];
            }
        }
        exprs.push(e);
    }
    for (j, expr) in exprs.iter().enumerate() {
        let (kind, vals) = &p.rows[j];
        let (lo, hi) = match (*kind, vals.as_slice()) {
            (0, x) if x.len() == 2 => (x[0], x[1]),
            (1, x) if x.len() == 1 => (f64::NEG_INFINITY, x[0]),
            (2, x) if x.len() == 1 => (x[0], f64::INFINITY),
            (3, _) => (f64::NEG_INFINITY, f64::INFINITY),
            (4, x) if x.len() == 1 => (x[0], x[0]),
            _ => return Err(invalid("r", "invalid row type")),
        };
        let name = rows.get(j).cloned().unwrap_or_else(|| format!("c{j}"));
        m.__add_constraint_interval(name, *expr, lo, hi);
    }
    if h.n_obj == 0 {
        if p.objective.is_some() {
            return Err(invalid("O", "objective record present with zero objectives"));
        }
        m.__feasibility();
    } else {
        let (sense, node) = p.objective.ok_or_else(|| invalid("O", "missing objective record"))?;
        let mut e = lower(&m, &vars, node, 0)?;
        for &(v, c) in &p.grad {
            if v >= vars.len() {
                return Err(invalid("G", "variable index out of range"));
            }
            e = e + c * vars[v];
        }
        match sense {
            ObjectiveSense::Minimize => m.__minimize(e),
            ObjectiveSense::Maximize => m.__maximize(e),
        }
    }
    Ok(m)
}

fn lower<'a>(m: &'a Model, vars: &[Expr<'a>], n: Node, depth: usize) -> Result<Expr<'a>, IoError> {
    match n {
        Node::Const(x) => Ok(m.__constant(x)),
        Node::Var(i) => {
            vars.get(i).copied().ok_or_else(|| invalid("expression", "variable index out of range"))
        }
        Node::Unary(c, a) => {
            let child_depth = next_expression_depth(depth)?;
            let x = lower(m, vars, *a, child_depth)?;
            Ok(match c {
                15 => x.abs(),
                16 => -x,
                41 => x.sin(),
                43 => x.log(),
                44 => x.exp(),
                46 => x.cos(),
                _ => return Err(invalid("expression", "unsupported unary opcode")),
            })
        }
        Node::Binary(c, a, b) => {
            let child_depth = next_expression_depth(depth)?;
            let x = lower(m, vars, *a, child_depth)?;
            let y = lower(m, vars, *b, child_depth)?;
            Ok(match c {
                0 => x + y,
                1 => x - y,
                2 => x * y,
                3 => x / y,
                5 => x.pow(y),
                _ => return Err(invalid("expression", "unsupported binary opcode")),
            })
        }
        Node::Sum(xs) => {
            let mut it = xs.into_iter();
            let Some(first) = it.next() else { return Ok(m.__constant(0.0)) };
            let child_depth = next_expression_depth(depth)?;
            let mut e = lower(m, vars, first, child_depth)?;
            for x in it {
                e = e + lower(m, vars, x, child_depth)?;
            }
            Ok(e)
        }
    }
}

fn domains(h: &Header) -> Vec<Domain> {
    let mut d = Vec::with_capacity(h.n_var);
    let both = h.nl_both;
    let c = h.nl_c.saturating_sub(both);
    let o = h.nl_o.saturating_sub(both);
    for (n, int) in [(both, h.nl_int_b), (c, h.nl_int_c), (o, h.nl_int_o)] {
        d.extend(std::iter::repeat_n(Domain::Real, n.saturating_sub(int)));
        d.extend(std::iter::repeat_n(Domain::Integer, int.min(n)));
    }
    let linear = h.n_var.saturating_sub(d.len() + h.n_bin + h.n_int);
    d.extend(std::iter::repeat_n(Domain::Real, linear));
    d.extend(std::iter::repeat_n(Domain::Binary, h.n_bin));
    d.extend(std::iter::repeat_n(Domain::Integer, h.n_int));
    d.truncate(h.n_var);
    d
}

fn record(lines: &[String], i: &mut usize, section: &str) -> Result<Vec<f64>, IoError> {
    if *i >= lines.len() {
        return Err(invalid(section, "truncated"));
    }
    let (s, _) = split_comment(&lines[*i]);
    *i += 1;
    s.split_whitespace().map(parse_float).collect()
}
fn decode_bound(v: &[f64]) -> Result<(f64, f64), IoError> {
    if v.is_empty() {
        return Err(invalid("b", "empty bound"));
    }
    let kind = as_index(v[0])?;
    Ok(match (kind, v.get(1).copied(), v.get(2).copied()) {
        (0, Some(a), Some(b)) => (a, b),
        (1, _, Some(b)) => (f64::NEG_INFINITY, b),
        (2, Some(a), _) => (a, f64::INFINITY),
        (3, _, _) => (f64::NEG_INFINITY, f64::INFINITY),
        (4, Some(a), _) => (a, a),
        _ => return Err(invalid("b", "invalid bound type")),
    })
}
fn parse_suffix(s: &str, c: char) -> Result<usize, IoError> {
    s.strip_prefix(c)
        .ok_or_else(|| invalid("segment", "bad header"))?
        .split_whitespace()
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| invalid("segment", "bad count"))
}
#[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn as_index(x: f64) -> Result<usize, IoError> {
    if !x.is_finite() || x < 0.0 || x.fract() != 0.0 {
        return Err(invalid("index", "not a nonnegative integer"));
    }
    if x > usize::MAX as f64 {
        return Err(invalid("index", "index out of range"));
    }
    Ok(x as usize)
}
fn nonneg(x: i32, field: &str) -> Result<usize, IoError> {
    if x < 0 {
        return Err(invalid("binary", format!("negative {field}")));
    }
    usize::try_from(
        u32::try_from(x).map_err(|_| invalid("binary", format!("{field} out of range")))?,
    )
    .map_err(|_| invalid("binary", format!("{field} out of range")))
}
fn parse_float(s: &str) -> Result<f64, IoError> {
    s.parse().map_err(|_| invalid("number", format!("invalid number {s}")))
}
fn split_comment(s: &str) -> (&str, Option<&str>) {
    match s.split_once('#') {
        Some((a, b)) => (a.trim(), Some(b.trim())),
        None => (s.trim(), None),
    }
}
fn ensure_unique(xs: &[String], what: &str) -> Result<(), IoError> {
    let mut seen = std::collections::HashSet::new();
    for x in xs {
        if !seen.insert(x) {
            return Err(invalid("sidecar", format!("duplicate {what} name {x}")));
        }
    }
    Ok(())
}
fn invalid(section: &str, message: impl Into<String>) -> IoError {
    IoError::InvalidNl { section: section.into(), message: message.into() }
}

fn bounded_capacity(requested: usize, remaining: usize) -> usize {
    requested.min(remaining)
}

fn next_expression_depth(depth: usize) -> Result<usize, IoError> {
    let next = depth.saturating_add(1);
    if next > MAX_EXPRESSION_DEPTH {
        Err(invalid("expression", "nesting too deep"))
    } else {
        Ok(next)
    }
}

fn start_values(starts: &[(usize, f64)], n_var: usize) -> Vec<Option<f64>> {
    let mut values = vec![None; n_var];
    for &(index, value) in starts {
        if let Some(slot) = values.get_mut(index) {
            if slot.is_none() {
                *slot = Some(value);
            }
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::Relate;

    fn header(n_var: usize, n_con: usize, n_obj: usize) -> String {
        format!(
            "g3 1 1 0\n {n_var} {n_con} {n_obj} 0 0\n 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 0 0 0 0 0\n 0 0\n 0 0\n 0 0 0 0 0\n"
        )
    }

    fn assert_invalid(result: Result<Model, IoError>, section: &str, message: &str) {
        match result {
            Err(IoError::InvalidNl { section: got_section, message: got_message }) => {
                assert_eq!(got_section, section);
                assert_eq!(got_message, message);
            }
            other => panic!("expected InvalidNl, got {other:?}"),
        }
    }

    fn assert_unsupported(result: Result<Model, IoError>, section: &str, feature: &str) {
        match result {
            Err(IoError::UnsupportedNl { section: got_section, feature: got_feature }) => {
                assert_eq!(got_section, section);
                assert_eq!(got_feature, feature);
            }
            other => panic!("expected UnsupportedNl, got {other:?}"),
        }
    }

    #[test]
    fn reads_netlib_diet_with_sidecars() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/nl/NETLIB/diet/diet.nl");
        let m = read_nl_file(path).expect("diet NL should parse");
        assert_eq!(m.num_variables(), 8);
        assert_eq!(m.num_constraints(), 4);
        assert_eq!(m.try_objective().unwrap().sense, ObjectiveSense::Minimize);
        assert_eq!(m.variables()[0].name, "Buy['BEEF']");
    }

    #[test]
    fn rejects_invalid_sidecar_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("names.col");
        std::fs::write(&path, "x\0\n").unwrap();
        match read_names(&path, "column") {
            Err(IoError::InvalidNl { section, message }) => {
                assert_eq!(section, "sidecar");
                assert_eq!(message, "column name contains a control character");
            }
            other => panic!("expected InvalidNl, got {other:?}"),
        }
    }

    #[test]
    fn reads_minlp_fixtures() {
        for rel in [
            "tests/fixtures/nl/MINLPlib/alkyl/alkyl.nl",
            "tests/fixtures/nl/MINLPlib/ann_compressor_exp/ann_compressor_exp.nl.txt",
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
            let m = read_nl_file(path).expect("MINLP fixture should parse");
            assert!(m.num_variables() > 0);
            assert!(m.num_constraints() > 0);
            assert!(!matches!(m.kind(), oximo_core::ModelKind::LP | oximo_core::ModelKind::MILP));
        }
    }

    #[test]
    fn reads_binary_writer_output() {
        let m = Model::new("binary");
        let x = m.__var("x").bounds(0.0, 10.0).build();
        let y = m.__var("y").bounds(-2.0, 2.0).build();
        m.__add_constraint("row", (x.sin() + 2.0 * y).le(4.0));
        m.__minimize(x.powi(2) + y);
        let mut bytes = Vec::new();
        super::super::write_nl_with(&m, &mut bytes, &super::super::WriteOptions::binary()).unwrap();
        let got = read_nl(&bytes[..]).expect("binary output should parse");
        assert_eq!(got.num_variables(), 2);
        assert_eq!(got.num_constraints(), 1);
        assert_eq!(got.try_objective().unwrap().sense, ObjectiveSense::Minimize);
    }

    #[test]
    fn preserves_range_rows() {
        let m = Model::new("range");
        let x = m.__var("x").bounds(-10.0, 10.0).build();
        m.__add_range("band", x, -1.0, 2.0);
        m.__minimize(x);
        let text = super::super::to_nl_string(&m).unwrap();
        let got = read_nl(text.as_bytes()).unwrap();
        assert_eq!(got.num_constraints(), 1);
        assert!(got.constraints().algebraic()[0].is_range());
    }

    #[test]
    fn restores_integer_domain_buckets() {
        let m = Model::new("integer");
        let x = m.__var("x").binary().build();
        let y = m.__var("y").integer().bounds(-3.0, 3.0).build();
        m.__add_constraint("row", (x + y).ge(0.0));
        m.__minimize(x + y);
        let text = super::super::to_nl_string(&m).unwrap();
        let got = read_nl(text.as_bytes()).unwrap();
        assert!(matches!(got.variables()[0].domain, Domain::Binary));
        assert!(matches!(got.variables()[1].domain, Domain::Integer));
    }

    #[test]
    fn rejects_truncated_header() {
        assert_invalid(read_nl("g3 1 1 0\n".as_bytes()), "header", "truncated header");
    }

    #[test]
    fn rejects_unknown_segment() {
        let text = format!("{}Z\n", header(0, 0, 0));
        assert_invalid(read_nl(text.as_bytes()), "body", "unknown segment Z");
    }

    #[test]
    fn rejects_out_of_range_j_row() {
        let text = format!("{}J 1 0\n", header(0, 1, 0));
        assert_invalid(read_nl(text.as_bytes()), "J", "row index out of range");
    }

    #[test]
    fn rejects_unsupported_opcode() {
        let text = format!("{}O 0 0\no999\n", header(0, 0, 1));
        assert_unsupported(read_nl(text.as_bytes()), "expression", "opcode 999");
    }

    #[test]
    fn rejects_deep_expression_nesting() {
        let mut text = format!("{}O 0 0\n", header(0, 0, 1));
        for _ in 0..=MAX_EXPRESSION_DEPTH {
            text.push_str("o16\n");
        }
        text.push_str("n0\n");
        assert_invalid(read_nl(text.as_bytes()), "expression", "nesting too deep");
    }

    #[test]
    fn rejects_unsupported_function_and_defined_variable_sections() {
        for (tag, feature) in [('F', "imported functions"), ('V', "defined variables")] {
            let text = format!("{}{}\n", header(0, 0, 0), tag);
            assert_unsupported(read_nl(text.as_bytes()), &tag.to_string(), feature);
        }
    }

    #[test]
    fn reads_feasibility_header_without_objective() {
        let m = read_nl(header(0, 0, 0).as_bytes()).expect("feasibility NL should parse");
        assert!(m.is_feasibility());
        assert!(m.try_objective().is_err());
    }
}
