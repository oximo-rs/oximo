//! MPS file format import and export.
//!
//! MPS is a widely supported text format for linear and quadratic optimization
//! problems. It is a common lingua franca for exchanging models between tools.
//!
//! [`read_mps`] and [`read_mps_file`] import whitespace-delimited MPS files.
//! [`write_mps`] exports linear and quadratic models to any `std::io::Write`.
//!
//! References:
//! - "MPS file format," lp_solve. <https://lpsolve.sourceforge.net/5.5/mps-format.htm> (accessed May 09, 2026).
//! - Gurobi, "Model File Formats." <https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html>
//! - IBM ILOG CPLEX, "Quadratically constrained programs (QCP) in MPS files." <https://www.ibm.com/docs/en/cofz/12.9.0?topic=extensions-quadratically-constrained-programs-qcp-in-mps-files>

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use oximo_core::{
    Constraint, Domain, Model, ModelKind, ObjectiveSense, Sense, SosConstraint, SosType, var_name,
};
use oximo_expr::{Expr, QuadraticTerms, VarId, describe_nonlinear_term, extract_quadratic};
use rustc_hash::FxHashMap;

use crate::error::IoError;

/// Coefficient convention used by quadratic-constraint MPS sections.
///
/// Objective sections always use the conventional `0.5 * x' Q x` scaling.
/// Gurobi-style `QCMATRIX` and constraint `QSECTION` records instead encode
/// polynomial coefficients directly, whereas CPLEX-style records use the
/// objective/Hessian convention.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MpsQuadraticFormat {
    /// Gurobi-compatible quadratic-constraint scaling.
    #[default]
    Gurobi,
    /// CPLEX-compatible quadratic-constraint scaling.
    Cplex,
    /// MOSEK-compatible `QSECTION` layout and scaling.
    Mosek,
}

/// Options controlling MPS import.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MpsReadOptions {
    /// How ambiguous quadratic-constraint matrix coefficients are scaled.
    pub quadratic_format: MpsQuadraticFormat,
}

/// Options controlling MPS export.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MpsWriteOptions {
    /// Solver-compatible layout and scaling for quadratic sections.
    pub quadratic_format: MpsQuadraticFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowKind {
    Free,
    Greater,
    Less,
    Equal,
}

struct ParsedRow {
    name: String,
    kind: RowKind,
    lower: f64,
    upper: f64,
    linear: FxHashMap<usize, f64>,
    quadratic: FxHashMap<(usize, usize), f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColumnKind {
    Continuous,
    Integer,
    Binary,
    SemiContinuous,
    SemiInteger,
}

struct ParsedColumn {
    name: String,
    lower: f64,
    upper: f64,
    kind: ColumnKind,
    default_bounds: bool,
    lower_explicit: bool,
    marker_placement: MarkerPlacement,
}

#[derive(Default)]
struct QuadraticTriangle {
    upper: Option<f64>,
    lower: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MarkerPlacement {
    #[default]
    Unseen,
    Outside,
    Inside,
    Mixed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum SectionRank {
    Name,
    ObjSense,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Quadratic,
    Sos,
    End,
}

#[derive(Clone, Debug)]
enum Section {
    Name,
    ObjSense,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    QuadObj,
    QMatrix,
    QcMatrix(String),
    QSec(String),
    Sos,
    End,
}

impl Section {
    fn rank(&self) -> SectionRank {
        match self {
            Self::Name => SectionRank::Name,
            Self::ObjSense => SectionRank::ObjSense,
            Self::Rows => SectionRank::Rows,
            Self::Columns => SectionRank::Columns,
            Self::Rhs => SectionRank::Rhs,
            Self::Ranges => SectionRank::Ranges,
            Self::Bounds => SectionRank::Bounds,
            Self::QuadObj | Self::QMatrix | Self::QcMatrix(_) | Self::QSec(_) => {
                SectionRank::Quadratic
            }
            Self::Sos => SectionRank::Sos,
            Self::End => SectionRank::End,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Name => "NAME",
            Self::ObjSense => "OBJSENSE",
            Self::Rows => "ROWS",
            Self::Columns => "COLUMNS",
            Self::Rhs => "RHS",
            Self::Ranges => "RANGES",
            Self::Bounds => "BOUNDS",
            Self::QuadObj => "QUADOBJ",
            Self::QMatrix => "QMATRIX",
            Self::QcMatrix(_) => "QCMATRIX",
            Self::QSec(_) => "QSECTION",
            Self::Sos => "SOS",
            Self::End => "ENDATA",
        }
    }
}

struct Field<'a> {
    text: &'a str,
    column: usize,
}

struct ParsedMps {
    name: String,
    objective_row: Option<String>,
    sense: Option<ObjectiveSense>,
    legacy_sense: Option<ObjectiveSense>,
    rows: Vec<ParsedRow>,
    row_index: FxHashMap<String, usize>,
    columns: Vec<ParsedColumn>,
    column_index: FxHashMap<String, usize>,
    objective_linear: Vec<f64>,
    objective_quadratic: FxHashMap<(usize, usize), f64>,
    quadratic_triangles: FxHashMap<(Option<usize>, usize, usize), QuadraticTriangle>,
    objective_constant: f64,
    intorg: bool,
    rhs_vector: Option<String>,
    range_vector: Option<String>,
    bounds_vector: Option<String>,
    objective_quadratic_source: Option<&'static str>,
    quadratic_rows: HashSet<String>,
    sos: Vec<ParsedSos>,
    current_sos: Option<usize>,
    seen_sections: u8,
}

struct ParsedSos {
    name: String,
    sos_type: SosType,
    members: Vec<(String, f64)>,
}

const MPS_INFINITY_SENTINEL: f64 = 1e30;
const SEEN_ROWS: u8 = 1;
const SEEN_COLUMNS: u8 = 2;
const SEEN_END: u8 = 4;

impl ParsedMps {
    fn new(fallback_name: &str) -> Self {
        Self {
            name: fallback_name.to_owned(),
            objective_row: None,
            sense: None,
            legacy_sense: None,
            rows: Vec::new(),
            row_index: FxHashMap::default(),
            columns: Vec::new(),
            column_index: FxHashMap::default(),
            objective_linear: Vec::new(),
            objective_quadratic: FxHashMap::default(),
            quadratic_triangles: FxHashMap::default(),
            objective_constant: 0.0,
            intorg: false,
            rhs_vector: None,
            range_vector: None,
            bounds_vector: None,
            objective_quadratic_source: None,
            quadratic_rows: HashSet::new(),
            sos: Vec::new(),
            current_sos: None,
            seen_sections: 0,
        }
    }

    fn add_column(&mut self, name: &str, inside_marker: bool) -> usize {
        let index = if let Some(index) = self.column_index.get(name) {
            *index
        } else {
            let index = self.columns.len();
            self.columns.push(ParsedColumn {
                name: name.to_owned(),
                lower: 0.0,
                upper: f64::INFINITY,
                kind: ColumnKind::Continuous,
                default_bounds: true,
                lower_explicit: false,
                marker_placement: MarkerPlacement::Unseen,
            });
            self.column_index.insert(name.to_owned(), index);
            self.objective_linear.push(0.0);
            index
        };
        let column = &mut self.columns[index];
        if inside_marker {
            column.marker_placement = match column.marker_placement {
                MarkerPlacement::Unseen | MarkerPlacement::Inside => MarkerPlacement::Inside,
                MarkerPlacement::Outside | MarkerPlacement::Mixed => MarkerPlacement::Mixed,
            };
            column.kind = ColumnKind::Integer;
            if column.default_bounds {
                column.upper = 1.0;
            }
        } else {
            column.marker_placement = match column.marker_placement {
                MarkerPlacement::Unseen | MarkerPlacement::Outside => MarkerPlacement::Outside,
                MarkerPlacement::Inside | MarkerPlacement::Mixed => MarkerPlacement::Mixed,
            };
        }
        index
    }
}

fn invalid_mps(line: usize, column: usize, message: impl Into<String>) -> IoError {
    IoError::InvalidMps { line, column, message: message.into() }
}

fn fields(line: &str) -> Vec<Field<'_>> {
    let mut out = Vec::new();
    let mut start = None;
    for (offset, ch) in line.char_indices() {
        if ch.is_whitespace() {
            if let Some(begin) = start.take() {
                out.push(Field { text: &line[begin..offset], column: begin + 1 });
            }
        } else if start.is_none() {
            start = Some(offset);
        }
    }
    if let Some(begin) = start {
        out.push(Field { text: &line[begin..], column: begin + 1 });
    }
    out
}

fn parse_number(field: &Field<'_>, line: usize) -> Result<f64, IoError> {
    let normalized;
    let text = if field.text.contains(['d', 'D']) {
        normalized = field.text.replace(['d', 'D'], "E");
        normalized.as_str()
    } else {
        field.text
    };
    let value = text
        .parse::<f64>()
        .map_err(|_| invalid_mps(line, field.column, format!("invalid number {:?}", field.text)))?;
    if !value.is_finite() {
        return Err(invalid_mps(line, field.column, "numeric fields must be finite"));
    }
    Ok(value)
}

fn looks_like_number(text: &str) -> bool {
    text.replace(['d', 'D'], "E").parse::<f64>().is_ok()
}

fn objective_sense(field: &Field<'_>, line: usize) -> Result<ObjectiveSense, IoError> {
    match field.text.to_ascii_uppercase().as_str() {
        "MIN" | "MINIMIZE" => Ok(ObjectiveSense::Minimize),
        "MAX" | "MAXIMIZE" => Ok(ObjectiveSense::Maximize),
        _ => Err(invalid_mps(line, field.column, "objective sense must be MIN or MAX")),
    }
}

fn parse_legacy_sense(line: &str) -> Option<ObjectiveSense> {
    let comment = line.trim_start_matches('*').trim();
    let value = comment.strip_prefix("sense:").or_else(|| comment.strip_prefix("SENSE:"))?;
    match value.trim().to_ascii_lowercase().as_str() {
        "minimize" | "min" => Some(ObjectiveSense::Minimize),
        "maximize" | "max" => Some(ObjectiveSense::Maximize),
        _ => None,
    }
}

fn header(items: &[Field<'_>], current: &Section) -> Option<Section> {
    let first = items.first()?.text.to_ascii_uppercase();
    match (first.as_str(), items.len()) {
        ("NAME", _) if matches!(current, Section::Name) => Some(Section::Name),
        ("OBJSENSE", 1 | 2) => Some(Section::ObjSense),
        ("ROWS", 1) => Some(Section::Rows),
        ("COLUMNS", 1) => Some(Section::Columns),
        ("RHS", 1) => Some(Section::Rhs),
        ("RANGES", 1) => Some(Section::Ranges),
        ("BOUNDS", 1) => Some(Section::Bounds),
        ("QUADOBJ", 1) => Some(Section::QuadObj),
        ("QMATRIX", 1) => Some(Section::QMatrix),
        ("QCMATRIX", 2) => Some(Section::QcMatrix(items[1].text.to_owned())),
        ("QSECTION", 2) => Some(Section::QSec(items[1].text.to_owned())),
        ("SOS", 1) => Some(Section::Sos),
        ("ENDATA", 1) => Some(Section::End),
        _ => None,
    }
}

fn select_vector(
    selected: &mut Option<String>,
    candidate: Option<&Field<'_>>,
    section: &str,
) -> Result<(), IoError> {
    let Some(candidate) = candidate else { return Ok(()) };
    if let Some(existing) = selected {
        if existing != candidate.text {
            return Err(IoError::UnsupportedMps {
                section: section.into(),
                feature: format!(
                    "multiple data vectors ({existing:?} and {:?}) are not supported",
                    candidate.text
                ),
            });
        }
    } else {
        *selected = Some(candidate.text.to_owned());
    }
    Ok(())
}

fn parse_row(data: &mut ParsedMps, items: &[Field<'_>], line: usize) -> Result<(), IoError> {
    if items.len() < 2 {
        return Err(invalid_mps(line, 1, "ROWS records require a sense and row name"));
    }
    let name = items[1].text;
    if data.objective_row.as_deref() == Some(name) || data.row_index.contains_key(name) {
        return Err(invalid_mps(line, items[1].column, format!("duplicate row name {name:?}")));
    }
    let kind = match items[0].text.to_ascii_uppercase().as_str() {
        "N" => RowKind::Free,
        "G" => RowKind::Greater,
        "L" => RowKind::Less,
        "E" => RowKind::Equal,
        _ => {
            return Err(invalid_mps(line, items[0].column, "row sense must be N, G, L, or E"));
        }
    };
    if kind == RowKind::Free && data.objective_row.is_none() {
        data.objective_row = Some(name.to_owned());
        return Ok(());
    }
    let (lower, upper) = match kind {
        RowKind::Free => (f64::NEG_INFINITY, f64::INFINITY),
        RowKind::Greater => (0.0, f64::INFINITY),
        RowKind::Less => (f64::NEG_INFINITY, 0.0),
        RowKind::Equal => (0.0, 0.0),
    };
    let index = data.rows.len();
    data.rows.push(ParsedRow {
        name: name.to_owned(),
        kind,
        lower,
        upper,
        linear: FxHashMap::default(),
        quadratic: FxHashMap::default(),
    });
    data.row_index.insert(name.to_owned(), index);
    Ok(())
}

fn parse_coefficient(
    data: &mut ParsedMps,
    column: usize,
    row: &Field<'_>,
    value: &Field<'_>,
    line: usize,
) -> Result<(), IoError> {
    let value = parse_number(value, line)?;
    if data.objective_row.as_deref() == Some(row.text) {
        data.objective_linear[column] += value;
        return Ok(());
    }
    let row_index = data.row_index.get(row.text).copied().ok_or_else(|| {
        invalid_mps(line, row.column, format!("unknown ROWS name {:?}", row.text))
    })?;
    *data.rows[row_index].linear.entry(column).or_insert(0.0) += value;
    Ok(())
}

fn unquote_marker(value: &str) -> String {
    value.trim_matches(|c| c == '\'' || c == '"').to_ascii_uppercase()
}

fn parse_columns(data: &mut ParsedMps, items: &[Field<'_>], line: usize) -> Result<(), IoError> {
    if items.len() == 3 && unquote_marker(items[1].text) == "MARKER" {
        match unquote_marker(items[2].text).as_str() {
            "INTORG" if !data.intorg => data.intorg = true,
            "INTEND" if data.intorg => data.intorg = false,
            "INTORG" => return Err(invalid_mps(line, items[2].column, "nested INTORG marker")),
            "INTEND" => {
                return Err(invalid_mps(line, items[2].column, "INTEND without INTORG"));
            }
            _ => return Err(invalid_mps(line, items[2].column, "unknown MARKER value")),
        }
        return Ok(());
    }
    if items.len() != 3 && items.len() != 5 {
        return Err(invalid_mps(line, 1, "COLUMNS records require three or five fields"));
    }
    let column = data.add_column(items[0].text, data.intorg);
    let column_data = &data.columns[column];
    if column_data.marker_placement == MarkerPlacement::Mixed {
        return Err(invalid_mps(
            line,
            items[0].column,
            format!("integer column {:?} also appears outside INTORG/INTEND", items[0].text),
        ));
    }
    parse_coefficient(data, column, &items[1], &items[2], line)?;
    if items.len() == 5 {
        parse_coefficient(data, column, &items[3], &items[4], line)?;
    }
    Ok(())
}

fn parse_rhs_value(
    data: &mut ParsedMps,
    row: &Field<'_>,
    value: &Field<'_>,
    line: usize,
) -> Result<(), IoError> {
    let value = parse_number(value, line)?;
    if data.objective_row.as_deref() == Some(row.text) {
        data.objective_constant = -value;
        return Ok(());
    }
    let index = data.row_index.get(row.text).copied().ok_or_else(|| {
        invalid_mps(line, row.column, format!("unknown ROWS name {:?}", row.text))
    })?;
    let parsed_row = &mut data.rows[index];
    match parsed_row.kind {
        RowKind::Greater => parsed_row.lower = value,
        RowKind::Less => parsed_row.upper = value,
        RowKind::Equal => {
            parsed_row.lower = value;
            parsed_row.upper = value;
        }
        RowKind::Free => {
            return Err(invalid_mps(line, row.column, "a free N row cannot have an RHS"));
        }
    }
    Ok(())
}

fn parse_rhs(data: &mut ParsedMps, items: &[Field<'_>], line: usize) -> Result<(), IoError> {
    let (vector, pairs): (Option<&Field<'_>>, &[Field<'_>]) = match items.len() {
        2 | 4 => (None, items),
        3 | 5 => (Some(&items[0]), &items[1..]),
        _ => return Err(invalid_mps(line, 1, "RHS records require two to five fields")),
    };
    select_vector(&mut data.rhs_vector, vector, "RHS")?;
    for pair in pairs.as_chunks::<2>().0 {
        parse_rhs_value(data, &pair[0], &pair[1], line)?;
    }
    Ok(())
}

fn parse_range_value(
    data: &mut ParsedMps,
    row: &Field<'_>,
    value: &Field<'_>,
    line: usize,
) -> Result<(), IoError> {
    let value = parse_number(value, line)?;
    let index = data.row_index.get(row.text).copied().ok_or_else(|| {
        invalid_mps(line, row.column, format!("unknown ROWS name {:?}", row.text))
    })?;
    let parsed_row = &mut data.rows[index];
    match parsed_row.kind {
        RowKind::Greater => parsed_row.upper = parsed_row.lower + value.abs(),
        RowKind::Less => parsed_row.lower = parsed_row.upper - value.abs(),
        RowKind::Equal if value >= 0.0 => parsed_row.upper = parsed_row.lower + value,
        RowKind::Equal => parsed_row.lower = parsed_row.upper + value,
        RowKind::Free => {
            return Err(invalid_mps(line, row.column, "a free N row cannot have a range"));
        }
    }
    Ok(())
}

fn parse_ranges(data: &mut ParsedMps, items: &[Field<'_>], line: usize) -> Result<(), IoError> {
    let (vector, pairs): (Option<&Field<'_>>, &[Field<'_>]) = match items.len() {
        2 | 4 => (None, items),
        3 | 5 => (Some(&items[0]), &items[1..]),
        _ => return Err(invalid_mps(line, 1, "RANGES records require two to five fields")),
    };
    select_vector(&mut data.range_vector, vector, "RANGES")?;
    for pair in pairs.as_chunks::<2>().0 {
        parse_range_value(data, &pair[0], &pair[1], line)?;
    }
    Ok(())
}

fn parse_bound(data: &mut ParsedMps, items: &[Field<'_>], line: usize) -> Result<(), IoError> {
    if !(2..=4).contains(&items.len()) {
        return Err(invalid_mps(line, 1, "BOUNDS records require two to four fields"));
    }
    let bound_type = items[0].text.to_ascii_uppercase();
    let requires_value =
        matches!(bound_type.as_str(), "FX" | "UP" | "LO" | "LI" | "UI" | "SC" | "SI");
    let (vector, column_field, value_field) = match (items.len(), requires_value) {
        (2, _) => (None, &items[1], None),
        (3, true) => (None, &items[1], Some(&items[2])),
        (3, false) => (Some(&items[1]), &items[2], None),
        (4, _) => (Some(&items[1]), &items[2], Some(&items[3])),
        _ => unreachable!(),
    };
    select_vector(&mut data.bounds_vector, vector, "BOUNDS")?;
    let column_index = data.column_index.get(column_field.text).copied().ok_or_else(|| {
        invalid_mps(line, column_field.column, format!("unknown column {:?}", column_field.text))
    })?;
    let value = value_field
        .map(|field| {
            parse_number(field, line)
                .map(|value| if value >= MPS_INFINITY_SENTINEL { f64::INFINITY } else { value })
        })
        .transpose()?;
    let column = &mut data.columns[column_index];
    if column.default_bounds && column.kind == ColumnKind::Integer {
        column.upper = f64::INFINITY;
    }
    column.default_bounds = false;
    match (bound_type.as_str(), value) {
        ("PL", None) => column.upper = f64::INFINITY,
        ("MI", None) => {
            column.lower = f64::NEG_INFINITY;
            column.lower_explicit = true;
        }
        ("FR", None | Some(_)) => {
            column.lower = f64::NEG_INFINITY;
            column.upper = f64::INFINITY;
            column.lower_explicit = true;
        }
        ("BV", None | Some(_)) => {
            column.lower = 0.0;
            column.upper = 1.0;
            column.kind = ColumnKind::Binary;
            column.lower_explicit = true;
        }
        ("FX", Some(value)) => {
            column.lower = value;
            column.upper = value;
            column.lower_explicit = true;
        }
        ("UP", Some(value)) => {
            if value < 0.0 && !column.lower_explicit {
                column.lower = f64::NEG_INFINITY;
            }
            column.upper = value;
        }
        ("LO", Some(value)) => {
            column.lower = value;
            column.lower_explicit = true;
        }
        ("LI", Some(value)) => {
            column.lower = value;
            column.kind = ColumnKind::Integer;
            column.lower_explicit = true;
        }
        ("UI", Some(value)) => {
            column.upper = value;
            column.kind = ColumnKind::Integer;
        }
        ("SC", Some(value)) => {
            if !column.lower_explicit {
                column.lower = 1.0;
            }
            column.upper = value;
            column.kind = ColumnKind::SemiContinuous;
        }
        ("SI", Some(value)) => {
            if !column.lower_explicit {
                column.lower = 1.0;
            }
            column.upper = value;
            column.kind = ColumnKind::SemiInteger;
        }
        _ => {
            return Err(invalid_mps(
                line,
                items[0].column,
                format!("invalid {bound_type} bound record"),
            ));
        }
    }
    Ok(())
}

fn quadratic_coefficient(
    format: MpsQuadraticFormat,
    diagonal: bool,
    objective: bool,
    value: f64,
) -> f64 {
    if objective || format != MpsQuadraticFormat::Gurobi {
        if diagonal { value / 2.0 } else { value }
    } else if diagonal {
        value
    } else {
        2.0 * value
    }
}

fn qsection_targets_objective(data: &ParsedMps, name: &str, line: usize) -> Result<bool, IoError> {
    if name == "OBJ" {
        if data.objective_row.as_deref() != Some("OBJ") && data.row_index.contains_key("OBJ") {
            return Err(invalid_mps(
                line,
                1,
                "QSECTION OBJ is ambiguous because OBJ is also a constraint row",
            ));
        }
        return Ok(true);
    }
    Ok(data.objective_row.as_deref() == Some(name))
}

fn parse_quadratic_record(
    data: &mut ParsedMps,
    section: &Section,
    items: &[Field<'_>],
    line: usize,
    options: MpsReadOptions,
) -> Result<(), IoError> {
    if items.len() != 3 {
        return Err(invalid_mps(line, 1, "quadratic records require three fields"));
    }
    let left = data.column_index.get(items[0].text).copied().ok_or_else(|| {
        invalid_mps(line, items[0].column, format!("unknown column {:?}", items[0].text))
    })?;
    let right = data.column_index.get(items[1].text).copied().ok_or_else(|| {
        invalid_mps(line, items[1].column, format!("unknown column {:?}", items[1].text))
    })?;
    let pair = if left <= right { (left, right) } else { (right, left) };
    let value = parse_number(&items[2], line)?;
    let objective = match section {
        Section::QuadObj | Section::QMatrix => true,
        Section::QSec(name) => qsection_targets_objective(data, name, line)?,
        Section::QcMatrix(_) => false,
        _ => unreachable!("quadratic parser called outside quadratic section"),
    };
    let row = if objective {
        None
    } else {
        let (Section::QcMatrix(row_name) | Section::QSec(row_name)) = section else {
            unreachable!()
        };
        Some(data.row_index.get(row_name).copied().ok_or_else(|| {
            invalid_mps(line, 1, format!("quadratic section names unknown row {row_name:?}"))
        })?)
    };
    let coefficient =
        quadratic_coefficient(options.quadratic_format, left == right, objective, value);
    let mut store_coefficient = true;
    if matches!(section, Section::QMatrix | Section::QcMatrix(_)) && left != right {
        let triangle = data.quadratic_triangles.entry((row, pair.0, pair.1)).or_default();
        let first_record = triangle.upper.is_none() && triangle.lower.is_none();
        let side = if left < right { &mut triangle.upper } else { &mut triangle.lower };
        let same_side = side.is_some();
        *side = Some(side.take().unwrap_or(0.0) + coefficient);
        store_coefficient = first_record || same_side;
    }
    if !store_coefficient {
        return Ok(());
    }
    if objective {
        *data.objective_quadratic.entry(pair).or_insert(0.0) += coefficient;
        return Ok(());
    }
    *data.rows[row.expect("quadratic constraint row")].quadratic.entry(pair).or_insert(0.0) +=
        coefficient;
    Ok(())
}

fn validate_quadratic_triangles(data: &ParsedMps, line: usize) -> Result<(), IoError> {
    for ((_, left, right), triangle) in &data.quadratic_triangles {
        if let (Some(upper), Some(lower)) = (triangle.upper, triangle.lower)
            && upper.total_cmp(&lower).is_ne()
        {
            return Err(invalid_mps(
                line,
                1,
                format!(
                    "asymmetric quadratic matrix entries for columns {:?} and {:?}",
                    data.columns[*left].name, data.columns[*right].name
                ),
            ));
        }
    }
    Ok(())
}

fn begin_quadratic_section(
    data: &mut ParsedMps,
    section: &Section,
    line: usize,
) -> Result<(), IoError> {
    let source = section.name();
    let objective = match section {
        Section::QuadObj | Section::QMatrix => true,
        Section::QSec(name) => qsection_targets_objective(data, name, line)?,
        Section::QcMatrix(_) => false,
        _ => return Ok(()),
    };
    if objective {
        if let Some(existing) = data.objective_quadratic_source {
            return Err(invalid_mps(
                line,
                1,
                format!("objective quadratic data already supplied by {existing}"),
            ));
        }
        data.objective_quadratic_source = Some(source);
        return Ok(());
    }
    let (Section::QcMatrix(row_name) | Section::QSec(row_name)) = section else { unreachable!() };
    if !data.row_index.contains_key(row_name) {
        return Err(invalid_mps(
            line,
            1,
            format!("quadratic section names unknown row {row_name:?}"),
        ));
    }
    if !data.quadratic_rows.insert(row_name.clone()) {
        return Err(invalid_mps(
            line,
            1,
            format!("duplicate quadratic section for row {row_name:?}"),
        ));
    }
    Ok(())
}

fn check_section_transition(
    previous: &Section,
    next: &Section,
    line: usize,
) -> Result<(), IoError> {
    let quadratic_after_sos = matches!(previous, Section::Sos)
        && matches!(
            next,
            Section::QuadObj | Section::QMatrix | Section::QcMatrix(_) | Section::QSec(_)
        );
    if next.rank() < previous.rank() && !quadratic_after_sos {
        return Err(invalid_mps(
            line,
            1,
            format!("{} section appears after {}", next.name(), previous.name()),
        ));
    }
    if next.rank() == previous.rank()
        && !matches!(next.rank(), SectionRank::Quadratic)
        && !matches!((previous, next), (Section::Name, Section::Name))
    {
        return Err(invalid_mps(line, 1, format!("duplicate {} section", next.name())));
    }
    Ok(())
}

fn parse_mps_line(
    data: &mut ParsedMps,
    section: &mut Section,
    saw_name: &mut bool,
    line: &str,
    line_no: usize,
    options: MpsReadOptions,
) -> Result<(), IoError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    if trimmed.starts_with('*') {
        if data.sense.is_none() {
            data.legacy_sense = parse_legacy_sense(trimmed).or(data.legacy_sense);
        }
        return Ok(());
    }
    if data.seen_sections & SEEN_END != 0 {
        return Err(invalid_mps(line_no, 1, "content after ENDATA"));
    }
    let items = fields(line);
    if let Some(first) = items.first()
        && items.len() == 1
        && first.text.eq_ignore_ascii_case("INDICATORS")
    {
        return Err(IoError::UnsupportedMps {
            section: "INDICATORS".into(),
            feature: "not represented by oximo-core".into(),
        });
    }
    if let Some(next) = header(&items, section) {
        if matches!(next, Section::Name) {
            if *saw_name {
                return Err(invalid_mps(line_no, 1, "duplicate NAME section"));
            }
            *saw_name = true;
            if items.len() > 1 {
                data.name = items[1..].iter().map(|field| field.text).collect::<Vec<_>>().join(" ");
            }
            return Ok(());
        }
        if !*saw_name {
            return Err(invalid_mps(line_no, 1, "the first data line must be NAME"));
        }
        check_section_transition(section, &next, line_no)?;
        if data.intorg && !matches!(next, Section::Columns) {
            return Err(invalid_mps(line_no, 1, "missing INTEND marker before COLUMNS ends"));
        }
        match &next {
            Section::ObjSense if items.len() == 2 => {
                data.sense = Some(objective_sense(&items[1], line_no)?);
            }
            Section::Rows => data.seen_sections |= SEEN_ROWS,
            Section::Columns => data.seen_sections |= SEEN_COLUMNS,
            Section::QuadObj | Section::QMatrix | Section::QcMatrix(_) | Section::QSec(_) => {
                begin_quadratic_section(data, &next, line_no)?;
            }
            Section::Sos => data.current_sos = None,
            Section::End => data.seen_sections |= SEEN_END,
            _ => {}
        }
        *section = next;
        return Ok(());
    }
    if !*saw_name {
        return Err(invalid_mps(line_no, 1, "the first data line must be NAME"));
    }
    match section {
        Section::ObjSense => {
            if items.len() != 1 {
                return Err(invalid_mps(line_no, 1, "OBJSENSE data requires one field"));
            }
            data.sense = Some(objective_sense(&items[0], line_no)?);
        }
        Section::Rows => parse_row(data, &items, line_no)?,
        Section::Columns => parse_columns(data, &items, line_no)?,
        Section::Rhs => parse_rhs(data, &items, line_no)?,
        Section::Ranges => parse_ranges(data, &items, line_no)?,
        Section::Bounds => parse_bound(data, &items, line_no)?,
        Section::QuadObj | Section::QMatrix | Section::QcMatrix(_) | Section::QSec(_) => {
            parse_quadratic_record(data, section, &items, line_no, options)?;
        }
        Section::Sos => parse_sos_record(data, &items, line_no)?,
        Section::Name => return Err(invalid_mps(line_no, 1, "expected NAME header")),
        Section::End => unreachable!(),
    }
    Ok(())
}

fn parse_sos_record(data: &mut ParsedMps, items: &[Field<'_>], line: usize) -> Result<(), IoError> {
    // Headers carry a set name, while members carry a numeric weight.
    // A variable named S1 or S2 must therefore follow the member path.
    let is_header = matches!(items.first().map(|item| item.text.to_ascii_uppercase()), Some(ty) if matches!(ty.as_str(), "S1" | "S2"))
        && (items.len() != 2 || !looks_like_number(items[1].text));
    if is_header {
        if items.len() != 2 {
            return Err(invalid_mps(line, 1, "SOS set headers require type and name"));
        }
        let sos_type = match items[0].text.to_ascii_uppercase().as_str() {
            "S1" => SosType::Sos1,
            _ => SosType::Sos2,
        };
        if items[1].text.is_empty() {
            return Err(invalid_mps(line, items[1].column, "SOS set needs a name"));
        }
        data.sos.push(ParsedSos { name: items[1].text.to_owned(), sos_type, members: Vec::new() });
        data.current_sos = Some(data.sos.len() - 1);
        return Ok(());
    }
    let Some(index) = data.current_sos else {
        return Err(invalid_mps(line, 1, "SOS member appears before an S1/S2 header"));
    };
    if items.len() != 2 {
        return Err(invalid_mps(line, 1, "SOS member needs variable name and weight"));
    }
    let weight = parse_number(&items[1], line)?;
    data.sos[index].members.push((items[0].text.to_owned(), weight));
    Ok(())
}

fn parse_mps<R: BufRead>(
    mut input: R,
    fallback_name: &str,
    options: MpsReadOptions,
) -> Result<Model, IoError> {
    let mut data = ParsedMps::new(fallback_name);
    let mut section = Section::Name;
    let mut saw_name = false;
    let mut line_buffer = String::new();
    let mut last_line = 0;
    loop {
        line_buffer.clear();
        if input.read_line(&mut line_buffer)? == 0 {
            break;
        }
        last_line += 1;
        let line = line_buffer.trim_end_matches(['\r', '\n']);
        parse_mps_line(&mut data, &mut section, &mut saw_name, line, last_line, options)?;
    }
    let last_line = last_line.max(1);
    if data.intorg {
        return Err(invalid_mps(last_line, 1, "missing INTEND marker"));
    }
    if data.seen_sections & SEEN_ROWS == 0 {
        return Err(invalid_mps(last_line, 1, "missing ROWS section"));
    }
    if data.seen_sections & SEEN_COLUMNS == 0 {
        return Err(invalid_mps(last_line, 1, "missing COLUMNS section"));
    }
    if data.seen_sections & SEEN_END == 0 {
        return Err(invalid_mps(last_line, 1, "missing ENDATA"));
    }
    validate_quadratic_triangles(&data, last_line)?;
    build_mps_model(data)
}

fn expression<'a>(
    model: &'a Model,
    variables: &[Expr<'a>],
    linear: impl IntoIterator<Item = (usize, f64)>,
    quadratic: impl IntoIterator<Item = ((usize, usize), f64)>,
    constant: f64,
) -> Expr<'a> {
    let mut expr = model.__constant(constant);
    for (column, coefficient) in linear {
        if coefficient != 0.0 {
            expr = expr + coefficient * variables[column];
        }
    }
    let mut quadratic: Vec<_> =
        quadratic.into_iter().filter(|(_, coefficient)| *coefficient != 0.0).collect();
    quadratic.sort_unstable_by_key(|((left, right), _)| (*left, *right));
    for ((left, right), coefficient) in quadratic {
        expr = expr + coefficient * variables[left] * variables[right];
    }
    expr
}

fn unique_mps_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
    fallback_prefix: &str,
    reserved: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let mut used: HashSet<String> = reserved.into_iter().map(str::to_owned).collect();
    names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let base: String = name
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                        ch
                    } else {
                        '_'
                    }
                })
                .collect();
            let base =
                if base.is_empty() { format!("{fallback_prefix}{}", index + 1) } else { base };
            let mut candidate = base.clone();
            let mut suffix = 1;
            while used.contains(&candidate) {
                candidate = format!("{base}_{suffix}");
                suffix += 1;
            }
            used.insert(candidate.clone());
            candidate
        })
        .collect()
}

fn write_quadratic_section<W: Write>(
    out: &mut W,
    header: &str,
    terms: &QuadraticTerms,
    variable_names: &[String],
    format: MpsQuadraticFormat,
    objective: bool,
) -> Result<(), IoError> {
    writeln!(out, "{header}")?;
    for &(left, right, hessian) in &terms.hessian {
        let (first, second) = if objective {
            // Gurobi and MOSEK document lower-triangular objective records.
            // CPLEX's QUADOBJ convention uses the upper triangle.
            if format == MpsQuadraticFormat::Cplex && left.index() > right.index() {
                (right, left)
            } else {
                (left, right)
            }
        } else if left.index() <= right.index() {
            (left, right)
        } else {
            (right, left)
        };
        let left_name = &variable_names[first.index()];
        let right_name = &variable_names[second.index()];
        let coefficient =
            if objective || format != MpsQuadraticFormat::Gurobi { hessian } else { hessian / 2.0 };
        writeln!(out, "    {left_name:<10} {right_name:<10} {coefficient}")?;
        if !objective && format != MpsQuadraticFormat::Mosek && first != second {
            // Gurobi and CPLEX require QCMATRIX to contain a symmetric Q.
            writeln!(out, "    {right_name:<10} {left_name:<10} {coefficient}")?;
        }
    }
    Ok(())
}

fn write_sos_section<W: Write>(
    out: &mut W,
    constraints: &[SosConstraint],
    variable_names: &[String],
) -> Result<(), IoError> {
    writeln!(out, "SOS")?;
    let active = || constraints.iter().filter(|constraint| constraint.active);
    let sos_names =
        unique_mps_names(active().map(|s| s.name.as_str()), "S", std::iter::empty::<&str>());
    for (set, set_name) in active().zip(sos_names.iter()) {
        writeln!(
            out,
            " S{} {}",
            match set.sos_type {
                SosType::Sos1 => 1,
                SosType::Sos2 => 2,
            },
            set_name
        )?;
        for member in &set.members {
            writeln!(out, "    {:<10} {}", variable_names[member.variable.index()], member.weight)?;
        }
    }
    Ok(())
}

fn build_mps_model(data: ParsedMps) -> Result<Model, IoError> {
    for column in &data.columns {
        if column.lower > column.upper {
            return Err(invalid_mps(
                1,
                1,
                format!("inconsistent bounds for column {:?}", column.name),
            ));
        }
        if matches!(column.kind, ColumnKind::SemiContinuous | ColumnKind::SemiInteger)
            && (!column.lower.is_finite() || column.lower < 0.0)
        {
            return Err(invalid_mps(
                1,
                1,
                format!("invalid semi-domain threshold for column {:?}", column.name),
            ));
        }
    }
    for row in &data.rows {
        if row.lower > row.upper {
            return Err(invalid_mps(1, 1, format!("inconsistent range for row {:?}", row.name)));
        }
    }
    let mut sos_names = HashSet::new();
    for set in &data.sos {
        if !sos_names.insert(set.name.clone()) {
            return Err(invalid_mps(1, 1, format!("duplicate SOS name {:?}", set.name)));
        }
        if set.members.is_empty() {
            return Err(invalid_mps(1, 1, format!("SOS {:?} has no members", set.name)));
        }
        let mut members = HashSet::new();
        let mut weights = Vec::new();
        for (var, weight) in &set.members {
            if !data.column_index.contains_key(var) {
                return Err(invalid_mps(1, 1, format!("unknown SOS variable {var:?}")));
            }
            if !members.insert(var) || weights.contains(weight) {
                return Err(invalid_mps(
                    1,
                    1,
                    format!("duplicate SOS member or weight in {:?}", set.name),
                ));
            }
            weights.push(*weight);
        }
    }
    let model = Model::new(data.name);
    let mut variables = Vec::with_capacity(data.columns.len());
    for column in &data.columns {
        let domain = match column.kind {
            ColumnKind::Continuous => Domain::Real,
            ColumnKind::Integer => Domain::Integer,
            ColumnKind::Binary => Domain::Binary,
            ColumnKind::SemiContinuous => Domain::SemiContinuous { threshold: column.lower },
            ColumnKind::SemiInteger => Domain::SemiInteger { threshold: column.lower },
        };
        let model_lower = match column.kind {
            ColumnKind::SemiContinuous | ColumnKind::SemiInteger => 0.0,
            _ => column.lower,
        };
        variables.push(
            model
                .__var(column.name.clone())
                .bounds(model_lower, column.upper)
                .domain(domain)
                .build(),
        );
    }
    for row in data.rows {
        let expr = expression(&model, &variables, row.linear, row.quadratic, 0.0);
        model.__add_constraint_interval(row.name, expr, row.lower, row.upper);
    }
    for set in data.sos {
        let members = set.members.into_iter().map(|(name, weight)| {
            let index = *data.column_index.get(&name).expect("validated SOS variable");
            (variables[index], weight)
        });
        model.add_sos_constraint(set.name, set.sos_type, members);
    }
    let objective = expression(
        &model,
        &variables,
        data.objective_linear.into_iter().enumerate(),
        data.objective_quadratic,
        data.objective_constant,
    );
    match data.sense.or(data.legacy_sense).unwrap_or(ObjectiveSense::Minimize) {
        ObjectiveSense::Minimize => model.__minimize(objective),
        ObjectiveSense::Maximize => model.__maximize(objective),
    }
    Ok(model)
}

/// Read an MPS byte stream with default options.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed syntax, or unsupported sections.
pub fn read_mps<R: Read>(input: R) -> Result<Model, IoError> {
    read_mps_with(input, &MpsReadOptions::default())
}

/// Read an MPS byte stream with explicit import options.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed syntax, or unsupported sections.
pub fn read_mps_with<R: Read>(input: R, options: &MpsReadOptions) -> Result<Model, IoError> {
    parse_mps(BufReader::new(input), "mps_model", *options)
}

/// Read an MPS file with default options.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed syntax, or unsupported sections.
pub fn read_mps_file(path: impl AsRef<Path>) -> Result<Model, IoError> {
    read_mps_file_with(path, &MpsReadOptions::default())
}

/// Read an MPS file with explicit import options.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed syntax, or unsupported sections.
pub fn read_mps_file_with(
    path: impl AsRef<Path>,
    options: &MpsReadOptions,
) -> Result<Model, IoError> {
    let path = path.as_ref();
    let fallback = path.file_stem().and_then(|name| name.to_str()).unwrap_or("mps_model");
    parse_mps(BufReader::new(File::open(path)?), fallback, *options)
}

/// Write `model` to `out` in fixed-format MPS.
///
/// MPS represents linear and quadratic LP/MILP/QP/QCP models. Higher-degree
/// or otherwise nonlinear expressions in the objective or constraints raise
/// [`IoError::Nonlinear`]. Second-order cone constraints raise [`IoError::Conic`].
/// The objective row is named `OBJ`.
/// Variable and constraint names have whitespace replaced by underscores and
/// are made unique within their respective MPS namespaces. The generated
/// objective row reserves the name `OBJ`.
///
/// # Errors
///
/// Returns [`IoError`] if there is an error writing the MPS data or if the model contains unsupported features.
///
pub fn write_mps<W: Write>(model: &Model, out: &mut W) -> Result<(), IoError> {
    write_mps_with(model, out, &MpsWriteOptions::default())
}

/// Write `model` to `out` with explicit quadratic MPS options.
///
/// `Gurobi` and `Cplex` emit `QUADOBJ` plus `QCMATRIX` sections. `Mosek` emits
/// `QSECTION` sections for the objective and quadratic constraints.
///
/// # Errors
///
/// Returns [`IoError`] if the model contains unsupported expressions or writing
/// the output fails.
#[expect(clippy::too_many_lines)]
pub fn write_mps_with<W: Write>(
    model: &Model,
    out: &mut W,
    options: &MpsWriteOptions,
) -> Result<(), IoError> {
    if model.has_active_sos_constraints() && options.quadratic_format == MpsQuadraticFormat::Mosek {
        return Err(IoError::UnsupportedMps {
            section: "SOS".into(),
            feature: "SOS is not supported by the MOSEK MPS dialect".into(),
        });
    }
    if model.num_soc_constraints() > 0
        || matches!(model.kind(), ModelKind::SOCP | ModelKind::MISOCP)
    {
        return Err(IoError::Conic);
    }
    let arena = model.arena();
    let vars = model.variables();
    let model_constraints = model.constraints();
    let constraints = model_constraints.algebraic();
    let objective = model.try_objective().map_err(|_| IoError::NoObjective)?;
    let variable_names = unique_mps_names(vars.iter().map(|v| v.name.as_str()), "C", []);
    let row_names = unique_mps_names(constraints.iter().map(|c| c.name.as_str()), "R", ["OBJ"]);

    let obj_terms =
        extract_quadratic(&arena, objective.expr).ok_or_else(|| IoError::Nonlinear {
            location: "the objective".into(),
            term: describe_nonlinear_term(&arena, objective.expr, &|v| var_name(&vars, v))
                .unwrap_or_else(|| "<nonlinear>".into()),
        })?;

    // Pre-compute quadratic terms once, reused for COLUMNS, RHS, and quadratic sections.
    let con_terms: Vec<QuadraticTerms> = constraints
        .iter()
        .map(|c| {
            extract_quadratic(&arena, c.lhs).ok_or_else(|| IoError::Nonlinear {
                location: format!("constraint {:?}", c.name),
                term: describe_nonlinear_term(&arena, c.lhs, &|v| var_name(&vars, v))
                    .unwrap_or_else(|| "<nonlinear>".into()),
            })
        })
        .collect::<Result<_, _>>()?;

    // Build column index: VarId to [(row_name, coef)] in row order (OBJ first, then constraints).
    let mut col_index: FxHashMap<VarId, Vec<(&str, f64)>> = FxHashMap::default();
    for (v, c) in &obj_terms.linear {
        col_index.entry(*v).or_default().push(("OBJ", *c));
    }
    for (row_name, terms) in row_names.iter().zip(con_terms.iter()) {
        for (v, coef) in &terms.linear {
            col_index.entry(*v).or_default().push((row_name.as_str(), *coef));
        }
    }

    writeln!(out, "* OXIMO MPS export")?;
    writeln!(
        out,
        "* sense: {}",
        match objective.sense {
            ObjectiveSense::Minimize => "minimize",
            ObjectiveSense::Maximize => "maximize",
        }
    )?;
    writeln!(out, "NAME          {}", model.name)?;
    writeln!(out, "OBJSENSE")?;
    writeln!(
        out,
        " {}",
        match objective.sense {
            ObjectiveSense::Minimize => "MIN",
            ObjectiveSense::Maximize => "MAX",
        }
    )?;

    writeln!(out, "ROWS")?;
    writeln!(out, " N  OBJ")?;
    for (c, row_name) in constraints.iter().zip(row_names.iter()) {
        let tag = match c.as_single() {
            Some((Sense::Le, _)) => 'L',
            Some((Sense::Ge, _)) => 'G',
            Some((Sense::Eq, _)) => 'E',
            // A two-sided range is an `L` row bounded by the `RANGES` section below.
            None if c.is_range() => 'L',
            // A free `[-inf, +inf]` row imposes nothing: emit an unconstraining
            // `N` row (no RHS) rather than an `L` row with a `+inf` bound.
            None => 'N',
        };
        writeln!(out, " {tag}  {row_name}")?;
    }

    writeln!(out, "COLUMNS")?;
    let mut int_open = false;
    for (v, column_name) in vars.iter().zip(variable_names.iter()) {
        // Binary and semi-integer columns carry their integrality via bounds.
        let needs_marker = matches!(v.domain, Domain::Integer);
        if needs_marker && !int_open {
            writeln!(out, "    MARKER                 'MARKER'                 'INTORG'")?;
            int_open = true;
        } else if !needs_marker && int_open {
            writeln!(out, "    MARKER                 'MARKER'                 'INTEND'")?;
            int_open = false;
        }
        if let Some(entries) = col_index.get(&v.id) {
            for (row_name, coef) in entries {
                writeln!(out, "    {column_name:<10} {row_name:<10} {coef}")?;
            }
        } else {
            writeln!(out, "    {column_name:<10} {:<10} 0", "OBJ")?;
        }
    }
    if int_open {
        writeln!(out, "    MARKER                 'MARKER'                 'INTEND'")?;
    }

    writeln!(out, "RHS")?;
    let obj_constant = obj_terms.constant;
    if obj_constant != 0.0 {
        writeln!(out, "    RHS       OBJ       {}", -obj_constant)?;
    }
    for ((c, row_name), t) in constraints.iter().zip(row_names.iter()).zip(con_terms.iter()) {
        // A range row's RHS is its upper bound (it is an `L` row), the `RANGES`
        // section then widens it down to the lower bound.
        let rhs = match c.as_single() {
            Some((_, rhs)) => rhs,
            None if c.is_range() => c.upper,
            // Free `N` row: carries no RHS.
            None => continue,
        };
        let adjusted = rhs - t.constant;
        if adjusted != 0.0 {
            writeln!(out, "    RHS       {row_name:<10} {adjusted}")?;
        }
    }

    if constraints.iter().any(Constraint::is_range) {
        writeln!(out, "RANGES")?;
        for (c, row_name) in constraints.iter().zip(row_names.iter()) {
            if c.is_range() {
                writeln!(out, "    RNG       {row_name:<10} {}", c.upper - c.lower)?;
            }
        }
    }

    writeln!(out, "BOUNDS")?;
    for (v, column_name) in vars.iter().zip(variable_names.iter()) {
        let lb = v.lb;
        let ub = v.ub;
        if matches!(v.domain, Domain::Binary) {
            writeln!(out, " BV BND       {column_name}")?;
            if lb != 0.0 {
                writeln!(out, " LO BND       {column_name:<10} {lb}")?;
            }
            if (ub - 1.0).abs() >= f64::EPSILON {
                writeln!(out, " UP BND       {column_name:<10} {ub}")?;
            }
            continue;
        }
        if let Some(thr) = v.domain.semi_threshold() {
            writeln!(out, " LO BND       {column_name:<10} {thr}")?;
            let semi_ub = if ub.is_finite() { ub } else { MPS_INFINITY_SENTINEL };
            // `is_integer()` distinguishes the two semi domains here.
            let code = if v.domain.is_integer() { "SI" } else { "SC" };
            writeln!(out, " {code} BND       {column_name:<10} {semi_ub}")?;
            continue;
        }
        if lb.is_finite() && (lb - ub).abs() < f64::EPSILON {
            writeln!(out, " FX BND       {column_name:<10} {lb}")?;
            continue;
        }
        let infinite_lo = lb == f64::NEG_INFINITY;
        let infinite_hi = ub == f64::INFINITY;
        match (infinite_lo, infinite_hi) {
            (true, true) => writeln!(out, " FR BND       {column_name}")?,
            (true, false) => {
                writeln!(out, " MI BND       {column_name}")?;
                writeln!(out, " UP BND       {column_name:<10} {ub}")?;
            }
            (false, true) => {
                if lb != 0.0 {
                    writeln!(out, " LO BND       {column_name:<10} {lb}")?;
                }
            }
            (false, false) => {
                if lb != 0.0 {
                    writeln!(out, " LO BND       {column_name:<10} {lb}")?;
                }
                writeln!(out, " UP BND       {column_name:<10} {ub}")?;
            }
        }
    }

    let sos = model.sos_constraints();
    let has_active_sos = sos.iter().any(|constraint| constraint.active);
    if options.quadratic_format == MpsQuadraticFormat::Cplex && has_active_sos {
        write_sos_section(out, &sos, &variable_names)?;
    }

    if !obj_terms.hessian.is_empty() {
        let header = if options.quadratic_format == MpsQuadraticFormat::Mosek {
            "QSECTION OBJ"
        } else {
            "QUADOBJ"
        };
        write_quadratic_section(
            out,
            header,
            &obj_terms,
            &variable_names,
            options.quadratic_format,
            true,
        )?;
    }
    for (row_name, terms) in row_names.iter().zip(con_terms.iter()) {
        if terms.hessian.is_empty() {
            continue;
        }
        let header = match options.quadratic_format {
            MpsQuadraticFormat::Mosek => format!("QSECTION {row_name}"),
            MpsQuadraticFormat::Gurobi | MpsQuadraticFormat::Cplex => {
                format!("QCMATRIX {row_name}")
            }
        };
        write_quadratic_section(
            out,
            &header,
            terms,
            &variable_names,
            options.quadratic_format,
            false,
        )?;
    }

    if options.quadratic_format != MpsQuadraticFormat::Cplex && has_active_sos {
        write_sos_section(out, &sos, &variable_names)?;
    }

    writeln!(out, "ENDATA")?;
    Ok(())
}

/// Convenience: render the MPS into a `String`.
///
/// # Errors
///
/// Returns [`IoError`] if writing the MPS data fails.
///
/// # Panics
///
/// Panics if the MPS writer internal buffer does not produce valid UTF-8 data.
pub fn to_mps_string(model: &Model) -> Result<String, IoError> {
    to_mps_string_with(model, &MpsWriteOptions::default())
}

/// Convenience: render the MPS into a `String` with explicit export options.
///
/// # Errors
///
/// Returns [`IoError`] if the model contains unsupported expressions or writing
/// the output fails.
///
/// # Panics
///
/// Panics if the MPS writer produces non-UTF-8 output.
pub fn to_mps_string_with(model: &Model, options: &MpsWriteOptions) -> Result<String, IoError> {
    let mut buf = Vec::new();
    write_mps_with(model, &mut buf, options)?;
    Ok(String::from_utf8(buf).expect("MPS writer emits ASCII"))
}
