//! CPLEX LP file format import and export.
//!
//! CPLEX LP is a widely supported text format for linear and quadratic
//! optimization problems. It supports a single objective, assumes variables
//! are non-negative by default, and uses an explicit `Bounds` section for free
//! variables. This module imports the LP/QP subset represented by `oximo-core`
//! and exports linear and quadratic LP/MILP/QP/QCP models.
//!
//! [`write_lp`] writes an oximo [`Model`] to any `std::io::Write`.
//! [`read_lp`] and [`read_lp_file`] import streams and files, respectively.
//!
//! References:
//! - "CPLEX lp files," lp_solve. <https://lpsolve.sourceforge.net/5.5/CPLEX-format.htm> (accessed May 11, 2026).
//! - IBM, "Quadratic terms in LP file format." <https://www.ibm.com/docs/en/icos/22.1.2?topic=representation-quadratic-terms-in-lp-file-format>
//! - IBM, "Constraints in LP file format." <https://www.ibm.com/docs/en/icos/22.1.0?topic=representation-constraints-in-lp-file-format>

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use oximo_core::{Domain, Model, ModelKind, ObjectiveSense, Relate, Sense, var_name};
use oximo_expr::{Expr, QuadraticTerms, describe_nonlinear_term, extract_quadratic};
use rustc_hash::FxHashSet;

use crate::error::IoError;

/// Read a CPLEX LP stream.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed syntax, or unsupported sections.
pub fn read_lp<R: Read>(input: R) -> Result<Model, IoError> {
    parse_lp(BufReader::new(input), "lp_model")
}

/// Read a CPLEX LP file. The file stem is used as the model name.
///
/// # Errors
///
/// Returns [`IoError`] for I/O failures, malformed syntax, or unsupported sections.
pub fn read_lp_file(path: impl AsRef<Path>) -> Result<Model, IoError> {
    let path = path.as_ref();
    let name = path.file_stem().and_then(|x| x.to_str()).unwrap_or("lp_model");
    parse_lp(BufReader::new(File::open(path)?), name)
}

#[derive(Clone, Debug)]
enum Ast {
    Const(f64),
    Var(String),
    Add(Box<Ast>, Box<Ast>),
    Sub(Box<Ast>, Box<Ast>),
    Mul(Box<Ast>, Box<Ast>),
    Div(Box<Ast>, Box<Ast>),
    Neg(Box<Ast>),
    Pow(Box<Ast>, u32),
    Bracket(Box<Ast>),
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Word(String),
    Number(f64),
    Plus,
    Minus,
    Star,
    Caret,
    Slash,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Colon,
    Le,
    Ge,
    Eq,
}

#[derive(Clone, Debug, PartialEq)]
struct Token {
    kind: Tok,
    column: usize,
}

fn invalid_lp(line: usize, column: usize, message: impl Into<String>) -> IoError {
    IoError::InvalidLp { line, column, message: message.into() }
}

#[expect(clippy::too_many_lines, reason = "tokenization")]
fn lex(line: &str, line_no: usize) -> Result<Vec<Token>, IoError> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let col = i + 1;
        let tok = match c {
            '+' => {
                i += 1;
                Tok::Plus
            }
            '-' => {
                i += 1;
                Tok::Minus
            }
            '*' => {
                i += 1;
                Tok::Star
            }
            '^' => {
                i += 1;
                Tok::Caret
            }
            '/' => {
                i += 1;
                Tok::Slash
            }
            '(' => {
                i += 1;
                Tok::LParen
            }
            ')' => {
                i += 1;
                Tok::RParen
            }
            '[' => {
                i += 1;
                Tok::LBracket
            }
            ']' => {
                i += 1;
                Tok::RBracket
            }
            ':' => {
                i += 1;
                Tok::Colon
            }
            '=' if chars.get(i + 1) == Some(&'<') => {
                i += 2;
                Tok::Le
            }
            '=' if chars.get(i + 1) == Some(&'>') => {
                i += 2;
                Tok::Ge
            }
            '=' if chars.get(i + 1) == Some(&'=') => {
                i += 2;
                Tok::Eq
            }
            '=' => {
                i += 1;
                Tok::Eq
            }
            '<' if chars.get(i + 1) == Some(&'=') => {
                i += 2;
                Tok::Le
            }
            '>' if chars.get(i + 1) == Some(&'=') => {
                i += 2;
                Tok::Ge
            }
            '<' => {
                i += 1;
                Tok::Le
            }
            '>' => {
                i += 1;
                Tok::Ge
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                i += 1;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || matches!(chars[i], '.' | 'e' | 'E' | '+' | '-'))
                {
                    if (chars[i] == '+' || chars[i] == '-')
                        && !matches!(chars.get(i.wrapping_sub(1)), Some('e' | 'E'))
                    {
                        break;
                    }
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let n = s
                    .parse::<f64>()
                    .map_err(|_| invalid_lp(line_no, col, format!("invalid number {s}")))?;
                if !n.is_finite() {
                    return Err(invalid_lp(line_no, col, "non-finite number"));
                }
                Tok::Number(n)
            }
            _ => {
                let start = i;
                i += 1;
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && !matches!(
                        chars[i],
                        '+' | '-' | '*' | '^' | '/' | '(' | ')' | '[' | ']' | ':' | '<' | '>' | '='
                    )
                {
                    i += 1;
                }
                Tok::Word(chars[start..i].iter().collect())
            }
        };
        out.push(Token { kind: tok, column: col });
    }
    Ok(out)
}

struct ExprParser {
    toks: Vec<Token>,
    pos: usize,
    line: usize,
}

impl ExprParser {
    fn new(toks: Vec<Token>, line: usize) -> Self {
        Self { toks, pos: 0, line }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|token| &token.kind)
    }

    fn take(&mut self) -> Option<Token> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += usize::from(t.is_some());
        t
    }

    fn parse(mut self) -> Result<Ast, IoError> {
        let e = self.sum()?;
        if self.pos != self.toks.len() {
            let column = self.toks.get(self.pos).map_or(self.pos + 1, |token| token.column);
            return Err(invalid_lp(self.line, column, "unexpected token"));
        }
        Ok(e)
    }

    fn sum(&mut self) -> Result<Ast, IoError> {
        let mut lhs = self.product()?;
        loop {
            lhs = match self.peek() {
                Some(Tok::Plus) => {
                    self.take();
                    Ast::Add(Box::new(lhs), Box::new(self.product()?))
                }
                Some(Tok::Minus) => {
                    self.take();
                    Ast::Sub(Box::new(lhs), Box::new(self.product()?))
                }
                _ => break,
            };
        }
        Ok(lhs)
    }

    fn product(&mut self) -> Result<Ast, IoError> {
        let mut lhs = self.unary()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.take();
                    lhs = Ast::Mul(Box::new(lhs), Box::new(self.unary()?));
                }
                Some(Tok::Slash) => {
                    self.take();
                    lhs = Ast::Div(Box::new(lhs), Box::new(self.unary()?));
                }
                Some(Tok::Number(_) | Tok::Word(_)) if matches!(&lhs, Ast::Const(_)) => {
                    lhs = Ast::Mul(Box::new(lhs), Box::new(self.unary()?));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Ast, IoError> {
        match self.peek() {
            Some(Tok::Plus) => {
                self.take();
                self.unary()
            }
            Some(Tok::Minus) => {
                self.take();
                Ok(Ast::Neg(Box::new(self.unary()?)))
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Ast, IoError> {
        let mut a = self.atom()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.take();
            let n = match self.take() {
                Some(Token { kind: Tok::Number(x), column }) if x >= 0.0 && x.fract() == 0.0 => x
                    .to_string()
                    .parse::<u32>()
                    .map_err(|_| invalid_lp(self.line, column, "power is too large"))?,
                Some(Token { column, .. }) => {
                    return Err(invalid_lp(
                        self.line,
                        column,
                        "power must be a nonnegative integer",
                    ));
                }
                None => {
                    return Err(invalid_lp(
                        self.line,
                        self.pos + 1,
                        "power must be a nonnegative integer",
                    ));
                }
            };
            a = Ast::Pow(Box::new(a), n);
        }
        Ok(a)
    }

    fn atom(&mut self) -> Result<Ast, IoError> {
        match self.take() {
            Some(Token { kind: Tok::Number(x), .. }) => Ok(Ast::Const(x)),
            Some(Token { kind: Tok::Word(s), column }) => {
                if s.eq_ignore_ascii_case("inf") || s.eq_ignore_ascii_case("infinity") {
                    return Err(invalid_lp(self.line, column, "infinity is valid only in bounds"));
                }
                Ok(Ast::Var(s))
            }
            Some(Token { kind: Tok::LParen, .. }) => {
                let e = self.sum()?;
                if let Some(token) = self.take() {
                    if !matches!(token.kind, Tok::RParen) {
                        return Err(invalid_lp(self.line, token.column, "missing ')'"));
                    }
                } else {
                    return Err(invalid_lp(self.line, self.pos, "missing ')'"));
                }
                Ok(e)
            }
            Some(Token { kind: Tok::LBracket, .. }) => {
                let e = self.sum()?;
                if let Some(token) = self.take() {
                    if !matches!(token.kind, Tok::RBracket) {
                        return Err(invalid_lp(self.line, token.column, "missing ']'"));
                    }
                } else {
                    return Err(invalid_lp(self.line, self.pos, "missing ']'"));
                }
                Ok(Ast::Bracket(Box::new(e)))
            }
            Some(Token { column, .. }) => Err(invalid_lp(self.line, column, "expected expression")),
            None => Err(invalid_lp(self.line, self.pos, "expected expression")),
        }
    }
}

fn degree(a: &Ast) -> u32 {
    match a {
        Ast::Const(_) => 0,
        Ast::Var(_) => 1,
        Ast::Add(a, b) | Ast::Sub(a, b) => degree(a).max(degree(b)),
        Ast::Mul(a, b) => degree(a).saturating_add(degree(b)),
        Ast::Div(a, b) => {
            if degree(b) == 0 {
                degree(a)
            } else {
                3
            }
        }
        Ast::Neg(a) | Ast::Bracket(a) => degree(a),
        Ast::Pow(a, n) => degree(a).saturating_mul(*n),
    }
}

fn lower<'a>(m: &'a Model, vars: &HashMap<String, Expr<'a>>, a: Ast) -> Result<Expr<'a>, IoError> {
    match a {
        Ast::Const(x) => Ok(m.__constant(x)),
        Ast::Var(v) => {
            vars.get(&v).copied().ok_or_else(|| invalid_lp(1, 1, format!("unknown variable {v}")))
        }
        Ast::Add(a, b) => Ok(lower(m, vars, *a)? + lower(m, vars, *b)?),
        Ast::Sub(a, b) => Ok(lower(m, vars, *a)? - lower(m, vars, *b)?),
        Ast::Mul(a, b) => Ok(lower(m, vars, *a)? * lower(m, vars, *b)?),
        Ast::Div(a, b) => Ok(lower(m, vars, *a)? / lower(m, vars, *b)?),
        Ast::Neg(a) => Ok(-lower(m, vars, *a)?),
        Ast::Pow(a, n) => Ok(lower(m, vars, *a)?.powi(i32::try_from(n).unwrap_or(i32::MAX))),
        Ast::Bracket(a) => lower(m, vars, *a),
    }
}

#[derive(Default)]
struct ParsedLp {
    sense: Option<ObjectiveSense>,
    rows: Vec<(String, Ast, Sense, f64)>,
    bounds: HashMap<String, (f64, f64)>,
    general: Vec<String>,
    binary: Vec<String>,
    semi: Vec<String>,
    vars: Vec<String>,
    var_set: FxHashSet<String>,
}

fn strip_comment(line: &str) -> &str {
    line.split_once('\\').map_or(line, |(x, _)| x)
}

fn has_comparison(text: &str) -> bool {
    text.contains("<=")
        || text.contains(">=")
        || text.contains("=<")
        || text.contains("=>")
        || text.contains("==")
        || text.contains('<')
        || text.contains('>')
        || text.contains('=')
}

fn has_rhs(text: &str) -> bool {
    let normalized = text.replace("=<", "<=").replace("=>", ">=").replace("==", "=");
    ["<=", ">=", "<", ">", "="]
        .iter()
        .find_map(|operator| normalized.find(operator).map(|pos| pos + operator.len()))
        .is_some_and(|end| !normalized[end..].trim().is_empty())
}

fn section(line: &str) -> Option<&'static str> {
    let s = line.trim().to_ascii_lowercase();
    match s.as_str() {
        "minimize" | "minimise" | "minimum" | "min" => Some("objective_min"),
        "maximize" | "maximise" | "maximum" | "max" => Some("objective_max"),
        "subject to" | "such that" | "st" | "s.t." | "st." => Some("rows"),
        "bounds" | "bound" => Some("bounds"),
        "general" | "generals" | "gen" | "integer" | "integers" => Some("general"),
        "binary" | "binaries" | "bin" => Some("binary"),
        "semi-continuous" | "semis" | "semi" => Some("semi"),
        "end" => Some("end"),
        "sos" | "indicators" | "lazy constraints" | "user cuts" | "pwl" | "multi-objective" => {
            Some("unsupported")
        }
        _ => None,
    }
}

fn parse_constraint_line(
    line: &str,
    line_no: usize,
    pending: &mut String,
    pending_line: &mut usize,
    p: &mut ParsedLp,
) -> Result<(), IoError> {
    if pending.is_empty() && (line.contains("::") || line.contains("->")) {
        return Err(IoError::UnsupportedLp {
            section: "Subject To".into(),
            feature: "SOS or indicator constraints are not represented by oximo-core".into(),
        });
    }
    if !pending.is_empty() && line.contains(':') && !has_comparison(pending) {
        return Err(invalid_lp(
            line_no,
            1,
            "new constraint begins before the previous constraint is complete",
        ));
    }
    if pending.is_empty() {
        *pending_line = line_no;
    }
    pending.push(' ');
    pending.push_str(line);
    if has_comparison(pending) && has_rhs(pending) {
        let (name, body) = pending
            .split_once(':')
            .map_or((String::new(), pending.as_str()), |(n, b)| (n.trim().to_string(), b));
        let (expr, sense, rhs) = parse_row(body, *pending_line)?;
        collect_vars(&expr, &mut p.vars, &mut p.var_set);
        p.rows.push((name, expr, sense, rhs));
        pending.clear();
    }
    Ok(())
}

fn parse_lp<R: BufRead>(mut input: R, model_name: &str) -> Result<Model, IoError> {
    let mut p = ParsedLp::default();
    let mut current = "";
    let mut objective_lines = Vec::new();
    let mut pending = String::new();
    let mut pending_line = 1;
    let mut ended = false;
    let mut line_buffer = String::new();
    let mut line_count = 0;
    loop {
        line_buffer.clear();
        if input.read_line(&mut line_buffer)? == 0 {
            break;
        }
        line_count += 1;
        let line = strip_comment(line_buffer.trim_end_matches(['\r', '\n'])).trim();
        if line.is_empty() {
            continue;
        }
        if ended {
            return Err(invalid_lp(line_count, 1, "content after End section"));
        }
        let lower_line = line.to_ascii_lowercase();
        let inline_objective = if ["minimize ", "minimise ", "minimum ", "min "]
            .iter()
            .any(|prefix| lower_line.starts_with(prefix))
        {
            Some((ObjectiveSense::Minimize, "objective_min"))
        } else if ["maximize ", "maximise ", "maximum ", "max "]
            .iter()
            .any(|prefix| lower_line.starts_with(prefix))
        {
            Some((ObjectiveSense::Maximize, "objective_max"))
        } else {
            None
        };
        if let Some((sense, section_name)) = inline_objective {
            p.sense = Some(sense);
            current = section_name;
            objective_lines.push(line[lower_line.find(' ').unwrap_or(0)..].trim().to_string());
            continue;
        }
        if let Some(s) = section(line) {
            if s == "end" {
                ended = true;
                continue;
            }
            if s == "unsupported" {
                // TODO: Add native SOS, indicator, PWL, and MO
                // representations before importing these sections.
                return Err(IoError::UnsupportedLp {
                    section: line.to_owned(),
                    feature: "not represented by oximo-core".into(),
                });
            }
            current = s;
            if s == "objective_min" {
                p.sense = Some(ObjectiveSense::Minimize);
                continue;
            }
            if s == "objective_max" {
                p.sense = Some(ObjectiveSense::Maximize);
                continue;
            }
            continue;
        }
        match current {
            "objective_min" | "objective_max" => objective_lines.push(line.to_owned()),
            "rows" => {
                parse_constraint_line(line, line_count, &mut pending, &mut pending_line, &mut p)?;
            }
            "bounds" => parse_bound(line, line_count, &mut p)?,
            "general" => p.general.extend(line.split_whitespace().map(str::to_owned)),
            "binary" => p.binary.extend(line.split_whitespace().map(str::to_owned)),
            "semi" => p.semi.extend(line.split_whitespace().map(str::to_owned)),
            _ => return Err(invalid_lp(line_count, 1, "content before an objective or after End")),
        }
    }
    if !ended {
        return Err(invalid_lp(line_count.max(1), 1, "missing End section"));
    }
    if !pending.is_empty() {
        return Err(invalid_lp(
            pending_line,
            1,
            "constraint is missing a comparison and right-hand side",
        ));
    }
    if p.sense.is_none() {
        return Err(invalid_lp(1, 1, "missing Minimize or Maximize section"));
    }
    build_model(p, objective_lines, model_name)
}

fn build_model(
    mut p: ParsedLp,
    objective_lines: Vec<String>,
    model_name: &str,
) -> Result<Model, IoError> {
    let obj_text = objective_lines.join(" ");
    let obj_text = obj_text.split_once(':').map_or(obj_text.as_str(), |(_, x)| x);
    let obj_toks = lex(obj_text, 1)?;
    let obj = ExprParser::new(obj_toks, 1).parse()?;
    if degree(&obj) > 2 {
        return Err(invalid_lp(1, 1, "higher-degree expressions are not valid CPLEX LP syntax"));
    }
    collect_vars(&obj, &mut p.vars, &mut p.var_set);
    let general_names: FxHashSet<String> = p.general.iter().cloned().collect();
    let binary_names: FxHashSet<String> = p.binary.iter().cloned().collect();
    let semi_names: FxHashSet<String> = p.semi.iter().cloned().collect();
    for n in p.bounds.keys().chain(p.general.iter()).chain(p.binary.iter()).chain(p.semi.iter()) {
        if p.var_set.insert(n.clone()) {
            p.vars.push(n.clone());
        }
    }
    for name in &p.vars {
        validate_lp_variable_name(name)?;
    }
    let m = Model::new(model_name);
    let mut vars = HashMap::new();
    for n in &p.vars {
        let (lb, mut ub) = p.bounds.get(n).copied().unwrap_or((0.0, f64::INFINITY));
        if binary_names.contains(n) && !p.bounds.contains_key(n) {
            ub = 1.0;
        }
        if lb > ub {
            return Err(invalid_lp(1, 1, format!("inconsistent bounds for variable {n}")));
        }
        let domain = if semi_names.contains(n) && general_names.contains(n) {
            Domain::SemiInteger { threshold: lb }
        } else if semi_names.contains(n) {
            Domain::SemiContinuous { threshold: lb }
        } else if binary_names.contains(n) {
            Domain::Binary
        } else if general_names.contains(n) {
            Domain::Integer
        } else {
            Domain::Real
        };
        vars.insert(n.clone(), m.__var(n.clone()).bounds(lb, ub).domain(domain).build());
    }
    let mut used_names = HashSet::new();
    for (name, _, _, _) in &p.rows {
        if name.is_empty() {
            continue;
        }
        validate_lp_row_name(name)?;
        if !used_names.insert(name.clone()) {
            return Err(invalid_lp(1, 1, format!("duplicate constraint name {name:?}")));
        }
    }
    let mut next_generated_name = 0;
    for (mut name, expr, sense, rhs) in p.rows {
        if name.is_empty() {
            loop {
                let candidate = format!("c{next_generated_name}");
                next_generated_name += 1;
                if used_names.insert(candidate.clone()) {
                    name = candidate;
                    break;
                }
            }
        }
        if degree(&expr) > 2 {
            return Err(invalid_lp(
                1,
                1,
                format!(
                    "higher-degree expression in constraint {name:?} is not valid CPLEX LP syntax"
                ),
            ));
        }
        let e = lower(&m, &vars, expr)?;
        m.__add_constraint(
            name,
            match sense {
                Sense::Le => e.le(rhs),
                Sense::Ge => e.ge(rhs),
                Sense::Eq => e.eq(rhs),
            },
        );
    }
    let e = lower(&m, &vars, obj)?;
    let sense = p.sense.ok_or_else(|| invalid_lp(1, 1, "missing Minimize or Maximize section"))?;
    match sense {
        ObjectiveSense::Minimize => m.__minimize(e),
        ObjectiveSense::Maximize => m.__maximize(e),
    }
    Ok(m)
}

fn validate_lp_variable_name(name: &str) -> Result<(), IoError> {
    if name.is_empty() || name.chars().any(char::is_control) || name.contains('\\') {
        return Err(invalid_lp(
            1,
            1,
            format!("variable name {name:?} contains characters not representable in LP syntax"),
        ));
    }
    if name.eq_ignore_ascii_case("inf") || name.eq_ignore_ascii_case("infinity") {
        return Err(invalid_lp(
            1,
            1,
            format!("variable name {name:?} is reserved for an LP bound"),
        ));
    }
    let tokens = lex(name, 1)?;
    if !matches!(tokens.as_slice(), [Token { kind: Tok::Word(word), .. }] if word == name) {
        return Err(invalid_lp(
            1,
            1,
            format!("variable name {name:?} cannot be represented as one LP identifier"),
        ));
    }
    Ok(())
}

fn validate_lp_row_name(name: &str) -> Result<(), IoError> {
    if name.chars().any(char::is_control) || name.contains('\\') {
        return Err(invalid_lp(
            1,
            1,
            format!("constraint name {name:?} contains characters not representable in LP syntax"),
        ));
    }
    Ok(())
}

fn collect_vars(a: &Ast, vars: &mut Vec<String>, seen: &mut FxHashSet<String>) {
    match a {
        Ast::Var(v) if seen.insert(v.clone()) => vars.push(v.clone()),
        Ast::Add(a, b) | Ast::Sub(a, b) | Ast::Mul(a, b) | Ast::Div(a, b) => {
            collect_vars(a, vars, seen);
            collect_vars(b, vars, seen);
        }
        Ast::Neg(a) | Ast::Pow(a, _) | Ast::Bracket(a) => collect_vars(a, vars, seen),
        Ast::Const(_) | Ast::Var(_) => {}
    }
}

fn parse_row(body: &str, line: usize) -> Result<(Ast, Sense, f64), IoError> {
    let normalized = body.replace("=<", "<=").replace("=>", ">=").replace("==", "=");
    let (pos, sense, width) = if let Some(x) = normalized.find("<=") {
        (x, Sense::Le, 2)
    } else if let Some(x) = normalized.find(">=") {
        (x, Sense::Ge, 2)
    } else if let Some(x) = normalized.find('<') {
        (x, Sense::Le, 1)
    } else if let Some(x) = normalized.find('>') {
        (x, Sense::Ge, 1)
    } else if let Some(x) = normalized.find('=') {
        (x, Sense::Eq, 1)
    } else {
        return Err(invalid_lp(line, 1, "constraint has no comparison"));
    };
    let lhs = ExprParser::new(lex(normalized[..pos].trim(), line)?, line).parse()?;
    let rhs_toks = lex(normalized[pos + width..].trim(), line)?;
    let mut rhs_pos = 0;
    let rhs = bound_value(&rhs_toks, &mut rhs_pos)
        .ok_or_else(|| invalid_lp(line, pos + width + 1, "right-hand side must be numeric"))?;
    if rhs_pos != rhs_toks.len() {
        return Err(invalid_lp(
            line,
            pos + width + rhs_pos + 1,
            "right-hand side must be one number",
        ));
    }
    Ok((lhs, sense, rhs))
}

fn bound_value(toks: &[Token], pos: &mut usize) -> Option<f64> {
    let mut sign = 1.0;
    if matches!(toks.get(*pos).map(|token| &token.kind), Some(Tok::Minus)) {
        sign = -1.0;
        *pos += 1;
    } else if matches!(toks.get(*pos).map(|token| &token.kind), Some(Tok::Plus)) {
        *pos += 1;
    }
    let value = match &toks.get(*pos)?.kind {
        Tok::Word(x) if x.eq_ignore_ascii_case("inf") || x.eq_ignore_ascii_case("infinity") => {
            f64::INFINITY
        }
        Tok::Number(x) => *x,
        _ => return None,
    };
    *pos += 1;
    Some(sign * value)
}

fn parse_bound(line: &str, line_no: usize, p: &mut ParsedLp) -> Result<(), IoError> {
    let toks = lex(line, line_no)?;
    if toks.len() == 2 && matches!(&toks[1].kind, Tok::Word(x) if x.eq_ignore_ascii_case("free")) {
        if let Tok::Word(n) = &toks[0].kind {
            p.bounds.insert(n.clone(), (f64::NEG_INFINITY, f64::INFINITY));
            return Ok(());
        }
    }
    if let Tok::Word(n) = &toks[0].kind {
        if matches!(toks.get(1).map(|token| &token.kind), Some(Tok::Eq)) {
            let mut i = 2;
            if let Some(v) = bound_value(&toks, &mut i)
                && i == toks.len()
            {
                p.bounds.insert(n.clone(), (v, v));
                return Ok(());
            }
        }
        if matches!(toks.get(1).map(|token| &token.kind), Some(Tok::Le | Tok::Ge)) {
            let mut i = 2;
            if let Some(v) = bound_value(&toks, &mut i)
                && i == toks.len()
            {
                let old = p.bounds.get(n).copied().unwrap_or((0.0, f64::INFINITY));
                if matches!(&toks[1].kind, Tok::Le) {
                    p.bounds.insert(n.clone(), (old.0, v));
                } else {
                    p.bounds.insert(n.clone(), (v, old.1));
                }
                return Ok(());
            }
        }
    }
    let mut i = 0;
    if let Some(lo) = bound_value(&toks, &mut i)
        && matches!(toks.get(i).map(|token| &token.kind), Some(Tok::Le))
    {
        i += 1;
        if let Some(Token { kind: Tok::Word(n), .. }) = toks.get(i) {
            i += 1;
            if matches!(toks.get(i).map(|token| &token.kind), Some(Tok::Le)) {
                i += 1;
                if let Some(hi) = bound_value(&toks, &mut i)
                    && i == toks.len()
                {
                    p.bounds.insert(n.clone(), (lo, hi));
                    return Ok(());
                }
            }
        }
    }
    let mut i = 0;
    if let Some(lo) = bound_value(&toks, &mut i)
        && matches!(toks.get(i).map(|token| &token.kind), Some(Tok::Le))
    {
        i += 1;
        if let Some(Token { kind: Tok::Word(n), .. }) = toks.get(i) {
            i += 1;
            if i == toks.len() {
                let old = p.bounds.get(n).copied().unwrap_or((0.0, f64::INFINITY));
                p.bounds.insert(n.clone(), (lo, old.1));
                return Ok(());
            }
        }
    }
    Err(invalid_lp(line_no, 1, "invalid bound declaration"))
}

/// Write `model` to `out` in LP format.
///
/// Sections emitted:
/// - `\* ... *\` header comment with model name and original sense
/// - `Minimize` / `Maximize` with `obj:` row
/// - `Subject To` with each constraint
/// - `Bounds` (only non-default bounds)
/// - `General` (non-binary integer vars)
/// - `Binaries` (binary vars)
/// - `End`
///
/// LP export supports linear and quadratic LP/MILP/QP/QCP expressions using
/// CPLEX bracket notation. Higher-degree or otherwise nonlinear nodes raise
/// [`IoError::Nonlinear`]; second-order cone constraints raise
/// [`IoError::Conic`].
///
/// # Errors
///
/// Returns [`IoError`] on I/O failure, missing objective, or nonlinear/conic
/// constructs.
#[expect(clippy::too_many_lines)]
pub fn write_lp<W: Write>(model: &Model, out: &mut W) -> Result<(), IoError> {
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

    let obj_terms =
        extract_quadratic(&arena, objective.expr).ok_or_else(|| IoError::Nonlinear {
            location: "the objective".into(),
            term: describe_nonlinear_term(&arena, objective.expr, &|v| var_name(&vars, v))
                .unwrap_or_else(|| "<nonlinear>".into()),
        })?;

    writeln!(out, "\\* OXIMO LP export - model: {} *\\", model.name)?;

    let sense_kw = match objective.sense {
        ObjectiveSense::Minimize => "Minimize",
        ObjectiveSense::Maximize => "Maximize",
    };
    writeln!(out, "{sense_kw}")?;
    write!(out, " obj:")?;
    write_quadratic(out, &obj_terms, &vars, false)?;
    if obj_terms.constant != 0.0 {
        let sign = if obj_terms.constant < 0.0 { '-' } else { '+' };
        write!(out, " {sign} {}", obj_terms.constant.abs())?;
    }
    writeln!(out)?;

    // A collapsed range row is split into `{name}_lo` / `{name}_hi` labels at
    // export time. Those derived labels can clash with another constraint named
    // literally `{name}_lo`, so disambiguate against every registered name.
    let mut used_labels: FxHashSet<String> =
        constraints.iter().map(|c| c.name.to_string()).collect();

    writeln!(out, "Subject To")?;
    for c in constraints {
        let t = extract_quadratic(&arena, c.lhs).ok_or_else(|| IoError::Nonlinear {
            location: format!("constraint {:?}", c.name),
            term: describe_nonlinear_term(&arena, c.lhs, &|v| var_name(&vars, v))
                .unwrap_or_else(|| "<nonlinear>".into()),
        })?;
        if let Some((sense, rhs)) = c.as_single() {
            let op = match sense {
                Sense::Le => "<=",
                Sense::Ge => ">=",
                Sense::Eq => "=",
            };
            let adjusted_rhs = rhs - t.constant;
            write!(out, " {}:", c.name)?;
            write_quadratic(out, &t, &vars, true)?;
            writeln!(out, " {op} {adjusted_rhs}")?;
        } else if c.is_range() {
            let lo = c.lower - t.constant;
            let hi = c.upper - t.constant;
            let lo_label = unique_label(&mut used_labels, &format!("{}_lo", c.name));
            write!(out, " {lo_label}:")?;
            write_quadratic(out, &t, &vars, true)?;
            writeln!(out, " >= {lo}")?;
            let hi_label = unique_label(&mut used_labels, &format!("{}_hi", c.name));
            write!(out, " {hi_label}:")?;
            write_quadratic(out, &t, &vars, true)?;
            writeln!(out, " <= {hi}")?;
        } else {
            // Free `[-inf, +inf]` row: imposes no constraint and has no valid LP
            // representation (`>= -inf` / `<= +inf` are illegal).
            // Leave a comment so the omission is traceable in the output.
            writeln!(out, "\\* skipped free row: {} *\\", c.name)?;
        }
    }

    let mut wrote_bounds_header = false;
    for v in vars.iter() {
        if matches!(v.domain, Domain::Binary) {
            continue;
        }
        // Semicont/semiint: the gap floor (`threshold`) is the LP lower bound.
        // The `Semi-Continuous` section below marks the gap.
        if let Some(thr) = v.domain.semi_threshold() {
            if !wrote_bounds_header {
                writeln!(out, "Bounds")?;
                wrote_bounds_header = true;
            }
            if v.ub == f64::INFINITY {
                writeln!(out, " {} >= {}", v.name, thr)?;
            } else {
                writeln!(out, " {} <= {} <= {}", thr, v.name, v.ub)?;
            }
            continue;
        }
        if v.lb.is_finite() && (v.lb - v.ub).abs() < f64::EPSILON {
            if !wrote_bounds_header {
                writeln!(out, "Bounds")?;
                wrote_bounds_header = true;
            }
            writeln!(out, " {} <= {} <= {}", v.lb, v.name, v.ub)?;
            continue;
        }
        let lb_default = v.lb == 0.0;
        let ub_default = v.ub == f64::INFINITY;
        if lb_default && ub_default {
            continue;
        }
        if !wrote_bounds_header {
            writeln!(out, "Bounds")?;
            wrote_bounds_header = true;
        }
        if v.lb == f64::NEG_INFINITY && ub_default {
            writeln!(out, " {} free", v.name)?;
        } else if v.lb == f64::NEG_INFINITY {
            writeln!(out, " -inf <= {} <= {}", v.name, v.ub)?;
        } else if ub_default {
            writeln!(out, " {} >= {}", v.name, v.lb)?;
        } else {
            writeln!(out, " {} <= {} <= {}", v.lb, v.name, v.ub)?;
        }
    }

    let general_vars: Vec<&str> = vars
        .iter()
        .filter(|v| matches!(v.domain, Domain::Integer | Domain::SemiInteger { .. }))
        .map(|v| v.name.as_str())
        .collect();
    if !general_vars.is_empty() {
        writeln!(out, "General")?;
        writeln!(out, " {}", general_vars.join(" "))?;
    }

    let binary_vars: Vec<&str> = vars
        .iter()
        .filter(|v| matches!(v.domain, Domain::Binary))
        .map(|v| v.name.as_str())
        .collect();
    if !binary_vars.is_empty() {
        writeln!(out, "Binaries")?;
        writeln!(out, " {}", binary_vars.join(" "))?;
    }

    // Semicontinuous and semi-integer vars. A var that is also in `General`
    // (the SemiInteger filter above) is read back as semi-integer.
    let semi_vars: Vec<&str> = vars
        .iter()
        .filter(|v| v.domain.semi_threshold().is_some())
        .map(|v| v.name.as_str())
        .collect();
    if !semi_vars.is_empty() {
        writeln!(out, "Semi-Continuous")?;
        writeln!(out, " {}", semi_vars.join(" "))?;
    }

    writeln!(out, "End")?;
    Ok(())
}

/// Convenience: render the LP into a `String`.
///
/// # Errors
///
/// Returns [`IoError`] if writing fails.
///
/// # Panics
///
/// Panics if the writer's internal buffer is not valid UTF-8
pub fn to_lp_string(model: &Model) -> Result<String, IoError> {
    let mut buf = Vec::new();
    write_lp(model, &mut buf)?;
    Ok(String::from_utf8(buf).expect("LP writer emits ASCII"))
}

/// Reserve a unique row label. Returns `base` if free, otherwise appends
/// `_2`, `_3`, ... until it no longer collides with a name in `used`. The chosen
/// label is recorded so later split rows cannot reuse it.
fn unique_label(used: &mut FxHashSet<String>, base: &str) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }
    let mut n = 2;
    loop {
        let candidate = format!("{base}_{n}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

/// Write a quadratic expression using CPLEX's bracket notation. Objective
/// brackets contain the Hessian coefficients and are divided by two; a
/// constraint bracket contains the polynomial coefficients directly.
fn write_quadratic<W: Write>(
    out: &mut W,
    t: &QuadraticTerms,
    vars: &[oximo_core::Variable],
    constraint: bool,
) -> std::io::Result<()> {
    let mut first = true;
    for &(v, coef) in &t.linear {
        write_signed_term(out, &mut first, coef, vars[v.index()].name.as_str())?;
    }
    if !t.hessian.is_empty() {
        if first {
            write!(out, " [")?;
        } else {
            write!(out, " + [")?;
        }
        let mut first_quad = true;
        for &(row, col, hessian) in &t.hessian {
            let coefficient = if constraint {
                if row == col { hessian / 2.0 } else { hessian }
            } else if row == col {
                hessian
            } else {
                2.0 * hessian
            };
            let body = if row == col {
                format!("{}^2", vars[row.index()].name)
            } else {
                format!("{} * {}", vars[col.index()].name, vars[row.index()].name)
            };
            write_signed_term(out, &mut first_quad, coefficient, &body)?;
        }
        write!(out, " ]")?;
        if !constraint {
            write!(out, "/2")?;
        }
        first = false;
    }
    if first {
        write!(out, " 0")?;
    }
    Ok(())
}

fn write_signed_term<W: Write>(
    out: &mut W,
    first: &mut bool,
    coefficient: f64,
    body: &str,
) -> std::io::Result<()> {
    if coefficient == 0.0 {
        return Ok(());
    }
    let negative = coefficient < 0.0;
    let magnitude = coefficient.abs();
    let coefficient_text = if (magnitude - 1.0).abs() < f64::EPSILON {
        String::new()
    } else {
        format!("{magnitude} ")
    };
    if *first {
        if negative {
            write!(out, " - {coefficient_text}{body}")?;
        } else {
            write!(out, " {coefficient_text}{body}")?;
        }
        *first = false;
    } else if negative {
        write!(out, " - {coefficient_text}{body}")?;
    } else {
        write!(out, " + {coefficient_text}{body}")?;
    }
    Ok(())
}
