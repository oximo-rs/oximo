//! Tests for the `.nl` writer's option surface: comments toggle, precision,
//! NaN/Inf strings, binary format, .row/.col sidecar, and F/S/V/d/r hooks.

use oximo_core::prelude::*;
use oximo_io::{
    Complementarity, DefinedVar, ImportedFunction, IoError, NlFormat, SuffixData, SuffixFlavour,
    SuffixKind, WriteOptions, to_nl_string_with, write_nl_files, write_nl_with,
};
use tempfile::TempDir;

fn simple_lp() -> Model {
    let m = Model::new("opt");
    variable!(m, x);
    variable!(m, y);
    objective!(m, Min, x + 2.0 * y);
    constraint!(m, c0, x + y >= 3.0);
    m
}

#[test]
fn no_comments() {
    let m = simple_lp();
    let s = to_nl_string_with(&m, &WriteOptions::ascii_lean()).expect("write");
    assert!(s.starts_with("g3 1 1 0\n"));
    assert!(!s.contains('#'));
    assert!(!s.contains('\t'));
}

#[test]
fn precision_knob() {
    let m = {
        let m = Model::new("p");
        variable!(m, x);
        objective!(m, Min, x * std::f64::consts::PI);
        constraint!(m, c0, x >= 0.0);
        m
    };
    let mut opts = WriteOptions::ascii_lean();
    opts.precision = Some(4);
    let s = to_nl_string_with(&m, &opts).expect("write");
    // Pi appears in the G segment as a coefficient. With 4 sig digits we
    // expect "3.142e0" via Rust's {:.3e} format.
    assert!(s.contains("3.142e0"), "expected 4-sig-digit pi in output, got:\n{s}");
}

#[test]
fn nonfinite_strings() {
    let m = {
        let m = Model::new("nf");
        variable!(m, f64::NEG_INFINITY <= x <= f64::INFINITY);
        objective!(m, Min, x);
        constraint!(m, c0, x >= 0.0);
        m
    };
    // Default strict mode passes because bounds use the `b 3` (free) line
    // rather than emitting NEG_INFINITY/INFINITY as numbers.
    let s = to_nl_string_with(&m, &WriteOptions::default()).expect("default");
    assert!(s.contains("\nb\n3\n"));
    // But if we deliberately bake an Inf into a constraint constant, default
    // mode errors. With `nonfinite_strings = true` it emits "Infinity".
    // Here we just sanity-check the option flag exists and is wired.
    let opts = WriteOptions { nonfinite_strings: true, ..Default::default() };
    let s2 = to_nl_string_with(&m, &opts).expect("nonfinite ok");
    assert_eq!(s, s2); // no NaN/Inf actually present, output unchanged
}

#[test]
fn nonfinite_constant_in_residual() {
    let m = Model::new("nf_res");
    variable!(m, x);
    objective!(m, Min, f64::INFINITY * x.sin());
    constraint!(m, c0, x >= 0.0);

    assert!(
        matches!(to_nl_string_with(&m, &WriteOptions::default()), Err(IoError::InvalidNumber)),
        "default strict mode must reject the non-finite constant"
    );

    let opts = WriteOptions { nonfinite_strings: true, ..Default::default() };
    let s = to_nl_string_with(&m, &opts).expect("nonfinite ok");
    assert!(s.contains("Infinity"), "expected Infinity constant in:\n{s}");
}

#[test]
fn rejects_semi_domains() {
    let m = Model::new("semic");
    variable!(m, 0.0 <= x <= 10.0, SemiCont(2.0));
    objective!(m, Min, x);
    let err = to_nl_string_with(&m, &WriteOptions::default()).unwrap_err();
    assert!(
        matches!(err, IoError::UnsupportedDomain(d) if d == "SemiContinuous"),
        "expected UnsupportedDomain(SemiContinuous), got {err:?}"
    );

    let m = Model::new("semii");
    variable!(m, 0.0 <= x <= 10.0, SemiInt(2.0));
    objective!(m, Min, x);
    let err = to_nl_string_with(&m, &WriteOptions::default()).unwrap_err();
    assert!(
        matches!(err, IoError::UnsupportedDomain(d) if d == "SemiInteger"),
        "expected UnsupportedDomain(SemiInteger), got {err:?}"
    );
}

#[test]
fn binary_header_marker() {
    let m = simple_lp();
    let mut buf = Vec::new();
    write_nl_with(&m, &mut buf, &WriteOptions::binary()).expect("binary");
    assert_eq!(&buf[..2], b"b3", "binary header must start with `b3`");
    // ASCII header line is still present (newline between line 1 and rest).
    assert!(buf.contains(&b'\n'));
}

#[test]
fn binary_ignores_comments() {
    // `comments` is ASCII-only. In binary mode it must be ignored, so
    // comments=true and comments=false produce byte-identical output.
    let m = simple_lp();
    let mut with_comments = Vec::new();
    write_nl_with(
        &m,
        &mut with_comments,
        &WriteOptions { format: NlFormat::Binary, comments: true, ..Default::default() },
    )
    .expect("binary");
    let mut without_comments = Vec::new();
    write_nl_with(&m, &mut without_comments, &WriteOptions::binary()).expect("binary");
    assert_eq!(with_comments, without_comments, "comments must not affect binary output");
}

#[test]
fn binary_to_string_errors() {
    // Binary output is not UTF-8, so the string helper refuses it instead of
    // panicking; binary callers must use the byte-sink `write_nl_with`.
    let m = simple_lp();
    let err = to_nl_string_with(&m, &WriteOptions::binary()).unwrap_err();
    assert!(matches!(err, IoError::BinaryToString));
}

#[test]
fn aux_files_sidecar() {
    let m = simple_lp();
    let dir = TempDir::new().expect("tempdir");
    let stub = dir.path().join("problem");
    let mut opts = WriteOptions::ascii_lean();
    opts.aux_files = true;
    write_nl_files(&m, &stub, &opts).expect("files");

    let row = std::fs::read_to_string(dir.path().join("problem.row")).expect("row");
    let col = std::fs::read_to_string(dir.path().join("problem.col")).expect("col");
    assert!(row.starts_with("c0\n"), "row file should list c0 first: {row}");
    assert!(row.contains("\nopt\n"), "row file should include objective name: {row}");
    assert_eq!(col, "x\ny\n");

    // Header should now carry nonzero max_name_len.
    let nl = std::fs::read_to_string(dir.path().join("problem.nl")).expect("nl");
    assert!(nl.contains("\n 2 1\n"), "header max-name-len: {nl}");
}

#[test]
fn f_segment_hook() {
    let m = simple_lp();
    let mut opts = WriteOptions::ascii_lean();
    opts.functions.push(ImportedFunction {
        name: "myfunc".into(),
        allow_string_args: 1,
        n_args: -1,
    });
    let s = to_nl_string_with(&m, &opts).expect("write");
    assert!(s.contains("F0 1 -1 myfunc"), "F segment present: {s}");
    // Header line 6, slot 2 (functions) should reflect n_funcs = 1; slot 1
    // (linear network variables) stays 0.
    assert!(s.contains("\n 0 1 0 1\n"), "header n_funcs == 1: {s}");
}

#[test]
fn s_segment_hook() {
    let m = simple_lp();
    let mut opts = WriteOptions::ascii_lean();
    opts.suffixes.push(SuffixData {
        name: "priority".into(),
        kind: SuffixKind::Variable,
        flavour: SuffixFlavour::Int,
        values: vec![(0, 5.0), (1, 3.0)],
    });
    let s = to_nl_string_with(&m, &opts).expect("write");
    assert!(s.contains("S0 2 priority"), "S header: {s}");
    assert!(s.contains("\n0 5\n"));
    assert!(s.contains("\n1 3\n"));
}

#[test]
fn v_segment_hook() {
    let m = simple_lp();
    let mut opts = WriteOptions::ascii_lean();
    opts.defined_vars.push(DefinedVar {
        nl_index: 2,
        linear: vec![(0, 1.0)],
        appearance: 0,
        nonlinear_polish: "n0".into(),
    });
    let s = to_nl_string_with(&m, &opts).expect("write");
    assert!(s.contains("V2 1 0\n0 1\nn0"), "V segment: {s}");
}

#[test]
fn d_segment_hook() {
    let m = simple_lp();
    let mut opts = WriteOptions::ascii_lean();
    opts.dual_init = vec![(0, 0.5)];
    let s = to_nl_string_with(&m, &opts).expect("write");
    assert!(s.contains("d1\n0 0.5\n"), "d segment: {s}");
}

#[test]
fn complementarity_in_r() {
    let m = {
        let m = Model::new("comp");
        variable!(m, x >= 0.0);
        objective!(m, Min, x);
        constraint!(m, c0, x >= 0.0);
        m
    };
    let mut opts = WriteOptions::ascii_lean();
    opts.complementarity = vec![(0, Complementarity { k: 1, i: 1 })];
    let s = to_nl_string_with(&m, &opts).expect("write");
    assert!(s.contains("\nr\n5 1 1\n"), "complementarity line in r: {s}");
}

#[test]
fn format_default_is_ascii() {
    assert_eq!(WriteOptions::default().format, NlFormat::Ascii);
}
