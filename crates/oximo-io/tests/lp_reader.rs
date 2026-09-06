use std::io::{self, Cursor, Read};

use oximo_core::prelude::*;
use oximo_io::{IoError, read_lp, read_lp_file, to_lp_string};

#[test]
fn wide_rows_lower_to_linear_size_arenas() {
    let expression = (0..10_000).map(|i| format!("x{i}")).collect::<Vec<_>>().join(" + ");
    let input = format!("Minimize\n obj: {expression}\nSubject To\n c: {expression} <= 10\nEnd\n");
    let model = read_lp(input.as_bytes()).unwrap();
    assert_eq!(model.num_variables(), 10_000);
    let arena = model.arena();
    assert!(arena.len() < 20_010, "wide import retained intermediate prefix nodes");
    let terms = oximo_expr::extract_linear(&arena, model.constraints().algebraic()[0].lhs).unwrap();
    assert_eq!(terms.coeffs.len(), 10_000);
    for (i, (v, c)) in terms.coeffs.iter().enumerate() {
        assert_eq!(v.index(), i);
        assert_eq!(c.to_bits(), 1.0_f64.to_bits());
    }
}

#[test]
fn flat_sum_preserves_subtraction_parentheses_and_variable_order() {
    let model = read_lp(
        b"Minimize\n obj: z - (x - 2 y) + 3 x - z + 7\nSubject To\n c: z + y - x <= 9\nEnd\n"
            .as_slice(),
    )
    .unwrap();
    let names: Vec<_> = model.variables().iter().map(|v| v.name.to_string()).collect();
    assert_eq!(names, ["z", "y", "x"]);
    let arena = model.arena();
    let objective = model.try_objective().unwrap();
    let terms = oximo_expr::extract_quadratic(&arena, objective.expr).unwrap();
    assert!(terms.hessian.is_empty());
    assert_eq!(terms.constant.to_bits(), 7.0_f64.to_bits());
    assert_eq!(terms.linear, vec![(oximo_expr::VarId(1), 2.0), (oximo_expr::VarId(2), 2.0)]);
}

#[test]
fn reads_milp_sections_bounds_and_comments() {
    let text = r"
\* a comment *\
Maximize
 obj: 2 x + 3 y + 4
Subject To
 c1: x + y <= 10
 c2: x - 2 y >= -3
 c3: x + z = 2
Bounds
 -infinity <= x <= 10
 y >= 2
  1 <= z
 z = 1
General
 y
Binaries
 z
End
";
    let model = read_lp(Cursor::new(text)).expect("LP should parse");
    assert_eq!(model.num_variables(), 3);
    assert_eq!(model.num_constraints(), 3);
    assert_eq!(model.variables()[0].name, "x");
    assert!(model.variables()[0].lb.is_infinite() && model.variables()[0].lb.is_sign_negative());
    assert!((model.variables()[0].ub - 10.0).abs() < f64::EPSILON);
    assert!(matches!(model.variables()[1].domain, Domain::Integer));
    assert!(matches!(model.variables()[2].domain, Domain::Binary));
    assert!((model.variables()[2].lb - 1.0).abs() < f64::EPSILON);
    assert!((model.variables()[2].ub - 1.0).abs() < f64::EPSILON);
    assert_eq!(model.try_objective().expect("objective").sense, ObjectiveSense::Maximize);
}

#[test]
fn reads_quadratic_objective_and_constraint() {
    let text = r"
Minimize
 obj: x + [ x^2 + 4 x*y ] / 2
Subject To
 q: x^2 + y^2 <= 9
End
";
    let model = read_lp(text.as_bytes()).expect("quadratic LP should parse");
    assert_eq!(model.num_variables(), 2);
    assert_eq!(model.num_constraints(), 1);
    assert!(matches!(model.kind(), ModelKind::QCP | ModelKind::QP));
}

#[test]
fn accepts_moi_style_aliases_and_comparisons() {
    let text = "minimum obj: x\nsubject to\n c1: x < 2\n c2: x > -3\n c3: x == 1\n c4: x =< 4\n c5: x => -4\ninteger\n x\nEnd\n";
    let model = read_lp(text.as_bytes()).expect("aliases should parse");
    assert_eq!(model.num_constraints(), 5);
    assert!(matches!(model.variables()[0].domain, Domain::Integer));
}

#[test]
fn constraint_rhs_can_continue_on_next_line() {
    let text = "Minimize\n obj: x\nSubject To\n c: x <=\n 5\nEnd\n";
    let model = read_lp(text.as_bytes()).expect("continued RHS should parse");
    assert_eq!(model.num_constraints(), 1);
    let constraints = model.constraints();
    let (sense, rhs) = constraints.algebraic()[0].as_single().expect("single row");
    assert_eq!(sense, Sense::Le);
    assert!((rhs - 5.0).abs() < f64::EPSILON);
}

#[test]
fn unnamed_constraint_names_skip_explicit_names() {
    let text = "Minimize\n obj: x\nSubject To\n x <= 5\n c0: x >= 0\n x = 2\nEnd\n";
    let model = read_lp(text.as_bytes()).expect("constraint names should be unique");
    let constraints = model.constraints();
    let names: Vec<_> = constraints.algebraic().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["c1", "c0", "c2"]);
}

#[test]
fn duplicate_constraint_names_are_rejected() {
    let text = "Minimize\n obj: x\nSubject To\n c: x <= 5\n c: x >= 0\nEnd\n";
    let err = read_lp(text.as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::InvalidLp { .. }));
}

#[test]
fn unrepresentable_variable_names_are_rejected() {
    let text = "Minimize\n obj: x\nBounds\n 1x >= 0\nEnd\n";
    let err = read_lp(text.as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::InvalidLp { .. }));
}

#[test]
fn negative_upper_bound_without_lower_bound_is_invalid() {
    let err = read_lp("Minimize\n obj: x\nBounds\n x <= -1\nEnd\n".as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::InvalidLp { .. }));
}

#[test]
fn writer_output_round_trips() {
    let m = Model::new("roundtrip");
    variable!(m, 0.0 <= x <= 5.0);
    variable!(m, y, Integer);
    constraint!(m, c, x + y >= 2.0);
    objective!(m, Min, x + 2.0 * y);
    let text = to_lp_string(&m).expect("write LP");
    let got = read_lp(text.as_bytes()).expect("read written LP");
    assert_eq!(got.num_variables(), 2);
    assert_eq!(got.num_constraints(), 1);
    assert!(matches!(got.variables()[1].domain, Domain::Integer));
}

#[test]
fn writer_preserves_objective_constant() {
    let m = Model::new("constant");
    variable!(m, x);
    objective!(m, Min, x + 4.0);
    let text = to_lp_string(&m).expect("write LP");
    assert!(text.contains("+ 4"), "{text}");
    let got = read_lp(text.as_bytes()).expect("read written LP");
    assert!(got.try_objective().is_ok());
}

#[test]
fn writer_formats_negative_objective_constant() {
    let m = Model::new("negative_constant");
    variable!(m, x);
    objective!(m, Min, x - 4.0);
    let text = to_lp_string(&m).expect("write LP");
    assert!(text.contains("obj: x - 4"), "{text}");
}

#[test]
fn file_reader_uses_stem_as_model_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("example.lp");
    std::fs::write(&path, "Minimize\n obj: x\nEnd\n").expect("write fixture");
    let model = read_lp_file(&path).expect("read fixture");
    assert_eq!(model.name, "example");
}

#[test]
fn stream_reader_preserves_read_errors() {
    struct FailingReader;

    impl Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("read failed"))
        }
    }

    let err = read_lp(FailingReader).unwrap_err();
    assert!(matches!(err, IoError::Io(_)));
}

#[test]
fn stream_reader_does_not_materialize_the_input() {
    struct ReadToStringFails(Cursor<&'static [u8]>);

    impl Read for ReadToStringFails {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(buf)
        }

        fn read_to_string(&mut self, _buf: &mut String) -> io::Result<usize> {
            Err(io::Error::other("read_to_string should not be called"))
        }
    }

    let input = ReadToStringFails(Cursor::new(b"Minimize\n obj: x\nEnd\n"));
    let model = read_lp(input).expect("LP reader should consume lines through BufReader");
    assert_eq!(model.num_variables(), 1);
}

#[test]
fn file_reader_preserves_open_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing.lp");
    let err = read_lp_file(path).unwrap_err();
    assert!(matches!(err, IoError::Io(_)));
}

#[test]
fn unsupported_sections_are_reported() {
    let err = read_lp("Minimize\n obj: x\nIndicators\n x\nEnd\n".as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::UnsupportedLp { .. }));
}

#[test]
fn inline_sos_and_indicators_are_explicitly_unsupported() {
    for row in [" c: S1:: x:1 y:2", " c: z = 1 -> x <= 2"] {
        let text = format!("Minimize\n obj: x\nSubject To\n{row}\nEnd\n");
        let err = read_lp(text.as_bytes()).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedLp { .. }), "{err:?}");
    }
}

#[test]
fn sos_sections_round_trip() {
    let text = "Minimize\n obj: x + y + z\nSOS\n s1: S1 :: x : 1 y : 2\n s2: S2 :: x : 1 y : 2 z : 3\nEnd\n";
    let model = read_lp(text.as_bytes()).expect("SOS LP");
    assert_eq!(model.num_sos_constraints(), 2);
    assert_eq!(model.kind(), oximo_core::ModelKind::MILP);
    let output = to_lp_string(&model).expect("write SOS LP");
    let roundtrip = read_lp(output.as_bytes()).expect("read written SOS LP");
    assert_eq!(roundtrip.num_sos_constraints(), 2);
    assert_eq!(roundtrip.sos_constraints()[0].name, "s1");
    assert_eq!(roundtrip.sos_constraints()[1].name, "s2");
    assert!((roundtrip.sos_constraints()[1].members[2].weight - 3.0).abs() < 1e-12);
}

#[test]
fn sos_name_with_colon_is_rejected_by_writer() {
    let m = Model::new("sos_colon_name");
    variable!(m, x);
    m.add_sos_constraint("choice:colon", SosType::Sos1, [(x, 1.0)]);
    objective!(m, Min, x);

    assert!(matches!(to_lp_string(&m), Err(IoError::InvalidLp { .. })));
}

#[test]
fn sos_signed_weights_round_trip() {
    let text = "Minimize\n obj: x + y + z\nSOS\n set: S2 :: x : -2.5 y : +1e-2 z : 3\nEnd\n";
    let model = read_lp(text.as_bytes()).expect("signed SOS weights should parse");
    let output = to_lp_string(&model).expect("write signed SOS weights");
    let roundtrip = read_lp(output.as_bytes()).expect("read signed SOS weights");
    let members = &roundtrip.sos_constraints()[0].members;
    assert!((members[0].weight + 2.5).abs() < 1e-12);
    assert!((members[1].weight - 1e-2).abs() < 1e-12);
}

#[test]
fn malformed_sos_rows_are_rejected() {
    for row in [
        "choice S1 :: x : 1",
        "choice:colon: S1 :: x : 1",
        "choice: S3 :: x : 1",
        "choice: S1 :: x",
        "choice: S1 :: x : nope",
        "choice: S1 :: x : NaN",
        "choice: S1 ::",
        "choice: S1 :: x : 1 x : 2",
        "choice: S1 :: x : 1 y : 1",
    ] {
        let text = format!("Minimize\n obj: x + y\nSOS\n {row}\nEnd\n");
        assert!(matches!(read_lp(text.as_bytes()), Err(IoError::InvalidLp { .. })), "{row}");
    }
}

#[test]
fn content_after_end_is_rejected() {
    let err = read_lp("Minimize\n obj: x\nEnd\nBounds\n x >= 0\n".as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::InvalidLp { .. }));
}

#[test]
fn malformed_input_has_lp_diagnostic() {
    let err =
        read_lp("Minimize\n obj: x\nSubject To\n bad: x <= nope\nEnd\n".as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::InvalidLp { .. }));
}

#[test]
fn parser_error_uses_offending_token_column() {
    let err = read_lp("Minimize\n obj: x + * y\nEnd\n".as_bytes()).unwrap_err();
    match err {
        IoError::InvalidLp { column, .. } => assert_eq!(column, 6),
        other => panic!("expected InvalidLp, got {other:?}"),
    }
}

#[test]
fn higher_degree_expression_is_invalid_lp_syntax() {
    let err = read_lp("Minimize\n obj: x^3\nEnd\n".as_bytes()).unwrap_err();
    assert!(matches!(err, IoError::InvalidLp { .. }));
}
