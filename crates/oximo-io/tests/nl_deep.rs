use oximo_core::prelude::*;
use oximo_expr::evaluate;
use oximo_io::{NlFormat, WriteOptions, read_nl, to_nl_string, write_nl_with};

#[test]
fn deferred_affine_sum_inside_nonlinear_objective_roundtrips() {
    let m = Model::new("deep affine NL");
    variable!(m, x[i in 0..20_000]);
    let sum = (0..20_000).map(|i| x[i]).reduce(|a, b| a + b).unwrap();
    objective!(m, Min, sum.sin());
    let ascii = to_nl_string(&m).unwrap();
    assert!(ascii.contains("O0 0\no41\n"));
    assert!(ascii.contains("G0 20000\n"));
    assert!(ascii.contains("v19999\n"));
    let mut binary = Vec::new();
    let options = WriteOptions { format: NlFormat::Binary, ..WriteOptions::default() };
    write_nl_with(&m, &mut binary, &options).unwrap();
    assert!(binary.starts_with(b"b3"));
    for bytes in [ascii.as_bytes(), binary.as_slice()] {
        let imported = read_nl(bytes).unwrap();
        let values = vec![1.0 / 20_000.0; 20_000];
        let objective = imported.objective();
        let actual =
            evaluate(&imported.arena(), objective.as_ref().unwrap().expr, &&values[..]).unwrap();
        assert!((actual - 1.0_f64.sin()).abs() < 1e-12, "{actual}");
    }
}

#[test]
fn deeply_nested_nonlinear_nodes_export_without_stack_overflow() {
    let m = Model::new("deep nonlinear NL");
    variable!(m, x);
    let mut expression = x.sin();
    for _ in 0..20_000 {
        expression = expression.sin();
    }
    objective!(m, Min, expression);
    let ascii = to_nl_string(&m).unwrap();
    assert_eq!(ascii.matches("o41\n").count(), 20_001);
    assert!(ascii.contains("G0 1\n0 0\n"));
}

#[test]
fn deferred_parameter_affine_subtree_roundtrips_after_rebinding() {
    let m = Model::new("parameter affine NL");
    variable!(m, x[i in 0..512]);
    param!(m, p = 2.0);
    let sum = (0..512).map(|i| p * x[i]).reduce(|a, b| a + b).unwrap();
    let expression = sum.sin();
    m.add_constraint("shared", expression.le(1.0));
    objective!(m, Min, expression);
    for coefficient in [2.0_f64, 3.0] {
        m.set_param(p, coefficient).unwrap();
        for format in [NlFormat::Ascii, NlFormat::Binary] {
            let mut bytes = Vec::new();
            write_nl_with(&m, &mut bytes, &WriteOptions { format, ..WriteOptions::default() })
                .unwrap();
            let imported = read_nl(bytes.as_slice()).unwrap();
            let values = vec![1.0 / 512.0; 512];
            let objective = imported.objective();
            let actual =
                evaluate(&imported.arena(), objective.as_ref().unwrap().expr, &&values[..])
                    .unwrap();
            assert!((actual - coefficient.sin()).abs() < 1e-12, "{actual}");
        }
    }
    m.set_param(p, f64::NAN).unwrap();
    assert!(to_nl_string(&m).is_err(), "a new export must revalidate changed parameters");
}

#[test]
fn mixed_additive_subtree_keeps_its_nonlinear_term_when_exported() {
    let m = Model::new("mixed inline NL");
    variable!(m, x[i in 0..512]);
    let sum = (0..512).map(|i| x[i]).reduce(|a, b| a + b).unwrap();
    objective!(m, Min, (sum + x[0].sin()).exp());
    let bytes = to_nl_string(&m).unwrap();
    let imported = read_nl(bytes.as_bytes()).unwrap();
    let values = vec![1.0 / 512.0; 512];
    let objective = imported.objective();
    let actual =
        evaluate(&imported.arena(), objective.as_ref().unwrap().expr, &&values[..]).unwrap();
    let expected = (1.0 + (1.0_f64 / 512.0).sin()).exp();
    assert!((actual - expected).abs() < 1e-12, "{actual}");
}
