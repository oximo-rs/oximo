use oximo_core::prelude::*;
use oximo_expr::evaluate;
use oximo_io::{IoError, WriteOptions, read_nl, to_nl_string, write_nl_with};

fn full_model() -> Model {
    let m = Model::new("nl_functions");
    variable!(m, -0.9 <= x <= 0.9);
    let positive = x + 2.0;
    let e = -x.sin()
        + x.abs()
        + positive.sqrt()
        + x.exp()
        + x.exp2()
        + positive.log()
        + positive.log10()
        + x.sin()
        + x.cos()
        + x.tan()
        + x.asin()
        + x.acos()
        + x.atan()
        + x.sinh()
        + x.cosh()
        + x.tanh()
        + x.asinh()
        + positive.acosh()
        + x.atanh()
        + x.atan2(positive)
        + x.min(positive)
        + x.max(positive);
    objective!(m, Min, e);
    m
}

fn objective_value(model: &Model, x: f64) -> f64 {
    let values: &[f64] = &[x];
    let objective_ref = model.objective();
    let objective = objective_ref.as_ref().unwrap();
    evaluate(&model.arena(), objective.expr, &values).unwrap()
}

#[test]
fn documented_unary_opcodes_are_exact() {
    macro_rules! assert_root_opcode {
        ($name:literal, $opcode:literal, |$arg:ident| $body:expr) => {{
            let m = Model::new($name);
            variable!(m, x);
            let $arg: oximo_expr::Expr<'_> = x;
            let expression = $body;
            objective!(m, Min, expression);

            let text = to_nl_string(&m).unwrap();
            let mut lines = text.lines();
            let objective_header = lines.find(|line| line.starts_with("O0 "));
            assert_eq!(objective_header, Some("O0 0"), "{name}: {text}", name = $name);
            assert_eq!(
                lines.next(),
                Some(concat!("o", stringify!($opcode))),
                "{name}: {text}",
                name = $name
            );
        }};
    }

    assert_root_opcode!("abs", 15, |x| x.abs());
    assert_root_opcode!("neg", 16, |x| x.sin().neg());
    assert_root_opcode!("tanh", 37, |x| x.tanh());
    assert_root_opcode!("tan", 38, |x| x.tan());
    assert_root_opcode!("sqrt", 39, |x| x.sqrt());
    assert_root_opcode!("sinh", 40, |x| x.sinh());
    assert_root_opcode!("sin", 41, |x| x.sin());
    assert_root_opcode!("log10", 42, |x| x.log10());
    assert_root_opcode!("log", 43, |x| x.log());
    assert_root_opcode!("exp", 44, |x| x.exp());
    assert_root_opcode!("cosh", 45, |x| x.cosh());
    assert_root_opcode!("cos", 46, |x| x.cos());
    assert_root_opcode!("atanh", 47, |x| x.atanh());
    assert_root_opcode!("atan", 49, |x| x.atan());
    assert_root_opcode!("asinh", 50, |x| x.asinh());
    assert_root_opcode!("asin", 51, |x| x.asin());
    assert_root_opcode!("acosh", 52, |x| x.acosh());
    assert_root_opcode!("acos", 53, |x| x.acos());
}

#[test]
fn documented_ascii_opcodes_round_trip() {
    let model = full_model();
    let text = to_nl_string(&model).unwrap();
    for opcode in
        [11_u32, 12, 15, 16, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    {
        assert!(text.lines().any(|line| line.starts_with(&format!("o{opcode}"))), "o{opcode}");
    }
    // exp2 is the exact documented pow encoding, `2 ^ x`.
    assert!(text.contains("o5\nn2\n"), "{text}");

    let decoded = read_nl(text.as_bytes()).unwrap();
    for x in [-0.5, 0.25, 0.75] {
        let expected = objective_value(&model, x);
        let actual = objective_value(&decoded, x);
        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }
}

#[test]
fn documented_binary_opcodes_round_trip() {
    let model = full_model();
    let mut bytes = Vec::new();
    write_nl_with(&model, &mut bytes, &WriteOptions::binary()).unwrap();
    let decoded = read_nl(bytes.as_slice()).unwrap();
    for x in [-0.25, 0.5] {
        let expected = objective_value(&model, x);
        let actual = objective_value(&decoded, x);
        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }
}

#[test]
fn undocumented_unary_families_return_typed_errors() {
    for name in ["cbrt", "expm1", "log1p", "log2"] {
        let m = Model::new(name);
        variable!(m, x);
        let expr = match name {
            "cbrt" => x.cbrt(),
            "expm1" => x.expm1(),
            "log1p" => x.log1p(),
            "log2" => x.log2(),
            _ => unreachable!(),
        };
        objective!(m, Min, expr);
        assert!(matches!(
            to_nl_string(&m),
            Err(IoError::UnsupportedNonlinearOperator { operator }) if operator == name
        ));
    }
}
