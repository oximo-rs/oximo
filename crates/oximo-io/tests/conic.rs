//! Every text format rejects models with explicit second-order cone
//! constraints: LP/MPS/NL have no conic representation.

use oximo_core::prelude::*;
use oximo_io::{
    IoError, MpsQuadraticFormat, MpsWriteOptions, to_lp_string, to_mps_string, to_mps_string_with,
    to_nl_string,
};

fn soc_model() -> Model {
    let m = Model::new("socp");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    m.add_soc_constraint("cone", [x, y], t);
    constraint!(m, c0, x + y >= 1.0);
    objective!(m, Min, t);
    m
}

#[test]
fn lp_writer_rejects_soc_constraints() {
    assert!(matches!(to_lp_string(&soc_model()), Err(IoError::Conic)));
}

#[test]
fn mps_writer_rejects_soc_constraints() {
    assert!(matches!(to_mps_string(&soc_model()), Err(IoError::Conic)));
}

#[test]
fn mps_writer_rejects_soc_constraints_when_indicators_are_present() {
    let m = Model::new("mixed_conic_indicator");
    variable!(m, x);
    variable!(m, y);
    variable!(m, t >= 0.0);
    variable!(m, enabled, Binary);
    m.add_soc_constraint("cone", [x, y], t);
    indicator_constraint!(m, gated, enabled == 1 => x <= 2.0);
    objective!(m, Min, t);

    assert!(matches!(to_mps_string(&m), Err(IoError::Conic)));
}

#[test]
fn nl_writer_rejects_soc_constraints() {
    assert!(matches!(to_nl_string(&soc_model()), Err(IoError::Conic)));
}

fn sos_model() -> Model {
    let m = Model::new("sos");
    variable!(m, x);
    variable!(m, y);
    sos_constraint!(m, choice, SOS1, [x, y]);
    objective!(m, Min, x + y);
    m
}

#[test]
fn nl_writer_rejects_sos_constraints() {
    assert!(matches!(
        to_nl_string(&sos_model()),
        Err(IoError::UnsupportedNl { section, .. }) if section == "SOS"
    ));
}

#[test]
fn mosek_mps_writer_rejects_sos_constraints() {
    let options = MpsWriteOptions { quadratic_format: MpsQuadraticFormat::Mosek };
    assert!(matches!(
        to_mps_string_with(&sos_model(), &options),
        Err(IoError::UnsupportedMps { section, .. }) if section == "SOS"
    ));
}

#[test]
fn text_writers_omit_reformulated_source_sos_constraints() {
    let transformed = sos_model()
        .to_reformulated_sos_model(SosReformulationOptions::default().with_fallback_big_m(10.0))
        .expect("SOS reformulation");

    let lp = to_lp_string(&transformed).expect("reformulated LP");
    assert!(!lp.contains("\nSOS\n"));

    for format in [MpsQuadraticFormat::Gurobi, MpsQuadraticFormat::Cplex, MpsQuadraticFormat::Mosek]
    {
        let options = MpsWriteOptions { quadratic_format: format };
        let mps = to_mps_string_with(&transformed, &options).expect("reformulated MPS");
        assert!(!mps.contains("\nSOS\n"));
    }
}
