//! Writer coverage for semicontinuous/semi-integer variables in the LP and
//! MPS exporters.

use oximo_core::prelude::*;
use oximo_io::{to_lp_string, to_mps_string};

/// The variable list written on the line after a section `header`.
fn section_list<'a>(text: &'a str, header: &str) -> Option<&'a str> {
    text.lines().skip_while(|l| !l.starts_with(header)).nth(1)
}

/// Build: min s + t  s.t. s + t >= 3, with s semicontinuous (0 or [2, 10]) and
/// t semi-integer (0 or integer in [1, 5]).
fn semi_model() -> Model {
    let m = Model::new("semi");
    variable!(m, s <= 10.0, SemiCont(2.0));
    variable!(m, t <= 5.0, SemiInt(1.0));
    objective!(m, Min, s + t);
    constraint!(m, c0, s + t >= 3.0);
    m
}

#[test]
fn lp_emits_semi_continuous_section_and_threshold_bounds() {
    let lp = to_lp_string(&semi_model()).expect("lp writer");

    assert!(lp.contains("2 <= s <= 10"), "missing semicont bound:\n{lp}");
    assert!(lp.contains("1 <= t <= 5"), "missing semiint bound:\n{lp}");

    let semi_line = section_list(&lp, "Semi-Continuous").expect("Semi-Continuous section");
    assert!(semi_line.contains('s') && semi_line.contains('t'), "semi list: {semi_line:?}");

    let general_line = section_list(&lp, "General").expect("General section");
    assert!(general_line.contains('t'), "general list: {general_line:?}");
}

#[test]
fn mps_emits_sc_and_si_bounds() {
    let mps = to_mps_string(&semi_model()).expect("mps writer");
    let has = |tokens: &[&str]| mps.lines().any(|l| tokens.iter().all(|t| l.contains(t)));

    assert!(has(&["SC BND", "s", "10"]), "missing SC for s:\n{mps}");
    assert!(has(&["LO BND", "s", "2"]), "missing LO threshold for s:\n{mps}");
    assert!(has(&["SI BND", "t", "5"]), "missing SI for t:\n{mps}");
    assert!(has(&["LO BND", "t", "1"]), "missing LO threshold for t:\n{mps}");
    assert!(!mps.contains("INTORG"), "semi-integer must not be integer-marked:\n{mps}");
}
