#![allow(unused_imports, unused_variables, dead_code)]

use oximo_gdp::prelude::*;
fn misuse(model: &Model, report: &GdpReformulationReportData) {
    let _ = model.variable_handle(report.hull_variables[0].source);
}
fn main() {}
