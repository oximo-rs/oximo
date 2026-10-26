#![allow(unused_imports, unused_variables, dead_code)]

use oximo_gdp::prelude::*;
fn misuse(model: &Model, report: &GdpReformulationReportData) {
    let _ = model.constraint_handle_from_id(report.rows[0].constraints[0]);
}
fn main() {}
