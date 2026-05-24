//! Logical inference for reaction path synthesis (MILP).
//!
//! Given 22 possible chemical reactions over 34 chemicals, determine whether
//! acetone (y06, ch3coch3) can be synthesized from a fixed set of raw materials
//! and catalysts.
//!
//! Binary variable `y[v] = 1` if chemical `v` is synthesizable. Available raw
//! materials are fixed to 1, unavailable chemicals to 0. For each reaction that
//! can produce `v` from reactants `vv`:
//!
//! ```text
//! sum_vv (1 - y[vv]) >= 1 - y[v]
//! ```
//!
//! This forces: if all reactants are present, the product must be present.
//! Minimizing `y[y06]` reveals whether acetone is synthesizable (optimal = 1).
//!
//! This example is translated from GAMS code in the REACTION model (SEQ=121).
//!
//! Reference:
//! Raman, R, and Grossmann, I E, "Relation between MINLP Modeling
//! and Logical Inference for Chemical Process Synthesis", Computers and
//! Chemical Engineering 15, 2 (1991), 73–84.
//!
//! Run with HiGHS (default):
//! ```text
//! cargo run --example reaction_path
//! ```
//!
//! Run with GAMS / CPLEX:
//! ```text
//! cargo run --example reaction_path --features gams
//! ```
//! Requires a licensed GAMS installation with CPLEX on PATH.

#![allow(clippy::cast_precision_loss)]

#[cfg(any(feature = "gams", feature = "highs"))]
use oximo::prelude::*;

#[cfg(feature = "gams")]
use oximo::gams::{GamsCplexOptions, GamsSolverConfig};
#[cfg(feature = "gams")]
use oximo::solvers::Gams;

#[cfg(all(feature = "highs", not(feature = "gams")))]
use oximo::solvers::Highs;

#[cfg(any(feature = "gams", feature = "highs"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 34 chemicals: index = yXX - 1 (y01 -> 0, ..., y34 -> 33).
    const CHEMICALS: [&str; 34] = [
        "y01", "y02", "y03", "y04", "y05", "y06", "y07", "y08", "y09", "y10", "y11", "y12", "y13",
        "y14", "y15", "y16", "y17", "y18", "y19", "y20", "y21", "y22", "y23", "y24", "y25", "y26",
        "y27", "y28", "y29", "y30", "y31", "y32", "y33", "y34",
    ];

    // (reaction label, product index, reactant indices) - all 0-based.
    // Transcribed from logicc set in GAMS REACTION model (SEQ=121).
    let logicc: &[(&str, usize, &[usize])] = &[
        ("rxn01", 3, &[0, 1, 2]),      // y04 <- y01 + y02 + y03
        ("rxn02", 5, &[3, 4]),         // y06 <- y04 + y05
        ("rxn03", 6, &[3, 4]),         // y07 <- y04 + y05
        ("rxn04", 2, &[3, 4]),         // y03 <- y04 + y05
        ("rxn05", 10, &[7, 8, 9]),     // y11 <- y08 + y09 + y10
        ("rxn06", 5, &[10, 11, 12]),   // y06 <- y11 + y12 + y13
        ("rxn07", 14, &[13, 8, 9, 4]), // y15 <- y14 + y09 + y10 + y05
        ("rxn08", 5, &[14, 15, 16]),   // y06 <- y15 + y16 + y17
        ("rxn09", 5, &[17, 18, 11]),   // y06 <- y18 + y19 + y12
        ("rxn10", 19, &[17, 18, 11]),  // y20 <- y18 + y19 + y12
        ("rxn11", 8, &[20, 21]),       // y09 <- y21 + y22
        ("rxn12", 23, &[8, 22]),       // y24 <- y09 + y23
        ("rxn13", 17, &[23, 16]),      // y18 <- y24 + y17
        ("rxn14", 20, &[24, 25]),      // y21 <- y25 + y26
        ("rxn15", 26, &[24, 25]),      // y27 <- y25 + y26
        ("rxn16", 13, &[2, 27, 28]),   // y14 <- y03 + y28 + y29
        ("rxn17", 31, &[29, 30, 11]),  // y32 <- y30 + y31 + y12
        ("rxn18", 7, &[29, 30, 11]),   // y08 <- y30 + y31 + y12
        ("rxn19", 29, &[24, 32]),      // y30 <- y25 + y33
        ("rxn20", 12, &[24, 32]),      // y13 <- y25 + y33
        ("rxn21", 0, &[33, 2]),        // y01 <- y34 + y03
        ("rxn22", 33, &[13, 27]),      // y34 <- y14 + y28
    ];

    // y02, y03, y05, y10, y12, y13, y17, y22, y25, y26, y28, y31, y33: fixed to 1.
    let available: &[usize] = &[1, 2, 4, 9, 11, 12, 16, 21, 24, 25, 27, 30, 32];
    // y16, y19: fixed to 0.
    let unavailable: &[usize] = &[15, 18];

    let m = Model::new("reaction_path");
    let chemicals = Set::strings(CHEMICALS);

    let y = m
        .indexed_var("y", &chemicals)
        .binary()
        .lb_by(
            |name: String| {
                if available.iter().any(|&i| CHEMICALS[i] == name) { 1.0 } else { 0.0 }
            },
        )
        .ub_by(
            |name: String| {
                if unavailable.iter().any(|&i| CHEMICALS[i] == name) { 0.0 } else { 1.0 }
            },
        )
        .build();

    // sum_vv (1 - y[vv]) >= 1 - y[v]
    //    <=>  y[v] - sum_vv y[vv] >= 1 - |reactants|
    for &(rx, prod, reactants) in logicc {
        let n = reactants.len() as f64;
        let reactant_sum = sum_over(reactants, |vv: usize| y[CHEMICALS[vv]]);
        m.constraint(format!("leq_{rx}"), (y[CHEMICALS[prod]] - reactant_sum).ge(1.0 - n));
    }

    m.minimize(y["y06"]); // acetone

    #[cfg(feature = "gams")]
    let result = {
        let opts = GamsOptions::default()
            .time_limit(std::time::Duration::from_secs(60))
            .solver(GamsSolverConfig::Cplex(GamsCplexOptions::default()))
            .verbose(true);
        let mut solver = Gams::new();
        solver.solve(&m, &opts)?
    };

    #[cfg(all(feature = "highs", not(feature = "gams")))]
    let result = Highs.solve(&m, &HighsOptions::default().verbose(true))?;

    println!("Status : {:?}", result.status);
    if let Some(obj) = result.objective {
        println!(
            "Acetone (y06, ch3coch3): {}",
            if (obj - 1.0).abs() < 1e-6 { "Synthesizable" } else { "Not synthesizable" }
        );
    }

    let synthesizable: Vec<&str> = CHEMICALS
        .iter()
        .copied()
        .filter(|name| (result.value_of(y[*name]).unwrap_or(0.0) - 1.0).abs() < 1e-6)
        .collect();
    if !synthesizable.is_empty() {
        println!("Synthesizable chemicals: {}", synthesizable.join(", "));
    }

    Ok(())
}

#[cfg(not(any(feature = "gams", feature = "highs")))]
fn main() {
    println!("Enable at least one solver feature:");
    println!("  cargo run --example reaction_path                  # HiGHS (default)");
    println!("  cargo run --example reaction_path --features gams  # GAMS/CPLEX");
}
