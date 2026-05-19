//! Capacitated lot-sizing MILP.
//!
//! A manufacturer plans monthly production over T = 12 periods to meet known
//! seasonal demand. Each period has a variable production cost, a fixed setup
//! cost (only incurred when the line runs), and a per-unit holding cost.
//! Production is bounded by a capacity limit, and a safety-stock requirement
//! must hold at the end of the horizon.
//!
//! This model is a simplified single-item version of the classic capacitated
//! lot-sizing problem, as described by Wilson et al. (2003) and other works.
//!
//! # Model
//!
//! ```text
//! Variables
//!   x[t] in [0, cap] - units produced in period t (continuous)
//!   h[t] in [0, +inf) - inventory at end of period t (continuous)
//!   s[t] in {0, 1} - 1 if and only if the production line runs in period t (binary)
//!
//! minimize
//!   sum_t  prod_cost[t]*x[t] + setup_cost*s[t] + hold_cost*h[t]
//!
//! s.t.
//!   h[0] - x[0] = initial_inventory - demand[0]
//!   h[t] - h[t-1] - x[t] = -demand[t], for t >= 1
//!   x[t] - capacity*s[t] <= 0, for all t
//!   h[T-1] >= safety_stock
//! ```
//!
//! Run with HiGHS (default):
//! ```text
//! cargo run --example lot_sizing
//! ```
//!
//! Run with Gurobi:
//! ```text
//! cargo run --example lot_sizing --features gurobi
//! ```
//!
//! References:
//! Karimi, B., Fatemi Ghomi, S.M.T., Wilson, J.M.
//! "The capacitated lot sizing problem: a review of models
//! and algorithms", Omega, 31(5), 365-378, 2003.

#![allow(clippy::many_single_char_names)]

#[cfg(any(feature = "gurobi", feature = "highs"))]
use oximo::prelude::*;

#[cfg(feature = "gurobi")]
use oximo::{GurobiOptions, solvers::Gurobi};

#[cfg(all(feature = "highs", not(feature = "gurobi")))]
use oximo::{HighsOptions, solvers::Highs};

#[cfg(any(feature = "gurobi", feature = "highs"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    const T: usize = 12;

    let demand: [f64; T] =
        [120.0, 90.0, 80.0, 140.0, 160.0, 200.0, 220.0, 190.0, 150.0, 130.0, 100.0, 170.0];
    let prod_cost: [f64; T] = [5.0, 5.0, 5.0, 5.5, 6.0, 6.5, 6.5, 6.0, 5.5, 5.0, 5.0, 5.5];
    let setup_cost = 500.0;
    let hold_cost = 2.0;
    let capacity = 300.0;
    let initial_inventory = 50.0;
    let safety_stock = 30.0;

    let m = Model::new("lot_sizing");
    let periods = Set::range(0..T);

    let x = m.indexed_var("x", &periods).lb(0.0).ub(capacity).build();
    let h = m.indexed_var("h", &periods).lb(0.0).build();
    let s = m.indexed_var("s", &periods).binary().build();

    m.constraint("inv_bal[0]", (h[0] - x[0]).eq(initial_inventory - demand[0]));
    m.add_constraints_over("inv_bal", &periods.filter(|k| k.as_i64().unwrap() > 0), |t: usize| {
        (h[t] - h[t - 1] - x[t]).eq(-demand[t])
    });
    m.add_constraints_over("setup", &periods, |t: usize| (x[t] - capacity * s[t]).le(0.0));
    m.constraint("safety_stock", h[T - 1].ge(safety_stock));

    let cost = sum(periods.iter().map(|k| {
        let t: usize = FromIndexKey::from_index_key(&k);
        prod_cost[t] * x[t] + setup_cost * s[t] + hold_cost * h[t]
    }));
    m.minimize(cost);

    #[cfg(feature = "gurobi")]
    let result = {
        let opts = GurobiOptions::default()
            .time_limit(std::time::Duration::from_secs(60))
            .mip_gap(1e-4)
            .verbose(true);
        Gurobi.solve(&m, &opts)?
    };

    #[cfg(all(feature = "highs", not(feature = "gurobi")))]
    let result = {
        let opts = HighsOptions::default()
            .time_limit(std::time::Duration::from_secs(60))
            .mip_gap(1e-4)
            .verbose(true);
        Highs.solve(&m, &opts)?
    };

    println!("\nLot-Sizing Result");
    println!("Status    : {:?}", result.status);
    if let Some(obj) = result.objective {
        println!("Total cost: {obj:.2}");
    }

    println!(
        "\n{:<8} {:>10} {:>10} {:>8} {:>12}",
        "Period", "Produce", "Inventory", "Active", "Period cost"
    );
    println!("{}", "-".repeat(55));

    let mut total_check = 0.0;
    for t in 0..T {
        let xt = result.value_of(x[t]).unwrap_or(0.0);
        let ht = result.value_of(h[t]).unwrap_or(0.0);
        let st = result.value_of(s[t]).unwrap_or(0.0);
        let period_cost = prod_cost[t] * xt + setup_cost * st + hold_cost * ht;
        total_check += period_cost;
        println!(
            "{:<8} {:>10.1} {:>10.1} {:>8} {:>12.2}",
            t + 1,
            xt,
            ht,
            if (st - 1.0).abs() < 1e-6 { "Yes" } else { "No" },
            period_cost
        );
    }
    println!("{}", "-".repeat(55));
    println!("{:<8} {:>10} {:>10} {:>8} {:>12.2}", "TOTAL", "", "", "", total_check);

    Ok(())
}

#[cfg(not(any(feature = "gurobi", feature = "highs")))]
fn main() {
    println!("Enable at least one solver feature:");
    println!("  cargo run --example lot_sizing                   # HiGHS (default)");
    println!("  cargo run --example lot_sizing --features gurobi # Gurobi");
}
