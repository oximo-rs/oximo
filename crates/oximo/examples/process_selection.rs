//! Structural optimization of a process flowsheet
//! This is the GAMS model library problem `PROCSEL`
//! (SEQ=116), from:
//!
//!   Kocis & Grossmann (1987), "Relaxation Strategy for the Structural
//!   Optimization of Process Flow Sheets", Ind. Eng. Chem. Res. 26(9),
//!   1869-1880; also Morari & Grossmann (eds.), "Chemical Engineering
//!   Optimization Models with GAMS" (1991).
//!
//! Chemical C is produced from B in unit 1. B is either purchased on the
//! external market (`bp`) or produced from raw material A through one of two
//! competing units (2 or 3). Binaries `y1,y2,y3` switch the three units on/off;
//! the goal is to maximise annual profit.
//!
//!         A2    +-----+  B2      BP
//!        +----->|  2  |----->+    |
//!   A    |      +-----+      |    |  B1    +-----+    C1
//!   ---->|                   +----+------->|  1  |-------->
//!        |      +-----+      |             +-----+
//!        +----->|  3  |----->+
//!         A3    +-----+  B3
//!
//! The input-output relations of units 2 and 3 are the (convexified) nonlinear
//! laws `exp(b2) - 1 = a2` and `exp(b3/1.2) - 1 = a3`, which make this an MINLP.
//!
//! Run with one nonlinear backend (enable exactly one, never both at once):
//!   cargo run --example process_selection --features gams
//!   cargo run --example process_selection --features gurobi

#[cfg(any(feature = "gams", feature = "gurobi"))]
mod model {
    use oximo::prelude::*;

    fn solve_and_report<S: Solver>(
        label: &str,
        mut solver: S,
        opts: &S::Options,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let m = Model::new("procsel");

        // Positive variables (consumptions, capacities, purchases).
        variable!(m, a2 >= 0.0);
        variable!(m, a3 >= 0.0);
        variable!(m, b2 >= 0.0);
        variable!(m, b3 >= 0.0);
        variable!(m, bp >= 0.0);
        variable!(m, b1 >= 0.0);
        variable!(m, 0.0 <= c1 <= 1.0);

        // Binaries: existence of each process unit.
        variable!(m, y1, Bin);
        variable!(m, y2, Bin);
        variable!(m, y3, Bin);

        constraint!(m, inout1, c1 == 0.9 * b1);
        constraint!(m, inout2, b2.exp() - 1.0 == a2);
        constraint!(m, inout3, (b3 / 1.2).exp() - 1.0 == a3);
        constraint!(m, mbalb, b1 == b2 + b3 + bp);
        constraint!(m, log1, c1 <= 2.0 * y1);
        constraint!(m, log2, b2 <= 4.0 * y2);
        constraint!(m, log3, b3 <= 5.0 * y3);

        // profit = sales - fixed investment - operating cost - purchases
        objective!(
            m,
            Max,
            11.0 * c1 - 3.5 * y1 - y2 - 1.5 * y3 - b2 - 1.2 * b3 - 1.8 * (a2 + a3) - 7.0 * bp
        );

        let result = solver.solve(&m, opts)?;

        if let Some(obj) = result.objective() {
            println!("--- {label} ---");
            println!("status = {:?}", result.termination);
            println!("profit = {obj:.4} M$/yr");

            println!("units selected:");
            for (name, y) in [("process 1", y1), ("process 2", y2), ("process 3", y3)] {
                println!(
                    "  {name}: {}",
                    if (result.value_of(y).unwrap_or(0.0) - 1.0).abs() < f64::EPSILON {
                        "ON"
                    } else {
                        "OFF"
                    }
                );
            }

            println!("flows:");
            println!("  c1 (C produced)   = {:.4}", result.value_of(c1).unwrap_or(0.0));
            println!("  b1 (B into 1)     = {:.4}", result.value_of(b1).unwrap_or(0.0));
            println!(
                "  b2, b3 (B from A) = {:.4}, {:.4}",
                result.value_of(b2).unwrap_or(0.0),
                result.value_of(b3).unwrap_or(0.0)
            );
            println!("  bp (B purchased)  = {:.4}", result.value_of(bp).unwrap_or(0.0));
            println!(
                "  a2, a3 (A used)   = {:.4}, {:.4}",
                result.value_of(a2).unwrap_or(0.0),
                result.value_of(a3).unwrap_or(0.0)
            );
        } else {
            println!("--- {label} ---");
            println!("status = {:?}", result.termination);
            println!("no objective value");
        }

        Ok(())
    }

    #[cfg(feature = "gams")]
    pub fn run_gams() -> Result<(), Box<dyn std::error::Error>> {
        use oximo::gams::{GamsBaronOptions, GamsSolverConfig};
        use oximo::solvers::Gams;
        use std::time::Duration;

        let opts = GamsOptions::default()
            .time_limit(Duration::from_secs(120))
            .solver(GamsSolverConfig::Baron(GamsBaronOptions {
                eps_r: Some(1e-4),
                ..Default::default()
            }))
            .verbose(true);
        solve_and_report("GAMS + BARON", Gams::new(), &opts)
    }

    #[cfg(feature = "gurobi")]
    pub fn run_gurobi() -> Result<(), Box<dyn std::error::Error>> {
        use oximo::solvers::Gurobi;
        use std::time::Duration;

        let opts = GurobiOptions::default().time_limit(Duration::from_secs(120));
        solve_and_report("Gurobi", Gurobi, &opts)
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "gams")]
        run_gams()?;
        #[cfg(feature = "gurobi")]
        run_gurobi()?;
        Ok(())
    }
}

#[cfg(any(feature = "gams", feature = "gurobi"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    model::run()
}

#[cfg(not(any(feature = "gams", feature = "gurobi")))]
fn main() {
    println!("Enable a nonlinear-capable backend feature:");
    println!("  cargo run --example process_selection --features gams");
    println!("  cargo run --example process_selection --features gurobi");
}
