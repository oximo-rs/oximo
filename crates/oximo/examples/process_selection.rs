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
        let a2 = m.var("a2").lb(0.0).build();
        let a3 = m.var("a3").lb(0.0).build();
        let b2 = m.var("b2").lb(0.0).build();
        let b3 = m.var("b3").lb(0.0).build();
        let bp = m.var("bp").lb(0.0).build();
        let b1 = m.var("b1").lb(0.0).build();
        let c1 = m.var("c1").lb(0.0).ub(1.0).build();

        // Binaries: existence of each process unit.
        let y1 = m.var("y1").binary().build();
        let y2 = m.var("y2").binary().build();
        let y3 = m.var("y3").binary().build();

        m.constraint("inout1", c1.eq(0.9 * b1));
        m.constraint("inout2", (b2.exp() - 1.0).eq(a2));
        m.constraint("inout3", ((b3 / 1.2).exp() - 1.0).eq(a3));
        m.constraint("mbalb", b1.eq(b2 + b3 + bp));
        m.constraint("log1", c1.le(2.0 * y1));
        m.constraint("log2", b2.le(4.0 * y2));
        m.constraint("log3", b3.le(5.0 * y3));

        // profit = sales - fixed investment - operating cost - purchases
        m.maximize(
            11.0 * c1 - 3.5 * y1 - y2 - 1.5 * y3 - b2 - 1.2 * b3 - 1.8 * (a2 + a3) - 7.0 * bp,
        );

        let result = solver.solve(&m, opts)?;

        if let Some(obj) = result.objective() {
            println!("--- {label} ---");
            println!("status = {:?}", result.status);
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
            println!("status = {:?}", result.status);
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
