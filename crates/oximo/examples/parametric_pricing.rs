//! Parametric pricing: re-solving one model across many parameter values.
//!
//! A small workshop makes two products from a shared pool of labor and
//! material. Product 2 sells at a fixed margin. Product 1's margin `p1` is a
//! *market parameter* we do not control. We want the optimal product mix as
//! `p1` sweeps over a range of market scenarios.
//!
//! The point of a [`Model::param`] is that it stays symbolic in the model: we
//! build the LP once, then call [`Model::set_param`] between solves to
//! re-bind `p1` without rebuilding any variables, constraints, or objective.
//! Each solve reads the parameter's current value, so the coefficient on `x1`
//! tracks the latest binding.
//!
//! References:
//! - Dantzig, G. B. (1998). Linear Programming and Extensions. 
//!   Princeton, NJ: Princeton University Press.
//! - Hillier F. S., & Lieberman G. J. (2010). Introduction to Operations Research.
//!   New York, NY: McGraw-Hill Higher Education.

use oximo::prelude::*;
use oximo::solvers::Highs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = Model::new("parametric_pricing");

    // Per-unit margin of product 1: a market price we re-bind per scenario.
    let p1 = m.param("p1", 0.0);

    // Production quantities (units)
    let x1 = m.var("x1").lb(0.0).build();
    let x2 = m.var("x2").lb(0.0).build();

    // Shared resources
    m.constraint("labor", (2.0 * x1 + x2).le(100.0));
    m.constraint("material", (x1 + 3.0 * x2).le(90.0));

    // profit = p1 * x1 + 5 * x2  (product 2's margin is fixed at 5)
    m.maximize(p1 * x1 + 5.0 * x2);

    // The model kind is inferred and does not change as we re-bind `p1`.
    assert_eq!(m.kind(), ModelKind::LP);
    println!("model kind: {:?}  (param * var stays linear)\n", m.kind());

    println!("  p1  |    x1    |    x2    |  profit");
    println!("------+----------+----------+---------");

    for price in [1.0, 1.6, 2.0, 5.0, 11.0] {
        // Re-bind `p1` to the current price
        m.set_param(p1, price);

        let result = Highs.solve(&m, &HighsOptions::default())?;
        assert_eq!(result.status, SolverStatus::Optimal);

        let x1v = result.value_of(x1).unwrap_or(0.0);
        let x2v = result.value_of(x2).unwrap_or(0.0);
        let profit = result.objective.unwrap_or(0.0);
        println!(" {price:>4.1} | {x1v:>8.2} | {x2v:>8.2} | {profit:>7.2}");
    }

    Ok(())
}
