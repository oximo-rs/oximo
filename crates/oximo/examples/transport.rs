//! Toy LP
//!
//! ```text
//! maximize  3x + 4y
//! s.t.   x + 2y <= 14
//!       3x -  y >=  0
//!        x -  y <=  2
//!        x >= 0,  0 <= y <= 4
//! ```
//!
//! Optimal: x = 6, y = 4, objective = 34.

use oximo::prelude::*;
use oximo::solvers::Highs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = Model::new("transport");

    let x = m.var("x").lb(0.0).build();
    let y = m.var("y").lb(0.0).ub(4.0).build();

    m.constraint("c1", (x + 2.0 * y).le(14.0));
    m.constraint("c2", (3.0 * x - y).ge(0.0));
    m.constraint("c3", (x - y).le(2.0));
    m.maximize(3.0 * x + 4.0 * y);

    let mut solver = Highs;
    let result = solver.solve(&m, &HighsOptions::default().verbose(true))?;

    println!("status    = {:?}", result.status);
    println!("objective = {:?}", result.objective);
    println!("x = {:?}", result.value_of(x));
    println!("y = {:?}", result.value_of(y));
    Ok(())
}
