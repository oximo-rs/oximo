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

    variable!(m, x >= 0.0);
    variable!(m, 0.0 <= y <= 4.0);

    constraint!(m, c1, x + 2.0 * y <= 14.0);
    constraint!(m, c2, 3.0 * x >= y);
    constraint!(m, c3, x <= y + 2.0);
    objective!(m, Max, 3.0 * x + 4.0 * y);

    let mut solver = Highs;
    let result = solver.solve(&m, &HighsOptions::default().verbose(true))?;

    print!("{}", result.report(&m));
    Ok(())
}
