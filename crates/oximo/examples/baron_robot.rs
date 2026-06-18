//! PUMA robot inverse-kinematics problem, solved with BARON.
//!
//! Find all configurations of a six-revolute PUMA manipulator that reach a
//! given hand position/orientation. The eight unknowns `x1..x8` are sines and
//! cosines of joint angles, constrained by trigonometric identities
//! (`xi^2 + xj^2 = 1`) and the kinematic loop equations (bilinear + linear).
//! It is a pure feasibility system (`minimize 0`) with multiple real solutions
//! (the BARON manual reports 14).
//!
//! This is the `robot.bar` example from the BARON manual. The manual writes
//! each equality as a pair of `<=` inequalities, here we use oximo's `.eq(..)`,
//! which is mathematically identical and emits the same constraint set. The
//! `num_sol(20)` option asks BARON to enumerate up to 20 distinct solutions.
//!
//! References:
//!   N. Sahinidis, BARON User Manual, version 2026.4.12.
//!   The Optimization Firm, LLC, Apr. 12, 2026.
//!
//!   Tsai, L., and Morgan, A. P. (June 1, 1985).
//!   "Solving the Kinematics of the Most General Six- and Five-Degree-of-Freedom
//!   Manipulators by Continuation Methods."
//!   ASME. J. Mech., Trans., and Automation. June 1985; 107(2): 189-200.
//!   <https://doi.org/10.1115/1.3258708>
//!
//! Run (requires a licensed BARON on PATH):
//!   cargo run -p oximo --example baron_robot --features baron

#![allow(clippy::unreadable_literal)]

#[cfg(feature = "baron")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oximo::prelude::*;
    use oximo::solvers::Baron;
    use std::time::Duration;

    let m = Model::new("robot");

    // Eight unknowns, each in [-1, 1].
    variable!(m, -1.0 <= x1 <= 1.0);
    variable!(m, -1.0 <= x2 <= 1.0);
    variable!(m, -1.0 <= x3 <= 1.0);
    variable!(m, -1.0 <= x4 <= 1.0);
    variable!(m, -1.0 <= x5 <= 1.0);
    variable!(m, -1.0 <= x6 <= 1.0);
    variable!(m, -1.0 <= x7 <= 1.0);
    variable!(m, -1.0 <= x8 <= 1.0);

    // Kinematic loop equations (linear + bilinear).
    constraint!(
        m,
        e1,
        0.004731 * x1 * x3 - 0.1238 * x1 - 0.3578 * x2 * x3 - 0.001637 * x2 - 0.9338 * x4 + x7
            == 0.3571
    );
    constraint!(
        m,
        e2,
        0.2238 * x1 * x3 + 0.2638 * x1 + 0.7623 * x2 * x3 - 0.07745 * x2 - 0.6734 * x4 - x7
            == 0.6022
    );
    constraint!(m, e3, x6 * x8 + 0.3578 * x1 + 0.004731 * x2 == 0.0);
    constraint!(m, e4, -0.7623 * x1 + 0.2238 * x2 == -0.3461);

    // Trigonometric identities: each (sin, cos) pair lies on the unit circle.
    constraint!(m, e5, x1.powi(2) + x2.powi(2) == 1.0);
    constraint!(m, e6, x3.powi(2) + x4.powi(2) == 1.0);
    constraint!(m, e7, x5.powi(2) + x6.powi(2) == 1.0);
    constraint!(m, e8, x7.powi(2) + x8.powi(2) == 1.0);

    // Pure feasibility problem: no objective set => `minimize 0`.

    let opts =
        BaronOptions::default().time_limit(Duration::from_secs(120)).num_sol(20).verbose(true);
    let result = Baron::new().solve(&m, &opts)?;

    println!("status = {:?}", result.status);
    println!("solutions found = {}", result.result_count());

    // BARON enumerates the distinct solutions of this feasibility system,
    // each is a separate point in `result.solutions`.
    let vars = [
        ("x1", x1),
        ("x2", x2),
        ("x3", x3),
        ("x4", x4),
        ("x5", x5),
        ("x6", x6),
        ("x7", x7),
        ("x8", x8),
    ];
    for i in 0..result.result_count() {
        let sol = result.solution(i).expect("index < result_count");
        println!("solution {}:", i + 1);
        for (name, x) in vars {
            println!("  {name} = {:?}", sol.value_of(x));
        }
    }
    Ok(())
}

#[cfg(not(feature = "baron"))]
fn main() {
    println!("Enable the BARON backend:");
    println!("  cargo run -p oximo --example baron_robot --features baron");
}
