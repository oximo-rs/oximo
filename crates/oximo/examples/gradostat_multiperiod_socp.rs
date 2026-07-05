//! Multi-period second-order cone optimization of the gradostat (model `P_RC`).
//!
//! Replication of Section 7.2 of Taylor & Rapaport (2021).
//! A four-tank gradostat with constant flows and diffusions is
//! optimized over `TAU = 1000` time periods to maximize cumulative
//! biogas production.
//!
//! # Model
//!
//! Growth parameters `mu_max = K = y = 1`, unit tank volumes `V = I`, Euler's
//! explicit method with time step `Delta = 1`, and a periodic horizon
//! (`S(1) = S(TAU+1)`, `X(1) = X(TAU+1)`).
//!
//! ```text
//! max   sum_{i,t}  T_i(t)                                             (eq. 24)
//! s.t.  S_i(t+1) - S_i(t) = -T_i(t) + [(M+L) S(t)]_i + Qin_i Sin_i(t) (25a)
//!       X_i(t+1) - X_i(t) =  T_i(t) + [(M+L) X(t)]_i + Qin_i Xin_i(t) (25b)
//!       || [S_i(t), T_i(t), X_i(t)] || <= X_i(t) + S_i(t) - T_i(t)    (eq. 9a)
//!       S_i(t) - T_i(t) >= 0                                          (eq. 9b)
//!       sum_i Qin_i Xin_i(t) <= 3                                     (biomass-inflow budget)
//!       S, X, T >= 0,   Xin >= 0
//! ```
//!
//! The substrate inflow `Sin(t)` is a fixed data profile, the biomass inflow
//! `Xin(t)` is the decision variable.
//!
//! # Reproduction vs. the authors' code
//!
//! This model matches the authors' reference implementation and
//! returns the same values.
//!
//! ## Details matched to the MATLAB code:
//!
//! 1. Diffusion `d = 0.3` on each existing pipe (`d = 0.3*(Q>0)`).
//! 2. Budget on the inflow `Xin(t)`: `Qin' Xin(t) <= 3`, not on the tank
//!    concentration `X(t)`.
//! 3. Time base: The paper indexes `t in {1..TAU}`, here periods are
//!    `p in {0..TAU-1}` with `t = p + 1` used in the `Sin` profiles.
//!
//! Run:
//!   cargo run -p oximo --example gradostat_multiperiod_socp --features clarabel
//!
//! References:
//! - Taylor, J. A., & Rapaport, A. (2021).
//!   Second-order cone optimization of the gradostat.
//!   Computers & Chemical Engineering, 151, 107347.
//!   <https://doi.org/10.1016/j.compchemeng.2021.107347>

#[cfg(feature = "clarabel")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oximo::ClarabelOptions;
    use oximo::prelude::*;
    use oximo::solvers::Clarabel;

    // Problem data
    const N: usize = 4; // tanks
    const TAU: usize = 1000; // time periods

    // Forced-flow + diffusion matrix M + L.
    const ML: [[f64; N]; N] = [
        [-2.3, 0.3, 0.0, 0.0],
        [1.3, -3.9, 0.3, 1.3],
        [0.0, 2.3, -3.6, 0.3],
        [0.0, 0.3, 1.3, -2.6],
    ];

    // C = diag(Qin), the external water inflow at each tank.
    const QIN: [f64; N] = [2.0, 1.0, 1.0, 1.0];

    // Total biomass mass allowed in the tanks per period: 1' Qin X(t) <= BUDGET.
    const BUDGET: f64 = 3.0;

    // Substrate inflow profiles Sin_i(t), indexed sin[i][p] with paper time t=p+1.
    #[allow(clippy::needless_range_loop, clippy::cast_precision_loss)]
    let sin: Vec<Vec<f64>> = {
        let pi = std::f64::consts::PI;
        let tau = TAU as f64;
        let mut sin = vec![vec![0.0f64; TAU]; N];
        for p in 0..TAU {
            let t = (p + 1) as f64;
            let ti = p + 1;
            sin[0][p] = 1.0 + (4.0 * pi * t / tau).sin();
            sin[1][p] = 0.0;
            sin[2][p] = if ti > TAU / 4 && ti <= 3 * TAU / 4 { 0.5 } else { 0.0 };
            sin[3][p] = 1.0 + (4.0 * pi * t / tau).cos();
        }
        sin
    };

    let m = Model::new("gradostat_multiperiod");
    let tanks = Set::range(0..N);
    let periods = Set::range(0..TAU);

    variable!(m, s[i in tanks, p in periods] >= 0.0); // substrate  S
    variable!(m, x[i in tanks, p in periods] >= 0.0); // biomass    X
    variable!(m, kin[i in tanks, p in periods] >= 0.0); // kinetics T
    variable!(m, xin[i in tanks, p in periods] >= 0.0); // biomass inflow Xin (decision)

    // Dynamic substrate and biomass balances (25a, 25b), Euler-explicit, V=y=1.
    constraint!(m, s_bal[i in tanks, p in periods],
        s[i, (p + 1) % TAU] - s[i, p]
            == -kin[i, p] + sum!(ML[i][j] * s[j, p] for j in tanks) + QIN[i] * sin[i][p]);
    constraint!(m, x_bal[i in tanks, p in periods],
        x[i, (p + 1) % TAU] - x[i, p]
            == kin[i, p] + sum!(ML[i][j] * x[j, p] for j in tanks) + QIN[i] * xin[i, p]);

    // Contois growth as a second-order cone (eq. 9, mu_max = K = 1).
    soc_constraint!(m, growth[i in tanks, p in periods],
        [s[i, p], kin[i, p], x[i, p]] <= x[i, p] + s[i, p] - kin[i, p]);
    constraint!(m, st_diff[i in tanks, p in periods], s[i, p] - kin[i, p] >= 0.0);

    // Per-period biomass-inflow budget, 1' Qin Xin(t) <= BUDGET.
    constraint!(m, bio_budget[p in periods], sum!(QIN[i] * xin[i, p] for i in tanks) <= BUDGET);

    // Maximize cumulative biogas, sum_{i,t} V_ii T_i with V = I (eq. 24, undiscounted).
    objective!(m, Max, sum!(kin[i, p] for i in tanks, p in periods));

    assert_eq!(m.kind(), ModelKind::SOCP);

    let res = Clarabel.solve(&m, &ClarabelOptions::default())?;
    assert_eq!(res.termination, TerminationStatus::Optimal);

    // Exactness metric E = max_{i,t} |r(S,X)-T|/r(S,X), with the Contois
    // kinetics r(S,X) = mu_max S X/(K X + S) evaluated at the solution.
    let mut e_max = 0.0f64;
    for i in 0..N {
        for p in 0..TAU {
            let sv = res.value_of(s[(i, p)]).unwrap();
            let xv = res.value_of(x[(i, p)]).unwrap();
            let tv = res.value_of(kin[(i, p)]).unwrap();
            let denom = sv + xv;
            if denom > 1e-9 {
                let r = sv * xv / denom;
                if r > 1e-9 {
                    e_max = e_max.max((r - tv).abs() / r);
                }
            }
        }
    }

    println!("Results:");
    println!("Termination: {:?}", res.termination);
    println!("Objective: {:.4}", res.objective().unwrap());
    println!("Exactness E: {e_max:.3e}");
    println!("Iterations: {}", res.iterations);
    println!("Solve time: {:.3} s", res.solve_time.as_secs_f64());

    Ok(())
}

#[cfg(not(feature = "clarabel"))]
fn main() {
    eprintln!("This example needs the clarabel feature:");
    eprintln!("  cargo run -p oximo --example gradostat_multiperiod_socp --features clarabel");
}
