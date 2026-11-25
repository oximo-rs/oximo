//! Exact quadratic Hull rows (GEHR and CEHR) using SCIP, Gurobi, BARON and MOSEK.
//!
//! ```text
//! cargo test -p oximo --features gdp,gurobi,baron,scip,mosek --test gdp_hull_quadratic
//! ```

#![cfg(all(
    feature = "gdp",
    any(feature = "gurobi", feature = "baron", feature = "scip", feature = "mosek")
))]

use oximo::prelude::*;

/// GEHR, and CEHR with explicit and algebraic cones.
fn exact_hulls() -> [Hull; 3] {
    [
        Hull::default().with_quadratic(HullQuadratic::Homogeneous),
        Hull::default().with_quadratic(HullQuadratic::Conic(HullConeForm::Explicit)),
        Hull::default().with_quadratic(HullQuadratic::Conic(HullConeForm::Algebraic)),
    ]
}

/// Choose one of two disks and optimize `x + w`.
fn solve_disks(
    hull: Hull,
    reference: Option<f64>,
    solve: impl Fn(&Model, GdpReformulationOptions) -> SolverResult,
) {
    for maximize in [true, false] {
        let model = Model::new("two disks");

        variable!(model, -5.0 <= x <= 5.0);
        variable!(model, -5.0 <= w <= 5.0);
        let upper = disjunct!(model, upper, |d| {
            constraint!(d, (x - 1.0) * (x - 1.0) + (w - 1.0) * (w - 1.0) <= 1.0);
        });
        let left = disjunct!(model, left, |d| {
            constraint!(d, (x + 2.0) * (x + 2.0) + w * w <= 0.5);
        });
        disjunction!(model, [upper, left]);
        if maximize {
            objective!(model, Max, x + w);
        } else {
            objective!(model, Min, x + w);
        }

        let mut options = GdpReformulationOptions::new(hull);
        if let Some(r) = reference {
            options = options.with_hull_reference_coordinate(x, r);
        }

        let result = solve(&model, options);
        assert_eq!(result.termination, TerminationStatus::Optimal);
        // Generated cones are not source coordinates.
        assert!(result.soc_dual.is_empty());

        let objective = result.value_of(x).unwrap().unwrap() + result.value_of(w).unwrap().unwrap();
        let (expected, chosen) =
            if maximize { (2.0 + 2.0_f64.sqrt(), upper) } else { (-3.0, left) };
        assert!((objective - expected).abs() < 1e-4, "{hull:?}: {objective} vs {expected}");
        assert_eq!(result.boolean_value_of(chosen).unwrap(), Some(true));
    }
}

#[cfg(feature = "gurobi")]
mod gurobi {
    use super::*;
    use oximo::GurobiOptions;
    use oximo::solvers::Gurobi;

    #[test]
    fn gurobi_solves_exact_quadratic_hulls() {
        for hull in exact_hulls() {
            solve_disks(hull, None, |model, options| {
                Gurobi.with_gdp(options).solve(model, &GurobiOptions::default()).unwrap()
            });
        }
    }

    #[test]
    fn gurobi_solves_shifted_conic_hulls() {
        for hull in &exact_hulls()[1..] {
            for reference in [None, Some(0.5)] {
                solve_disks(*hull, reference, |model, options| {
                    Gurobi.with_gdp(options).solve(model, &GurobiOptions::default()).unwrap()
                });
            }
        }
    }
}

#[cfg(feature = "baron")]
mod baron {
    use super::*;
    use oximo::BaronOptions;
    use oximo::solvers::Baron;
    use std::time::Duration;

    #[test]
    fn baron_solves_exact_quadratic_hulls() {
        for hull in exact_hulls() {
            solve_disks(hull, Some(0.5), |model, options| {
                Baron::new()
                    .with_gdp(options)
                    .solve(model, &BaronOptions::default().time_limit(Duration::from_secs(60)))
                    .unwrap()
            });
        }
    }
}

#[cfg(feature = "scip")]
mod scip {
    use super::*;
    use oximo::ScipOptions;
    use oximo::solvers::Scip;

    #[test]
    fn scip_solves_exact_quadratic_hulls() {
        for hull in exact_hulls() {
            for reference in [None, Some(0.5)] {
                solve_disks(hull, reference, |model, options| {
                    Scip::default().with_gdp(options).solve(model, &ScipOptions::default()).unwrap()
                });
            }
        }
    }
}

#[cfg(feature = "mosek")]
mod mosek {
    use super::*;
    use oximo::MosekOptions;
    use oximo::solvers::Mosek;

    #[test]
    fn mosek_solves_conic_hulls() {
        for hull in &exact_hulls()[1..] {
            for reference in [None, Some(0.5)] {
                solve_disks(*hull, reference, |model, options| {
                    Mosek.with_gdp(options).solve(model, &MosekOptions::default()).unwrap()
                });
            }
        }
    }
}
