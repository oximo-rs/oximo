//! Alkylation process optimization (NLP), solved with POUNCE.
//!
//! An alkylation unit reacts an olefin feed with isobutane over an acid
//! catalyst to produce alkylate, a high-octane gasoline blending stock.
//! Choose the olefin feed, isobutane recycle/makeup, and acid addition rate
//! to maximize daily profit, where the alkylate yield, motor octane number,
//! acid dilution factor, and F-4 performance number are tied together by
//! nonlinear regression correlations.
//!
//! This example is translated from GAMS code in the PROCESS model (SEQ=20),
//! using its `process` variant where every regression holds exactly.
//!
//! # Model
//!
//! ```text
//! Variables (with bounds)
//!   olefin   olefin feed (bpd)                [10, 2000]
//!   isor     isobutane recycle (bpd)          [0, 16000]
//!   acid     acid addition rate (1000 lb/day) [0, 120]
//!   alkylate alkylate yield (bpd)             [0, 5000]
//!   isom     isobutane makeup (bpd)           [0, 2000]
//!   strength acid strength (weight pct)       [85, 93]
//!   octane   motor octane number              [90, 95]
//!   ratio    external isobutane-to-olefin     [3, 12]
//!   dilute   acid dilution factor             [1.2, 4]
//!   f4       F-4 performance number           [145, 162]
//!
//! maximize
//!   0.063 alkylate octane - 5.04 olefin - 0.035 isor - 10 acid - 3.36 isom
//!
//! s.t.
//!   alkylate = olefin (1.12 + 0.13167 ratio - 0.00667 ratio^2)
//!   alkylate = olefin + isom - 0.22 alkylate
//!   acid = alkylate dilute strength / (98 - strength) / 1000
//!   octane = 86.35 + 1.098 ratio - 0.038 ratio^2 - 0.325 (89 - strength)
//!   ratio = (isor + isom) / olefin
//!   dilute = 35.82 - 0.222 f4
//!   f4 = -133 + 3 octane
//! ```
//!
//! After the base solve, the alkylate price coefficient (0.063 $/octane-bbl)
//! is swept over a small range through a model parameter, re-solving with a
//! persistent POUNCE handle that keeps its state between solves.
//!
//! References:
//!   Bracken, J, and McCormick, G P, "Optimization of an Alkylation Process",
//!   Chapter 4 in Selected Applications of Nonlinear Programming,
//!   John Wiley and Sons, New York, 1968.
//!
//! Run using finite differences:
//! ```text
//! cargo run -p oximo --example pounce_alkylation --features pounce
//! ```
//!
//! With exact derivatives via Enzyme (requires a nightly Enzyme toolchain).
//! ```text
//! RUSTFLAGS="-Zautodiff=Enable" cargo +nightly run -p oximo --example pounce_alkylation \
//!     --no-default-features --features pounce-enzyme --profile enzyme
//! ```

#[cfg(feature = "pounce")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oximo::prelude::*;
    use oximo::solvers::Pounce;

    let m = Model::new("alkylation");

    // Process streams and operating conditions.
    variable!(m, 10.0 <= olefin <= 2000.0, initial = 1745.0);
    variable!(m, 0.0 <= isor <= 16000.0, initial = 12000.0);
    variable!(m, 0.0 <= acid <= 120.0, initial = 110.0);
    variable!(m, 0.0 <= alkylate <= 5000.0, initial = 3048.0);
    variable!(m, 0.0 <= isom <= 2000.0, initial = 1974.0);
    variable!(m, 85.0 <= strength <= 93.0, initial = 89.2);
    variable!(m, 90.0 <= octane <= 95.0, initial = 92.8);
    variable!(m, 3.0 <= ratio <= 12.0, initial = 8.0);
    variable!(m, 1.2 <= dilute <= 4.0, initial = 3.6);
    variable!(m, 145.0 <= f4 <= 162.0, initial = 145.0);

    // Alkylate yield regression on the isobutane-to-olefin ratio.
    constraint!(m, yield_, alkylate == olefin * (1.12 + 0.13167 * ratio - 0.00667 * ratio.powi(2)));

    // Volumetric balance: 22% shrinkage of the combined feed.
    constraint!(m, makeup, alkylate == olefin + isom - 0.22 * alkylate);

    // Acid consumption from dilution at the operating strength.
    constraint!(m, sdef, acid == alkylate * dilute * strength / (98.0 - strength) / 1000.0);

    // Motor octane regression on ratio and acid strength.
    constraint!(
        m,
        motor,
        octane == 86.35 + 1.098 * ratio - 0.038 * ratio.powi(2) - 0.325 * (89.0 - strength)
    );

    // External isobutane-to-olefin ratio definition.
    constraint!(m, drat, ratio == (isor + isom) / olefin);

    // Acid dilution factor and F-4 performance number correlations.
    constraint!(m, ddil, dilute == 35.82 - 0.222 * f4);
    constraint!(m, df4, f4 == -133.0 + 3.0 * octane);

    // Profit.
    param!(m, alk_price = 0.063);
    objective!(
        m,
        Max,
        alk_price * alkylate * octane - 5.04 * olefin - 0.035 * isor - 10.0 * acid - 3.36 * isom
    );

    // On stable Rust, we use finite differences, which caps
    // the reachable dual infeasibility near 1e-5.
    #[cfg(not(feature = "pounce-enzyme"))]
    let opts = PounceOptions::default().tol(1e-5);
    #[cfg(feature = "pounce-enzyme")]
    let opts = PounceOptions::default();
    let result = Pounce.solve(&m, &opts)?;

    println!("status  = {:?}", result.termination);
    println!("profit  = {:.2} $/day", result.objective().unwrap_or(f64::NAN));
    println!();
    let streams = [
        ("olefin feed", olefin, "bpd"),
        ("isobutane recycle", isor, "bpd"),
        ("isobutane makeup", isom, "bpd"),
        ("alkylate yield", alkylate, "bpd"),
        ("acid addition", acid, "1000 lb/day"),
        ("acid strength", strength, "wt pct"),
        ("motor octane", octane, ""),
        ("i/o ratio", ratio, ""),
        ("dilution factor", dilute, ""),
        ("F-4 performance", f4, ""),
    ];
    for (name, x, unit) in streams {
        println!("  {name:<18} = {:>9.2} {unit}", result.value_of(x).unwrap_or(f64::NAN));
    }

    // Price sensitivity: sweep the alkylate price and re-solve.
    println!();
    println!("alkylate price sweep:");
    let mut solver = Pounce.persistent();
    for price in [0.055, 0.059, 0.063, 0.067, 0.071] {
        alk_price.set_param_value(price);
        let res = solver.solve(&m, &opts)?;
        println!(
            "  price {price:.3} $/oct-bbl -> profit {:>8.2} $/day, olefin {:>8.2} bpd, octane {:.2}",
            res.objective().unwrap_or(f64::NAN),
            res.value_of(olefin).unwrap_or(f64::NAN),
            res.value_of(octane).unwrap_or(f64::NAN),
        );
    }

    Ok(())
}

#[cfg(not(feature = "pounce"))]
fn main() {
    println!("Enable the POUNCE backend:");
    println!("  cargo run -p oximo --example pounce_alkylation --features pounce");
}
