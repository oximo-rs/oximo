//! Writer coverage for two-sided range constraints (`lo <= e <= hi`).
//!
//! A range with constant bounds is one `Constraint`. MPS represents it natively
//! via a `RANGES` section. CPLEX LP has no portable single-row range so the
//! writer expands it to two rows. The AMPL `.nl` writer uses bound-type code 0.

use oximo_core::prelude::*;
use oximo_io::{to_lp_string, to_mps_string};

/// max x + y  s.t.  1 <= x + y <= 4.
fn range_model() -> Model {
    let m = Model::new("rng");
    variable!(m, x >= 0.0);
    variable!(m, y >= 0.0);
    constraint!(m, band, 1.0 <= x + y <= 4.0);
    objective!(m, Max, x + y);
    assert_eq!(m.num_constraints(), 1);
    m
}

#[test]
fn mps_emits_ranges_section() {
    let s = to_mps_string(&range_model()).expect("mps writer");
    // The range is an `L` row whose RHS is the upper bound
    assert!(s.contains(" L  band"), "{s}");
    assert!(s.contains("RHS       band      4"), "{s}");
    // widened down to the lower bound by RANGES: R = upper - lower = 3.
    assert!(s.contains("RANGES"), "{s}");
    assert!(s.contains("RNG       band      3"), "{s}");
}

#[test]
fn lp_expands_range_to_two_rows() {
    let s = to_lp_string(&range_model()).expect("lp writer");
    assert!(s.contains("band_lo:"), "{s}");
    assert!(s.contains("band_hi:"), "{s}");
    assert!(s.contains(">= 1"), "{s}");
    assert!(s.contains("<= 4"), "{s}");
}
