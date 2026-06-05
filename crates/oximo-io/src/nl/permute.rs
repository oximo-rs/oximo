//! Standard AMPL/ASL variable + constraint permutation.
//!
//! ASL-linked solvers expect variables ordered into these buckets (D. M. Gay,
//! "Hooking your solver to AMPL" Tables 3-4). The nonlinear block is grouped by
//! appearance (both, constraints-only, objective-only), and within each group
//! continuous variables precede integer ones:
//!
//! 1. nonlinear continuous in BOTH constraints and objective
//! 2. nonlinear integer in BOTH
//! 3. nonlinear continuous in constraints only
//! 4. nonlinear integer in constraints only
//! 5. nonlinear continuous in objective only
//! 6. nonlinear integer in objective only
//! 7. linear continuous
//! 8. linear binary
//! 9. linear other-integer
//!
//! Within each bucket the original `VarId` order is preserved.

use oximo_core::{Domain, Variable};
use oximo_expr::VarId;
use rustc_hash::FxHashMap;

use super::analyze::Analysis;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Bucket {
    NlBothC,
    NlOnlyC,
    NlOnlyO,
    NlBothI,
    NlOnlyCI,
    NlOnlyOI,
    LinC,
    LinB,
    LinI,
}

impl Bucket {
    fn order(self) -> u8 {
        // Gay Table 4: group by appearance (both, con-only, obj-only) and place
        // continuous before integer within each group.
        match self {
            Bucket::NlBothC => 0,
            Bucket::NlBothI => 1,
            Bucket::NlOnlyC => 2,
            Bucket::NlOnlyCI => 3,
            Bucket::NlOnlyO => 4,
            Bucket::NlOnlyOI => 5,
            Bucket::LinC => 6,
            Bucket::LinB => 7,
            Bucket::LinI => 8,
        }
    }
}

fn bucket_for(v: &Variable, analysis: &Analysis) -> Bucket {
    let in_c = analysis.nl_vars_c.contains(&v.id);
    let in_o = analysis.nl_vars_o.contains(&v.id);
    let is_int = v.domain.is_integer();
    let is_bin = matches!(v.domain, Domain::Binary);
    match (in_c, in_o, is_int) {
        (true, true, false) => Bucket::NlBothC,
        (true, false, false) => Bucket::NlOnlyC,
        (false, true, false) => Bucket::NlOnlyO,
        (true, true, true) => Bucket::NlBothI,
        (true, false, true) => Bucket::NlOnlyCI,
        (false, true, true) => Bucket::NlOnlyOI,
        (false, false, false) => Bucket::LinC,
        (false, false, true) => {
            if is_bin {
                Bucket::LinB
            } else {
                Bucket::LinI
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct Permutation {
    /// NL-index -> original VarId.
    pub(crate) var_order: Vec<VarId>,
    /// VarId -> NL-index.
    pub(crate) var_index: FxHashMap<VarId, u32>,
    /// NL-index -> original constraint index.
    pub(crate) con_order: Vec<usize>,
}

impl Permutation {
    pub(crate) fn build(vars: &[Variable], analysis: &Analysis) -> Self {
        let mut indexed: Vec<(usize, Bucket)> =
            vars.iter().enumerate().map(|(i, v)| (i, bucket_for(v, analysis))).collect();
        indexed.sort_by_key(|(i, b)| (b.order(), *i));

        let var_order: Vec<VarId> = indexed.iter().map(|(i, _)| vars[*i].id).collect();
        let mut var_index: FxHashMap<VarId, u32> = FxHashMap::default();
        for (nl_idx, vid) in var_order.iter().enumerate() {
            var_index.insert(*vid, u32::try_from(nl_idx).expect("var count overflow"));
        }

        // Constraints: nonlinear first (original order), then linear (original order).
        let mut nl_idx: Vec<usize> = Vec::new();
        let mut lin_idx: Vec<usize> = Vec::new();
        for (i, form) in analysis.cons.iter().enumerate() {
            if form.is_nonlinear() {
                nl_idx.push(i);
            } else {
                lin_idx.push(i);
            }
        }
        let con_order: Vec<usize> = nl_idx.into_iter().chain(lin_idx).collect();

        Self { var_order, var_index, con_order }
    }
}
