//! Internal helpers shared by model reformulations.

use smol_str::SmolStr;

use crate::domain::Domain;
use crate::model::Model;
use crate::var::Variable;

/// Bounds of the variable's domain, including the zero branch of semi-domains.
pub(crate) fn raw_effective_bounds(variable: &Variable) -> (f64, f64) {
    match variable.domain {
        Domain::SemiContinuous { threshold } | Domain::SemiInteger { threshold } => {
            (threshold.min(0.0), variable.ub.max(0.0))
        }
        Domain::Real | Domain::Integer | Domain::Binary => (variable.lb, variable.ub),
    }
}

pub(crate) fn unique_variable_name(model: &Model, base: &str) -> SmolStr {
    unique_name(base, |candidate| model.variable_id(candidate).is_some())
}

pub(crate) fn unique_constraint_name(model: &Model, base: &str) -> SmolStr {
    unique_name(base, |candidate| model.constraint_id(candidate).is_some())
}

fn unique_name(base: &str, exists: impl Fn(&str) -> bool) -> SmolStr {
    if !exists(base) {
        return base.into();
    }
    for suffix in 1_u64.. {
        let candidate = format!("{base}_{suffix}");
        if !exists(&candidate) {
            return candidate.into();
        }
    }
    unreachable!("u64 name suffix space exhausted")
}
