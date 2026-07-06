use std::hash::Hasher;

use oximo_core::{Model, ModelKind, ObjectiveSense, Variable};
use oximo_expr::{ExprArena, VarId, extract_linear};
use rustc_hash::FxHasher;

use crate::status::SolverError;

/// Column-aligned snapshot of the quantities a persistent backend can push to a
/// resident model without rebuilding it (objective linear coefficients, the
/// objective constant, and variable bounds) plus a structural `fingerprint` of
/// everything it cannot push.
///
/// Two snapshots of the same model whose `fingerprint`s match differ only in
/// pushable quantities, so a backend may update those in place and warm-start. A
/// fingerprint mismatch means the structure changed and the model must be rebuilt.
/// All vectors are indexed in [`Model::variables`] (column) order.
///
/// This is solver-agnostic, each backend's persistent handle computes a baseline
/// snapshot when it (re)builds and a fresh one on each re-solve, pushes the diffs
/// its API supports, and rebuilds on a fingerprint mismatch.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Objective linear coefficient per variable, in column order.
    pub obj_costs: Vec<f64>,
    /// Constant term of the (linear) objective.
    pub obj_constant: f64,
    /// Lower bound per variable, in column order.
    pub lb: Vec<f64>,
    /// Upper bound per variable, in column order.
    pub ub: Vec<f64>,
    /// Hash of the structural parts that the fast path cannot push.
    pub fingerprint: u64,
}

/// Compute the incremental [`Snapshot`] of a linear model (`LP`/`MILP`).
///
/// The objective and every constraint must be linear, the snapshot is the basis
/// for a persistent backend's warm re-solve fast path and is only meaningful for
/// linear models (a quadratic/nonlinear model always rebuilds).
///
/// # Errors
///
/// Returns [`SolverError::Nonlinear`] if the objective or any constraint is not
/// linear, or [`SolverError::UnsupportedKind`] if the model is a second-order
/// cone program (explicit [`oximo_core::SocConstraint`]s or SOC-shaped
/// quadratic constraints detected by [`Model::kind`]).
pub fn snapshot(model: &Model) -> Result<Snapshot, SolverError> {
    model.ensure_objective_declared().map_err(SolverError::Core)?;
    let kind = model.kind();
    if model.num_soc_constraints() > 0 || matches!(kind, ModelKind::SOCP | ModelKind::MISOCP) {
        return Err(SolverError::UnsupportedKind(kind));
    }
    let arena = model.arena();
    let vars = model.variables();
    let constraints = model.constraints();

    let objective = model.objective();
    let obj = objective.as_ref();
    let sense = obj.map_or(ObjectiveSense::Minimize, |o| o.sense);
    let (obj_by_id, obj_constant) = match obj {
        Some(o) => {
            let lin = extract_linear(&arena, o.expr).ok_or(SolverError::Nonlinear)?;
            let mut by_id = vec![0.0; vars.len()];
            for (v, c) in &lin.coeffs {
                by_id[v.index()] = *c;
            }
            (by_id, lin.constant)
        }
        None => (vec![0.0; vars.len()], 0.0),
    };

    let mut obj_costs = Vec::with_capacity(vars.len());
    let mut lb = Vec::with_capacity(vars.len());
    let mut ub = Vec::with_capacity(vars.len());
    let mut hasher = FxHasher::default();
    hash_header(&mut hasher, &vars, sense);
    for v in vars.iter() {
        obj_costs.push(obj_by_id[v.id.index()]);
        lb.push(v.lb);
        ub.push(v.ub);
    }

    let arena_ref: &ExprArena = &arena;
    for c in constraints.iter() {
        let t = extract_linear(arena_ref, c.lhs).ok_or(SolverError::Nonlinear)?;
        hash_row(&mut hasher, c.lower - t.constant, c.upper - t.constant, &t.coeffs);
    }

    Ok(Snapshot { obj_costs, obj_constant, lb, ub, fingerprint: hasher.finish() })
}

/// Hash the parts that decide column count, integrality, and objective sense.
fn hash_header(h: &mut FxHasher, vars: &[Variable], sense: ObjectiveSense) {
    h.write_usize(vars.len());
    h.write_u8(match sense {
        ObjectiveSense::Minimize => 0,
        ObjectiveSense::Maximize => 1,
    });
    for v in vars {
        h.write_u8(u8::from(v.domain.is_integer()));
    }
}

/// Hash one constraint row: its (constant-folded) bounds and its `(column, coeff)`
/// terms, sorted so the hash is independent of extraction order.
fn hash_row(h: &mut FxHasher, lower: f64, upper: f64, coeffs: &[(VarId, f64)]) {
    h.write_u64(lower.to_bits());
    h.write_u64(upper.to_bits());
    let mut terms: Vec<(usize, u64)> =
        coeffs.iter().map(|(v, c)| (v.index(), c.to_bits())).collect();
    terms.sort_unstable();
    for (vi, cb) in terms {
        h.write_usize(vi);
        h.write_u64(cb);
    }
}

#[cfg(test)]
mod tests {
    use oximo_core::prelude::*;

    use super::snapshot;

    #[test]
    fn objective_coeff_change_keeps_fingerprint() {
        let m = Model::new("t");
        param!(m, p = 1.0);
        variable!(m, x >= 0.0);
        variable!(m, y >= 0.0);
        constraint!(m, c, x + y <= 10.0);
        objective!(m, Max, p * x + 2.0 * y);

        let s1 = snapshot(&m).unwrap();
        p.set_param_value(5.0);
        let s2 = snapshot(&m).unwrap();
        assert_eq!(s1.fingerprint, s2.fingerprint, "structure unchanged");
        assert_ne!(s1.obj_costs, s2.obj_costs, "coefficient moved");
    }

    #[test]
    fn bound_change_keeps_fingerprint() {
        let m = Model::new("t");
        variable!(m, x >= 0.0);
        constraint!(m, c, x <= 10.0);
        objective!(m, Max, x);

        let s1 = snapshot(&m).unwrap();
        m.fix(x, 3.0);
        let s2 = snapshot(&m).unwrap();
        assert_eq!(s1.fingerprint, s2.fingerprint, "structure unchanged");
        assert_ne!(s1.ub, s2.ub, "bound moved");
    }

    #[test]
    fn constraint_rhs_change_breaks_fingerprint() {
        let m = Model::new("t");
        param!(m, cap = 10.0);
        variable!(m, x >= 0.0);
        constraint!(m, c, x <= cap);
        objective!(m, Max, x);

        let s1 = snapshot(&m).unwrap();
        cap.set_param_value(20.0);
        let s2 = snapshot(&m).unwrap();
        assert_ne!(s1.fingerprint, s2.fingerprint, "row bound changed");
    }

    #[test]
    fn constraint_coeff_change_breaks_fingerprint() {
        let m = Model::new("t");
        param!(m, a = 1.0);
        variable!(m, x >= 0.0);
        variable!(m, y >= 0.0);
        constraint!(m, c, a * x + y <= 10.0);
        objective!(m, Max, x + y);

        let s1 = snapshot(&m).unwrap();
        a.set_param_value(3.0);
        let s2 = snapshot(&m).unwrap();
        assert_ne!(s1.fingerprint, s2.fingerprint, "matrix coefficient changed");
    }

    #[test]
    fn nonlinear_objective_is_rejected() {
        let m = Model::new("t");
        variable!(m, x >= 0.0);
        objective!(m, Min, x.powi(2));
        assert!(snapshot(&m).is_err());
    }

    #[test]
    fn soc_constraint_is_rejected() {
        let m = Model::new("t");
        variable!(m, x >= 0.0);
        variable!(m, t >= 0.0);
        m.add_soc_constraint("cone", [x], t);
        objective!(m, Min, t);
        assert!(matches!(
            snapshot(&m),
            Err(crate::status::SolverError::UnsupportedKind(ModelKind::SOCP))
        ));
    }

    #[test]
    fn no_objective_is_a_zero_objective() {
        let m = Model::new("feas");
        variable!(m, x >= 0.0);
        variable!(m, y >= 0.0);
        constraint!(m, c, x + y == 5.0);

        let s = snapshot(&m).expect("feasibility model snapshots");
        assert!(s.obj_costs.iter().all(|&c| c.abs() < 1e-12), "costs = {:?}", s.obj_costs);
        assert!(s.obj_constant.abs() < 1e-12, "constant = {}", s.obj_constant);
    }
}
