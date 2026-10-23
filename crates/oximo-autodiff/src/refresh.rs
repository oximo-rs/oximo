//! Transactional selected-slot refresh, independent of the derivative engine.
use oximo_core::ParameterTarget;
use oximo_expr::{ExprArena, ExprId};
use rustc_hash::FxHashMap;

use crate::{FunctionSlot, sparsity::hessian_lagrangian_structure};

pub(crate) struct SelectedSlots {
    pub objective: Option<FunctionSlot>,
    pub constraints: FxHashMap<usize, FunctionSlot>,
}

pub(crate) fn prepare_selected(
    arena: &ExprArena,
    targets: &[ParameterTarget],
    objective_expr: Option<ExprId>,
    constraint_exprs: &[ExprId],
    objective: &FunctionSlot,
    constraints: &[FunctionSlot],
    hessian: &[(usize, usize)],
) -> Option<SelectedSlots> {
    let mut fresh = SelectedSlots { objective: None, constraints: FxHashMap::default() };
    for target in targets {
        match target {
            ParameterTarget::Objective => {
                if let Some(expr) = objective_expr {
                    fresh.objective = Some(objective.reclassify(arena, expr));
                }
            }
            ParameterTarget::Algebraic(id) => {
                let slot = constraints.get(id.index())?;
                let &expr = constraint_exprs.get(id.index())?;
                fresh.constraints.entry(id.index()).or_insert_with(|| slot.reclassify(arena, expr));
            }
            ParameterTarget::VariableBound(_) => {}
            _ => return None,
        }
    }
    if fresh.constraints.iter().any(|(&i, slot)| slot.support != constraints[i].support) {
        return None;
    }
    let mut hessian_changed = false;
    for (new, old) in fresh
        .objective
        .iter()
        .map(|slot| (slot, objective))
        .chain(fresh.constraints.iter().map(|(&i, slot)| (slot, &constraints[i])))
    {
        // The Lagrangian tape owns the original nonlinear sources.
        // Changing membership requires retaping.
        if new.is_nonlinear() != old.is_nonlinear() {
            return None;
        }
        hessian_changed |=
            hessian_lagrangian_structure([new]) != hessian_lagrangian_structure([old]);
    }
    if hessian_changed {
        let slots = std::iter::once(fresh.objective.as_ref().unwrap_or(objective)).chain(
            constraints
                .iter()
                .enumerate()
                .map(|(i, slot)| fresh.constraints.get(&i).unwrap_or(slot)),
        );
        if hessian_lagrangian_structure(slots) != hessian {
            return None;
        }
    }
    Some(fresh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_core::prelude::*;

    #[test]
    fn selected_refresh_rejects_changed_nonlinear_membership_with_the_same_hessian() {
        let m = Model::new("nonlinear tape membership");
        variable!(m, x);
        objective!(m, Min, x * x);
        let expr = m.objective().as_ref().unwrap().expr;

        // A retained tape for this expression has the same derivative support
        // as the new cf.
        let objective = FunctionSlot {
            kind: crate::SlotKind::Nonlinear(crate::Tape::compile(&m.arena(), expr)),
            support: vec![0],
            hess_pairs: vec![(0, 0)],
        };

        let hessian = hessian_lagrangian_structure([&objective]);
        assert_eq!(
            hessian,
            hessian_lagrangian_structure([&objective.reclassify(&m.arena(), expr)])
        );
        assert!(
            prepare_selected(
                &m.arena(),
                &[ParameterTarget::Objective],
                Some(expr),
                &[],
                &objective,
                &[],
                &hessian
            )
            .is_none()
        );
    }

    #[test]
    fn selected_refresh_preserves_unselected_slots_and_checks_zero_crossings() {
        let m = Model::new("selected derivative slots");

        param!(m, p = 1.0);
        variable!(m, x);
        variable!(m, y);
        constraint!(m, first, p * x * x + x <= 4.0);
        constraint!(m, second, x * x + y <= 4.0);
        objective!(m, Min, p * x);

        let exprs: Vec<_> = m.constraints().algebraic().iter().map(|c| c.lhs).collect();
        let objective_expr = m.objective().as_ref().unwrap().expr;
        let arena = m.arena();
        let objective = FunctionSlot::classify(&arena, objective_expr);
        let constraints: Vec<_> =
            exprs.iter().map(|&e| FunctionSlot::classify(&arena, e)).collect();
        let hessian = hessian_lagrangian_structure(std::iter::once(&objective).chain(&constraints));
        drop(arena);
        let targets = [ParameterTarget::Algebraic(ConstraintId(0)), ParameterTarget::Objective];
        for value in [2.0, 0.0] {
            p.set_param_value(value).unwrap();
            let fresh = prepare_selected(
                &m.arena(),
                &targets,
                Some(objective_expr),
                &exprs,
                &objective,
                &constraints,
                &hessian,
            )
            .unwrap();
            assert_eq!(fresh.constraints.len(), 1);
            assert!(!fresh.constraints.contains_key(&1));
            let crate::SlotKind::Linear(terms) = fresh.objective.unwrap().kind else {
                panic!("linear objective")
            };
            assert_eq!(terms.coeffs.iter().map(|(_, c)| c).sum::<f64>(), value);
        }
        assert!(
            prepare_selected(
                &m.arena(),
                &targets,
                Some(objective_expr),
                &exprs[..1],
                &objective,
                &constraints[..1],
                &hessian
            )
            .is_none()
        );

        let reused = prepare_selected(
            &m.arena(),
            &[],
            Some(objective_expr),
            &exprs,
            &objective,
            &constraints,
            &hessian,
        )
        .unwrap();
        assert!(reused.objective.is_none() && reused.constraints.is_empty());
        assert!(
            matches!(constraints[0].kind, crate::SlotKind::Quadratic(_)),
            "failed refresh leaves original slots intact"
        );
    }

    #[test]
    fn selected_refresh_rejects_new_jacobian_support_and_unsupported_targets() {
        let m = Model::new("changed Jacobian");

        param!(m, p = 0.0);
        variable!(m, x);
        constraint!(m, row, p * x * x <= 1.0);
        objective!(m, Min, x);

        let expr = m.constraints().algebraic()[0].lhs;
        let objective = FunctionSlot::zero();
        let slots = [FunctionSlot::classify(&m.arena(), expr)];
        p.set_param_value(1.0).unwrap();
        assert!(
            prepare_selected(
                &m.arena(),
                &[ParameterTarget::Algebraic(ConstraintId(0))],
                None,
                &[expr],
                &objective,
                &slots,
                &[]
            )
            .is_none()
        );
        assert!(
            prepare_selected(
                &m.arena(),
                &[ParameterTarget::SecondOrderCone(SocConstraintId(0))],
                None,
                &[expr],
                &objective,
                &slots,
                &[]
            )
            .is_none()
        );
    }
}
