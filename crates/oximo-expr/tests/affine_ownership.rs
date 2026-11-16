use std::panic::{AssertUnwindSafe, catch_unwind};

use oximo_expr::{
    Affine, Children, Expr, ExprArena, ExprArenaCell, ExprId, ExprNode, VarId, VariableHandle,
    evaluate, extract_linear,
};

fn variables(arena: &ExprArenaCell, count: u32) -> Vec<VariableHandle<'_>> {
    (0..count).map(|i| Expr::from_var(arena, VarId(i))).collect()
}

fn last_node_id(arena: &ExprArenaCell) -> ExprId {
    ExprId(u32::try_from(arena.__node_count() - 1).unwrap())
}

fn linear_bits(arena: &ExprArena, id: ExprId) -> (Vec<(VarId, u64)>, u64) {
    let terms = extract_linear(arena, id).expect("expression must remain affine");
    (
        terms.coeffs.iter().map(|&(var, coefficient)| (var, coefficient.to_bits())).collect(),
        terms.constant.to_bits(),
    )
}

fn assert_linear(arena: &ExprArena, id: ExprId, coeffs: &[(VarId, f64)], constant: f64) {
    assert_eq!(
        linear_bits(arena, id),
        (
            coeffs.iter().map(|&(var, coefficient)| (var, coefficient.to_bits())).collect(),
            constant.to_bits(),
        )
    );
}

#[test]
fn consuming_three_copy_leaves_reuses_the_linear_root() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let x = Expr::from_var(&arena, VarId(0));
    let y = Expr::from_var(&arena, VarId(1));
    let z = Expr::from_var(&arena, VarId(2));
    let prefix = x + y;
    let nodes_before_growth = arena.__node_count();
    let expression = prefix + z;
    assert_eq!(arena.__node_count(), nodes_before_growth, "the owned root is reused");

    let snapshot = arena.borrow();
    assert_eq!(snapshot.len(), 4, "three variable nodes and one reused Linear node");
    assert_eq!(snapshot.nodes().iter().filter(|n| matches!(n, ExprNode::Var(_))).count(), 3);
    assert_eq!(snapshot.nodes().iter().filter(|n| matches!(n, ExprNode::Linear { .. })).count(), 1);
    assert_linear(
        &snapshot,
        expression.id(),
        &[(VarId(0), 1.0), (VarId(1), 1.0), (VarId(2), 1.0)],
        0.0,
    );
    assert_eq!(evaluate(&snapshot, expression.id(), &&[2.0, 3.0, 5.0][..]).unwrap(), 10.0);
    drop(snapshot);

    // from_var returns Copy identities.
    let again = x + y + z;
    assert_linear(
        &arena.borrow(),
        again.id(),
        &[(VarId(0), 1.0), (VarId(1), 1.0), (VarId(2), 1.0)],
        0.0,
    );
}

#[test]
fn dynamic_consuming_handles_and_checked_narrowing_preserve_ownership() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 4);
    let prefix = (vars[0] + vars[1]).into_dynamic();
    let nodes_before_growth = arena.__node_count();
    let prefix = prefix + vars[2].into_dynamic();
    assert_eq!(arena.__node_count(), nodes_before_growth);

    let nodes_before_narrowed_growth = arena.__node_count();
    let expression = prefix.into_affine().unwrap() + vars[3];
    assert_eq!(arena.__node_count(), nodes_before_narrowed_growth);

    let snapshot = arena.borrow();
    assert_eq!(snapshot.len(), 5);
    assert_linear(
        &snapshot,
        expression.id(),
        &[(VarId(0), 1.0), (VarId(1), 1.0), (VarId(2), 1.0), (VarId(3), 1.0)],
        0.0,
    );
}

#[test]
fn consuming_add_of_a_constant_expression_reuses_the_root() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 2);
    let prefix = vars[0] + vars[1];
    let nodes_before_constant = arena.__node_count();
    let expression = prefix + Expr::constant(&arena, 7.0);
    assert_eq!(arena.__node_count(), nodes_before_constant + 1, "only the constant is appended");
    assert_linear(&arena.borrow(), expression.id(), &[(VarId(0), 1.0), (VarId(1), 1.0)], 7.0);
}

#[test]
fn subtraction_results_remain_eligible_for_consuming_addition() {
    for scalar_offset in [false, true] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars = variables(&arena, 3);
        let difference = if scalar_offset { vars[0] - 2.0 } else { vars[0] - vars[1] };
        let before = arena.__node_count();
        let result = difference + vars[2];
        assert_eq!(arena.__node_count(), before, "subtraction-created root should be reused");

        let snapshot = arena.borrow();
        let expected = if scalar_offset {
            &[(VarId(0), 1.0), (VarId(2), 1.0)][..]
        } else {
            &[(VarId(0), 1.0), (VarId(1), -1.0), (VarId(2), 1.0)][..]
        };
        let constant = if scalar_offset { -2.0 } else { 0.0 };
        assert_linear(&snapshot, result.id(), expected, constant);
    }
}

#[test]
fn clone_permanently_protects_the_original_and_all_branches() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 4);
    let prefix = vars[0] + vars[1];
    let root = last_node_id(&arena);
    let alias = prefix.clone();
    let left = prefix + vars[2];
    let right = alias + vars[3];
    assert_ne!(left.id(), root);
    assert_ne!(right.id(), root);

    let snapshot = arena.borrow();
    assert_linear(&snapshot, root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
    assert_linear(&snapshot, left.id(), &[(VarId(0), 1.0), (VarId(1), 1.0), (VarId(2), 1.0)], 0.0);
    assert_linear(&snapshot, right.id(), &[(VarId(0), 1.0), (VarId(1), 1.0), (VarId(3), 1.0)], 0.0);
}

#[test]
fn dropping_a_clone_does_not_restore_unique_ownership() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 3);
    let prefix = vars[0] + vars[1];
    let root = last_node_id(&arena);
    drop(prefix.clone());
    let expression = prefix + vars[2];
    assert_ne!(expression.id(), root);
    assert_linear(&arena.borrow(), root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
}

#[test]
fn adding_aliases_does_not_mutate_the_shared_input() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 2);
    let prefix = vars[0] + vars[1];
    let root = last_node_id(&arena);
    let alias = prefix.clone();
    let expression = prefix + alias;
    assert_ne!(expression.id(), root);

    let snapshot = arena.borrow();
    assert_linear(&snapshot, root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
    assert_linear(&snapshot, expression.id(), &[(VarId(0), 2.0), (VarId(1), 2.0)], 0.0);
}

#[test]
#[expect(clippy::op_ref, reason = "exercise the borrowed RHS operator even for Copy leaves")]
fn every_borrowed_add_combination_protects_the_borrowed_root() {
    for mode in 0..3 {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars = variables(&arena, 4);
        let prefix = vars[0] + vars[1];
        let root = last_node_id(&arena);
        let branch = match mode {
            0 => &prefix + vars[2],
            1 => vars[2] + &prefix,
            _ => &prefix + &vars[2],
        };
        let expression = prefix + vars[3];
        assert_ne!(expression.id(), root, "borrow mode {mode}");

        let snapshot = arena.borrow();
        assert_linear(&snapshot, root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
        assert_eq!(evaluate(&snapshot, branch.id(), &&[2.0, 3.0, 5.0, 7.0][..]).unwrap(), 10.0);
        assert_eq!(evaluate(&snapshot, expression.id(), &&[2.0, 3.0, 5.0, 7.0][..]).unwrap(), 12.0);
    }
}

#[test]
fn borrowed_dynamic_view_seals_the_static_owner() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 3);
    let prefix = vars[0] + vars[1];
    let root = last_node_id(&arena);
    let view = prefix.as_dynamic();
    let expression = prefix + vars[2];
    assert_ne!(expression.id(), root);
    assert_linear(&arena.borrow(), view.id(), &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
}

#[test]
fn snapshot_seals_all_roots_even_when_only_one_is_inspected() {
    for retain_snapshot in [false, true] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars = variables(&arena, 5);
        let first = vars[0] + vars[1];
        let first_id = last_node_id(&arena);
        let second = vars[2] + vars[3];
        let second_id = last_node_id(&arena);
        let snapshot = arena.borrow();
        let old_len = snapshot.len();
        assert_linear(&snapshot, first_id, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
        let saved = if retain_snapshot {
            Some(snapshot)
        } else {
            drop(snapshot);
            None
        };

        let first = first + vars[4];
        let second = second + vars[4];
        assert_ne!(first.id(), first_id);
        assert_ne!(second.id(), second_id, "snapshot must also seal uninspected roots");

        let latest = arena.borrow();
        assert_linear(&latest, first_id, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
        assert_linear(&latest, second_id, &[(VarId(2), 1.0), (VarId(3), 1.0)], 0.0);
        assert_eq!(evaluate(&latest, first.id(), &&[1.0; 5][..]).unwrap(), 3.0);
        assert_eq!(evaluate(&latest, second.id(), &&[1.0; 5][..]).unwrap(), 3.0);

        if let Some(saved) = saved {
            assert_eq!(saved.len(), old_len);
            assert_linear(&saved, second_id, &[(VarId(2), 1.0), (VarId(3), 1.0)], 0.0);
        }
    }
}

#[test]
fn publishing_an_id_or_reconstructing_a_handle_protects_the_old_result() {
    for reconstruct in [false, true] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars = variables(&arena, 3);
        let prefix = vars[0] + vars[1];
        let root = last_node_id(&arena);
        let published = if reconstruct {
            Some(Expr::new(root, &arena))
        } else {
            assert_eq!(prefix.publish_id(), root);
            assert_eq!(prefix.publish_id(), root, "publication is idempotent");
            None
        };

        let expression = prefix + vars[2];
        assert_ne!(expression.id(), root);

        if let Some(published) = published {
            assert_eq!(published.into_id(), root);
        }
        assert_linear(&arena.borrow(), root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
    }
}

#[test]
fn into_id_and_raw_parent_publication_keep_roots_immutable() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 3);
    let root = (vars[0] + vars[1]).into_id();
    let expression = Expr::new(root, &arena) + vars[2];
    assert_ne!(expression.id(), root);
    assert_linear(&arena.borrow(), root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);

    let prefix = vars[0] + vars[1];
    let root = last_node_id(&arena);
    let parent = arena.borrow_mut().push(ExprNode::Add(Children::from_slice(&[root, root])));
    let expression = prefix + vars[2];
    assert_ne!(expression.id(), root);

    let snapshot = arena.borrow();
    assert_linear(&snapshot, parent, &[(VarId(0), 2.0), (VarId(1), 2.0)], 0.0);
    assert_linear(
        &snapshot,
        expression.id(),
        &[(VarId(0), 1.0), (VarId(1), 1.0), (VarId(2), 1.0)],
        0.0,
    );
}

#[test]
fn coefficient_boundaries_reuse_through_32_and_defer_33() {
    for count in [31_u32, 32, 33] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let vars = variables(&arena, count);
        let mut expression = vars[0] + vars[1];
        let mut root_nodes = arena.__node_count();
        for (i, variable) in vars.iter().enumerate().skip(2) {
            expression = expression + *variable;
            if i < 32 {
                assert_eq!(arena.__node_count(), root_nodes, "{} coefficients", i + 1);
            } else {
                assert_eq!(arena.__node_count(), root_nodes + 1, "33 coefficients defer to Add");
                root_nodes += 1;
            }
        }

        let snapshot = arena.borrow();
        let expected: Vec<_> = (0..count).map(|i| (VarId(i), 1.0)).collect();
        assert_linear(&snapshot, expression.id(), &expected, 0.0);
        if count <= 32 {
            assert_eq!(snapshot.len(), count as usize + 1);
            assert!(snapshot.nodes().iter().any(
                |node| matches!(node, ExprNode::Linear { coeffs, .. } if coeffs.len() == count as usize)
            ));
        } else {
            assert!(matches!(snapshot.get(expression.id()), ExprNode::Add(_)));
            assert_eq!(snapshot.len(), count as usize + 2);
            assert!(snapshot.nodes().iter().any(|node| {
                matches!(node, ExprNode::Linear { coeffs, .. } if coeffs.len() == 32)
            }));
        }
    }
}

#[test]
fn duplicate_at_the_coefficient_limit_does_not_force_a_new_root() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 32);
    let mut expression = vars[0] + vars[1];
    for variable in &vars[2..] {
        expression = expression + *variable;
    }
    let nodes_before_duplicate = arena.__node_count();
    let expression = expression + vars[0];
    assert_eq!(arena.__node_count(), nodes_before_duplicate);

    let snapshot = arena.borrow();
    let mut expected: Vec<_> = (0..32).map(|i| (VarId(i), 1.0)).collect();
    expected[0].1 = 2.0;
    assert_linear(&snapshot, expression.id(), &expected, 0.0);
    assert_eq!(snapshot.len(), 33);
}

#[test]
fn oversized_linear_operand_falls_back_without_partially_mutating_the_prefix() {
    let arena = ExprArenaCell::new(ExprArena::new());
    let vars = variables(&arena, 34);
    let mut prefix = vars[0] + vars[1];
    for variable in &vars[2..32] {
        prefix = prefix + *variable;
    }

    let root = last_node_id(&arena);

    // Merge a duplicate before overflowing.
    let rhs = vars[0] + vars[32] + vars[33];
    let expression = prefix + rhs;
    assert_ne!(expression.id(), root);

    let snapshot = arena.borrow();
    assert!(matches!(snapshot.get(expression.id()), ExprNode::Add(_)));

    let old: Vec<_> = (0..32).map(|i| (VarId(i), 1.0)).collect();
    assert_linear(&snapshot, root, &old, 0.0);

    let mut expected = old;
    expected[0].1 = 2.0;
    expected.extend([(VarId(32), 1.0), (VarId(33), 1.0)]);
    assert_linear(&snapshot, expression.id(), &expected, 0.0);
}

fn numeric_chain<'a>(
    arena: &'a ExprArenaCell,
    terms: &[(Vec<(VarId, f64)>, f64)],
    right_growing: bool,
    forced_shared: bool,
) -> Expr<'a> {
    let operands: Vec<_> = terms
        .iter()
        .map(|(coeffs, constant)| {
            let id = arena
                .borrow_mut()
                .push(ExprNode::Linear { coeffs: coeffs.clone(), constant: *constant });
            Expr::new(id, arena)
        })
        .collect();

    let mut operands = operands.into_iter();
    let mut expression = operands.next().unwrap();
    for operand in operands {
        // Clone before each step, but never snapshot the optimized prefix.
        if forced_shared {
            drop(expression.clone());
        }
        expression = if right_growing { operand + expression } else { expression + operand };
    }

    expression
}

fn compare_numeric_chain(terms: &[(Vec<(VarId, f64)>, f64)], right_growing: bool) {
    let owned_arena = ExprArenaCell::new(ExprArena::new());
    let shared_arena = ExprArenaCell::new(ExprArena::new());
    let owned = numeric_chain(&owned_arena, terms, right_growing, false);
    let shared = numeric_chain(&shared_arena, terms, right_growing, true);
    let owned_snapshot = owned_arena.borrow();
    let shared_snapshot = shared_arena.borrow();
    assert_eq!(
        linear_bits(&owned_snapshot, owned.id()),
        linear_bits(&shared_snapshot, shared.id()),
        "right_growing={right_growing}, terms={terms:?}"
    );

    for values in [[0.0, -0.0, 1.0, -1.0], [1.0, 2.0, 3.0, 4.0], [-3.0, 0.5, -2.0, 7.0]] {
        assert_eq!(
            evaluate(&owned_snapshot, owned.id(), &&values[..]).unwrap().to_bits(),
            evaluate(&shared_snapshot, shared.id(), &&values[..]).unwrap().to_bits(),
            "evaluation: right_growing={right_growing}, terms={terms:?}, values={values:?}"
        );
    }
}

#[test]
fn floating_point_chains_match_the_forced_shared_fallback_bitwise() {
    let cases = [
        vec![
            (vec![(VarId(2), 1e16), (VarId(0), 1.0)], 1e16),
            (vec![(VarId(2), 1.0), (VarId(1), 3.0)], 1.0),
            (vec![(VarId(2), -1e16), (VarId(0), 2.0)], -1e16),
            (vec![(VarId(3), 4.0), (VarId(2), 1.0)], 1.0),
        ],
        vec![
            (vec![(VarId(0), f64::MAX), (VarId(1), f64::MIN_POSITIVE)], f64::MAX),
            (vec![(VarId(0), f64::MAX), (VarId(1), f64::from_bits(1))], f64::MAX),
            (vec![(VarId(0), -f64::MAX), (VarId(2), 2.0)], -f64::MAX),
        ],
    ];

    for terms in cases {
        for right in [false, true] {
            compare_numeric_chain(&terms, right);
        }
    }
}

#[test]
fn duplicate_entries_cancellation_and_signed_zeros_match_the_fallback_bitwise() {
    let cases = [
        vec![
            (vec![(VarId(2), 4.0), (VarId(0), -0.0), (VarId(2), -1.0)], -0.0),
            (vec![(VarId(2), -3.0), (VarId(1), -0.0)], -0.0),
            (vec![(VarId(2), 2.0), (VarId(0), 0.0)], 0.0),
            (vec![(VarId(2), -2.0)], -0.0),
        ],
        vec![
            (vec![(VarId(0), 1.0)], -0.0),
            (vec![(VarId(0), -1.0)], -0.0),
            (vec![(VarId(1), 2.0)], -0.0),
            (vec![(VarId(1), -2.0)], 0.0),
            (vec![(VarId(0), 3.0)], -0.0),
        ],
    ];

    for terms in cases {
        for right in [false, true] {
            compare_numeric_chain(&terms, right);
        }
    }
}

#[test]
fn nonfinite_coefficients_and_constants_match_the_fallback_bitwise() {
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)] {
        let terms = vec![
            (vec![(VarId(0), 2.0), (VarId(1), -1.0)], -0.0),
            (vec![(VarId(0), value), (VarId(2), 3.0)], value),
            (vec![(VarId(0), -value), (VarId(1), value)], -value),
            (vec![(VarId(3), 1.0)], 1.0),
        ];

        for right in [false, true] {
            compare_numeric_chain(&terms, right);
        }
    }
}

#[test]
fn right_growing_chains_preserve_first_seen_coefficient_order() {
    let terms = vec![
        (vec![(VarId(0), 1.0)], 1.0),
        (vec![(VarId(1), 2.0)], 2.0),
        (vec![(VarId(2), 3.0), (VarId(0), 4.0)], 3.0),
        (vec![(VarId(3), 5.0), (VarId(1), 6.0)], 4.0),
    ];

    compare_numeric_chain(&terms, true);
    let arena = ExprArenaCell::new(ExprArena::new());
    let expression = numeric_chain(&arena, &terms, true, false);
    assert_linear(
        &arena.borrow(),
        expression.id(),
        &[(VarId(3), 5.0), (VarId(1), 8.0), (VarId(2), 3.0), (VarId(0), 5.0)],
        10.0,
    );
}

#[test]
fn symbolic_fallbacks_publish_the_numeric_prefix_and_parameters_stay_live() {
    for parameter_coefficient in [false, true] {
        for right in [false, true] {
            let arena = ExprArenaCell::new(ExprArena::new());
            let vars = variables(&arena, 4);
            let param = arena.borrow_mut().new_param(2.0);
            let p = Expr::from_param(&arena, param);
            let prefix = vars[0] + vars[1];
            let root = last_node_id(&arena);
            let symbolic: Expr<'_, Affine> =
                if parameter_coefficient { p * vars[2] } else { p.into() };
            let expression = if right { symbolic + prefix } else { prefix + symbolic };
            assert_ne!(expression.id(), root);

            let expression = expression + vars[3];
            for value in [2.0, -3.0, 0.0, 7.0] {
                p.set_param_value(value).unwrap();
                let snapshot = arena.borrow();
                assert_linear(&snapshot, root, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
                assert!(matches!(snapshot.get(expression.id()), ExprNode::Add(_)));

                let expected =
                    2.0 + 3.0 + 7.0 + if parameter_coefficient { value * 5.0 } else { value };
                assert_eq!(
                    evaluate(&snapshot, expression.id(), &&[2.0, 3.0, 5.0, 7.0][..])
                        .unwrap()
                        .to_bits(),
                    expected.to_bits()
                );

                let terms = extract_linear(&snapshot, expression.id()).unwrap();
                assert_eq!(terms.constant, if parameter_coefficient { 0.0 } else { value });

                if parameter_coefficient {
                    assert_eq!(
                        terms.coeffs.iter().find(|(var, _)| *var == VarId(2)).unwrap().1,
                        value
                    );
                }
            }
        }
    }
}

#[test]
fn cross_model_add_panics_before_mutating_either_owned_root() {
    for dynamic in [false, true] {
        let arena = ExprArenaCell::new(ExprArena::new());
        let foreign = ExprArenaCell::new(ExprArena::new());
        let local_vars = variables(&arena, 2);
        let foreign_vars = variables(&foreign, 2);
        let lhs = local_vars[0] + local_vars[1];
        let rhs = foreign_vars[0] + foreign_vars[1];
        let lhs_id = lhs.id();
        let rhs_id = rhs.id();
        let result = catch_unwind(AssertUnwindSafe(|| {
            if dynamic {
                let _ = lhs.into_dynamic() + rhs.into_dynamic();
            } else {
                let _ = lhs + rhs;
            }
        }));

        let panic = result.expect_err("cross-model addition must panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(message.contains("expressions belong to different arenas"), "{message}");

        let local = arena.borrow();
        let remote = foreign.borrow();
        assert_eq!(local.len(), 3);
        assert_eq!(remote.len(), 3);
        assert_linear(&local, lhs_id, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
        assert_linear(&remote, rhs_id, &[(VarId(0), 1.0), (VarId(1), 1.0)], 0.0);
    }
}
