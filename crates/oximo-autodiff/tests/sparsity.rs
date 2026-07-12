//! Stable tests for the exact structural Hessian pattern and the star-coloring
//! compression of the Hessian into Hessian-vector-product seeds.
#![allow(clippy::unreadable_literal, clippy::cast_precision_loss)]

use oximo_autodiff::sparsity::{HessianColoring, hessian_pattern, star_hessian_coloring};
use oximo_expr::{ExprArena, ExprNode, VarId, extract_quadratic};
use rustc_hash::FxHashSet;

fn var(arena: &mut ExprArena, i: u32) -> oximo_expr::ExprId {
    arena.var(VarId(i))
}

#[test]
fn separable_sum_of_sins_is_diagonal() {
    let mut arena = ExprArena::new();
    let sins: Vec<_> = (0..5)
        .map(|i| {
            let x = var(&mut arena, i);
            arena.push(ExprNode::Sin(x))
        })
        .collect();
    let root = arena.push(ExprNode::Add(sins.into_iter().collect()));
    assert_eq!(hessian_pattern(&arena, root), vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);
}

#[test]
fn product_and_unary_patterns() {
    let mut arena = ExprArena::new();
    let x0 = var(&mut arena, 0);
    let x1 = var(&mut arena, 1);
    let x2 = var(&mut arena, 2);

    let mul = arena.push(ExprNode::Mul([x0, x1].into_iter().collect()));
    let sin = arena.push(ExprNode::Sin(x2));
    let root = arena.push(ExprNode::Add([mul, sin].into_iter().collect()));
    assert_eq!(hessian_pattern(&arena, root), vec![(1, 0), (2, 2)]);

    let add = arena.push(ExprNode::Add([x0, x1].into_iter().collect()));
    let two = arena.constant(2.0);
    let sq = arena.push(ExprNode::Pow(add, two));
    assert_eq!(hessian_pattern(&arena, sq), vec![(0, 0), (1, 0), (1, 1)]);

    let div = arena.push(ExprNode::Div(x0, x1));
    assert_eq!(hessian_pattern(&arena, div), vec![(1, 0), (1, 1)]);

    let pow = arena.push(ExprNode::Pow(x0, x1));
    assert_eq!(hessian_pattern(&arena, pow), vec![(0, 0), (1, 0), (1, 1)]);
}

#[test]
fn abs_passes_through_its_argument_pattern() {
    let mut arena = ExprArena::new();
    let lin = arena.linear(vec![(VarId(0), 2.0), (VarId(1), -1.0)], 0.0);
    let abs_lin = arena.push(ExprNode::Abs(lin));
    assert_eq!(hessian_pattern(&arena, abs_lin), vec![]);

    let x0 = var(&mut arena, 0);
    let three = arena.constant(3.0);
    let cube = arena.push(ExprNode::Pow(x0, three));
    let abs_cube = arena.push(ExprNode::Abs(cube));
    assert_eq!(hessian_pattern(&arena, abs_cube), vec![(0, 0)]);
}

#[test]
fn dag_shared_subexpression() {
    let mut arena = ExprArena::new();
    let x0 = var(&mut arena, 0);
    let s = arena.push(ExprNode::Sin(x0));
    let mul = arena.push(ExprNode::Mul([s, s].into_iter().collect()));
    let root = arena.push(ExprNode::Add([mul, s].into_iter().collect()));
    assert_eq!(hessian_pattern(&arena, root), vec![(0, 0)]);
}

#[test]
fn pattern_of_quadratic_matches_extract_quadratic() {
    let mut arena = ExprArena::new();
    let x0 = var(&mut arena, 0);
    let x1 = var(&mut arena, 1);
    let c3 = arena.constant(3.0);
    let sq0 = arena.push(ExprNode::Mul([c3, x0, x0].into_iter().collect()));
    let cross = arena.push(ExprNode::Mul([x0, x1].into_iter().collect()));
    let sq1 = arena.push(ExprNode::Mul([x1, x1].into_iter().collect()));
    let lin = arena.linear(vec![(VarId(0), 2.0), (VarId(1), -1.0)], 0.0);
    let root = arena.push(ExprNode::Add([sq0, cross, sq1, lin].into_iter().collect()));

    let mut from_extract: Vec<(u32, u32)> = extract_quadratic(&arena, root)
        .expect("degree 2")
        .hessian
        .iter()
        .map(|&(r, c, _)| (r.0, c.0))
        .collect();
    from_extract.sort_unstable();
    assert_eq!(hessian_pattern(&arena, root), from_extract);
}

#[test]
fn triple_product_has_no_diagonal() {
    let mut arena = ExprArena::new();
    let x = var(&mut arena, 0);
    let y = var(&mut arena, 1);
    let z = var(&mut arena, 2);
    let prod = arena.push(ExprNode::Mul([x, y, z].into_iter().collect()));
    assert_eq!(hessian_pattern(&arena, prod), vec![(1, 0), (2, 0), (2, 1)]);
}

#[test]
fn sum_of_variables_has_empty_hessian() {
    let mut arena = ExprArena::new();
    let x = var(&mut arena, 0);
    let y = var(&mut arena, 1);
    let sum = arena.push(ExprNode::Add([x, y].into_iter().collect()));
    assert_eq!(hessian_pattern(&arena, sum), vec![]);
}

#[test]
fn sin_of_sum_is_a_full_clique() {
    let mut arena = ExprArena::new();
    let x = var(&mut arena, 0);
    let y = var(&mut arena, 1);
    let sum = arena.push(ExprNode::Add([x, y].into_iter().collect()));
    let s = arena.push(ExprNode::Sin(sum));
    assert_eq!(hessian_pattern(&arena, s), vec![(0, 0), (1, 0), (1, 1)]);
}

#[test]
fn square_of_sum_matches_the_smooth_unary_clique() {
    let mut arena = ExprArena::new();
    let x = var(&mut arena, 0);
    let y = var(&mut arena, 1);
    let sum = arena.push(ExprNode::Add([x, y].into_iter().collect()));
    let two = arena.constant(2.0);
    let sq = arena.push(ExprNode::Pow(sum, two));
    assert_eq!(hessian_pattern(&arena, sq), vec![(0, 0), (1, 0), (1, 1)]);
}

#[test]
fn self_division_keeps_its_structural_entry() {
    let mut arena = ExprArena::new();
    let x = var(&mut arena, 0);
    let div = arena.push(ExprNode::Div(x, x));
    assert_eq!(hessian_pattern(&arena, div), vec![(0, 0)]);
}

/// Exact recovery check, fill a deterministic (pseudo-random, collision-free)
/// symmetric matrix on `pattern`, simulate one HVP per group
/// (`b[g][row] = sum_{col in group} H[row][col]`), and confirm the coloring's
/// recovery plan reproduces every entry.
fn assert_recovers(pattern: &[(usize, usize)]) -> HessianColoring {
    let coloring = star_hessian_coloring(pattern);
    let n = pattern.iter().flat_map(|&(r, c)| [r, c]).max().map_or(0, |m| m + 1);

    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 + 0.5
    };
    let mut h = vec![vec![0.0f64; n]; n];
    for &(r, c) in pattern {
        let v = rng();
        h[r][c] = v;
        h[c][r] = v;
    }

    let b: Vec<Vec<f64>> = coloring
        .groups
        .iter()
        .map(|cols| (0..n).map(|row| cols.iter().map(|&col| h[row][col]).sum()).collect())
        .collect();

    for (i, &(r, c)) in pattern.iter().enumerate() {
        let (g, row) = coloring.recover[i];
        assert!(
            (b[g][row] - h[r][c]).abs() < 1e-12,
            "entry {:?}: recovered {}, want {}",
            (r, c),
            b[g][row],
            h[r][c]
        );
    }
    coloring
}

#[test]
fn diagonal_pattern_needs_one_group() {
    let pattern: Vec<(usize, usize)> = (0..6).map(|i| (i, i)).collect();
    let coloring = assert_recovers(&pattern);
    assert_eq!(coloring.groups.len(), 1);
    assert_eq!(coloring.groups[0], vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn tridiagonal_pattern_needs_at_most_three_groups() {
    let mut pattern: Vec<(usize, usize)> = (0..6).map(|i| (i, i)).collect();
    pattern.extend((1..6).map(|i| (i, i - 1)));
    pattern.sort_unstable();
    let coloring = assert_recovers(&pattern);
    assert!(coloring.groups.len() <= 3, "tridiagonal needs <= 3 groups, got {:?}", coloring.groups);
}

/// The pattern {(1,0),(2,1),(2,2)} is the classic case where treating columns 0
/// and 2 as independent corrupts entry (1,0) via `H[1,2] = H[2,1]`. The
/// star-coloring recovery plan must reproduce every entry exactly.
#[test]
fn adjacent_columns_recover_exactly() {
    assert_recovers(&[(1, 0), (2, 1), (2, 2)]);
}

#[test]
fn arrow_pattern_collapses_to_two_seeds() {
    // Hub column 0 couples to every spoke. Distance-2 grouping needed one seed
    // per column (n); star coloring recovers the whole arrow in two HVPs.
    let n = 5;
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    pattern.extend((1..n).map(|i| (i, 0)));
    pattern.sort_unstable();
    let coloring = assert_recovers(&pattern);
    assert_eq!(coloring.groups.len(), 2, "star coloring compresses the hub");
}

#[test]
fn randomized_patterns_recover_exactly() {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move |bound: usize| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % bound
    };
    for _ in 0..50 {
        let n = 4 + next(12);
        let nnz = 1 + next(2 * n);
        let mut pattern: FxHashSet<(usize, usize)> = FxHashSet::default();
        for _ in 0..nnz {
            let i = next(n);
            let j = next(n);
            pattern.insert((i.max(j), i.min(j)));
        }
        let mut pattern: Vec<(usize, usize)> = pattern.into_iter().collect();
        pattern.sort_unstable();
        assert_recovers(&pattern);
    }
}

/// Build a normalized lower-triangle Hessian pattern for an `n`-vertex graph.
fn graph_pattern(n: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut set: FxHashSet<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    for &(a, b) in edges {
        set.insert((a.max(b), a.min(b)));
    }
    let mut pattern: Vec<(usize, usize)> = set.into_iter().collect();
    pattern.sort_unstable();
    pattern
}

#[test]
fn path_graph_recovers_exactly() {
    let edges: Vec<(usize, usize)> = (1..6).map(|i| (i, i - 1)).collect();
    let coloring = assert_recovers(&graph_pattern(6, &edges));
    assert!(coloring.groups.len() <= 3, "path needs <= 3 groups, got {}", coloring.groups.len());
}

#[test]
fn cycle_graph_recovers_exactly() {
    let mut edges: Vec<(usize, usize)> = (1..6).map(|i| (i, i - 1)).collect();
    edges.push((5, 0));
    assert_recovers(&graph_pattern(6, &edges));
}

#[test]
fn complete_graph_recovers_exactly() {
    let n = 5;
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..i {
            edges.push((i, j));
        }
    }
    let coloring = assert_recovers(&graph_pattern(n, &edges));
    assert_eq!(coloring.groups.len(), n, "dense block needs one seed per column");
}

#[test]
fn disconnected_graph_recovers_exactly() {
    let edges = [(1, 0), (2, 0), (2, 1), (4, 3), (5, 3), (5, 4)];
    assert_recovers(&graph_pattern(6, &edges));
}
