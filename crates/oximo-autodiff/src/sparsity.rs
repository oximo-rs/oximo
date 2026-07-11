//! Structural sparsity analysis: which variables an expression touches, the
//! exact second-order interaction pattern, and the Jacobian/Hessian
//! patterns derivative-based solvers ask for up front.

use oximo_expr::{ExprArena, ExprId, ExprNode, Visitor, walk};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::slot::{FunctionSlot, SlotKind};

/// Sorted, deduplicated indices of the variables appearing under `root`.
pub fn variable_support(arena: &ExprArena, root: ExprId) -> Vec<u32> {
    struct Support(FxHashSet<u32>);
    impl Visitor for Support {
        fn visit(&mut self, _arena: &ExprArena, _id: ExprId, node: &ExprNode) {
            match node {
                ExprNode::Var(v) => {
                    self.0.insert(v.0);
                }
                ExprNode::Linear { coeffs, .. } => {
                    self.0.extend(coeffs.iter().map(|(v, _)| v.0));
                }
                _ => {}
            }
        }
    }
    let mut visitor = Support(FxHashSet::default());
    walk(arena, root, &mut visitor);
    let mut support: Vec<u32> = visitor.0.into_iter().collect();
    support.sort_unstable();
    support
}

/// Per-node first/second-order structural sparsity.
/// `vars` is the gradient support, `pairs` the normalized
/// lower-triangle second-partial support.
#[derive(Clone, Debug, Default)]
struct NodeSparsity {
    vars: FxHashSet<u32>,
    pairs: FxHashSet<(u32, u32)>,
}

fn norm(i: u32, j: u32) -> (u32, u32) {
    if i >= j { (i, j) } else { (j, i) }
}

fn add_clique(vars: &FxHashSet<u32>, pairs: &mut FxHashSet<(u32, u32)>) {
    let mut sorted: Vec<u32> = vars.iter().copied().collect();
    sorted.sort_unstable();
    for (i, &row) in sorted.iter().enumerate() {
        for &col in &sorted[..=i] {
            pairs.insert((row, col));
        }
    }
}

fn add_cross(a: &FxHashSet<u32>, b: &FxHashSet<u32>, pairs: &mut FxHashSet<(u32, u32)>) {
    for &i in a {
        for &j in b {
            pairs.insert(norm(i, j));
        }
    }
}

/// Exact structural lower-triangle Hessian pattern of the expression rooted
/// at `root`. Normalized `(row, col)` index pairs with `row >= col`, sorted
/// and deduplicated.
///
/// "Exact structural" means a superset of the numerically nonzero second
/// partials that ignores value cancellation. Parameters stay symbolic, so the
/// pattern is independent of current parameter values.
/// `Abs` contributes only its argument's pattern.
pub fn hessian_pattern(arena: &ExprArena, root: ExprId) -> Vec<(u32, u32)> {
    let mut memo: FxHashMap<ExprId, NodeSparsity> = FxHashMap::default();
    let result = node_sparsity(arena, root, &mut memo);
    let mut pattern: Vec<(u32, u32)> = result.pairs.iter().copied().collect();
    pattern.sort_unstable();
    pattern
}

fn node_sparsity<'m>(
    arena: &ExprArena,
    id: ExprId,
    memo: &'m mut FxHashMap<ExprId, NodeSparsity>,
) -> &'m NodeSparsity {
    if !memo.contains_key(&id) {
        let computed = compute_node_sparsity(arena, id, memo);
        memo.insert(id, computed);
    }
    &memo[&id]
}

// Exact 0.0/1.0 exponent bucketing matches the semantics of `classify` and
// the tape's PowC lowering.
#[allow(clippy::float_cmp)]
fn compute_node_sparsity(
    arena: &ExprArena,
    id: ExprId,
    memo: &mut FxHashMap<ExprId, NodeSparsity>,
) -> NodeSparsity {
    match arena.get(id) {
        ExprNode::Const(_) | ExprNode::Param(_) => NodeSparsity::default(),
        ExprNode::Var(v) => {
            NodeSparsity { vars: std::iter::once(v.0).collect(), pairs: FxHashSet::default() }
        }
        ExprNode::Linear { coeffs, .. } => NodeSparsity {
            vars: coeffs.iter().map(|(v, _)| v.0).collect(),
            pairs: FxHashSet::default(),
        },
        ExprNode::Neg(inner) | ExprNode::Abs(inner) => node_sparsity(arena, *inner, memo).clone(),
        ExprNode::Add(children) => {
            let mut acc = NodeSparsity::default();
            for &c in children {
                let s = node_sparsity(arena, c, memo);
                acc.vars.extend(s.vars.iter().copied());
                acc.pairs.extend(s.pairs.iter().copied());
            }
            acc
        }
        // Pairwise left fold is exactly the n-ary rule, at each step the
        // accumulated vars are the union of earlier factors, so the cross
        // products cover every distinct factor pair.
        ExprNode::Mul(children) => {
            let mut acc = NodeSparsity::default();
            for &c in children {
                let s = node_sparsity(arena, c, memo);
                add_cross(&acc.vars, &s.vars, &mut acc.pairs);
                acc.vars.extend(s.vars.iter().copied());
                acc.pairs.extend(s.pairs.iter().copied());
            }
            acc
        }
        // a/b = a * (1/b), and 1/b is nonlinear in all of b's variables.
        ExprNode::Div(num, den) => {
            let n = node_sparsity(arena, *num, memo).clone();
            let d = node_sparsity(arena, *den, memo);
            let mut acc = NodeSparsity {
                vars: n.vars.union(&d.vars).copied().collect(),
                pairs: n.pairs.union(&d.pairs).copied().collect(),
            };
            add_clique(&d.vars, &mut acc.pairs);
            add_cross(&n.vars, &d.vars, &mut acc.pairs);
            acc
        }
        // phi(g) for smooth nonlinear phi: phi''*g_i'g_j' + phi'·g''_ij.
        ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Log(inner) => smooth_unary(arena, *inner, memo),
        ExprNode::Pow(base, exp) => {
            // Constant-exponent detection mirrors the tape's PowC check.
            if let ExprNode::Const(e) = arena.get(*exp) {
                if *e == 0.0 {
                    NodeSparsity::default()
                } else if *e == 1.0 {
                    node_sparsity(arena, *base, memo).clone()
                } else {
                    smooth_unary(arena, *base, memo)
                }
            } else {
                // g^e = exp(e*ln g): the first-derivative products alone fill
                // the clique over vars(g) U vars(e).
                let b = node_sparsity(arena, *base, memo).clone();
                let e = node_sparsity(arena, *exp, memo);
                let mut acc = NodeSparsity {
                    vars: b.vars.union(&e.vars).copied().collect(),
                    pairs: b.pairs.union(&e.pairs).copied().collect(),
                };
                add_clique(&acc.vars, &mut acc.pairs);
                acc
            }
        }
    }
}

fn smooth_unary(
    arena: &ExprArena,
    inner: ExprId,
    memo: &mut FxHashMap<ExprId, NodeSparsity>,
) -> NodeSparsity {
    let mut acc = node_sparsity(arena, inner, memo).clone();
    add_clique(&acc.vars, &mut acc.pairs);
    acc
}

/// Constraint Jacobian pattern as `(constraint, variable)` index pairs in
/// row-major order. Row `i`'s entries are exactly `slots[i].support`.
pub fn jacobian_structure(slots: &[FunctionSlot]) -> Vec<(usize, usize)> {
    let mut entries = Vec::with_capacity(slots.iter().map(|s| s.support.len()).sum());
    for (row, slot) in slots.iter().enumerate() {
        entries.extend(slot.support.iter().map(|&v| (row, v as usize)));
    }
    entries
}

/// Lower-triangle Hessian-of-the-Lagrangian pattern (`row >= col`), sorted and
/// deduplicated, over the objective and all constraints.
///
/// Quadratic slots contribute their exact constant-Hessian entries, nonlinear
/// slots their exact structural pattern (`FunctionSlot::hess_pairs`, computed
/// by [`hessian_pattern`]).
pub fn hessian_lagrangian_structure<'a, I>(slots: I) -> Vec<(usize, usize)>
where
    I: IntoIterator<Item = &'a FunctionSlot>,
{
    let mut entries = FxHashSet::default();
    for slot in slots {
        match &slot.kind {
            SlotKind::Linear(_) => {}
            SlotKind::Quadratic(q) => {
                for &(r, c, _) in &q.hessian {
                    entries.insert((r.index(), c.index()));
                }
            }
            SlotKind::Nonlinear(_) => {
                entries.extend(slot.hess_pairs.iter().map(|&(r, c)| (r as usize, c as usize)));
            }
        }
    }
    let mut entries: Vec<(usize, usize)> = entries.into_iter().collect();
    entries.sort_unstable();
    entries
}

/// Direct-recovery coloring of a symmetric Hessian pattern.
/// One Hessian-vector product per group, then each entry is read
/// from a single group/row with no linear solve.
#[derive(Clone, Debug)]
pub struct HessianColoring {
    /// Columns seeded together, the caller performs one HVP per group.
    pub groups: Vec<Vec<usize>>,
    /// Aligned with the input `pattern`: entry `i` is recovered directly as
    /// `value = hv_of[group][row]`, where `(group, row) = recover[i]`.
    pub recover: Vec<(usize, usize)>,
}

/// Star-coloring compression of a symmetric Hessian `pattern` for direct
/// recovery from Hessian-vector products.
///
/// Builds the adjacency graph of the pattern (vertices=variables, edges=
/// off-diagonal structural nonzeros), star-colors it, and seeds one HVP per
/// color class that some entry reads from. Recovery is direct, since a diagonal
/// `(i, i)` is read from `i`'s own color at row `i` (proper coloring isolates
/// it); an off-diagonal `(u, w)` is read from whichever endpoint has the other
/// as its only neighbor of that color falling back to a lone seed of `u`
/// if neither does, so the returned is exact regardless of coloring quality.
///
/// `pattern` is a normalized lower-triangle pattern (`row >= col`).
pub fn star_hessian_coloring(pattern: &[(usize, usize)]) -> HessianColoring {
    let mut adj: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
    for &(r, c) in pattern {
        adj.entry(r).or_default(); // ensure diagonal-only vertices are colored
        if r != c {
            adj.entry(r).or_default().insert(c);
            adj.entry(c).or_default().insert(r);
        }
    }

    let color = greedy_star_coloring(&adj);

    let mut nbr_colors: FxHashMap<usize, FxHashMap<usize, usize>> = FxHashMap::default();
    for (&v, nbrs) in &adj {
        let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
        for &w in nbrs {
            *counts.entry(color[&w]).or_insert(0) += 1;
        }
        nbr_colors.insert(v, counts);
    }

    // Members of each color class (sorted), materialized into a seed group only
    // when some entry reads from that class.
    let mut class: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (&v, &col) in &color {
        class.entry(col).or_default().push(v);
    }
    for members in class.values_mut() {
        members.sort_unstable();
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of_color: FxHashMap<usize, usize> = FxHashMap::default();
    let mut singleton_of: FxHashMap<usize, usize> = FxHashMap::default();
    let mut recover: Vec<(usize, usize)> = Vec::with_capacity(pattern.len());

    let unique = |v: usize, col: usize| nbr_colors[&v].get(&col) == Some(&1);

    for &(r, c) in pattern {
        let entry = if r == c {
            (color_group(color[&r], &class, &mut groups, &mut group_of_color), r)
        } else if unique(r, color[&c]) {
            // `c` is `r`'s only color[c] neighbor -> seed color[c], read row r.
            (color_group(color[&c], &class, &mut groups, &mut group_of_color), r)
        } else if unique(c, color[&r]) {
            (color_group(color[&r], &class, &mut groups, &mut group_of_color), c)
        } else {
            // No clean class read (a non-star edge).
            (singleton_group(r, &mut groups, &mut singleton_of), c)
        };
        recover.push(entry);
    }

    HessianColoring { groups, recover }
}

/// Greedy star coloring of `adj`, a proper coloring in which no path on four
/// vertices is two-colored, so every pair of colors induces a star forest and
/// the Hessian is directly recoverable.
fn greedy_star_coloring(adj: &FxHashMap<usize, FxHashSet<usize>>) -> FxHashMap<usize, usize> {
    let mut order: Vec<usize> = adj.keys().copied().collect();
    order.sort_unstable_by_key(|&v| (usize::MAX - adj[&v].len(), v));

    let mut color: FxHashMap<usize, usize> = FxHashMap::default();
    for &v in &order {
        let nbrs = &adj[&v];
        // Colored-neighbor color multiplicities, for the internal-P4 rule.
        let mut nbr_count: FxHashMap<usize, usize> = FxHashMap::default();
        for &w in nbrs {
            if let Some(&cw) = color.get(&w) {
                *nbr_count.entry(cw).or_insert(0) += 1;
            }
        }
        // Proper coloring forbids neighbor colors outright.
        let mut forbidden: FxHashSet<usize> = nbr_count.keys().copied().collect();

        for &w in nbrs {
            let Some(&b) = color.get(&w) else { continue };
            // Endpoint P4  v-w-x-y  colored c,b,c,b: forbid color(x) = c when x has
            // another b-colored neighbor y != w (assigning c to v closes the P4).
            for &x in &adj[&w] {
                if x == v {
                    continue;
                }
                let Some(&cx) = color.get(&x) else { continue };
                if adj[&x].iter().any(|&y| y != w && color.get(&y) == Some(&b)) {
                    forbidden.insert(cx);
                }
            }
            // Internal P4  u-v-w-x  colored b,c,b,c: if v already has another
            // b-colored neighbor (u), forbid color(x) for x adjacent to w.
            if nbr_count[&b] >= 2 {
                for &x in &adj[&w] {
                    if x != v {
                        if let Some(&cx) = color.get(&x) {
                            forbidden.insert(cx);
                        }
                    }
                }
            }
        }

        let mut c = 0;
        while forbidden.contains(&c) {
            c += 1;
        }
        color.insert(v, c);
    }
    color
}

/// Index of the seed group for color `col`, creating it on first use.
fn color_group(
    col: usize,
    class: &FxHashMap<usize, Vec<usize>>,
    groups: &mut Vec<Vec<usize>>,
    group_of_color: &mut FxHashMap<usize, usize>,
) -> usize {
    *group_of_color.entry(col).or_insert_with(|| {
        let idx = groups.len();
        groups.push(class[&col].clone());
        idx
    })
}

/// Index of a lone-column seed group for `v`, created once per vertex.
fn singleton_group(
    v: usize,
    groups: &mut Vec<Vec<usize>>,
    singleton_of: &mut FxHashMap<usize, usize>,
) -> usize {
    *singleton_of.entry(v).or_insert_with(|| {
        let idx = groups.len();
        groups.push(vec![v]);
        idx
    })
}
