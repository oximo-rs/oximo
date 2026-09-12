//! Structural sparsity analysis: which variables an expression touches, the
//! exact second-order interaction pattern, and the Jacobian/Hessian
//! patterns derivative-based solvers ask for up front.

use std::ops::Range;

use oximo_expr::{ExprArena, ExprId, ExprNode, UnaryOp};
use rustc_hash::FxHashSet;

use crate::slot::{FunctionSlot, SlotKind};

// Keeps the dense triangular pair bitmap at or below 64 KiB. Wider expressions
// use sparse rows and pairs so memory follows actual structural nonzeros.
const DENSE_SPARSITY_MAX_VARS: usize = 1_024;
type HessianPattern = Vec<(usize, usize)>;
type HessianPatterns = (HessianPattern, Option<HessianPattern>);

/// Sorted, deduplicated indices of the variables appearing under `root`.
pub fn variable_support(arena: &ExprArena, root: ExprId) -> Vec<u32> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut support = Vec::new();

    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.index()], true) {
            continue;
        }
        match arena.get(id) {
            ExprNode::Var(v) => support.push(v.0),
            ExprNode::Linear { coeffs, .. } => {
                support.extend(coeffs.iter().map(|(v, _)| v.0));
            }
            ExprNode::Add(children)
            | ExprNode::Mul(children)
            | ExprNode::Min(children)
            | ExprNode::Max(children) => {
                stack.extend(children.iter().copied());
            }
            ExprNode::Unary(_, inner) => stack.push(*inner),
            ExprNode::Pow(base, exp) | ExprNode::Div(base, exp) | ExprNode::Atan2(base, exp) => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprNode::Const(_) | ExprNode::Param(_) => {}
        }
    }

    support.sort_unstable();
    support.dedup();
    support
}

pub(crate) struct StructuralSparsity {
    pub(crate) support: Vec<u32>,
    pub(crate) hess_pairs: Vec<(u32, u32)>,
}

pub(crate) fn structural_sparsity(arena: &ExprArena, root: ExprId) -> StructuralSparsity {
    structural_sparsity_with_workspace(arena, root, &mut SparsityWorkspace::default())
}

#[derive(Clone, Copy, Default)]
struct NodeMeta {
    syntax_epoch: u32,
    active_epoch: u32,
    active_state: u8,
    row: usize,
}

#[derive(Clone, Copy)]
enum WalkAction {
    ActiveEnter(ExprId),
    ActiveExit(ExprId),
    Syntax(ExprId),
}

#[derive(Default)]
pub(crate) struct SparsityWorkspace {
    epoch: u32,
    meta: Vec<NodeMeta>,
    walk: Vec<WalkAction>,
    order: Vec<ExprId>,
    syntax_vars: Vec<u32>,
    active_vars: Vec<u32>,
    supports: Vec<u64>,
    clique_covered: Vec<bool>,
    pairs: Vec<u64>,
    sparse_supports: Vec<usize>,
    sparse_rows: Vec<Range<usize>>,
    sparse_scratch: Vec<usize>,
    sparse_pairs: FxHashSet<(usize, usize)>,
}

impl SparsityWorkspace {
    fn begin(&mut self, arena_len: usize) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.meta.fill(NodeMeta::default());
            self.epoch = 1;
        }
        self.meta.resize(arena_len, NodeMeta::default());
        self.walk.clear();
        self.order.clear();
        self.syntax_vars.clear();
        self.active_vars.clear();
        self.supports.clear();
        self.clique_covered.clear();
        self.pairs.clear();
        self.sparse_supports.clear();
        self.sparse_rows.clear();
        self.sparse_scratch.clear();
        self.sparse_pairs.clear();
    }

    fn mark_syntax(&mut self, arena: &ExprArena, id: ExprId) -> bool {
        let meta = &mut self.meta[id.index()];
        if meta.syntax_epoch == self.epoch {
            return false;
        }
        meta.syntax_epoch = self.epoch;
        collect_node_vars(arena.get(id), &mut self.syntax_vars);
        true
    }

    fn alloc_row(&mut self, words: usize) -> usize {
        let row = self.clique_covered.len();
        let new_len = self
            .supports
            .len()
            .checked_add(words)
            .expect("dense sparsity support storage size overflow");
        self.supports.resize(new_len, 0);
        self.clique_covered.push(false);
        row
    }

    fn add_clique_once(&mut self, row: usize, words: usize) {
        if !self.clique_covered[row] {
            add_clique(&self.supports, row, words, &mut self.pairs);
            self.clique_covered[row] = true;
        }
    }

    fn add_sparse_clique_once(&mut self, row: usize) {
        if !self.clique_covered[row] {
            let range = self.sparse_rows[row].clone();
            add_sparse_clique(&self.sparse_supports[range], &mut self.sparse_pairs);
            self.clique_covered[row] = true;
        }
    }
}

#[derive(Clone, Copy)]
enum StorageMode {
    Dense { words: usize },
    Sparse,
}

#[derive(Clone, Copy)]
enum ConstantExponent {
    Zero,
    One,
    Other,
}

fn constant_exponent(arena: &ExprArena, exp: ExprId) -> Option<ConstantExponent> {
    let ExprNode::Const(value) = arena.get(exp) else { return None };
    let bits = value.to_bits();
    if bits & !(1_u64 << 63) == 0 {
        Some(ConstantExponent::Zero)
    } else if bits == 1.0_f64.to_bits() {
        Some(ConstantExponent::One)
    } else {
        Some(ConstantExponent::Other)
    }
}

#[inline]
fn words_for(bits: usize) -> usize {
    bits.div_ceil(u64::BITS as usize)
}

#[inline]
fn set_bit(bits: &mut [u64], bit: usize) {
    bits[bit / u64::BITS as usize] |= 1 << (bit % u64::BITS as usize);
}

#[inline]
fn has_bit(bits: &[u64], bit: usize) -> bool {
    bits[bit / u64::BITS as usize] & (1 << (bit % u64::BITS as usize)) != 0
}

#[inline]
fn row_start(row: usize, words: usize) -> usize {
    row * words
}

fn union_rows(supports: &mut [u64], dst: usize, src: usize, words: usize) {
    let dst = row_start(dst, words);
    let src = row_start(src, words);
    for word in 0..words {
        supports[dst + word] |= supports[src + word];
    }
}

fn set_pair(pairs: &mut [u64], i: usize, j: usize) {
    let (row, col) = if i >= j { (i, j) } else { (j, i) };
    set_bit(pairs, row * (row + 1) / 2 + col);
}

fn add_cross(supports: &[u64], a: usize, b: usize, words: usize, pairs: &mut [u64]) {
    let a = row_start(a, words);
    let b = row_start(b, words);
    for aw in 0..words {
        let mut a_bits = supports[a + aw];
        while a_bits != 0 {
            let i = aw * u64::BITS as usize + a_bits.trailing_zeros() as usize;
            a_bits &= a_bits - 1;
            for bw in 0..words {
                let mut b_bits = supports[b + bw];
                while b_bits != 0 {
                    let j = bw * u64::BITS as usize + b_bits.trailing_zeros() as usize;
                    b_bits &= b_bits - 1;
                    set_pair(pairs, i, j);
                }
            }
        }
    }
}

fn add_clique(supports: &[u64], row: usize, words: usize, pairs: &mut [u64]) {
    let start = row_start(row, words);
    for iw in 0..words {
        let mut i_bits = supports[start + iw];
        while i_bits != 0 {
            let i = iw * u64::BITS as usize + i_bits.trailing_zeros() as usize;
            i_bits &= i_bits - 1;
            for jw in 0..=iw {
                let mut j_bits = supports[start + jw];
                while j_bits != 0 {
                    let j = jw * u64::BITS as usize + j_bits.trailing_zeros() as usize;
                    j_bits &= j_bits - 1;
                    if j <= i {
                        set_pair(pairs, i, j);
                    }
                }
            }
        }
    }
}

fn add_sparse_cross(a: &[usize], b: &[usize], pairs: &mut FxHashSet<(usize, usize)>) {
    for &i in a {
        for &j in b {
            pairs.insert(if i >= j { (i, j) } else { (j, i) });
        }
    }
}

fn add_sparse_clique(vars: &[usize], pairs: &mut FxHashSet<(usize, usize)>) {
    for (index, &row) in vars.iter().enumerate() {
        for &col in &vars[..=index] {
            pairs.insert((row, col));
        }
    }
}

fn append_sparse_row(
    supports: &mut Vec<usize>,
    rows: &mut Vec<Range<usize>>,
    clique_covered: &mut Vec<bool>,
    values: &[usize],
) -> usize {
    let row = rows.len();
    let start = supports.len();
    supports.extend_from_slice(values);
    rows.push(start..supports.len());
    clique_covered.push(false);
    row
}

fn extend_sparse_row(flat: &[usize], range: Range<usize>, scratch: &mut Vec<usize>) {
    scratch.extend_from_slice(&flat[range]);
}

fn finish_sparse_union(scratch: &mut Vec<usize>) {
    scratch.sort_unstable();
    scratch.dedup();
}

/// Exact structural lower-triangle Hessian pattern of the expression rooted
/// at `root`. Normalized `(row, col)` index pairs with `row >= col`, sorted
/// and deduplicated.
///
/// "Exact structural" means a superset of the numerically nonzero second
/// partials that ignores value cancellation. Parameters stay symbolic, so the
/// pattern is independent of current parameter values.
/// `Abs` and `Min`/`Max` conservatively union branch patterns without adding
/// synthetic cross terms between branches.
pub fn hessian_pattern(arena: &ExprArena, root: ExprId) -> Vec<(u32, u32)> {
    structural_sparsity(arena, root).hess_pairs
}

fn collect_node_vars(node: &ExprNode, vars: &mut Vec<u32>) {
    match node {
        ExprNode::Var(v) => vars.push(v.0),
        ExprNode::Linear { coeffs, .. } => {
            vars.extend(coeffs.iter().map(|(v, _)| v.0));
        }
        _ => {}
    }
}

fn push_syntax_children(node: &ExprNode, walk: &mut Vec<WalkAction>) {
    match node {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => {
            walk.extend(children.iter().copied().map(WalkAction::Syntax));
        }
        ExprNode::Unary(_, inner) => walk.push(WalkAction::Syntax(*inner)),
        ExprNode::Pow(base, exp) | ExprNode::Div(base, exp) | ExprNode::Atan2(base, exp) => {
            walk.push(WalkAction::Syntax(*base));
            walk.push(WalkAction::Syntax(*exp));
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {}
    }
}

fn push_active_children(arena: &ExprArena, id: ExprId, walk: &mut Vec<WalkAction>) {
    match arena.get(id) {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => {
            walk.extend(children.iter().rev().copied().map(WalkAction::ActiveEnter));
        }
        ExprNode::Unary(_, inner) => walk.push(WalkAction::ActiveEnter(*inner)),
        ExprNode::Div(num, den) => {
            walk.push(WalkAction::ActiveEnter(*den));
            walk.push(WalkAction::ActiveEnter(*num));
        }
        ExprNode::Pow(base, exp) => match arena.get(*exp) {
            ExprNode::Const(e) if *e == 0.0 => {
                walk.push(WalkAction::Syntax(*exp));
                walk.push(WalkAction::Syntax(*base));
            }
            ExprNode::Const(_) => walk.push(WalkAction::ActiveEnter(*base)),
            _ => {
                walk.push(WalkAction::ActiveEnter(*exp));
                walk.push(WalkAction::ActiveEnter(*base));
            }
        },
        ExprNode::Atan2(y, x) => {
            walk.push(WalkAction::ActiveEnter(*x));
            walk.push(WalkAction::ActiveEnter(*y));
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {}
    }
}

fn build_order_and_variables(arena: &ExprArena, root: ExprId, workspace: &mut SparsityWorkspace) {
    workspace.walk.push(WalkAction::ActiveEnter(root));
    while let Some(action) = workspace.walk.pop() {
        match action {
            WalkAction::Syntax(id) => {
                if workspace.mark_syntax(arena, id) {
                    push_syntax_children(arena.get(id), &mut workspace.walk);
                }
            }
            WalkAction::ActiveEnter(id) => {
                workspace.mark_syntax(arena, id);
                let meta = &mut workspace.meta[id.index()];
                if meta.active_epoch == workspace.epoch {
                    continue;
                }
                meta.active_epoch = workspace.epoch;
                meta.active_state = 1;
                collect_node_vars(arena.get(id), &mut workspace.active_vars);
                workspace.walk.push(WalkAction::ActiveExit(id));
                push_active_children(arena, id, &mut workspace.walk);
            }
            WalkAction::ActiveExit(id) => {
                let meta = &mut workspace.meta[id.index()];
                if meta.active_state != 2 {
                    meta.active_state = 2;
                    workspace.order.push(id);
                }
            }
        }
    }
    workspace.syntax_vars.sort_unstable();
    workspace.syntax_vars.dedup();
    workspace.active_vars.sort_unstable();
    workspace.active_vars.dedup();
}

fn prepare_storage(workspace: &mut SparsityWorkspace) -> StorageMode {
    if workspace.active_vars.len() > DENSE_SPARSITY_MAX_VARS {
        append_sparse_row(
            &mut workspace.sparse_supports,
            &mut workspace.sparse_rows,
            &mut workspace.clique_covered,
            &[],
        );
        return StorageMode::Sparse;
    }

    let words = words_for(workspace.active_vars.len());
    let pair_count = workspace
        .active_vars
        .len()
        .checked_add(1)
        .and_then(|next| workspace.active_vars.len().checked_mul(next))
        .expect("Hessian sparsity bitset size overflow")
        / 2;
    workspace.pairs.resize(words_for(pair_count), 0);
    workspace.pairs.fill(0);
    workspace.alloc_row(words); // shared empty support row
    StorageMode::Dense { words }
}

#[expect(clippy::too_many_lines)]
fn build_dense_support_rows(arena: &ExprArena, workspace: &mut SparsityWorkspace, words: usize) {
    for order_index in 0..workspace.order.len() {
        let id = workspace.order[order_index];
        let row = match arena.get(id) {
            ExprNode::Const(_) | ExprNode::Param(_) => 0,
            ExprNode::Var(v) => {
                let row = workspace.alloc_row(words);
                let bit = workspace.active_vars.binary_search(&v.0).expect("collected variable");
                set_bit(&mut workspace.supports[row_start(row, words)..][..words], bit);
                row
            }
            ExprNode::Linear { coeffs, .. } => {
                let row = workspace.alloc_row(words);
                let dst = &mut workspace.supports[row_start(row, words)..][..words];
                for (v, _) in coeffs {
                    let bit =
                        workspace.active_vars.binary_search(&v.0).expect("collected variable");
                    set_bit(dst, bit);
                }
                row
            }
            ExprNode::Unary(UnaryOp::Neg | UnaryOp::Abs, inner) => {
                workspace.meta[inner.index()].row
            }
            ExprNode::Add(children) if children.len() == 1 => {
                workspace.meta[children[0].index()].row
            }
            ExprNode::Add(children) | ExprNode::Min(children) | ExprNode::Max(children) => {
                let row = workspace.alloc_row(words);
                for child in children {
                    union_rows(
                        &mut workspace.supports,
                        row,
                        workspace.meta[child.index()].row,
                        words,
                    );
                }
                row
            }
            ExprNode::Mul(children) => {
                let row = workspace.alloc_row(words);
                for child in children {
                    let child = workspace.meta[child.index()].row;
                    add_cross(&workspace.supports, row, child, words, &mut workspace.pairs);
                    union_rows(&mut workspace.supports, row, child, words);
                }
                row
            }
            ExprNode::Div(num, den) => {
                let row = workspace.alloc_row(words);
                let num = workspace.meta[num.index()].row;
                let den = workspace.meta[den.index()].row;
                union_rows(&mut workspace.supports, row, num, words);
                workspace.add_clique_once(den, words);
                add_cross(&workspace.supports, row, den, words, &mut workspace.pairs);
                union_rows(&mut workspace.supports, row, den, words);
                row
            }
            ExprNode::Unary(_, inner) => {
                let row = workspace.meta[inner.index()].row;
                workspace.add_clique_once(row, words);
                row
            }
            ExprNode::Atan2(y, x) => {
                let row = workspace.alloc_row(words);
                union_rows(&mut workspace.supports, row, workspace.meta[y.index()].row, words);
                union_rows(&mut workspace.supports, row, workspace.meta[x.index()].row, words);
                workspace.add_clique_once(row, words);
                row
            }
            ExprNode::Pow(base, exp) => {
                if let Some(exponent) = constant_exponent(arena, *exp) {
                    match exponent {
                        ConstantExponent::Zero => 0,
                        ConstantExponent::One => workspace.meta[base.index()].row,
                        ConstantExponent::Other => {
                            let row = workspace.meta[base.index()].row;
                            workspace.add_clique_once(row, words);
                            row
                        }
                    }
                } else {
                    let row = workspace.alloc_row(words);
                    union_rows(
                        &mut workspace.supports,
                        row,
                        workspace.meta[base.index()].row,
                        words,
                    );
                    union_rows(
                        &mut workspace.supports,
                        row,
                        workspace.meta[exp.index()].row,
                        words,
                    );
                    workspace.add_clique_once(row, words);
                    row
                }
            }
        };
        workspace.meta[id.index()].row = row;
    }
}

fn store_sparse_scratch(workspace: &mut SparsityWorkspace) -> usize {
    append_sparse_row(
        &mut workspace.sparse_supports,
        &mut workspace.sparse_rows,
        &mut workspace.clique_covered,
        &workspace.sparse_scratch,
    )
}

fn sparse_union_row(
    workspace: &mut SparsityWorkspace,
    children: impl IntoIterator<Item = ExprId>,
) -> usize {
    workspace.sparse_scratch.clear();
    for child in children {
        let range = workspace.sparse_rows[workspace.meta[child.index()].row].clone();
        extend_sparse_row(&workspace.sparse_supports, range, &mut workspace.sparse_scratch);
    }
    finish_sparse_union(&mut workspace.sparse_scratch);
    store_sparse_scratch(workspace)
}

fn sparse_product_row(
    workspace: &mut SparsityWorkspace,
    children: impl IntoIterator<Item = ExprId>,
) -> usize {
    workspace.sparse_scratch.clear();
    for child in children {
        let range = workspace.sparse_rows[workspace.meta[child.index()].row].clone();
        let child = &workspace.sparse_supports[range.clone()];
        add_sparse_cross(&workspace.sparse_scratch, child, &mut workspace.sparse_pairs);
        extend_sparse_row(&workspace.sparse_supports, range, &mut workspace.sparse_scratch);
        finish_sparse_union(&mut workspace.sparse_scratch);
    }
    store_sparse_scratch(workspace)
}

fn sparse_division_row(workspace: &mut SparsityWorkspace, num: ExprId, den: ExprId) -> usize {
    workspace.sparse_scratch.clear();
    let num = workspace.sparse_rows[workspace.meta[num.index()].row].clone();
    let den_row = workspace.meta[den.index()].row;
    let den = workspace.sparse_rows[den_row].clone();
    extend_sparse_row(&workspace.sparse_supports, num, &mut workspace.sparse_scratch);
    workspace.add_sparse_clique_once(den_row);
    add_sparse_cross(
        &workspace.sparse_scratch,
        &workspace.sparse_supports[den.clone()],
        &mut workspace.sparse_pairs,
    );
    extend_sparse_row(&workspace.sparse_supports, den, &mut workspace.sparse_scratch);
    finish_sparse_union(&mut workspace.sparse_scratch);
    store_sparse_scratch(workspace)
}

fn sparse_power_row(
    arena: &ExprArena,
    workspace: &mut SparsityWorkspace,
    base: ExprId,
    exp: ExprId,
) -> usize {
    match constant_exponent(arena, exp) {
        Some(ConstantExponent::Zero) => 0,
        Some(ConstantExponent::One) => workspace.meta[base.index()].row,
        Some(ConstantExponent::Other) => {
            let row = workspace.meta[base.index()].row;
            workspace.add_sparse_clique_once(row);
            row
        }
        None => {
            let row = sparse_union_row(workspace, [base, exp]);
            workspace.add_sparse_clique_once(row);
            row
        }
    }
}

fn build_sparse_support_rows(arena: &ExprArena, workspace: &mut SparsityWorkspace) {
    for order_index in 0..workspace.order.len() {
        let id = workspace.order[order_index];
        let row = match arena.get(id) {
            ExprNode::Const(_) | ExprNode::Param(_) => 0,
            ExprNode::Var(v) => {
                let bit = workspace.active_vars.binary_search(&v.0).expect("collected variable");
                append_sparse_row(
                    &mut workspace.sparse_supports,
                    &mut workspace.sparse_rows,
                    &mut workspace.clique_covered,
                    &[bit],
                )
            }
            ExprNode::Linear { coeffs, .. } => {
                workspace.sparse_scratch.clear();
                workspace.sparse_scratch.extend(coeffs.iter().map(|(v, _)| {
                    workspace.active_vars.binary_search(&v.0).expect("collected variable")
                }));
                finish_sparse_union(&mut workspace.sparse_scratch);
                store_sparse_scratch(workspace)
            }
            ExprNode::Unary(UnaryOp::Neg | UnaryOp::Abs, inner) => {
                workspace.meta[inner.index()].row
            }
            ExprNode::Add(children) if children.len() == 1 => {
                workspace.meta[children[0].index()].row
            }
            ExprNode::Add(children) => sparse_union_row(workspace, children.iter().copied()),
            ExprNode::Min(children) | ExprNode::Max(children) => {
                sparse_union_row(workspace, children.iter().copied())
            }
            ExprNode::Mul(children) => sparse_product_row(workspace, children.iter().copied()),
            ExprNode::Div(num, den) => sparse_division_row(workspace, *num, *den),
            ExprNode::Unary(_, inner) => {
                let row = workspace.meta[inner.index()].row;
                workspace.add_sparse_clique_once(row);
                row
            }
            ExprNode::Atan2(y, x) => {
                let row = sparse_union_row(workspace, [*y, *x]);
                workspace.add_sparse_clique_once(row);
                row
            }
            ExprNode::Pow(base, exp) => sparse_power_row(arena, workspace, *base, *exp),
        };
        workspace.meta[id.index()].row = row;
    }
}

fn materialize_pattern(workspace: &SparsityWorkspace, storage: StorageMode) -> Vec<(u32, u32)> {
    match storage {
        StorageMode::Dense { .. } => {
            let mut pattern = Vec::new();
            for row in 0..workspace.active_vars.len() {
                for col in 0..=row {
                    let bit = row * (row + 1) / 2 + col;
                    if has_bit(&workspace.pairs, bit) {
                        pattern.push((workspace.active_vars[row], workspace.active_vars[col]));
                    }
                }
            }
            pattern
        }
        StorageMode::Sparse => {
            let mut pattern: Vec<(u32, u32)> = workspace
                .sparse_pairs
                .iter()
                .map(|&(row, col)| (workspace.active_vars[row], workspace.active_vars[col]))
                .collect();
            pattern.sort_unstable();
            pattern
        }
    }
}

pub(crate) fn structural_sparsity_with_workspace(
    arena: &ExprArena,
    root: ExprId,
    workspace: &mut SparsityWorkspace,
) -> StructuralSparsity {
    workspace.begin(arena.len());
    build_order_and_variables(arena, root, workspace);
    let storage = prepare_storage(workspace);
    match storage {
        StorageMode::Dense { words } => build_dense_support_rows(arena, workspace, words),
        StorageMode::Sparse => build_sparse_support_rows(arena, workspace),
    }
    StructuralSparsity {
        support: workspace.syntax_vars.clone(),
        hess_pairs: materialize_pattern(workspace, storage),
    }
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
    let slots: Vec<_> = slots.into_iter().collect();
    hessian_patterns_impl(&slots, false).0
}

/// Build the merged Hessian pattern and the nonlinear-only subset in one pass
/// over the slots.
#[cfg(feature = "enzyme")]
pub(crate) fn hessian_lagrangian_patterns<'a, I>(slots: I) -> (HessianPattern, HessianPattern)
where
    I: IntoIterator<Item = &'a FunctionSlot>,
{
    let slots: Vec<_> = slots.into_iter().collect();
    let (full, nonlinear) = hessian_patterns_impl(&slots, true);
    (full, nonlinear.expect("nonlinear pattern requested"))
}

fn hessian_patterns_impl(slots: &[&FunctionSlot], include_nonlinear: bool) -> HessianPatterns {
    let mut max_variable = None;
    for slot in slots {
        match &slot.kind {
            SlotKind::Linear(_) => {}
            SlotKind::Quadratic(q) => {
                for &(r, c, _) in &q.hessian {
                    max_variable = max_variable.max(Some(r.index().max(c.index())));
                }
            }
            SlotKind::Nonlinear(_) => {
                for &(r, c) in &slot.hess_pairs {
                    max_variable = max_variable.max(Some((r as usize).max(c as usize)));
                }
            }
        }
    }

    if max_variable.is_none_or(|index| index < DENSE_SPARSITY_MAX_VARS) {
        let variables = max_variable.map_or(0, |index| index + 1);
        let pair_count = variables * (variables + 1) / 2;
        let mut pairs = vec![0; words_for(pair_count)];
        let mut nonlinear_pairs = include_nonlinear.then(|| vec![0; words_for(pair_count)]);
        for slot in slots {
            match &slot.kind {
                SlotKind::Linear(_) => {}
                SlotKind::Quadratic(q) => {
                    for &(r, c, _) in &q.hessian {
                        set_pair(&mut pairs, r.index(), c.index());
                    }
                }
                SlotKind::Nonlinear(_) => {
                    for &(r, c) in &slot.hess_pairs {
                        set_pair(&mut pairs, r as usize, c as usize);
                        if let Some(ref mut nonlinear_pairs) = nonlinear_pairs {
                            set_pair(nonlinear_pairs, r as usize, c as usize);
                        }
                    }
                }
            }
        }
        let mut entries = Vec::new();
        for row in 0..variables {
            for col in 0..=row {
                if has_bit(&pairs, row * (row + 1) / 2 + col) {
                    entries.push((row, col));
                }
            }
        }
        let nonlinear_entries = nonlinear_pairs.map(|pairs| {
            let mut entries = Vec::new();
            for row in 0..variables {
                for col in 0..=row {
                    if has_bit(&pairs, row * (row + 1) / 2 + col) {
                        entries.push((row, col));
                    }
                }
            }
            entries
        });
        return (entries, nonlinear_entries);
    }

    let mut entries = FxHashSet::default();
    let mut nonlinear_entries = include_nonlinear.then(FxHashSet::default);
    for slot in slots {
        match &slot.kind {
            SlotKind::Linear(_) => {}
            SlotKind::Quadratic(q) => {
                for &(r, c, _) in &q.hessian {
                    entries.insert((r.index(), c.index()));
                }
            }
            SlotKind::Nonlinear(_) => {
                let pairs = slot.hess_pairs.iter().map(|&(r, c)| (r as usize, c as usize));
                entries.extend(pairs.clone());
                if let Some(ref mut nonlinear_entries) = nonlinear_entries {
                    nonlinear_entries.extend(pairs);
                }
            }
        }
    }
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_unstable();
    let nonlinear_entries = nonlinear_entries.map(|entries| {
        let mut entries: Vec<_> = entries.into_iter().collect();
        entries.sort_unstable();
        entries
    });
    (entries, nonlinear_entries)
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
    let graph = CsrGraph::from_pattern(pattern);
    let color = greedy_star_coloring(&graph);
    let neighbor_colors = NeighborColorCounts::new(&graph, &color);
    let classes = ColorClasses::new(&graph, &color);

    let mut groups = Vec::new();
    let mut group_of_color = vec![None; classes.len()];
    let mut singleton_of = vec![None; graph.len()];
    let mut recover = Vec::with_capacity(pattern.len());

    for (&(r, c), &(dense_r, dense_c)) in pattern.iter().zip(&graph.dense_pattern) {
        let entry = if r == c {
            (color_group(color[dense_r], &classes, &mut groups, &mut group_of_color), r)
        } else if neighbor_colors.count(dense_r, color[dense_c]) == 1 {
            (color_group(color[dense_c], &classes, &mut groups, &mut group_of_color), r)
        } else if neighbor_colors.count(dense_c, color[dense_r]) == 1 {
            (color_group(color[dense_r], &classes, &mut groups, &mut group_of_color), c)
        } else {
            (singleton_group(dense_r, &graph, &mut groups, &mut singleton_of), c)
        };
        recover.push(entry);
    }

    HessianColoring { groups, recover }
}

struct CsrGraph {
    vertices: Vec<usize>,
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    dense_pattern: Vec<(usize, usize)>,
}

impl CsrGraph {
    fn from_pattern(pattern: &[(usize, usize)]) -> Self {
        let mut vertices: Vec<_> = pattern.iter().flat_map(|&(r, c)| [r, c]).collect();
        vertices.sort_unstable();
        vertices.dedup();

        let dense_pattern: Vec<_> = pattern
            .iter()
            .map(|&(r, c)| {
                (
                    vertices.binary_search(&r).expect("pattern vertex was collected"),
                    vertices.binary_search(&c).expect("pattern vertex was collected"),
                )
            })
            .collect();

        let mut degrees = vec![0_usize; vertices.len()];
        for &(r, c) in &dense_pattern {
            if r != c {
                degrees[r] += 1;
                degrees[c] += 1;
            }
        }

        let mut raw_offsets = Vec::with_capacity(vertices.len() + 1);
        raw_offsets.push(0);
        for degree in degrees {
            raw_offsets.push(raw_offsets.last().copied().unwrap_or(0) + degree);
        }
        let mut raw_neighbors = vec![0; raw_offsets.last().copied().unwrap_or(0)];
        let mut cursors = raw_offsets[..vertices.len()].to_vec();
        for &(r, c) in &dense_pattern {
            if r != c {
                raw_neighbors[cursors[r]] = c;
                cursors[r] += 1;
                raw_neighbors[cursors[c]] = r;
                cursors[c] += 1;
            }
        }

        let mut offsets = Vec::with_capacity(vertices.len() + 1);
        let mut neighbors = Vec::with_capacity(raw_neighbors.len());
        offsets.push(0);
        for vertex in 0..vertices.len() {
            let row = &mut raw_neighbors[raw_offsets[vertex]..raw_offsets[vertex + 1]];
            row.sort_unstable();
            let mut previous = None;
            for &neighbor in row.iter() {
                if previous != Some(neighbor) {
                    neighbors.push(neighbor);
                    previous = Some(neighbor);
                }
            }
            offsets.push(neighbors.len());
        }

        Self { vertices, offsets, neighbors, dense_pattern }
    }

    fn len(&self) -> usize {
        self.vertices.len()
    }

    fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.neighbors[self.offsets[vertex]..self.offsets[vertex + 1]]
    }
}

const UNCOLORED: usize = usize::MAX;

/// Greedy star coloring of a CSR graph: a proper coloring in which no path on
/// four vertices is two-colored.
fn greedy_star_coloring(graph: &CsrGraph) -> Vec<usize> {
    let mut order: Vec<_> = (0..graph.len()).collect();
    order.sort_unstable_by_key(|&vertex| {
        (std::cmp::Reverse(graph.neighbors(vertex).len()), graph.vertices[vertex])
    });

    let mut color = vec![UNCOLORED; graph.len()];
    let mut neighbor_color_count = vec![0_usize; graph.len()];
    let mut touched_colors = Vec::new();
    let mut forbidden_epoch = vec![0_u32; graph.len()];
    let mut epoch = 0_u32;

    for vertex in order {
        epoch = epoch.wrapping_add(1);
        if epoch == 0 {
            forbidden_epoch.fill(0);
            epoch = 1;
        }
        touched_colors.clear();

        for &neighbor in graph.neighbors(vertex) {
            let neighbor_color = color[neighbor];
            if neighbor_color == UNCOLORED {
                continue;
            }
            if neighbor_color_count[neighbor_color] == 0 {
                touched_colors.push(neighbor_color);
            }
            neighbor_color_count[neighbor_color] += 1;
            forbidden_epoch[neighbor_color] = epoch;
        }

        for &neighbor in graph.neighbors(vertex) {
            let neighbor_color = color[neighbor];
            if neighbor_color == UNCOLORED {
                continue;
            }
            forbid_endpoint_p4(
                graph,
                &color,
                vertex,
                neighbor,
                neighbor_color,
                &mut forbidden_epoch,
                epoch,
            );
            forbid_internal_p4(
                graph,
                &color,
                vertex,
                neighbor,
                neighbor_color_count[neighbor_color] >= 2,
                &mut forbidden_epoch,
                epoch,
            );
        }

        let selected =
            (0..graph.len()).find(|&candidate| forbidden_epoch[candidate] != epoch).unwrap_or(0);
        color[vertex] = selected;
        for &touched in &touched_colors {
            neighbor_color_count[touched] = 0;
        }
    }
    color
}

/// Endpoint-P4 rule while choosing `vertex`'s color.
fn forbid_endpoint_p4(
    graph: &CsrGraph,
    color: &[usize],
    vertex: usize,
    neighbor: usize,
    neighbor_color: usize,
    forbidden_epoch: &mut [u32],
    epoch: u32,
) {
    for &next in graph.neighbors(neighbor) {
        if next == vertex {
            continue;
        }
        let next_color = color[next];
        if next_color == UNCOLORED {
            continue;
        }
        if graph
            .neighbors(next)
            .iter()
            .any(|&last| last != neighbor && color[last] == neighbor_color)
        {
            forbidden_epoch[next_color] = epoch;
        }
    }
}

/// Internal-P4 rule while choosing `vertex`'s color.
fn forbid_internal_p4(
    graph: &CsrGraph,
    color: &[usize],
    vertex: usize,
    neighbor: usize,
    vertex_has_another_neighbor_of_color: bool,
    forbidden_epoch: &mut [u32],
    epoch: u32,
) {
    if !vertex_has_another_neighbor_of_color {
        return;
    }
    for &next in graph.neighbors(neighbor) {
        if next != vertex && color[next] != UNCOLORED {
            forbidden_epoch[color[next]] = epoch;
        }
    }
}

struct NeighborColorCounts {
    offsets: Vec<usize>,
    entries: Vec<(usize, usize)>,
}

impl NeighborColorCounts {
    fn new(graph: &CsrGraph, color: &[usize]) -> Self {
        let mut offsets = Vec::with_capacity(graph.len() + 1);
        let mut entries = Vec::new();
        let mut scratch = Vec::new();
        offsets.push(0);

        for vertex in 0..graph.len() {
            scratch.clear();
            scratch.extend(graph.neighbors(vertex).iter().map(|&neighbor| color[neighbor]));
            scratch.sort_unstable();

            let mut index = 0;
            while index < scratch.len() {
                let current = scratch[index];
                let start = index;
                while index < scratch.len() && scratch[index] == current {
                    index += 1;
                }
                entries.push((current, index - start));
            }
            offsets.push(entries.len());
        }
        Self { offsets, entries }
    }

    fn count(&self, vertex: usize, color: usize) -> usize {
        let entries = &self.entries[self.offsets[vertex]..self.offsets[vertex + 1]];
        entries
            .binary_search_by_key(&color, |&(entry_color, _)| entry_color)
            .map_or(0, |index| entries[index].1)
    }
}

struct ColorClasses {
    offsets: Vec<usize>,
    members: Vec<usize>,
}

impl ColorClasses {
    fn new(graph: &CsrGraph, color: &[usize]) -> Self {
        let count = color.iter().copied().max().map_or(0, |maximum| maximum + 1);
        let mut sizes = vec![0_usize; count];
        for &entry in color {
            sizes[entry] += 1;
        }

        let mut offsets = Vec::with_capacity(count + 1);
        offsets.push(0);
        for size in sizes {
            offsets.push(offsets.last().copied().unwrap_or(0) + size);
        }

        let mut cursors = offsets[..count].to_vec();
        let mut members = vec![0; graph.len()];
        for (vertex, &entry) in color.iter().enumerate() {
            members[cursors[entry]] = graph.vertices[vertex];
            cursors[entry] += 1;
        }
        Self { offsets, members }
    }

    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn members(&self, color: usize) -> &[usize] {
        &self.members[self.offsets[color]..self.offsets[color + 1]]
    }
}

/// Index of the seed group for `color`, creating it on first use.
fn color_group(
    color: usize,
    classes: &ColorClasses,
    groups: &mut Vec<Vec<usize>>,
    group_of_color: &mut [Option<usize>],
) -> usize {
    if let Some(group) = group_of_color[color] {
        return group;
    }
    let group = groups.len();
    groups.push(classes.members(color).to_vec());
    group_of_color[color] = Some(group);
    group
}

/// Index of a lone-column seed group for `vertex`, created once per vertex.
fn singleton_group(
    vertex: usize,
    graph: &CsrGraph,
    groups: &mut Vec<Vec<usize>>,
    singleton_of: &mut [Option<usize>],
) -> usize {
    if let Some(group) = singleton_of[vertex] {
        return group;
    }
    let group = groups.len();
    groups.push(vec![graph.vertices[vertex]]);
    singleton_of[vertex] = Some(group);
    group
}
