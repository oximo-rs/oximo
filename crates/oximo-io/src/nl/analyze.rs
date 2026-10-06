//! Per-row classification.
//!
//! For each constraint and the objective we run `split_linear` on the body.
//! The result is a `Row { linear, residual }` where:
//!
//! - `linear` carries the linear coefficients (goes into `J`/`G` segments
//!   and into the RHS constant shift).
//! - `residual` is `Some(ExprId)` when nonlinear summands remain (emitted in
//!   the corresponding `C`/`O` segment), `None` when the whole body is
//!   purely linear.
//!
//! Each row also caches the union of variables appearing anywhere in the body
//! (linear OR nonlinear), used to size the Jacobian/gradient sparsity. The
//! variable sets `nl_vars_c`, `nl_vars_o` only count vars appearing inside the
//! nonlinear residual.

use oximo_core::{AlgebraicConstraint, Domain, Objective, Variable};
use oximo_expr::{
    ExprArena, ExprId, ExprNode, LinearTerms, SignedExpr, VarId, Visitor, split_linear, walk_shared,
};
use rustc_hash::FxHashSet;

use crate::error::IoError;

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) linear: LinearTerms<'static>,
    /// Nonlinear summands of the body, empty when the row is purely linear.
    pub(crate) residual: Vec<SignedExpr>,
}

#[derive(Default)]
struct ResidualCache {
    entries: Vec<(ExprId, FxHashSet<VarId>)>,
}

impl ResidualCache {
    fn variables(
        &mut self,
        arena: &ExprArena,
        id: ExprId,
        nonfinite_strings: bool,
    ) -> Result<&FxHashSet<VarId>, IoError> {
        if let Some(index) = self.entries.iter().position(|(root, _)| *root == id) {
            return Ok(&self.entries[index].1);
        }
        let mut vars = FxHashSet::default();
        analyze_residual(arena, id, nonfinite_strings, &mut vars)?;
        if self.entries.len() == 8 {
            self.entries.remove(0);
        }
        self.entries.push((id, vars));
        Ok(&self.entries.last().expect("inserted residual").1)
    }
}

impl Row {
    pub(crate) fn is_nonlinear(&self) -> bool {
        !self.residual.is_empty()
    }
}

#[derive(Debug)]
pub(crate) struct Analysis {
    pub(crate) cons: Vec<Row>,
    /// `None` for a feasibility model, which NL writes with zero objectives.
    pub(crate) obj: Option<Row>,
    /// Concatenated sorted variable supports for all constraint rows.
    cons_vars: Vec<VarId>,
    /// Row starts into `cons_vars`.
    cons_var_offsets: Vec<usize>,
    pub(crate) obj_vars: Vec<VarId>,
    pub(crate) nl_vars_c: FxHashSet<VarId>,
    pub(crate) nl_vars_o: FxHashSet<VarId>,
}

impl Analysis {
    pub(crate) fn constraint_vars(&self, row: usize) -> &[VarId] {
        &self.cons_vars[self.cons_var_offsets[row]..self.cons_var_offsets[row + 1]]
    }

    pub(crate) fn constraint_vars_iter(&self) -> impl Iterator<Item = &[VarId]> {
        self.cons_var_offsets.windows(2).map(|range| &self.cons_vars[range[0]..range[1]])
    }

    pub(crate) fn jacobian_nnz(&self) -> usize {
        self.cons_vars.len()
    }

    pub(crate) fn build(
        arena: &ExprArena,
        vars: &[Variable],
        constraints: &[AlgebraicConstraint],
        objective: Option<&Objective>,
        nonfinite_strings: bool,
    ) -> Result<Self, IoError> {
        for v in vars {
            match v.domain {
                Domain::Real | Domain::Integer | Domain::Binary => {}
                Domain::SemiContinuous { .. } => {
                    return Err(IoError::UnsupportedDomain("SemiContinuous"));
                }
                Domain::SemiInteger { .. } => {
                    return Err(IoError::UnsupportedDomain("SemiInteger"));
                }
            }
        }

        let mut nl_vars_c: FxHashSet<VarId> = FxHashSet::default();
        // Scratch lives only for this immutable arena snapshot and options.
        let mut residual_cache = ResidualCache::default();
        let mut nl_vars_o: FxHashSet<VarId> = FxHashSet::default();
        let mut cons: Vec<Row> = Vec::with_capacity(constraints.len());
        let mut cons_vars = Vec::new();
        let mut cons_var_offsets = Vec::with_capacity(constraints.len() + 1);
        cons_var_offsets.push(0);

        for c in constraints {
            let (linear, residual) = split_linear(arena, c.lhs);
            let mut all = FxHashSet::default();
            for (v, _) in linear.coeffs.iter() {
                all.insert(*v);
            }
            if !residual.is_empty() {
                for r in &residual {
                    for v in residual_cache.variables(arena, r.id, nonfinite_strings)? {
                        nl_vars_c.insert(*v);
                        all.insert(*v);
                    }
                }
            }
            cons.push(Row { linear: linear.into_owned(), residual });
            let start = cons_vars.len();
            cons_vars.extend(all);
            cons_vars[start..].sort_by_key(|v| v.0);
            cons_var_offsets.push(cons_vars.len());
        }

        let mut obj_all = FxHashSet::default();
        let obj = match objective {
            Some(objective) => {
                let (obj_linear, obj_residual) = split_linear(arena, objective.expr);
                for (v, _) in obj_linear.coeffs.iter() {
                    obj_all.insert(*v);
                }
                if !obj_residual.is_empty() {
                    for r in &obj_residual {
                        for v in residual_cache.variables(arena, r.id, nonfinite_strings)? {
                            nl_vars_o.insert(*v);
                            obj_all.insert(*v);
                        }
                    }
                }
                Some(Row { linear: obj_linear.into_owned(), residual: obj_residual })
            }
            None => None,
        };

        Ok(Self {
            cons,
            obj,
            cons_vars,
            cons_var_offsets,
            obj_vars: sorted(obj_all),
            nl_vars_c,
            nl_vars_o,
        })
    }
}

fn sorted(set: FxHashSet<VarId>) -> Vec<VarId> {
    let mut v: Vec<VarId> = set.into_iter().collect();
    v.sort_by_key(|v| v.0);
    v
}

/// Walk a nonlinear residual, rejecting nodes the writer cannot emit. Non-finite
/// constants are an error only when `nonfinite_strings` is off. When on, the
/// writer emits them as `Infinity`/`NaN`, so they are allowed through this function
/// to keep `WriteOptions::nonfinite_strings` effective for expression constants.
fn analyze_residual(
    arena: &ExprArena,
    id: ExprId,
    nonfinite_strings: bool,
    vars: &mut FxHashSet<VarId>,
) -> Result<(), IoError> {
    struct ResidualVisitor<'a> {
        vars: &'a mut FxHashSet<VarId>,
        nonfinite_strings: bool,
        error: Option<IoError>,
    }
    impl Visitor for ResidualVisitor<'_> {
        fn visit(&mut self, arena: &ExprArena, _id: ExprId, node: &ExprNode) {
            if self.error.is_some() {
                return;
            }
            let number = match node {
                ExprNode::Const(value) => Some((*value, "an expression constant")),
                ExprNode::Param(p) => Some((arena.param_value(*p), "a parameter")),
                ExprNode::Var(v) => {
                    self.vars.insert(*v);
                    None
                }
                ExprNode::Linear { coeffs, constant } => {
                    self.vars.extend(coeffs.iter().map(|&(v, _)| v));
                    Some((*constant, "a linear expression constant"))
                }
                _ => None,
            };
            if let Some((value, location)) = number
                && !self.nonfinite_strings
                && !value.is_finite()
            {
                self.error = Some(IoError::InvalidNumber { value, location: location.into() });
            }
        }
    }
    let mut visitor = ResidualVisitor { vars, nonfinite_strings, error: None };
    walk_shared(arena, id, &mut visitor);
    visitor.error.map_or(Ok(()), Err)
}

#[cfg(feature = "benchmark-support")]
#[doc(hidden)]
#[expect(clippy::cast_precision_loss)]
#[allow(clippy::wildcard_imports)]
pub mod benchmark_support {
    use oximo_core::Model;
    use oximo_core::constraint::Relate;
    use rayon::prelude::*;

    use super::*;

    /// Crossover candidate used only to size the preprocessing benchmark cases.
    pub const THRESHOLD: usize = 1_024;

    pub fn model(rows: usize, degree: usize) -> Model {
        let model = Model::new("analysis_bench");
        let x = model.__var("x").lb(-5.0).ub(5.0).build();
        let y = model.__var("y").lb(-5.0).ub(5.0).build();
        let z = model.__var("z").lb(-5.0).ub(5.0).build();
        model.__minimize(x + y + z);
        for i in 0..rows {
            let lhs = match degree {
                1 => (x + 2.0 * y - z).erase(),
                2 => x.powi(2) + y * z,
                _ => (x * y * z + x.sin()).erase(),
            };
            model.__add_constraint_auto(lhs.le(i as f64 + 10.0));
        }
        model
    }

    pub fn analyze(model: &Model, parallel: bool) -> Result<usize, IoError> {
        let arena = model.arena();
        let vars = model.variables();
        let model_constraints = model.constraints();
        let constraints = model_constraints.algebraic();
        let objective = model.try_objective().map_err(|_| IoError::NoObjective)?;
        let arena_ref = &*arena;
        let analysis = build(arena_ref, &vars, constraints, &objective, parallel)?;
        Ok(analysis.cons.len()
            + analysis.jacobian_nnz()
            + analysis.obj_vars.len()
            + analysis.nl_vars_c.len()
            + analysis.nl_vars_o.len())
    }

    fn row(
        arena: &ExprArena,
        c: &AlgebraicConstraint,
    ) -> Result<(Row, Vec<VarId>, FxHashSet<VarId>), IoError> {
        let (linear, residual) = split_linear(arena, c.lhs);
        let mut all: FxHashSet<VarId> = linear.coeffs.iter().map(|(v, _)| *v).collect();
        let mut nonlinear = FxHashSet::default();
        for r in &residual {
            analyze_residual(arena, r.id, false, &mut nonlinear)?;
        }
        all.extend(nonlinear.iter().copied());
        Ok((Row { linear: linear.into_owned(), residual }, sorted(all), nonlinear))
    }

    fn build(
        arena: &ExprArena,
        vars: &[Variable],
        constraints: &[AlgebraicConstraint],
        objective: &Objective,
        parallel: bool,
    ) -> Result<Analysis, IoError> {
        for v in vars {
            if matches!(v.domain, Domain::SemiContinuous { .. }) {
                return Err(IoError::UnsupportedDomain("SemiContinuous"));
            }
            if matches!(v.domain, Domain::SemiInteger { .. }) {
                return Err(IoError::UnsupportedDomain("SemiInteger"));
            }
        }
        let rows = if parallel {
            constraints.par_iter().map(|c| row(arena, c)).collect::<Result<Vec<_>, _>>()?
        } else {
            constraints.iter().map(|c| row(arena, c)).collect::<Result<Vec<_>, _>>()?
        };
        let mut cons = Vec::with_capacity(rows.len());
        let mut cons_vars = Vec::new();
        let mut cons_var_offsets = Vec::with_capacity(rows.len() + 1);
        let mut nl_vars_c = FxHashSet::default();
        cons_var_offsets.push(0);
        for (row, support, used) in rows {
            cons.push(row);
            cons_vars.extend(support);
            cons_var_offsets.push(cons_vars.len());
            nl_vars_c.extend(used);
        }
        let (obj_linear, obj_residual) = split_linear(arena, objective.expr);
        let mut obj_all: FxHashSet<VarId> = obj_linear.coeffs.iter().map(|(v, _)| *v).collect();
        let mut nl_vars_o = FxHashSet::default();
        for residual in &obj_residual {
            analyze_residual(arena, residual.id, false, &mut nl_vars_o)?;
        }
        obj_all.extend(nl_vars_o.iter().copied());
        let obj = Some(Row { linear: obj_linear.into_owned(), residual: obj_residual });
        let obj_vars = sorted(obj_all);
        Ok(Analysis { cons, obj, cons_vars, cons_var_offsets, obj_vars, nl_vars_c, nl_vars_o })
    }
}
