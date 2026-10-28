use std::sync::Arc;

use oximo_expr::{
    ExprArena, ExprId, ExprNode, ModelId, ModelMismatchError, ParamId, ParameterSnapshot, VarId,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::Model;
use crate::{
    ConstraintId, IndicatorConstraintId, PsdConstraintId, SemanticConstraintId, SocConstraintId,
};

/// A native model entity that can depend on a parameter.
///
/// These are model-level targets, not backend coefficient slots. Backend
/// lowering decides the concrete matrix, right-hand-side, and Hessian slots.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParameterTarget {
    /// The model objective, including its semantic declaration.
    Objective,
    /// A variable's lower or upper bound.
    VariableBound(VarId),
    /// A stored algebraic row.
    Algebraic(ConstraintId),
    /// A stored second-order-cone constraint.
    SecondOrderCone(SocConstraintId),
    /// A stored positive-semidefinite constraint.
    PositiveSemidefinite(PsdConstraintId),
    /// A stored indicator constraint.
    Indicator(IndicatorConstraintId),
}

/// Immutable parameter values captured with their model identity.
///
/// Capture through [`Model::parameter_snapshot`]. Unlike an arena's raw
/// [`ParameterSnapshot`], this can be checked against a dependency index.
#[derive(Clone, Debug)]
pub struct ModelParameterSnapshot {
    model_id: ModelId,
    parameters: ParameterSnapshot,
}

impl ModelParameterSnapshot {
    /// Model whose parameter values were captured.
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Captured values for raw expression evaluation and comparison.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSnapshot {
        &self.parameters
    }
}

/// An immutable snapshot of the parameters used by each model entity.
///
/// The index is built from the current model structure. It is not live: rebuild
/// it after changing roots, bounds, or declarations. Parameters that affect no
/// model entity are omitted.
#[derive(Clone, Debug)]
pub struct ParameterDependencies {
    model_id: ModelId,
    parameters: Vec<ParamId>,
    targets: FxHashMap<ParamId, Vec<ParameterTarget>>,
    target_parameters: FxHashMap<ParameterTarget, Vec<ParamId>>,
    semantic_targets: FxHashMap<ParamId, Vec<SemanticConstraintId>>,
    semantic_parameters: FxHashMap<SemanticConstraintId, Vec<ParamId>>,
}

impl ParameterDependencies {
    /// Model whose structure produced this snapshot.
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Number of parameters with at least one native or semantic dependency.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Whether no parameter affects any model entity.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Parameters with at least one native or semantic dependency, in ID order.
    pub fn parameters(&self) -> impl ExactSizeIterator<Item = ParamId> + '_ {
        self.parameters.iter().copied()
    }

    /// Each indexed parameter and its native targets, in parameter-ID order.
    /// A semantic-only parameter has an empty native-target slice.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (ParamId, &[ParameterTarget])> + '_ {
        self.parameters().map(|parameter| (parameter, self.targets(parameter)))
    }

    /// Whether `target` depends on at least one parameter.
    #[must_use]
    pub fn contains(&self, target: ParameterTarget) -> bool {
        self.target_parameters.contains_key(&target)
    }

    /// Native entities affected by `parameter`.
    ///
    /// Returns an empty slice when the parameter is unused or unknown.
    #[must_use]
    pub fn targets(&self, parameter: ParamId) -> &[ParameterTarget] {
        self.targets.get(&parameter).map_or(&[], Vec::as_slice)
    }

    /// Parameters that affect `target`, in model scan order.
    #[must_use]
    pub fn parameters_for(&self, target: ParameterTarget) -> &[ParamId] {
        self.target_parameters.get(&target).map_or(&[], Vec::as_slice)
    }

    /// Semantic declarations affected by `parameter`.
    ///
    /// This includes direct declarations and their original function/set pairs.
    /// Use [`Self::targets`] for stored native rows.
    #[must_use]
    pub fn semantic_constraints(&self, parameter: ParamId) -> &[SemanticConstraintId] {
        self.semantic_targets.get(&parameter).map_or(&[], Vec::as_slice)
    }

    /// Parameters that affect a semantic declaration, in model scan order.
    #[must_use]
    pub fn parameters_for_semantic(&self, constraint: SemanticConstraintId) -> &[ParamId] {
        self.semantic_parameters.get(&constraint).map_or(&[], Vec::as_slice)
    }

    /// Native entities affected by parameters whose exact values changed.
    ///
    /// This reports data dependencies, not dual validity. A dual for an
    /// unaffected row can still become invalid when another row changes.
    /// Results follow changed parameter-ID order, then per-parameter model scan
    /// order, with duplicates removed at their first occurrence. They are not
    /// globally sorted by target ID.
    ///
    /// # Errors
    /// Returns an ownership error if either snapshot belongs to another model.
    pub fn changed_targets(
        &self,
        previous: &ModelParameterSnapshot,
        current: &ModelParameterSnapshot,
    ) -> Result<Vec<ParameterTarget>, ModelMismatchError> {
        Ok(self.changed_targets_iter(previous, current)?.collect())
    }

    /// Lazily query changed native targets with the ordering of [`Self::changed_targets`].
    ///
    /// # Errors
    /// Returns an ownership error if either snapshot belongs to another model.
    pub fn changed_targets_iter<'a>(
        &'a self,
        previous: &'a ModelParameterSnapshot,
        current: &'a ModelParameterSnapshot,
    ) -> Result<impl Iterator<Item = ParameterTarget> + 'a, ModelMismatchError> {
        self.validate_snapshots(previous, current)?;
        Ok(unique(
            self.changed_parameters(previous, current)
                .flat_map(|parameter| self.targets(parameter).iter().copied()),
        ))
    }

    /// Semantic declarations affected by parameters whose exact values changed.
    ///
    /// Results follow changed parameter-ID order, then model scan order, with
    /// duplicates removed at their first occurrence; they are not globally sorted.
    ///
    /// # Errors
    /// Returns an ownership error if either snapshot belongs to another model.
    pub fn changed_semantic_constraints(
        &self,
        previous: &ModelParameterSnapshot,
        current: &ModelParameterSnapshot,
    ) -> Result<Vec<SemanticConstraintId>, ModelMismatchError> {
        Ok(self.changed_semantic_constraints_iter(previous, current)?.collect())
    }

    /// Lazily query semantic changes with the ordering of [`Self::changed_semantic_constraints`].
    ///
    /// # Errors
    /// Returns an ownership error if either snapshot belongs to another model.
    pub fn changed_semantic_constraints_iter<'a>(
        &'a self,
        previous: &'a ModelParameterSnapshot,
        current: &'a ModelParameterSnapshot,
    ) -> Result<impl Iterator<Item = SemanticConstraintId> + 'a, ModelMismatchError> {
        self.validate_snapshots(previous, current)?;
        Ok(unique(
            self.changed_parameters(previous, current)
                .flat_map(|parameter| self.semantic_constraints(parameter).iter().copied()),
        ))
    }

    fn changed_parameters<'a>(
        &'a self,
        previous: &'a ModelParameterSnapshot,
        current: &'a ModelParameterSnapshot,
    ) -> impl Iterator<Item = ParamId> + 'a {
        self.parameters().filter(|&id| {
            previous.parameters.value(id).map(f64::to_bits)
                != current.parameters.value(id).map(f64::to_bits)
        })
    }

    fn validate_snapshots(
        &self,
        previous: &ModelParameterSnapshot,
        current: &ModelParameterSnapshot,
    ) -> Result<(), ModelMismatchError> {
        for snapshot in [previous, current] {
            if snapshot.model_id != self.model_id {
                return Err(ModelMismatchError::new(self.model_id, snapshot.model_id));
            }
        }
        Ok(())
    }
}

// Preserve encounter order, with inline deduplication for the common small
// update and hash-based lookup for large fan-outs.
fn unique<T: Copy + Eq + std::hash::Hash>(
    items: impl Iterator<Item = T>,
) -> impl Iterator<Item = T> {
    let mut inline = SmallVec::<[T; 4]>::new();
    let mut overflow: Option<FxHashSet<T>> = None;
    items.filter(move |&item| {
        if let Some(seen) = &mut overflow {
            return seen.insert(item);
        }
        if inline.contains(&item) {
            return false;
        }
        if inline.len() < 4 {
            inline.push(item);
        } else {
            let mut seen: FxHashSet<_> = inline.iter().copied().collect();
            seen.insert(item);
            overflow = Some(seen);
        }
        true
    })
}

impl Model {
    /// Capture parameter values with this model's identity for dependency queries.
    #[must_use]
    pub fn parameter_snapshot(&self) -> ModelParameterSnapshot {
        self.__capture_parameters().0
    }

    /// Capture tagged parameters and expression data at the same instant.
    #[doc(hidden)]
    pub fn __capture_parameters(
        &self,
    ) -> (ModelParameterSnapshot, oximo_expr::ExprArenaSnapshot<'_>) {
        let arena = self.arena();
        let parameters =
            ModelParameterSnapshot { model_id: self.id(), parameters: arena.parameter_snapshot() };
        (parameters, arena)
    }

    /// Build an immutable parameter-dependency snapshot.
    ///
    /// The index covers the objective, parameterized variable bounds, stored
    /// algebraic/SOC/PSD/indicator constraints, and semantic declarations.
    /// Traversal is stack-safe and deduplicates shared paths.
    ///
    /// # Panics
    /// Panics if a registry contains more entities than its ID type can address.
    /// Registration already enforces these bounds.
    #[must_use]
    pub fn parameter_dependencies(&self) -> ParameterDependencies {
        let arena = self.arena();
        let mut builder = Builder::new(&arena);

        let mut objective_roots = Vec::new();
        {
            let objective = self.objective.borrow();
            if let Some(objective) = objective.as_ref() {
                objective_roots.push(objective.expr);
            }

            let semantic = self.semantic_objective.borrow();
            if let Some(objective) = semantic.as_deref().and_then(Option::as_ref) {
                objective.function.visit_expressions(&mut |root| objective_roots.push(root));
            }
        }
        let parameters = builder.parameters_for_roots(objective_roots);
        builder.add_target(ParameterTarget::Objective, parameters);

        let variables = self.variables();
        for variable in variables.iter() {
            let mut parameters = Vec::new();
            if let Some(parameter) = variable.lb.parameter() {
                parameters.push(parameter);
            }
            if let Some(parameter) = variable.ub.parameter()
                && !parameters.contains(&parameter)
            {
                parameters.push(parameter);
            }
            builder.add_target(ParameterTarget::VariableBound(variable.id), parameters);
        }
        drop(variables);

        {
            let constraints = self.constraints();
            let inactive = self.inactive_scalar_views.borrow();

            for (index, constraint) in constraints.algebraic().iter().enumerate() {
                let id = ConstraintId(u32::try_from(index).expect("constraint count fits u32"));
                let root = inactive
                    .as_deref()
                    .and_then(|views| views.get(&id))
                    .map_or(constraint.lhs, |view| view.lhs);
                let parameters = builder.parameters_for_roots([root]);
                builder.add_target(ParameterTarget::Algebraic(id), parameters);
            }

            for (index, constraint) in constraints.second_order_cones().iter().enumerate() {
                let id = SocConstraintId(u32::try_from(index).expect("SOC count fits u32"));
                let parameters = builder.parameters_for_roots(
                    constraint.terms.iter().copied().chain(std::iter::once(constraint.bound)),
                );
                builder.add_target(ParameterTarget::SecondOrderCone(id), parameters);
            }

            for (index, constraint) in constraints.positive_semidefinite().iter().enumerate() {
                let id = PsdConstraintId(u32::try_from(index).expect("PSD count fits u32"));
                let parameters = builder
                    .parameters_for_roots(constraint.matrix.upper_triangle().iter().copied());
                builder.add_target(ParameterTarget::PositiveSemidefinite(id), parameters);
            }

            // SOS members carry variables and numeric weights, not expressions.
            for (index, constraint) in constraints.indicators().iter().enumerate() {
                let id =
                    IndicatorConstraintId(u32::try_from(index).expect("indicator count fits u32"));
                let parameters = builder.parameters_for_roots([constraint.lhs]);
                builder.add_target(ParameterTarget::Indicator(id), parameters);
            }
        }

        {
            let semantics = self.semantic_constraints_ref();
            for constraint in semantics.iter() {
                let mut roots = Vec::new();
                constraint.pair.function.visit_expressions(&mut |root| roots.push(root));
                let parameters = builder.parameters_for_roots(roots);
                builder.add_semantic(constraint.id, parameters);
            }
        }

        builder.finish(self.id())
    }
}

struct Builder<'a> {
    arena: &'a ExprArena,
    roots: FxHashMap<ExprId, Arc<[ParamId]>>,
    shared: FxHashMap<ExprId, Arc<[ParamId]>>,
    shared_parameters: usize,
    seen: FxHashSet<ExprId>,
    visited: FxHashSet<ExprId>,
    targets: FxHashMap<ParamId, Vec<ParameterTarget>>,
    target_parameters: FxHashMap<ParameterTarget, Vec<ParamId>>,
    semantic_targets: FxHashMap<ParamId, Vec<SemanticConstraintId>>,
    semantic_parameters: FxHashMap<SemanticConstraintId, Vec<ParamId>>,
}

impl<'a> Builder<'a> {
    fn new(arena: &'a ExprArena) -> Self {
        Self {
            arena,
            roots: FxHashMap::default(),
            shared: FxHashMap::default(),
            shared_parameters: 0,
            seen: FxHashSet::default(),
            visited: FxHashSet::default(),
            targets: FxHashMap::default(),
            target_parameters: FxHashMap::default(),
            semantic_targets: FxHashMap::default(),
            semantic_parameters: FxHashMap::default(),
        }
    }

    fn parameters_for_roots(&mut self, roots: impl IntoIterator<Item = ExprId>) -> Vec<ParamId> {
        let mut parameters = FxHashSet::default();

        for root in roots {
            let root_parameters = self.parameters_at(root);
            parameters.extend(root_parameters.iter().copied());
        }

        let mut parameters: Vec<_> = parameters.into_iter().collect();
        parameters.sort_unstable_by_key(|parameter| parameter.index());
        parameters
    }

    fn parameters_at(&mut self, root: ExprId) -> Arc<[ParamId]> {
        if let Some(parameters) = self.cached(root) {
            return Arc::clone(parameters);
        }

        let mut parameters = FxHashSet::default();
        let mut stack = vec![root];
        let mut visited = FxHashSet::default();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }

            if let Some(cached) = self.cached(id) {
                parameters.extend(cached.iter().copied());
                continue;
            }

            let node = self.arena.get(id);
            if let ExprNode::Param(parameter) = node {
                parameters.insert(*parameter);
            } else if matches!(
                node,
                ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
            ) {
                // Leaves have no reusable traversal to cache.
            } else if self.seen.insert(id) {
                for_each_child(node, |child| stack.push(child));
            } else {
                // A previous root already visited this subtree.
                let shared = self.collect_parameters(id);
                // Bound auxiliary cached IDs by the arena size.
                if shared.len() <= self.arena.len() - self.shared_parameters {
                    self.shared_parameters += shared.len();
                    self.shared.insert(id, Arc::clone(&shared));
                }
                parameters.extend(shared.iter().copied());
            }
        }

        let mut parameters: Vec<_> = parameters.into_iter().collect();
        parameters.sort_unstable_by_key(|parameter| parameter.index());
        let parameters: Arc<[ParamId]> = Arc::from(parameters);
        self.roots.insert(root, Arc::clone(&parameters));
        parameters
    }

    fn cached(&self, root: ExprId) -> Option<&Arc<[ParamId]>> {
        self.roots.get(&root).or_else(|| self.shared.get(&root))
    }

    fn collect_parameters(&mut self, root: ExprId) -> Arc<[ParamId]> {
        let mut parameters = FxHashSet::default();
        let mut stack = vec![root];
        self.visited.clear();
        while let Some(id) = stack.pop() {
            if !self.visited.insert(id) {
                continue;
            }
            if let Some(cached) = self.cached(id) {
                parameters.extend(cached.iter().copied());
                continue;
            }
            let node = self.arena.get(id);
            if let ExprNode::Param(parameter) = node {
                parameters.insert(*parameter);
            } else {
                self.seen.insert(id);
                for_each_child(node, |child| stack.push(child));
            }
        }
        let mut parameters: Vec<_> = parameters.into_iter().collect();
        parameters.sort_unstable_by_key(|parameter| parameter.index());
        Arc::from(parameters)
    }

    fn add_target(&mut self, target: ParameterTarget, parameters: Vec<ParamId>) {
        for parameter in parameters {
            self.targets.entry(parameter).or_default().push(target);
            self.target_parameters.entry(target).or_default().push(parameter);
        }
    }

    fn add_semantic(&mut self, constraint: SemanticConstraintId, parameters: Vec<ParamId>) {
        for parameter in parameters {
            self.semantic_targets.entry(parameter).or_default().push(constraint);
            self.semantic_parameters.entry(constraint).or_default().push(parameter);
        }
    }

    fn finish(self, model_id: ModelId) -> ParameterDependencies {
        let mut parameters: Vec<_> =
            self.targets.keys().chain(self.semantic_targets.keys()).copied().collect();
        parameters.sort_unstable_by_key(|parameter| parameter.index());
        parameters.dedup();

        ParameterDependencies {
            model_id,
            parameters,
            targets: self.targets,
            target_parameters: self.target_parameters,
            semantic_targets: self.semantic_targets,
            semantic_parameters: self.semantic_parameters,
        }
    }
}

fn for_each_child(node: &ExprNode, mut visitor: impl FnMut(ExprId)) {
    match node {
        ExprNode::Add(children)
        | ExprNode::Mul(children)
        | ExprNode::Min(children)
        | ExprNode::Max(children) => {
            for &child in children {
                visitor(child);
            }
        }
        ExprNode::Unary(_, child) => visitor(*child),
        ExprNode::Pow(left, right) | ExprNode::Div(left, right) | ExprNode::Atan2(left, right) => {
            visitor(*left);
            visitor(*right);
        }
        ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Linear { .. } => {}
    }
}
