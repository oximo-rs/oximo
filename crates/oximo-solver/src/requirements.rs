//! Solver-independent requirements at one model and parameter assignment.

use oximo_core::{
    Domain, FunctionRepresentationRef, Model, ModelKind, ObjectiveSense, SemanticConstraintId,
    SemanticType, SosType, VarId,
};
use oximo_expr::{ExprArena, ExprClass, ExprId, ExprNode};

use crate::capability::{ConstraintSignature, scalar_set, scalar_signature};

/// Variable domains, independently of their numeric thresholds and bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableDomain {
    Real,
    Integer,
    Binary,
    SemiContinuous,
    SemiInteger,
}

impl From<Domain> for VariableDomain {
    fn from(value: Domain) -> Self {
        match value {
            Domain::Real => Self::Real,
            Domain::Integer => Self::Integer,
            Domain::Binary => Self::Binary,
            Domain::SemiContinuous { .. } => Self::SemiContinuous,
            Domain::SemiInteger { .. } => Self::SemiInteger,
        }
    }
}

pub use oximo_expr::ExpressionOperator;

/// Facts retained in addition to the dominant model kind for combination checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent mathematical facts can coexist in any combination"
)]
pub struct ModelFeatures {
    pub kind: ModelKind,
    pub objective_class: ExprClass,
    pub has_psd: bool,
    pub has_explicit_soc: bool,
    pub has_indicators: bool,
    pub has_sos: bool,
    pub has_discrete_variables: bool,
    pub has_non_affine_rows: bool,
    pub has_variable_nonsmooth: bool,
}

/// A mathematical interface or prerequisite required by a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Requirement {
    ObjectiveDeclared,
    GdpReformulated,
    Domain(VariableDomain),
    Constraint(ConstraintSignature),
    Objective {
        function: SemanticType,
        sense: ObjectiveSense,
    },
    Sos(SosType),
    Operator(ExpressionOperator),
    Combination(ModelFeatures),
    /// A locally checked restriction that cannot be inferred from signatures alone.
    AdapterRestriction(&'static str),
}

impl std::fmt::Display for Requirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ObjectiveDeclared => f.write_str("objective or feasibility declaration"),
            Self::GdpReformulated => f.write_str("GDP reformulation"),
            Self::Domain(domain) => write!(f, "{domain:?} variable domain"),
            Self::Constraint(pair) => write!(f, "{} in {}", pair.function.name(), pair.set.name()),
            Self::Objective { function, sense } => {
                write!(f, "{sense:?} {} objective", function.name())
            }
            Self::Sos(kind) => write!(
                f,
                "native {kind:?}; explicitly use Model::reformulate_sos or Model::to_reformulated_sos_model"
            ),
            Self::Operator(op) => write!(f, "operator {op}"),
            Self::Combination(features) => {
                write!(f, "combination of domains, objective and constraints ({})", features.kind)
            }
            Self::AdapterRestriction(name) => f.write_str(name),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequirementEntity {
    Model,
    Objective,
    Variable(VarId),
    Constraint(SemanticConstraintId),
}

/// An ID and a captured human-readable location; names need not be unique.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequirementOrigin {
    pub entity: RequirementEntity,
    pub location: String,
    /// Original function/set identity.
    pub declared_constraint: Option<ConstraintSignature>,
}

/// Any alternative satisfies this entry.
/// All entries must be satisfied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequirementEntry {
    pub alternatives: Vec<Requirement>,
    pub origins: Vec<RequirementOrigin>,
}

/// An owned informational snapshot.
/// Solves always check the current model requirements.
#[derive(Clone, Debug, Default)]
pub struct ModelRequirements {
    pub entries: Vec<RequirementEntry>,
}

impl ModelRequirements {
    /// Inspect current active native views.
    pub fn infer(model: &Model) -> Self {
        Self::infer_with_details(model, true, None).0
    }

    fn infer_with_details(
        model: &Model,
        details: bool,
        prepared: Option<&crate::prepare::LoweringContext<'_>>,
    ) -> (Self, RequirementFacts, OperatorSummaries) {
        let mut result = Self::default();

        let model_origin = RequirementOrigin {
            entity: RequirementEntity::Model,
            location: if details { "model".into() } else { String::new() },
            declared_constraint: None,
        };
        if model.ensure_objective_declared().is_err() {
            result.add([Requirement::ObjectiveDeclared], model_origin.clone());
        }
        if model.has_unreformulated_gdp() {
            result.add([Requirement::GdpReformulated], model_origin.clone());
        }

        let live_arena = model.arena();
        let arena = prepared.map_or(&*live_arena, |p| p.arena());
        let resolved;
        let variables = if let Some(prepared) = prepared {
            prepared.variables()
        } else {
            resolved = model.variables().iter().map(|v| v.resolved(arena)).collect::<Vec<_>>();
            &resolved
        };
        let mut expressions = ExpressionRequirements {
            polynomials: prepared.map(std::ops::Deref::deref),
            ..ExpressionRequirements::default()
        };

        for variable in variables {
            result.add(
                [Requirement::Domain(variable.domain.into())],
                RequirementOrigin {
                    entity: RequirementEntity::Variable(variable.id),
                    location: if details {
                        format!("variable {:?}", variable.name)
                    } else {
                        String::new()
                    },
                    declared_constraint: None,
                },
            );
        }

        result.infer_constraints(model, arena, variables, &mut expressions, details);
        if let Some(objective) = model.semantic_objective().as_ref() {
            expressions.facts.uncacheable |= !builtin_scalar(objective.function.signature());
            let origin = RequirementOrigin {
                entity: RequirementEntity::Objective,
                location: if details { "objective".into() } else { String::new() },
                declared_constraint: None,
            };
            let mut alternatives: smallvec::SmallVec<[Requirement; 2]> =
                smallvec::smallvec![Requirement::Objective {
                    function: objective.function.signature(),
                    sense: objective.sense,
                }];
            if let FunctionRepresentationRef::Scalar(root) = objective.function.representation() {
                alternatives.push(Requirement::Objective {
                    function: scalar_signature(oximo_expr::classify(arena, root)),
                    sense: objective.sense,
                });
            }
            alternatives.dedup();
            result.add(alternatives, origin.clone());
            // Consume the native objective.
            if let Some(native) = model.objective().as_ref() {
                expressions.collect(arena, &[native.expr], &origin, &mut result);
            }
        }

        let features = ModelFeatures {
            kind: prepared.map_or_else(
                || model.__kind_at(arena, variables),
                crate::prepare::LoweringContext::kind,
            ),
            objective_class: model
                .objective()
                .as_ref()
                .map_or(ExprClass::Linear, |o| oximo_expr::classify(arena, o.expr)),
            has_psd: model.has_active_psd_constraints(),
            has_explicit_soc: model.has_active_soc_constraints(),
            has_indicators: model.has_active_indicator_constraints(),
            has_sos: model.has_active_sos_constraints(),
            has_discrete_variables: variables.iter().any(|v| v.domain.requires_discrete_choice()),
            has_non_affine_rows: model
                .constraints()
                .algebraic()
                .iter()
                .any(|r| r.active && oximo_expr::classify(arena, r.lhs) != ExprClass::Linear),
            has_variable_nonsmooth: expressions.nonsmooth,
        };
        result.add([Requirement::Combination(features)], model_origin);
        expressions.capture_numeric_parameters(model, arena);
        (result, expressions.facts, expressions.operators)
    }

    fn infer_constraints(
        &mut self,
        model: &Model,
        arena: &ExprArena,
        variables: &[oximo_core::ResolvedVariable],
        expressions: &mut ExpressionRequirements,
        details: bool,
    ) {
        let declarations = model.semantic_constraints_ref();
        if details {
            for record in declarations.iter().filter(|record| record.active) {
                self.infer_constraint(model, &record, arena, variables, expressions, true);
            }
        } else {
            for position in 0..declarations.len() {
                if let Some(scalar) = declarations.scalar_metadata(position)
                    && self.infer_flat_scalar(model, arena, scalar, expressions)
                {
                    continue;
                }

                let record = declarations.get(position).expect("registered declaration");
                if record.active {
                    self.infer_constraint(model, &record, arena, variables, expressions, false);
                }
            }
        }
    }

    fn infer_constraint(
        &mut self,
        model: &Model,
        record: &oximo_core::SemanticConstraintRef<'_>,
        arena: &ExprArena,
        variables: &[oximo_core::ResolvedVariable],
        expressions: &mut ExpressionRequirements,
        details: bool,
    ) {
        let declared = ConstraintSignature {
            function: record.pair.function.signature(),
            set: record.pair.set.signature(),
        };
        let selected = model.alternative_lowering_view(record.id);
        let representation = selected.is_none().then(|| record.pair.function.representation());
        if !details && expressions.repeated_flat_pair(arena, representation, declared) {
            return;
        }

        expressions.facts.uncacheable |= !builtin_pair(declared);
        let origin = RequirementOrigin {
            entity: RequirementEntity::Constraint(record.id),
            location: if details { format!("constraint {:?}", record.name) } else { String::new() },
            declared_constraint: Some(declared),
        };

        let mut alternatives: smallvec::SmallVec<[Requirement; 3]> =
            smallvec::smallvec![Requirement::Constraint(declared)];
        let mut roots = smallvec::SmallVec::<[ExprId; 4]>::new();
        let mut native_class = None;
        if let Some(view) = selected {
            // A selected lowering replaces, rather than adds to, source semantics.
            let signature = if matches!(record.id, SemanticConstraintId::Indicator(_)) {
                ConstraintSignature::of::<oximo_core::IndicatorFunctionData, oximo_core::IndicatorSet>(
                )
            } else {
                let class = oximo_expr::classify(arena, view.lhs);
                native_class = Some(class);

                ConstraintSignature {
                    function: scalar_signature(class),
                    set: native_scalar_set(view),
                }
            };
            alternatives = smallvec::smallvec![Requirement::Constraint(signature)];
            roots.push(view.lhs);
        } else {
            match representation.expect("unselected source representation") {
                FunctionRepresentationRef::Scalar(root) => {
                    let class = oximo_expr::classify(arena, root);
                    native_class = Some(class);
                    alternatives.push(Requirement::Constraint(ConstraintSignature {
                        function: scalar_signature(class),
                        set: declared.set,
                    }));
                    roots.push(root);
                }
                FunctionRepresentationRef::VectorAffine(components) => {
                    alternatives.push(Requirement::Constraint(ConstraintSignature {
                        function: SemanticType::of::<oximo_core::VectorAffineData>(),
                        set: declared.set,
                    }));
                    roots.extend_from_slice(components);
                }
                FunctionRepresentationRef::SymmetricAffine(matrix) => {
                    alternatives.push(Requirement::Constraint(ConstraintSignature {
                        function: SemanticType::of::<oximo_core::SymmetricAffineData>(),
                        set: declared.set,
                    }));
                    roots.extend_from_slice(matrix.upper_triangle());
                }
                FunctionRepresentationRef::Variables(_) => {
                    if let Some(kind) = record.pair.set.sos_kind() {
                        // The concrete kind is mandatory: a generic SOS pair
                        // must not bypass a kind-specific denial or condition.
                        alternatives = smallvec::smallvec![Requirement::Sos(kind)];
                    }
                }
                FunctionRepresentationRef::Indicator { consequent, .. } => {
                    alternatives.push(Requirement::Constraint(ConstraintSignature::of::<
                        oximo_core::IndicatorFunctionData,
                        oximo_core::IndicatorSet,
                    >()));
                    consequent.visit_expressions(&mut |root| roots.push(root));
                }
                _ => {}
            }
        }
        if let SemanticConstraintId::Algebraic(id) = record.id
            && native_class == Some(ExprClass::Quadratic)
            && let Some(row) = model.constraints().algebraic().get(id.index())
        {
            expressions.facts.parameter_sensitive = true;
            expressions.collect_numeric_parameters(arena, row.lhs);
            if let Some(quadratic) = expressions.polynomials.map_or_else(
                || {
                    oximo_expr::extract_quadratic(arena, row.lhs)
                        .map(crate::prepare::Extracted::Owned)
                },
                |p| p.quadratic(row.lhs),
            ) && oximo_core::__is_detected_soc_from_quadratic(variables, row, &quadratic)
            {
                alternatives.push(Requirement::Constraint(ConstraintSignature::of::<
                    oximo_core::VectorAffineData,
                    oximo_core::SecondOrderCone,
                >()));
            }
        }

        alternatives.dedup();
        self.add(alternatives, origin.clone());
        expressions.collect(arena, &roots, &origin, self);
    }

    fn infer_flat_scalar(
        &mut self,
        model: &Model,
        arena: &ExprArena,
        scalar: oximo_core::ScalarConstraintMetadata<'_>,
        expressions: &mut ExpressionRequirements,
    ) -> bool {
        if !scalar.active {
            return true;
        }

        if !matches!(
            arena.get(scalar.view.lhs),
            ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Var(_) | ExprNode::Linear { .. }
        ) || model.alternative_lowering_view(scalar.id.into()).is_some()
        {
            return false;
        }

        let declared = ConstraintSignature { function: scalar.function, set: scalar.set };
        if expressions.last_flat_pair != Some(declared) {
            expressions.last_flat_pair = Some(declared);
            expressions.facts.uncacheable |= !builtin_pair(declared);
            let native = ConstraintSignature {
                function: scalar_signature(ExprClass::Linear),
                set: scalar.set,
            };
            let origin = RequirementOrigin {
                entity: RequirementEntity::Constraint(scalar.id.into()),
                location: String::new(),
                declared_constraint: Some(declared),
            };
            if declared == native {
                self.add([Requirement::Constraint(declared)], origin);
            } else {
                self.add(
                    [Requirement::Constraint(declared), Requirement::Constraint(native)],
                    origin,
                );
            }
        }
        true
    }

    fn add(&mut self, alternatives: impl AsRef<[Requirement]>, origin: RequirementOrigin) {
        let alternatives = alternatives.as_ref();
        if let Some(entry) =
            self.entries.iter_mut().find(|entry| entry.alternatives.as_slice() == alternatives)
        {
            if !origin.location.is_empty() && entry.origins.last() != Some(&origin) {
                entry.origins.push(origin);
            }
        } else {
            self.entries.push(RequirementEntry {
                alternatives: alternatives.to_vec(),
                origins: if origin.location.is_empty() { Vec::new() } else { vec![origin] },
            });
        }
    }
}

#[expect(clippy::float_cmp, reason = "native equality rows use exact matching bound values")]
fn native_scalar_set(view: oximo_core::ScalarConstraintView) -> SemanticType {
    if view.lower == view.upper {
        SemanticType::of::<oximo_core::EqualTo>()
    } else if view.lower == f64::NEG_INFINITY {
        SemanticType::of::<oximo_core::LessThan>()
    } else if view.upper == f64::INFINITY {
        SemanticType::of::<oximo_core::GreaterThan>()
    } else {
        SemanticType::of::<oximo_core::Interval>()
    }
}

const _: () = assert!(ExpressionOperator::ALL.len() <= u64::BITS as usize);

/// Summaries cover only reachable compound nodes.
#[derive(Default)]
struct ExpressionRequirements<'a> {
    polynomials: Option<&'a crate::prepare::PreparedExpressions>,
    analysis: oximo_expr::ExpressionAnalysis,
    operators: OperatorSummaries,
    nonsmooth: bool,
    facts: RequirementFacts,
    last_flat_pair: Option<ConstraintSignature>,
    numeric_parameters: rustc_hash::FxHashSet<oximo_expr::ParamId>,
    numeric_visited: rustc_hash::FxHashSet<ExprId>,
    numeric_stack: Vec<ExprId>,
}

/// Immutable after inference and shared with preparation on both cache hits and misses.
#[derive(Debug, Default)]
pub(crate) struct OperatorSummaries {
    roots: rustc_hash::FxHashMap<ExprId, u64>,
}

const OPERATOR_ROOTS: usize = 16_384;

impl OperatorSummaries {
    fn record(&mut self, root: ExprId, operators: u64) {
        if operators != 0 && self.roots.len() < OPERATOR_ROOTS {
            self.roots.insert(root, operators);
        }
    }

    pub(crate) fn get(&self, root: ExprId) -> Option<u64> {
        self.roots.get(&root).copied()
    }

    fn retained_bytes(&self) -> usize {
        // Include spare hash buckets and control bytes conservatively.
        std::mem::size_of::<Self>()
            + 2 * std::mem::size_of::<usize>()
            + self.roots.capacity()
                * (std::mem::size_of::<(ExprId, u64)>() + std::mem::size_of::<usize>())
    }
}

#[derive(Clone, Default)]
struct RequirementFacts {
    uncacheable: bool,
    parameter_sensitive: bool,
    parameter_revision: u64,
    parameter_values: Vec<(oximo_expr::ParamId, u64)>,
}

fn children(node: &ExprNode, mut visit: impl FnMut(ExprId)) {
    match node {
        ExprNode::Unary(_, child) => visit(*child),
        ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
            visit(*a);
            visit(*b);
        }
        ExprNode::Add(items)
        | ExprNode::Mul(items)
        | ExprNode::Min(items)
        | ExprNode::Max(items) => {
            for &child in items {
                visit(child);
            }
        }
        ExprNode::Const(_)
        | ExprNode::Param(_)
        | ExprNode::Var(_)
        | ExprNode::Linear { .. }
        | ExprNode::Quadratic(_) => {}
    }
}

impl ExpressionRequirements<'_> {
    fn collect_numeric_parameters(&mut self, arena: &ExprArena, root: ExprId) {
        // The operator pass already visits this DAG.
        if !self.summarize(arena, root).depends_on_parameter {
            return;
        }
        self.numeric_stack.push(root);
        while let Some(id) = self.numeric_stack.pop() {
            if !self.numeric_visited.insert(id) {
                continue;
            }
            if let ExprNode::Param(parameter) = arena.get(id) {
                self.numeric_parameters.insert(*parameter);
            } else {
                children(arena.get(id), |child| {
                    if self.known(arena, child).expect("summarized child").depends_on_parameter {
                        self.numeric_stack.push(child);
                    }
                });
            }
        }
    }

    fn capture_numeric_parameters(&mut self, model: &Model, arena: &ExprArena) {
        if self.facts.parameter_sensitive {
            // Cone recognition can depend on a variable's bound sign even when
            // its quadratic row contains no parameters.
            // We include all bound parameters conservatively, while ignoring objective/affine edits.
            for variable in model.variables().iter() {
                for bound in [variable.lb, variable.ub] {
                    if let oximo_core::Bound::Parameter(parameter) = bound {
                        self.numeric_parameters.insert(parameter);
                    }
                }
            }
        }

        self.facts.parameter_values = self
            .numeric_parameters
            .iter()
            .map(|&parameter| (parameter, arena.param_value(parameter).to_bits()))
            .collect();
        self.facts.parameter_values.sort_unstable_by_key(|&(parameter, _)| parameter.index());
        self.facts.parameter_sensitive = !self.facts.parameter_values.is_empty();
        self.facts.parameter_revision = arena.parameter_revision();
    }

    fn repeated_flat_pair(
        &mut self,
        arena: &ExprArena,
        representation: Option<FunctionRepresentationRef<'_>>,
        declared: ConstraintSignature,
    ) -> bool {
        if let Some(FunctionRepresentationRef::Scalar(root)) = representation
            && matches!(
                arena.get(root),
                ExprNode::Const(_)
                    | ExprNode::Param(_)
                    | ExprNode::Var(_)
                    | ExprNode::Linear { .. }
            )
        {
            // Flat leaves add no operator or value-dependent requirements.
            // Their clause depends only on the declared pair, so we retain it once.
            let repeated = self.last_flat_pair == Some(declared);
            self.last_flat_pair = Some(declared);
            repeated
        } else {
            false
        }
    }

    fn known(&self, arena: &ExprArena, id: ExprId) -> Option<oximo_expr::ExpressionSummary> {
        arena.__cached_expression_summary(id).or_else(|| self.analysis.known(arena, id))
    }

    fn summarize(&mut self, arena: &ExprArena, root: ExprId) -> oximo_expr::ExpressionSummary {
        arena
            .__preparation_expression_summary(root)
            .unwrap_or_else(|| self.analysis.summarize(arena, root))
    }

    fn collect(
        &mut self,
        arena: &ExprArena,
        roots: &[ExprId],
        origin: &RequirementOrigin,
        requirements: &mut ModelRequirements,
    ) {
        let mut operators = 0;
        for &root in roots {
            let summary = self.summarize(arena, root);
            self.operators.record(root, summary.operators);
            operators |= summary.operators;
            self.nonsmooth |= summary.nonsmooth;
        }

        if operators == 0 {
            return;
        }
        for &operator in ExpressionOperator::ALL {
            if operators & operator.bit() != 0 {
                requirements.add([Requirement::Operator(operator)], origin.clone());
            }
        }
    }
}

// Successful validation borrows an immutable summary instead of rebuilding and
// cloning diagnostic names on every solve.
const CACHE_MODELS: usize = 4;
const CACHE_BYTES: usize = 4 * 1024 * 1024;

struct CachedRequirements {
    model: oximo_core::ModelId,
    key: oximo_core::model::ModelUpdateKey,
    parameters: u64,
    pending_gdp: bool,
    parameter_sensitive: bool,
    parameter_values: Vec<(oximo_expr::ParamId, u64)>,
    detailed: bool,
    bytes: usize,
    requirements: std::rc::Rc<ModelRequirements>,
    operators: std::sync::Arc<OperatorSummaries>,
}

thread_local! {
    static VALIDATION_REQUIREMENTS: std::cell::RefCell<std::collections::VecDeque<CachedRequirements>> = const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

/// Model mutations invalidate the summary; only proven parameter-independent
/// built-in interfaces may reuse it across parameter changes.
pub(crate) fn for_validation(model: &Model) -> std::rc::Rc<ModelRequirements> {
    cached(model, true, None)
}

pub(crate) fn for_matching(model: &Model) -> std::rc::Rc<ModelRequirements> {
    cached(model, false, None)
}

pub(crate) fn for_prepared(
    model: &Model,
    prepared: &crate::prepare::LoweringContext<'_>,
    detailed: bool,
) -> std::rc::Rc<ModelRequirements> {
    cached(model, detailed, Some(prepared))
}

fn cached(
    model: &Model,
    detailed: bool,
    prepared: Option<&crate::prepare::LoweringContext<'_>>,
) -> std::rc::Rc<ModelRequirements> {
    let key = model.__update_key();
    let arena = model.arena();
    let parameters = arena.parameter_revision();
    let pending_gdp = model.has_unreformulated_gdp();
    if let Some((requirements, operators)) = VALIDATION_REQUIREMENTS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let index = cache.iter().position(|entry| {
            entry.key == key
                && (!detailed || entry.detailed)
                && entry.pending_gdp == pending_gdp
                && (!entry.parameter_sensitive
                    || entry.parameters == parameters
                    || entry
                        .parameter_values
                        .iter()
                        .all(|&(parameter, bits)| arena.param_value(parameter).to_bits() == bits))
        })?;
        let mut entry = cache.remove(index)?;
        entry.parameters = parameters;
        let requirements = std::rc::Rc::clone(&entry.requirements);
        let operators = std::sync::Arc::clone(&entry.operators);
        cache.push_front(entry);
        Some((requirements, operators))
    }) {
        if let Some(prepared) = prepared {
            prepared.set_operator_summaries(operators);
        }
        return requirements;
    }
    drop(arena);

    // Infer outside the RefCell borrow since semantic views can invoke user code.
    let (requirements, facts, operators) =
        ModelRequirements::infer_with_details(model, detailed, prepared);
    let requirements = std::rc::Rc::new(requirements);
    let operators = std::sync::Arc::new(operators);
    if let Some(prepared) = prepared {
        prepared.set_operator_summaries(std::sync::Arc::clone(&operators));
    }
    if facts.uncacheable {
        return requirements;
    }

    let parameter_sensitive = facts.parameter_sensitive;
    let mut bytes = std::mem::size_of::<CachedRequirements>()
        + std::mem::size_of::<ModelRequirements>()
        + 4 * std::mem::size_of::<usize>()
        + model.name.len()
        + operators.retained_bytes()
        + facts.parameter_values.capacity() * std::mem::size_of::<(oximo_expr::ParamId, u64)>()
        + requirements.entries.capacity() * std::mem::size_of::<RequirementEntry>();

    for entry in &requirements.entries {
        bytes += entry.origins.capacity() * std::mem::size_of::<RequirementOrigin>()
            + entry.alternatives.capacity() * std::mem::size_of::<Requirement>();
        for origin in &entry.origins {
            bytes += origin.location.capacity();
        }
    }

    if bytes <= CACHE_BYTES {
        VALIDATION_REQUIREMENTS.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.retain(|entry| entry.model != model.id());
            while cache.len() >= CACHE_MODELS
                || cache.iter().map(|entry| entry.bytes).sum::<usize>() + bytes > CACHE_BYTES
            {
                cache.pop_back();
            }
            cache.push_front(CachedRequirements {
                model: model.id(),
                key,
                parameters: facts.parameter_revision,
                pending_gdp,
                parameter_sensitive,
                parameter_values: facts.parameter_values,
                detailed,
                bytes,
                requirements: std::rc::Rc::clone(&requirements),
                operators,
            });
        });
    }

    requirements
}

fn builtin_scalar(function: SemanticType) -> bool {
    function.is::<oximo_core::ScalarFunctionData<oximo_expr::Constant>>()
        || function.is::<oximo_core::ScalarFunctionData<oximo_core::Affine>>()
        || function.is::<oximo_core::ScalarFunctionData<oximo_core::Quadratic>>()
        || function.is::<oximo_core::ScalarFunctionData<oximo_core::Nonlinear>>()
        || function.is::<oximo_core::ScalarFunctionData<oximo_expr::Dynamic>>()
        || function.is::<oximo_core::AnyScalarFunctionData>()
}

fn builtin_pair(pair: ConstraintSignature) -> bool {
    use oximo_core::{
        IndicatorFunctionData, IndicatorSet, PositiveSemidefiniteCone, SecondOrderCone,
        SpecialOrderedSet, SymmetricAffineData, VariableVectorData, VectorAffineData,
    };
    (builtin_scalar(pair.function) && scalar_set(pair.set))
        || (pair.function.is::<VectorAffineData>() && pair.set.is::<SecondOrderCone>())
        || (pair.function.is::<SymmetricAffineData>() && pair.set.is::<PositiveSemidefiniteCone>())
        || (pair.function.is::<IndicatorFunctionData>() && pair.set.is::<IndicatorSet>())
        || (pair.function.is::<VariableVectorData>() && pair.set.is::<SpecialOrderedSet>())
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use oximo_core::prelude::*;
    use std::rc::Rc;

    #[test]
    fn operator_bits_match_the_complete_public_vocabulary() {
        let mut bits = 0;
        for (i, &op) in ExpressionOperator::ALL.iter().enumerate() {
            assert_eq!(op.bit(), 1_u64 << i);
            assert_eq!(bits & op.bit(), 0);
            bits |= op.bit();
        }
    }

    #[test]
    fn prepared_and_local_summaries_preserve_operators_and_numeric_dependencies() {
        let model = Model::new("summary scratch");
        variable!(model, x);
        param!(model, p = 2.0);

        let shared = (p * x).sin() + x.abs();
        let root = shared.clone() * shared + x.pow(x) + (p * x * x);
        let arena = model.arena();
        let snapshot = (*arena).clone();
        snapshot.__enable_preparation_classifications();
        let mut local = ExpressionRequirements::default();
        let mut prepared = ExpressionRequirements::default();
        assert_eq!(local.summarize(&arena, root.id()), prepared.summarize(&snapshot, root.id()));

        local.collect_numeric_parameters(&arena, root.id());
        prepared.collect_numeric_parameters(&snapshot, root.id());
        assert_eq!(local.numeric_parameters, prepared.numeric_parameters);
        assert!(snapshot.__cached_expression_summary(root.id()).is_some());
    }

    #[test]
    fn numeric_parameters_reuse_summaries_and_skip_parameter_free_dags() {
        let model = Model::new("shared numeric analysis");
        variable!(model, x);
        param!(model, p = 2.0);
        param!(model, q = 3.0);

        let shared = p * x * x;
        let root = shared.clone() + shared + q * x + x * x;
        let arena = model.arena();
        let mut expressions = ExpressionRequirements::default();
        expressions.collect_numeric_parameters(&arena, root.id());
        assert_eq!(expressions.numeric_parameters.len(), 2);
        assert!(expressions.numeric_parameters.contains(&p.param_id().unwrap()));
        assert!(expressions.numeric_parameters.contains(&q.param_id().unwrap()));

        let summary = expressions.summarize(&arena, root.id());
        let visited = expressions.numeric_visited.len();
        expressions.collect_numeric_parameters(&arena, root.id());
        assert_eq!(expressions.summarize(&arena, root.id()), summary);
        assert_eq!(expressions.numeric_visited.len(), visited);
    }

    #[test]
    fn prepared_operator_proofs_are_snapshot_local_and_keep_hidden_operators() {
        let model = Model::new("operator proof");
        variable!(model, x);
        param!(model, p = 1.0);
        let root = p * x * x;
        objective!(model, Min, &root);
        let prepared = crate::prepare::LoweringContext::new(&model).unwrap();
        assert!(!prepared.has_only_polynomial_operators(root.id()));

        let first = for_prepared(&model, &prepared, false);
        assert!(prepared.has_only_polynomial_operators(root.id()));

        let next = crate::prepare::LoweringContext::new(&model).unwrap();
        assert!(!next.has_only_polynomial_operators(root.id()));
        assert!(Rc::ptr_eq(&first, &for_prepared(&model, &next, false)));
        assert!(next.has_only_polynomial_operators(root.id()));

        drop(next);
        drop(prepared);
        let hidden = x.sin().powi(0);
        objective!(model, Min, &hidden);
        let prepared = crate::prepare::LoweringContext::new(&model).unwrap();
        for_prepared(&model, &prepared, false);
        assert!(!prepared.has_only_polynomial_operators(hidden.id()));
    }

    #[test]
    fn standalone_cache_hits_restore_operator_proofs_and_keep_numeric_validation() {
        let model = Model::new("standalone operator cache");
        variable!(model, x);
        param!(model, p = 1.0);
        let polynomial = p * x * x;
        let division = x / p;
        constraint!(model, &division <= 2.0);
        objective!(model, Min, &polynomial);

        let profile = crate::SolverProfile::new("unsupported");
        for detailed in [false, true] {
            let first = if detailed { for_validation(&model) } else { for_matching(&model) };
            for value in [1.0, 0.0, 2.0] {
                model.set_param(p, value).unwrap();
                let prepared = crate::prepare::LoweringContext::new(&model).unwrap();
                assert!(Rc::ptr_eq(&first, &for_prepared(&model, &prepared, detailed)));
                assert!(prepared.has_only_polynomial_operators(polynomial.id()));
                assert!(!prepared.has_only_polynomial_operators(division.id()));
            }
        }

        let hidden = x.asin().powi(0);
        objective!(model, Min, &hidden);
        let expected = profile.check_model(&model);
        for _ in 0..2 {
            let prepared = crate::prepare::LoweringContext::new(&model).unwrap();
            let report = profile.check_prepared(&model, &prepared).unwrap();
            assert_eq!(format!("{expected:?}"), format!("{report:?}"));
            assert!(!prepared.has_only_polynomial_operators(hidden.id()));
        }
    }

    #[test]
    fn runtime_classified_builtin_functions_reuse_and_invalidate_the_summary() {
        use oximo_core::AnyScalarFunction;
        for class in [ExprClass::Linear, ExprClass::Quadratic, ExprClass::Nonlinear] {
            let model = Model::new("runtime-classified builtin");

            variable!(model, x);
            param!(model, p = 1.0);
            let function = match class {
                ExprClass::Linear => AnyScalarFunction::Affine((p * x).into_function()),
                ExprClass::Quadratic => {
                    AnyScalarFunction::Quadratic((p * x.square()).into_function())
                }
                ExprClass::Nonlinear => AnyScalarFunction::Nonlinear((p * x.exp()).into_function()),
            };
            model.minimize(function.clone());

            let row = model.add_constraint("runtime_row", Constraint::new(function, LessThan(2.0)));
            let first = for_matching(&model);
            assert!(Rc::ptr_eq(&first, &for_matching(&model)));

            let detailed = for_validation(&model);
            assert!(Rc::ptr_eq(&detailed, &for_validation(&model)));
            assert_eq!(detailed.entries, ModelRequirements::infer(&model).entries);

            p.set_param_value(2.0).unwrap();
            let changed = for_validation(&model);
            assert_eq!(changed.entries, ModelRequirements::infer(&model).entries);

            if class != ExprClass::Quadratic {
                assert!(Rc::ptr_eq(&detailed, &changed));
            }
            model.set_constraint_active(row, false).unwrap();
            let inactive = for_validation(&model);
            assert!(!Rc::ptr_eq(&changed, &inactive));
            assert_eq!(inactive.entries, ModelRequirements::infer(&model).entries);
        }
    }

    #[test]
    fn full_validation_keeps_a_large_supported_model_on_the_compact_cache_path() {
        use crate::{Solver, SolverProfile};
        struct LocalChecks(std::cell::Cell<usize>);
        impl Solver for LocalChecks {
            type Options = ();
            fn name(&self) -> &str {
                "local checks"
            }

            fn profile(&self, (): &()) -> std::borrow::Cow<'_, SolverProfile> {
                std::borrow::Cow::Owned({
                    SolverProfile::new(self.name())
                        .with_kinds([ModelKind::LP])
                        .with_scalar_constraints([ExprClass::Linear])
                        .with_scalar_objectives([ExprClass::Linear])
                        .with_domains([VariableDomain::Real])
                        .with_operators(ExpressionOperator::ALL.iter().copied())
                })
            }

            fn validate_requirements(
                &self,
                _: &Model,
                (): &(),
                requirements: &ModelRequirements,
            ) -> Vec<crate::RequirementDiagnostic> {
                assert!(requirements.entries.iter().all(|entry| entry.origins.is_empty()));
                self.0.set(self.0.get() + 1);
                Vec::new()
            }

            fn solve(
                &mut self,
                _: &Model,
                (): &(),
            ) -> Result<crate::SolverResult, crate::SolverError> {
                unreachable!()
            }
        }

        let model = Model::new("large supported model");

        variable!(model, x);
        objective!(model, Min, x);
        for i in 0..50_000 {
            model.add_constraint(format!("row_{i}"), x.le(2.0));
        }

        let solver = LocalChecks(std::cell::Cell::new(0));
        assert!(solver.validate_model(&model, &()).is_confirmed());

        let first = for_matching(&model);
        for _ in 0..3 {
            assert!(solver.validate_model(&model, &()).is_confirmed());
            assert!(Rc::ptr_eq(&first, &for_matching(&model)));
        }
        assert_eq!(solver.0.get(), 4);

        VALIDATION_REQUIREMENTS.with(|cache| {
            let cache = cache.borrow();
            let entry = cache.iter().find(|entry| entry.model == model.id()).unwrap();
            assert!(!entry.detailed);
            assert!(entry.bytes <= CACHE_BYTES);
        });
    }

    #[test]
    fn compact_clauses_match_full_inference_for_repeated_and_mixed_scalar_roots() {
        let model = Model::new("compact scalar clauses");

        variable!(model, x);
        variable!(model, y);
        param!(model, p = 2.0);
        for i in 0..20 {
            model.add_constraint(format!("affine_{i}"), (p * x + y).le(f64::from(i)));
            model.add_constraint(format!("dynamic_{i}"), x.erase().le(f64::from(i)));
        }
        constraint!(model, nonlinear, x.sin().erase() <= 2.0);
        constraint!(model, quadratic, x.square() <= 4.0);
        constraint!(model, other_set, x == y);
        objective!(model, Min, x);

        let (compact, compact_facts, _) =
            ModelRequirements::infer_with_details(&model, false, None);
        let (mut full, full_facts, _) = ModelRequirements::infer_with_details(&model, true, None);
        assert!(compact.entries.iter().all(|entry| entry.origins.is_empty()));

        for entry in &mut full.entries {
            entry.origins.clear();
        }
        assert_eq!(compact.entries, full.entries);
        assert_eq!(compact_facts.parameter_sensitive, full_facts.parameter_sensitive);
        assert_eq!(compact_facts.uncacheable, full_facts.uncacheable);
    }

    #[test]
    fn affine_updates_and_unused_nodes_reuse_the_same_summary() {
        let model = Model::new("parameter-independent requirements");

        variable!(model, x);
        param!(model, p = 1.0);
        constraint!(model, p * x <= 2.0);
        objective!(model, Min, p * x);
        let first = for_validation(&model);
        p.set_param_value(2.0).unwrap();
        assert!(Rc::ptr_eq(&first, &for_validation(&model)));

        let mut unused = x.erase();
        for _ in 0..20_000 {
            unused = unused.sin().erase();
        }
        assert!(Rc::ptr_eq(&first, &for_validation(&model)));
        assert_eq!(first.entries, ModelRequirements::infer(&model).entries);
    }

    #[test]
    fn quadratic_parameters_and_soc_bound_signs_refresh_the_summary() {
        let model = Model::new("numeric cone requirements");

        param!(model, p = 1.0);
        param!(model, lower = 0.0);
        param!(model, unused = 0.0);
        variable!(model, -1.0 <= x <= 1.0);
        variable!(model, lower <= t <= 5.0);
        constraint!(model, p * x.square() <= t.square());
        objective!(model, Min, t);

        let first = for_validation(&model);
        assert!(Rc::ptr_eq(&first, &for_validation(&model)));

        unused.set_param_value(1.0).unwrap();
        assert!(Rc::ptr_eq(&first, &for_validation(&model)));

        p.set_param_value(-1.0).unwrap();
        let second = for_validation(&model);
        assert!(!Rc::ptr_eq(&first, &second));
        assert_eq!(second.entries, ModelRequirements::infer(&model).entries);

        p.set_param_value(1.0).unwrap();
        lower.set_param_value(-1.0).unwrap();
        let third = for_validation(&model);
        assert!(!Rc::ptr_eq(&second, &third));
        assert_eq!(third.entries, ModelRequirements::infer(&model).entries);
    }

    #[test]
    fn constant_quadratics_reuse_requirements_after_objective_and_affine_updates() {
        let model = Model::new("independent quadratic requirements");

        param!(model, cost = 1.0);
        param!(model, demand = 1.0);
        variable!(model, -1.0 <= x <= 1.0);
        variable!(model, 0.0 <= t <= 5.0);
        constraint!(model, cone, x.square() <= t.square());
        constraint!(model, affine, t >= demand);
        objective!(model, Min, cost * t.exp());

        let first = for_validation(&model);
        cost.set_param_value(1.01).unwrap();
        demand.set_param_value(2.0).unwrap();
        assert!(Rc::ptr_eq(&first, &for_validation(&model)));
        assert_eq!(first.entries, ModelRequirements::infer(&model).entries);
    }

    #[test]
    fn parameterized_bounds_invalidate_constant_quadratic_cone_requirements() {
        let model = Model::new("bound-dependent cone requirements");

        param!(model, lower = 0.0);
        variable!(model, -1.0 <= x <= 1.0);
        variable!(model, lower <= t <= 5.0);
        constraint!(model, cone, x.square() <= t.square());
        objective!(model, Min, t);

        let first = for_validation(&model);
        lower.set_param_value(-1.0).unwrap();
        let changed = for_validation(&model);
        assert!(!Rc::ptr_eq(&first, &changed));
        assert_ne!(first.entries, changed.entries);
        assert_eq!(changed.entries, ModelRequirements::infer(&model).entries);
    }

    #[test]
    fn quadratic_constant_terms_invalidate_cone_requirements() {
        let model = Model::new("parameterized quadratic right side");

        param!(model, cap = 0.0);
        variable!(model, -1.0 <= x <= 1.0);
        variable!(model, 0.0 <= t <= 5.0);
        constraint!(model, cone, x.square() <= t.square() + cap);
        objective!(model, Min, t);

        let first = for_validation(&model);
        cap.set_param_value(1.0).unwrap();
        let changed = for_validation(&model);
        assert!(!Rc::ptr_eq(&first, &changed));
        assert_eq!(changed.entries, ModelRequirements::infer(&model).entries);

        cap.set_param_value(2.0).unwrap();
        cap.set_param_value(1.0).unwrap();
        assert!(Rc::ptr_eq(&changed, &for_validation(&model)));
    }

    #[test]
    fn edits_forks_and_activity_changes_invalidate_diagnostic_metadata() {
        let model = Model::new("structural cache");

        variable!(model, x);
        let row = constraint!(model, active, x.sin() <= 2.0);
        objective!(model, Min, x);
        let first = for_validation(&model);
        let fork = model.fork();
        assert!(!Rc::ptr_eq(&first, &for_validation(&fork)));

        model.set_constraint_active(row, false).unwrap();
        let second = for_validation(&model);
        assert!(!Rc::ptr_eq(&first, &second));
        assert_eq!(second.entries, ModelRequirements::infer(&model).entries);

        variable!(model, z, Integer);
        let third = for_validation(&model);
        assert!(!Rc::ptr_eq(&second, &third));
        assert_eq!(third.entries, ModelRequirements::infer(&model).entries);

        objective!(model, Max, z.square());
        let fourth = for_validation(&model);
        assert!(!Rc::ptr_eq(&third, &fourth));
        assert_eq!(fourth.entries, ModelRequirements::infer(&model).entries);
    }

    #[test]
    fn cache_retention_is_bounded() {
        VALIDATION_REQUIREMENTS.with(|cache| cache.borrow_mut().clear());
        for i in 0..12 {
            let model = Model::new(format!("model {i}"));
            variable!(model, x);
            objective!(model, Min, x);
            for_validation(&model);
        }
        VALIDATION_REQUIREMENTS.with(|cache| {
            let cache = cache.borrow();
            assert!(cache.len() <= CACHE_MODELS);
            assert!(cache.iter().map(|entry| entry.bytes).sum::<usize>() <= CACHE_BYTES);
        });
    }

    #[test]
    fn shared_subgraphs_are_summarized_once_for_all_origins() {
        let model = Model::new("shared operators");

        variable!(model, x);
        let mut shared = x.erase();
        for _ in 0..1000 {
            shared = shared.sin().erase();
        }
        for i in 0..100 {
            model.add_constraint(format!("row {i}"), shared.clone().le(2.0));
        }
        objective!(model, Min, x);

        let arena = model.arena();
        let mut expressions = ExpressionRequirements::default();
        let mut requirements = ModelRequirements::default();

        for record in model.semantic_constraints_ref().iter() {
            let origin = RequirementOrigin {
                entity: RequirementEntity::Constraint(record.id),
                location: record.name.to_string(),
                declared_constraint: None,
            };
            expressions.collect(&arena, &[shared.id()], &origin, &mut requirements);
        }
        assert_eq!(expressions.analysis.cached_nodes(), 1000);
        assert_eq!(requirements.entries.len(), 1);
        assert_eq!(requirements.entries[0].origins.len(), 100);
    }

    #[test]
    fn custom_semantic_objects_are_reexamined_on_every_check() {
        use oximo_core::{ScalarFunctionData, SemanticFunction};

        #[derive(Clone, Debug)]
        struct CustomData(ScalarFunctionData<Affine>);
        impl SemanticFunction for CustomData {
            fn model_id(&self) -> oximo_core::ModelId {
                self.0.model_id()
            }
            fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
                self.0.visit_expressions(visitor);
            }
            fn remap(&mut self, remap: oximo_expr::ExprIdRemap) {
                self.0.remap(remap);
            }
            fn rebind_model(&mut self, id: oximo_core::ModelId) {
                self.0.rebind_model(id);
            }
            fn representation(&self) -> FunctionRepresentationRef<'_> {
                self.0.representation()
            }
        }
        struct Custom<'a>(Expr<'a, Affine>);
        impl Function for Custom<'_> {
            type Owned = CustomData;
            fn semantic_data(&self) -> CustomData {
                CustomData(ScalarFunctionData::from_expression(&self.0))
            }
        }
        impl FunctionInSet<LessThan> for Custom<'_> {}
        impl<'a> LowerConstraint<LessThan> for Custom<'a> {
            type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
            fn lower(self, set: LessThan) -> Self::Ir {
                self.0.le(set.0).into_ir()
            }
        }

        let model = Model::new("custom declaration");

        variable!(model, x);
        objective!(model, Min, x);
        model.add_constraint("custom", Constraint::new(Custom(x.into()), LessThan(2.0)));
        let first = for_validation(&model);
        let second = for_validation(&model);
        assert!(!Rc::ptr_eq(&first, &second));
        assert_eq!(first.entries, second.entries);
    }
}
