//! Declared solver profiles and accumulated compatibility reports.

use crate::{
    ExpressionOperator, ModelFeatures, ModelRequirements, Requirement, RequirementEntity,
    RequirementOrigin, VariableDomain,
};
use oximo_core::{
    Affine, Domain, FunctionKind, ModelKind, Nonlinear, ObjectiveSense, Quadratic,
    ScalarFunctionData, SemanticConstraintId, SemanticType, SosType,
};
use oximo_expr::ExprClass;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintSignature {
    pub function: SemanticType,
    pub set: SemanticType,
}
impl ConstraintSignature {
    pub fn of<F: 'static, S: 'static>() -> Self {
        Self { function: SemanticType::of::<F>(), set: SemanticType::of::<S>() }
    }
}

/// External uncertainty is explicit and does not imply confirmed compatibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequirementSupport {
    Supported,
    Unsupported,
    Conditional(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequirementAlternative {
    pub requirement: Requirement,
    pub support: RequirementSupport,
}

#[derive(Clone, Debug)]
pub struct RequirementDiagnostic {
    /// Primary failure, or the alternative supplying a conditional acceptance.
    pub requirement: Requirement,
    pub origins: Vec<RequirementOrigin>,
    /// Every considered representation and its resolved support result.
    pub alternatives: Vec<RequirementAlternative>,
    pub reason: String,
}
impl RequirementDiagnostic {
    pub fn new(
        requirement: Requirement,
        origins: Vec<RequirementOrigin>,
        reason: impl Into<String>,
    ) -> Self {
        let alternatives = vec![RequirementAlternative {
            requirement: requirement.clone(),
            support: RequirementSupport::Unsupported,
        }];
        Self { requirement, origins, alternatives, reason: reason.into() }
    }

    /// First constraint origin, when one exists.
    pub fn constraint(&self) -> Option<SemanticConstraintId> {
        self.origins.iter().find_map(|origin| match origin.entity {
            RequirementEntity::Constraint(id) => Some(id),
            _ => None,
        })
    }

    /// All captured source locations, in declaration order.
    pub fn location(&self) -> String {
        self.origins.iter().map(|origin| origin.location.as_str()).collect::<Vec<_>>().join(", ")
    }

    pub fn adapter(location: &'static str, reason: impl Into<String>) -> Self {
        Self::new(
            Requirement::AdapterRestriction(location),
            vec![RequirementOrigin {
                entity: RequirementEntity::Model,
                location: location.into(),
                declared_constraint: None,
            }],
            reason,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompatibilityReport {
    pub solver: String,
    pub errors: Vec<RequirementDiagnostic>,
    pub conditions: Vec<RequirementDiagnostic>,
}
impl CompatibilityReport {
    /// Whether a solve attempt is allowed; external uncertainty may remain.
    pub fn is_supported(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn is_confirmed(&self) -> bool {
        self.errors.is_empty() && self.conditions.is_empty()
    }

    /// # Errors
    /// Returns the complete typed compatibility report when requirements are unmet.
    pub fn into_result(self) -> Result<(), crate::SolverError> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(crate::SolverError::IncompatibleModel(Box::new(self)))
        }
    }
}
impl std::fmt::Display for CompatibilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} model compatibility", self.solver)?;
        for diagnostic in &self.errors {
            write!(
                f,
                "\n{}: unmet {}: {}",
                diagnostic.location(),
                diagnostic.requirement,
                diagnostic.reason
            )?;
            display_alternatives(f, diagnostic)?;
        }
        for diagnostic in &self.conditions {
            write!(
                f,
                "\n{}: conditional {}: {}",
                diagnostic.location(),
                diagnostic.requirement,
                diagnostic.reason
            )?;
            display_alternatives(f, diagnostic)?;
        }
        Ok(())
    }
}

fn display_alternatives(
    f: &mut std::fmt::Formatter<'_>,
    diagnostic: &RequirementDiagnostic,
) -> std::fmt::Result {
    if diagnostic.alternatives.len() > 1 {
        for alternative in &diagnostic.alternatives {
            write!(f, "\n  alternative {}: ", alternative.requirement)?;
            match alternative.support {
                RequirementSupport::Supported => f.write_str("supported")?,
                RequirementSupport::Unsupported => f.write_str("unsupported")?,
                RequirementSupport::Conditional(reason) => write!(f, "conditional: {reason}")?,
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CombinationRule {
    rejects: fn(&ModelFeatures) -> bool,
    reason: &'static str,
}

/// A self-contained declaration for one backend configuration.
/// Unlisted domains, operators, and interfaces are unsupported.
#[derive(Clone, Debug)]
pub struct SolverProfile {
    pub name: String,
    kinds: Vec<ModelKind>,
    scalar_constraints: Vec<ExprClass>,
    scalar_objectives: Vec<ExprClass>,
    domains: Vec<VariableDomain>,
    sos: Vec<SosType>,
    indicators: bool,
    soc: bool,
    psd: bool,
    operators: Vec<ExpressionOperator>,
    overrides: Vec<(Requirement, RequirementSupport)>,
    rules: Vec<CombinationRule>,
    uncertainty: Option<&'static str>,
}

/// All summary kinds, useful when declaring a profile from an adapter table.
pub const MODEL_KINDS: [ModelKind; 12] = [
    ModelKind::LP,
    ModelKind::MILP,
    ModelKind::QP,
    ModelKind::MIQP,
    ModelKind::QCP,
    ModelKind::MIQCP,
    ModelKind::SOCP,
    ModelKind::MISOCP,
    ModelKind::SDP,
    ModelKind::MISDP,
    ModelKind::NLP,
    ModelKind::MINLP,
];

impl SolverProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kinds: Vec::new(),
            scalar_constraints: Vec::new(),
            scalar_objectives: Vec::new(),
            domains: Vec::new(),
            sos: Vec::new(),
            indicators: false,
            soc: false,
            psd: false,
            operators: Vec::new(),
            overrides: Vec::new(),
            rules: Vec::new(),
            uncertainty: None,
        }
    }

    /// Declare supported whole-model combinations.
    #[must_use]
    pub fn with_kinds(mut self, kinds: impl IntoIterator<Item = ModelKind>) -> Self {
        self.kinds.extend(kinds);
        self
    }

    /// Enable scalar inequalities, equalities, and intervals of these classes.
    #[must_use]
    pub fn with_scalar_constraints(mut self, classes: impl IntoIterator<Item = ExprClass>) -> Self {
        self.scalar_constraints.extend(classes);
        self
    }

    /// Enable scalar objectives of these classes, for both objective senses.
    /// Use `with_support` for an exact type/sense restriction.
    #[must_use]
    pub fn with_scalar_objectives(mut self, classes: impl IntoIterator<Item = ExprClass>) -> Self {
        self.scalar_objectives.extend(classes);
        self
    }

    #[must_use]
    pub fn with_soc(mut self, enabled: bool) -> Self {
        self.soc = enabled;
        self
    }

    #[must_use]
    pub fn with_domains(mut self, domains: impl IntoIterator<Item = VariableDomain>) -> Self {
        self.domains.extend(domains);
        self
    }

    #[must_use]
    pub fn with_sos(mut self, kinds: impl IntoIterator<Item = SosType>) -> Self {
        self.sos.extend(kinds);
        self
    }

    #[must_use]
    pub fn with_indicators(mut self, enabled: bool) -> Self {
        self.indicators = enabled;
        self
    }

    #[must_use]
    pub fn with_psd(mut self, enabled: bool) -> Self {
        self.psd = enabled;
        self
    }

    #[must_use]
    pub fn with_operators(
        mut self,
        operators: impl IntoIterator<Item = ExpressionOperator>,
    ) -> Self {
        self.operators.extend(operators);
        self
    }

    /// Override an individual interface, independently of whole-model combinations.
    #[must_use]
    pub fn with_support(mut self, requirement: Requirement, support: RequirementSupport) -> Self {
        self.overrides.push((requirement, support));
        self
    }

    #[must_use]
    pub fn reject_combination(
        mut self,
        rejects: fn(&ModelFeatures) -> bool,
        reason: &'static str,
    ) -> Self {
        self.rules.push(CombinationRule { rejects, reason });
        self
    }

    #[must_use]
    pub fn with_external_uncertainty(mut self, reason: &'static str) -> Self {
        self.uncertainty = Some(reason);
        self
    }

    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn supports_kind(&self, kind: ModelKind) -> bool {
        self.kinds.contains(&kind)
    }

    /// Whether support for this SOS kind is confirmed, after overrides.
    /// Use `sos_support` to retain conditional support details.
    pub fn supports_sos(&self, kind: SosType) -> bool {
        matches!(self.sos_support(kind), RequirementSupport::Supported)
    }

    /// Resolve kind-specific SOS support, including overrides and conditions.
    pub fn sos_support(&self, kind: SosType) -> RequirementSupport {
        self.support(&Requirement::Sos(kind))
    }

    pub fn constraint_support(&self, signature: ConstraintSignature) -> RequirementSupport {
        self.support(&Requirement::Constraint(signature))
    }

    pub fn objective_support(
        &self,
        function: SemanticType,
        sense: ObjectiveSense,
    ) -> RequirementSupport {
        self.support(&Requirement::Objective { function, sense })
    }

    pub fn domain_support(&self, domain: Domain) -> RequirementSupport {
        self.support(&Requirement::Domain(domain.into()))
    }

    pub fn support(&self, requirement: &Requirement) -> RequirementSupport {
        if let Some((_, support)) = self.overrides.iter().rev().find(|(r, _)| r == requirement) {
            return support.clone();
        }

        let supported = match requirement {
            Requirement::ObjectiveDeclared
            | Requirement::GdpReformulated
            | Requirement::AdapterRestriction(_) => false,
            Requirement::Domain(domain) => self.domains.contains(domain),
            Requirement::Constraint(signature) => return query_constraint(self, *signature),
            Requirement::Objective { function, sense } => {
                return query_objective(self, *function, *sense);
            }
            Requirement::Sos(kind) => self.sos.contains(kind),
            Requirement::Operator(operator) => self.operators.contains(operator),
            Requirement::Combination(features) => {
                self.supports_kind(features.kind)
                    && !self.rules.iter().any(|rule| (rule.rejects)(features))
            }
        };
        if supported {
            if let Some(reason) =
                self.uncertainty.filter(|_| matches!(requirement, Requirement::Combination(_)))
            {
                return RequirementSupport::Conditional(reason);
            }
            RequirementSupport::Supported
        } else {
            RequirementSupport::Unsupported
        }
    }

    /// Check the current model's structural requirements, reusing only revision-valid summaries.
    /// Local numeric and adapter route checks belong to `Solver::validate_model`.
    pub fn check_model(&self, model: &oximo_core::Model) -> CompatibilityReport {
        let requirements = crate::requirements::for_matching(model);
        self.check_with_origins(model, &requirements)
    }

    /// Match the same requirements and retain the same diagnostic origins using
    /// a preparation's captured numeric analysis. Reject stale or foreign views.
    ///
    /// # Errors
    /// Returns an error when the preparation no longer describes this model.
    #[doc(hidden)]
    pub fn check_prepared(
        &self,
        model: &oximo_core::Model,
        prepared: &crate::prepare::LoweringContext<'_>,
    ) -> Result<CompatibilityReport, crate::SolverError> {
        prepared.ensure_matches(model)?;
        let requirements = crate::requirements::for_prepared(model, prepared, false);
        let report = self.check_with_prepared_origins(model, &requirements, Some(prepared));
        prepared.ensure_matches(model)?;
        Ok(report)
    }

    pub(crate) fn check_with_origins(
        &self,
        model: &oximo_core::Model,
        requirements: &ModelRequirements,
    ) -> CompatibilityReport {
        self.check_with_prepared_origins(model, requirements, None)
    }

    /// Check requirements and ensure errors and conditions identify their model sources.
    ///
    /// Return the initial report when every diagnostic already has origins with
    /// nonempty locations. Otherwise, rebuild requirements with source names and
    /// check again, reusing captured analysis from `prepared` when supplied or
    /// reading the current model when it is absent.
    fn check_with_prepared_origins(
        &self,
        model: &oximo_core::Model,
        requirements: &ModelRequirements,
        prepared: Option<&crate::prepare::LoweringContext<'_>>,
    ) -> CompatibilityReport {
        let report = self.check(requirements);
        if report.errors.iter().chain(&report.conditions).all(|diagnostic| {
            !diagnostic.origins.is_empty()
                && diagnostic.origins.iter().all(|origin| !origin.location.is_empty())
        }) {
            report
        } else {
            // Materialize source names only when we need them.
            self.check(&prepared.map_or_else(
                || crate::requirements::for_validation(model),
                |p| crate::requirements::for_prepared(model, p, true),
            ))
        }
    }

    pub fn check(&self, requirements: &ModelRequirements) -> CompatibilityReport {
        let mut report =
            CompatibilityReport { solver: self.name.clone(), ..CompatibilityReport::default() };

        for entry in &requirements.entries {
            let mut condition = None;
            let mut supports = smallvec::SmallVec::<[RequirementSupport; 3]>::new();
            if entry.alternatives.iter().enumerate().any(|(index, r)| {
                let support = self.support(r);
                supports.push(support.clone());
                match support {
                    RequirementSupport::Supported => true,
                    RequirementSupport::Conditional(reason) => {
                        condition = Some((index, reason));
                        false
                    }
                    RequirementSupport::Unsupported => false,
                }
            }) {
                continue;
            }
            let alternatives = entry
                .alternatives
                .iter()
                .cloned()
                .zip(supports)
                .map(|(requirement, support)| RequirementAlternative { requirement, support })
                .collect();

            if let Some((index, reason)) = condition {
                report.conditions.push(RequirementDiagnostic {
                    requirement: entry.alternatives[index].clone(),
                    origins: entry.origins.clone(),
                    alternatives,
                    reason: reason.into(),
                });
                continue;
            }
            let Some(requirement) = entry.alternatives.first() else {
                report.errors.push(RequirementDiagnostic::adapter(
                    "requirement alternatives",
                    "no representation alternatives supplied",
                ));
                continue;
            };

            let reason = match requirement {
                Requirement::ObjectiveDeclared => "no objective or feasibility direction declared".into(),
                Requirement::GdpReformulated => "GDP requires explicit reformulation".into(),
                Requirement::Constraint(signature) if signature.set.is::<oximo_core::IndicatorSet>() => "explicitly use Model::reformulate_indicators or Model::to_reformulated_indicator_model".into(),
                Requirement::Constraint(_) => "unsupported function/set pair".into(),
                Requirement::Combination(features) => self.rules.iter().filter(|rule| (rule.rejects)(features)).map(|rule| rule.reason).collect::<Vec<_>>().join("; "),
                _ => "unsupported by this solver configuration".into(),
            };
            report.errors.push(RequirementDiagnostic {
                requirement: requirement.clone(),
                origins: entry.origins.clone(),
                alternatives,
                reason: if reason.is_empty() {
                    "unsupported model combination".into()
                } else {
                    reason
                },
            });
        }
        report
    }
}

pub(crate) fn scalar_signature(class: ExprClass) -> SemanticType {
    match class {
        ExprClass::Linear => SemanticType::of::<ScalarFunctionData<Affine>>(),
        ExprClass::Quadratic => SemanticType::of::<ScalarFunctionData<Quadratic>>(),
        ExprClass::Nonlinear => SemanticType::of::<ScalarFunctionData<Nonlinear>>(),
    }
}

/// Return a scalar function's expression class, or `None` if it is unclassified
/// or the signature describes another function kind.
pub(crate) fn scalar_class(signature: SemanticType) -> Option<ExprClass> {
    match signature.function_kind() {
        FunctionKind::Scalar(class) => class,
        _ => None,
    }
}

/// Check whether the signature is a scalar inequality, equality, or interval set.
pub(crate) fn scalar_set(signature: SemanticType) -> bool {
    signature.is::<oximo_core::LessThan>()
        || signature.is::<oximo_core::GreaterThan>()
        || signature.is::<oximo_core::EqualTo>()
        || signature.is::<oximo_core::Interval>()
}

/// Determine support for a function/set pair independently of model combinations.
fn query_constraint(solver: &SolverProfile, signature: ConstraintSignature) -> RequirementSupport {
    let kind = signature.function.function_kind();
    if scalar_set(signature.set) {
        if let Some(class) = scalar_class(signature.function) {
            let supported = solver.scalar_constraints.contains(&class);
            return if supported {
                RequirementSupport::Supported
            } else {
                RequirementSupport::Unsupported
            };
        }

        if matches!(kind, FunctionKind::Scalar(None)) {
            return RequirementSupport::Unsupported;
        }
    }

    if kind == FunctionKind::VectorAffine && signature.set.is::<oximo_core::SecondOrderCone>() {
        return if solver.soc {
            RequirementSupport::Supported
        } else {
            RequirementSupport::Unsupported
        };
    }

    if kind == FunctionKind::SymmetricAffine
        && signature.set.is::<oximo_core::PositiveSemidefiniteCone>()
    {
        return if solver.psd {
            RequirementSupport::Supported
        } else {
            RequirementSupport::Unsupported
        };
    }

    if kind == FunctionKind::Indicator && signature.set.is::<oximo_core::IndicatorSet>() {
        return if solver.indicators {
            RequirementSupport::Supported
        } else {
            RequirementSupport::Unsupported
        };
    }

    if kind == FunctionKind::Variables && signature.set.is::<oximo_core::SpecialOrderedSet>() {
        // An unqualified SOS signature promises both kinds.
        // A concrete model instead requires its resolved kind.
        return match (solver.sos_support(SosType::Sos1), solver.sos_support(SosType::Sos2)) {
            (RequirementSupport::Unsupported, _) | (_, RequirementSupport::Unsupported) => {
                RequirementSupport::Unsupported
            }
            (RequirementSupport::Supported, RequirementSupport::Supported) => {
                RequirementSupport::Supported
            }
            (RequirementSupport::Conditional(a), RequirementSupport::Conditional(b)) if a != b => {
                RequirementSupport::Conditional(
                    "SOS1 and SOS2 have separate support conditions; query sos_support for each kind",
                )
            }
            (RequirementSupport::Conditional(reason), _)
            | (_, RequirementSupport::Conditional(reason)) => {
                RequirementSupport::Conditional(reason)
            }
        };
    }
    RequirementSupport::Unsupported
}

/// Determine objective support independently of model combinations.
fn query_objective(
    solver: &SolverProfile,
    signature: SemanticType,
    _sense: ObjectiveSense,
) -> RequirementSupport {
    match scalar_class(signature) {
        Some(class) => {
            if solver.scalar_objectives.contains(&class) {
                RequirementSupport::Supported
            } else {
                RequirementSupport::Unsupported
            }
        }
        None => RequirementSupport::Unsupported,
    }
}
