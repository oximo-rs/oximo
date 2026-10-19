//! Solver capability queries and whole-model compatibility validation.
//!
//! Queries report support for constraint function/set pairs, objectives, and variable
//! domains. Validation checks their combinations and runs adapter-specific data checks.

use oximo_expr::ExprClass;

use oximo_core::{
    Affine, Domain, FunctionKind, FunctionRepresentationRef, Model, ModelKind, Nonlinear,
    ObjectiveSense, Quadratic, ScalarFunctionData, SemanticConstraintId, SemanticType,
    SymmetricAffineData, VectorAffineData,
};

use crate::Solver;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintSignature {
    pub function: SemanticType,
    pub set: SemanticType,
}

impl ConstraintSignature {
    /// Build a constraint signature from its function and set types.
    pub fn of<F: 'static, S: 'static>() -> Self {
        Self { function: SemanticType::of::<F>(), set: SemanticType::of::<S>() }
    }
}

/// A solver's support for a constraint signature, objective, or variable domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Capability {
    /// The adapter implements this interface.
    Native,

    /// This query found no supported interface.
    Unsupported,

    /// The signature alone is insufficient to determine support.
    RequiresValidation(&'static str),
}

impl Capability {
    /// Return no reason for native support, the supplied message for unsupported
    /// constructs, or the stored reason when further validation is required.
    fn rejection_reason(self, unsupported: &str) -> Option<String> {
        match self {
            Self::Native => None,
            Self::Unsupported => Some(unsupported.into()),
            Self::RequiresValidation(reason) => Some(reason.into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CapabilityDiagnostic {
    pub constraint: Option<SemanticConstraintId>,
    pub location: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default)]
pub struct ModelValidation {
    pub errors: Vec<CapabilityDiagnostic>,
}

impl ModelValidation {
    /// Return whether validation collected no compatibility errors.
    pub fn is_supported(&self) -> bool {
        self.errors.is_empty()
    }

    /// Consume the validation report, returning `Ok(())` if it contains no errors.
    ///
    /// # Errors
    /// Returns [`crate::SolverError::Backend`] with each diagnostic's location and
    /// reason joined by newlines when compatibility errors were collected.
    pub fn into_result(self) -> Result<(), crate::SolverError> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(crate::SolverError::Backend(
                self.errors
                    .into_iter()
                    .map(|d| format!("{}: {}", d.location, d.reason))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        }
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

/// Determine support for a function/set pair from the solver's advertised model
/// kinds and constraint features, flagging cases that need data checks.
pub(crate) fn query_constraint<S: Solver + ?Sized>(
    solver: &S,
    signature: ConstraintSignature,
) -> Capability {
    let kind = signature.function.function_kind();
    if scalar_set(signature.set) {
        if let Some(class) = scalar_class(signature.function) {
            let supported = match class {
                ExprClass::Linear => solver.supports(ModelKind::LP),
                ExprClass::Quadratic => {
                    solver.supports(ModelKind::QCP) || solver.supports(ModelKind::MIQCP)
                }
                ExprClass::Nonlinear => {
                    solver.supports(ModelKind::NLP) || solver.supports(ModelKind::MINLP)
                }
            };
            return if supported { Capability::Native } else { Capability::Unsupported };
        }

        if matches!(kind, FunctionKind::Scalar(None)) {
            return Capability::RequiresValidation(
                "dynamic function needs classification or narrowing",
            );
        }
    }

    if kind == FunctionKind::VectorAffine && signature.set.is::<oximo_core::SecondOrderCone>() {
        return if solver.supports(ModelKind::SOCP) || solver.supports(ModelKind::MISOCP) {
            Capability::Native
        } else {
            Capability::Unsupported
        };
    }

    if kind == FunctionKind::SymmetricAffine
        && signature.set.is::<oximo_core::PositiveSemidefiniteCone>()
    {
        return if solver.supports_psd() { Capability::Native } else { Capability::Unsupported };
    }

    if kind == FunctionKind::Indicator && signature.set.is::<oximo_core::IndicatorSet>() {
        return if solver.supports_indicators() {
            Capability::Native
        } else {
            Capability::Unsupported
        };
    }

    if kind == FunctionKind::Variables && signature.set.is::<oximo_core::SpecialOrderedSet>() {
        return if solver.supports_sos(oximo_core::SosType::Sos1)
            && solver.supports_sos(oximo_core::SosType::Sos2)
        {
            Capability::Native
        } else if solver.supports_sos(oximo_core::SosType::Sos1)
            || solver.supports_sos(oximo_core::SosType::Sos2)
        {
            Capability::RequiresValidation("SOS kind needs checking")
        } else {
            Capability::Unsupported
        };
    }
    Capability::Unsupported
}

/// Determine objective support from its scalar expression class and the solver's
/// advertised model kinds.
pub(crate) fn query_objective<S: Solver + ?Sized>(
    solver: &S,
    signature: SemanticType,
    _sense: ObjectiveSense,
) -> Capability {
    match scalar_class(signature) {
        Some(ExprClass::Linear) => {
            if solver.supports(ModelKind::LP) {
                Capability::Native
            } else {
                Capability::Unsupported
            }
        }

        Some(ExprClass::Quadratic) => {
            if solver.supports(ModelKind::QP) || solver.supports(ModelKind::MIQP) {
                Capability::Native
            } else {
                Capability::Unsupported
            }
        }

        Some(ExprClass::Nonlinear) => {
            if solver.supports(ModelKind::NLP) || solver.supports(ModelKind::MINLP) {
                Capability::Native
            } else {
                Capability::Unsupported
            }
        }
        None => Capability::RequiresValidation("objective has no supported scalar representation"),
    }
}

/// Determine variable-domain support from the solver's advertised model kinds,
/// leaving semicontinuous and semi-integer domains to backend-specific validation.
pub(crate) fn query_domain<S: Solver + ?Sized>(solver: &S, domain: Domain) -> Capability {
    match domain {
        Domain::Real => Capability::Native,
        Domain::Integer | Domain::Binary => {
            if solver.supports(ModelKind::MILP)
                || solver.supports(ModelKind::MINLP)
                || solver.supports(ModelKind::MIQP)
                || solver.supports(ModelKind::MIQCP)
                || solver.supports(ModelKind::MISOCP)
                || solver.supports(ModelKind::MISDP)
            {
                Capability::Native
            } else {
                Capability::Unsupported
            }
        }
        Domain::SemiContinuous { .. } | Domain::SemiInteger { .. } => {
            Capability::RequiresValidation(
                "semicontinuous domains need backend-specific validation",
            )
        }
    }
}

/// Validate structural interfaces and combinations, then run the adapter's data checks.
/// Numerical validity, termination and solution availability are not predicted here.
pub(crate) fn validate<S: Solver + ?Sized>(
    solver: &S,
    model: &Model,
    opts: &S::Options,
) -> ModelValidation {
    let mut report = ModelValidation::default();
    let mut reject = |constraint, location: String, reason: String| {
        report.errors.push(CapabilityDiagnostic { constraint, location, reason });
    };

    if model.ensure_objective_declared().is_err() {
        reject(None, "model".into(), "no objective or feasibility direction declared".into());
    }

    if model.has_unreformulated_gdp() {
        reject(None, "model".into(), "GDP requires reformulation".into());
    }

    let semantic = model.semantic_constraints_ref();
    let arena = model.arena();

    for c in semantic.iter().filter(|c| c.active) {
        if let Some(reason) = support_for_record(solver, &c, model, &arena, opts)
            .rejection_reason("unsupported function/set pair")
        {
            reject(
                Some(c.id),
                format!("constraint {:?}", c.name),
                format!(
                    "{reason}; declared {} in {}",
                    c.pair.function.signature().name(),
                    c.pair.set.signature().name()
                ),
            );
        }
    }

    if let Some(objective) = model.semantic_objective().as_ref() {
        let declared =
            solver.objective_support(objective.function.signature(), objective.sense, opts);
        let support = if matches!(declared, Capability::Native) {
            declared
        } else if let Some(root) = match objective.function.representation() {
            FunctionRepresentationRef::Scalar(root) => Some(root),
            _ => None,
        } {
            solver.objective_support(
                signature_for_class(oximo_expr::classify(&model.arena(), root)),
                objective.sense,
                opts,
            )
        } else {
            declared
        };

        if let Some(reason) = support.rejection_reason("unsupported objective function and sense") {
            reject(None, "objective".into(), reason);
        }
    }

    for variable in &model.resolved_variables() {
        if let Some(reason) =
            solver.domain_support(variable.domain).rejection_reason("unsupported domain")
        {
            reject(None, format!("variable {:?}", variable.name), reason);
        }
    }

    if !solver.supports(model.kind()) {
        reject(
            None,
            "model".into(),
            format!(
                "unsupported combination of domains, objective and constraints ({})",
                model.kind()
            ),
        );
    }

    if !solver.supports_model_extra(model) {
        reject(None, "model".into(), "backend-specific structural or data restriction".into());
    }

    report.errors.extend(solver.validate_model_extra(model, opts));

    report
}

/// Map an expression class to the corresponding built-in scalar function signature.
fn signature_for_class(class: ExprClass) -> SemanticType {
    match class {
        ExprClass::Linear => SemanticType::of::<ScalarFunctionData<Affine>>(),
        ExprClass::Quadratic => SemanticType::of::<ScalarFunctionData<Quadratic>>(),
        ExprClass::Nonlinear => SemanticType::of::<ScalarFunctionData<Nonlinear>>(),
    }
}

/// Determine support for a constraint using its selected alternative lowering,
/// declared signature, or canonical representation.
pub(crate) fn support_for_record<S: Solver + ?Sized>(
    solver: &S,
    record: &oximo_core::SemanticConstraintRef<'_>,
    model: &Model,
    arena: &oximo_expr::ExprArena,
    opts: &S::Options,
) -> Capability {
    if let Some(view) = model.alternative_lowering_view(record.id) {
        let signature = match record.id {
            SemanticConstraintId::Algebraic(_) => ConstraintSignature {
                function: signature_for_class(oximo_expr::classify(arena, view.lhs)),
                set: native_scalar_set(view),
            },
            SemanticConstraintId::Indicator(_) => {
                ConstraintSignature::of::<oximo_core::IndicatorFunctionData, oximo_core::IndicatorSet>(
                )
            }
            _ => unreachable!("alternative scalar views belong to rows or indicator consequents"),
        };
        return support_for_native_signature(solver, signature, record.id, model, arena, opts);
    }

    let mut signature = ConstraintSignature {
        function: record.pair.function.signature(),
        set: record.pair.set.signature(),
    };

    let declared = solver.constraint_support(signature, opts);

    if matches!(declared, Capability::Native) {
        return declared;
    }

    match record.pair.function.representation() {
        FunctionRepresentationRef::Scalar(root) => {
            signature.function = signature_for_class(oximo_expr::classify(arena, root));
        }
        FunctionRepresentationRef::VectorAffine(_) => {
            signature.function = SemanticType::of::<VectorAffineData>();
        }
        FunctionRepresentationRef::SymmetricAffine(_) => {
            signature.function = SemanticType::of::<SymmetricAffineData>();
        }
        FunctionRepresentationRef::Variables(_) => {
            if let Some(kind) = record.pair.set.sos_kind() {
                return if solver.supports_sos(kind) {
                    Capability::Native
                } else {
                    Capability::Unsupported
                };
            }
            return declared;
        }
        FunctionRepresentationRef::Indicator { .. } => {
            signature.function = SemanticType::of::<oximo_core::IndicatorFunctionData>();
        }
        _ => return declared,
    }

    support_for_native_signature(solver, signature, record.id, model, arena, opts)
}

/// Recover the canonical bound set of a native scalar interval row.
#[expect(clippy::float_cmp, reason = "native equality rows use exact matching bound values")]
fn native_scalar_set(view: oximo_core::ScalarConstraintView) -> SemanticType {
    if view.lower == view.upper {
        SemanticType::of::<oximo_core::EqualTo>()
    } else if view.lower == f64::NEG_INFINITY && view.upper != f64::INFINITY {
        SemanticType::of::<oximo_core::LessThan>()
    } else if view.upper == f64::INFINITY && view.lower != f64::NEG_INFINITY {
        SemanticType::of::<oximo_core::GreaterThan>()
    } else {
        SemanticType::of::<oximo_core::Interval>()
    }
}

/// Query the selected native pair, including supported quadratic-to-SOC detection.
fn support_for_native_signature<S: Solver + ?Sized>(
    solver: &S,
    signature: ConstraintSignature,
    id: SemanticConstraintId,
    model: &Model,
    arena: &oximo_expr::ExprArena,
    opts: &S::Options,
) -> Capability {
    let narrowed = solver.constraint_support(signature, opts);

    if !matches!(narrowed, Capability::Native)
        && scalar_set(signature.set)
        && matches!(
            solver.constraint_support(
                ConstraintSignature::of::<VectorAffineData, oximo_core::SecondOrderCone>(),
                opts
            ),
            Capability::Native
        )
        && let SemanticConstraintId::Algebraic(id) = id
        && let Some(row) = model.constraints().algebraic().get(id.index())
        && let Some(quadratic) = oximo_expr::extract_quadratic(arena, row.lhs)
        && oximo_core::__detect_soc_from_quadratic(&model.resolved_variables(), row, &quadratic)
            .is_some()
    {
        return Capability::Native;
    }
    narrowed
}
