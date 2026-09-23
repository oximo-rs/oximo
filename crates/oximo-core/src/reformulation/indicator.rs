//! Explicit indicator-to-linear Big-M reformulation.

use std::cell::Ref;

use oximo_expr::{Expr, ExprId, ExprNode, extract_linear};
use smol_str::SmolStr;

use crate::constraint::{ConstraintId, Relate};
use crate::domain::Domain;
use crate::indicator::{IndicatorConstraint, IndicatorConstraintHandle, IndicatorConstraintId};
use crate::model::Model;
use crate::reformulation::sos::{ReformulatedModel, ReformulationError};
use crate::var::Variable;

/// Settings for explicit indicator-to-MILP reformulation.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct IndicatorReformulationOptions {
    fallback_big_m: Option<f64>,
}

impl IndicatorReformulationOptions {
    /// Use `big_m` as the row M only where finite variable bounds cannot derive one.
    #[must_use]
    pub const fn with_fallback_big_m(mut self, big_m: f64) -> Self {
        self.fallback_big_m = Some(big_m);
        self
    }

    #[must_use]
    pub const fn fallback_big_m(self) -> Option<f64> {
        self.fallback_big_m
    }
}

/// IDs appended while replacing one source indicator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndicatorReformulationArtifacts {
    pub source: IndicatorConstraintId,
    pub constraints: Vec<ConstraintId>,
}

#[derive(Clone, Debug)]
struct PlannedIndicator {
    source: IndicatorConstraintId,
    lower_m: Option<f64>,
    upper_m: Option<f64>,
}

#[derive(Debug)]
struct IndicatorPlan(Vec<PlannedIndicator>);

impl IndicatorPlan {
    fn one(
        model: &Model,
        id: IndicatorConstraintId,
        options: IndicatorReformulationOptions,
    ) -> Result<Self, ReformulationError> {
        validate_options(options)?;
        let constraints = model.indicator_constraints.borrow();
        let source = constraints
            .get(id.index())
            .ok_or(ReformulationError::UnknownIndicatorConstraint(id.index()))?;
        let entries =
            if source.active { vec![plan_one(model, id, source, options)?] } else { Vec::new() };
        Ok(Self(entries))
    }

    fn all(
        model: &Model,
        options: IndicatorReformulationOptions,
    ) -> Result<Self, ReformulationError> {
        validate_options(options)?;
        let constraints = model.indicator_constraints.borrow();
        let mut entries = Vec::new();
        for (index, source) in constraints.iter().enumerate() {
            if source.active {
                entries.push(plan_one(
                    model,
                    IndicatorConstraintId(u32::try_from(index).expect("indicator ID overflow")),
                    source,
                    options,
                )?);
            }
        }
        Ok(Self(entries))
    }

    fn apply(self, model: &Model) -> Vec<IndicatorReformulationArtifacts> {
        let mut artifacts = Vec::with_capacity(self.0.len());
        for planned in self.0 {
            artifacts.push(planned.apply(model));
        }
        if !artifacts.is_empty() {
            model.invalidate_kind();
        }
        model.indicator_reformulations.borrow_mut().extend(artifacts.iter().cloned());
        artifacts
    }
}

fn validate_options(options: IndicatorReformulationOptions) -> Result<(), ReformulationError> {
    if let Some(big_m) = options.fallback_big_m
        && (!big_m.is_finite() || big_m <= 0.0)
    {
        return Err(ReformulationError::InvalidFallbackBigM(big_m));
    }
    Ok(())
}

fn plan_one(
    model: &Model,
    id: IndicatorConstraintId,
    source: &IndicatorConstraint,
    options: IndicatorReformulationOptions,
) -> Result<PlannedIndicator, ReformulationError> {
    if source.lower == f64::INFINITY || source.upper == f64::NEG_INFINITY {
        return Err(ReformulationError::InvalidIndicatorBounds { constraint: source.name.clone() });
    }
    let arena = model.arena.borrow();
    if contains_parameter(&arena, source.lhs) {
        return Err(ReformulationError::ParameterDependentIndicator {
            constraint: source.name.clone(),
        });
    }
    let terms = extract_linear(&arena, source.lhs).ok_or_else(|| {
        ReformulationError::InvalidIndicatorExpression { constraint: source.name.clone() }
    })?;
    if !terms.constant.is_finite() || terms.coeffs.iter().any(|(_, c)| !c.is_finite()) {
        return Err(ReformulationError::InvalidIndicatorExpression {
            constraint: source.name.clone(),
        });
    }
    let variables = model.variables.borrow();
    let inactive = if source.active_value { 0.0 } else { 1.0 };
    let mut minimum = terms.constant;
    let mut maximum = terms.constant;
    for &(variable, coefficient) in terms.coeffs.iter() {
        if coefficient == 0.0 {
            continue;
        }
        let (lower, upper) = effective_bounds(&variables[variable.index()]);
        let (lower, upper) =
            if variable == source.trigger { (inactive, inactive) } else { (lower, upper) };
        let (min_bound, max_bound) =
            if coefficient >= 0.0 { (lower, upper) } else { (upper, lower) };
        minimum += coefficient * min_bound;
        maximum += coefficient * max_bound;
    }
    if minimum.is_nan() || maximum.is_nan() {
        return Err(ReformulationError::InvalidIndicatorExpression {
            constraint: source.name.clone(),
        });
    }
    let lower_m = if source.lower.is_finite() {
        Some(m_for_side(source.lower - minimum, source, "lower", options)?)
    } else {
        None
    };
    let upper_m = if source.upper.is_finite() {
        Some(m_for_side(maximum - source.upper, source, "upper", options)?)
    } else {
        None
    };
    Ok(PlannedIndicator { source: id, lower_m, upper_m })
}

fn m_for_side(
    required: f64,
    source: &IndicatorConstraint,
    side: &'static str,
    options: IndicatorReformulationOptions,
) -> Result<f64, ReformulationError> {
    if required.is_finite() {
        Ok(required.max(0.0))
    } else if let Some(big_m) = options.fallback_big_m {
        Ok(big_m)
    } else {
        Err(ReformulationError::MissingIndicatorBigM { constraint: source.name.clone(), side })
    }
}

fn effective_bounds(variable: &Variable) -> (f64, f64) {
    match variable.domain {
        Domain::SemiContinuous { threshold } | Domain::SemiInteger { threshold } => {
            (threshold.min(0.0), variable.ub.max(0.0))
        }
        Domain::Real | Domain::Integer | Domain::Binary => (variable.lb, variable.ub),
    }
}

fn contains_parameter(arena: &oximo_expr::ExprArena, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match arena.get(id) {
            ExprNode::Param(_) => return true,
            ExprNode::Add(children)
            | ExprNode::Mul(children)
            | ExprNode::Min(children)
            | ExprNode::Max(children) => stack.extend(children.iter().copied()),
            ExprNode::Unary(_, child) => stack.push(*child),
            ExprNode::Pow(left, right)
            | ExprNode::Div(left, right)
            | ExprNode::Atan2(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            ExprNode::Const(_) | ExprNode::Var(_) | ExprNode::Linear { .. } => {}
        }
    }
    false
}

impl PlannedIndicator {
    fn apply(self, model: &Model) -> IndicatorReformulationArtifacts {
        let source = model.indicator_constraints.borrow()[self.source.index()].clone();
        let lhs = Expr::new(source.lhs, &model.arena);
        let trigger = Expr::from_var(&model.arena, source.trigger);
        let inactive = if source.active_value { 1.0 - trigger } else { trigger };
        let mut generated = Vec::with_capacity(2);
        // A zero M still needs a row: the body can differ between trigger values.
        if let Some(big_m) = self.lower_m {
            let name = unique_constraint_name(
                model,
                &format!("__oximo_indicator{}_lower", self.source.index()),
            );
            generated
                .push(model.__add_constraint(name, lhs.ge(source.lower - big_m * inactive)).id());
        }
        if let Some(big_m) = self.upper_m {
            let name = unique_constraint_name(
                model,
                &format!("__oximo_indicator{}_upper", self.source.index()),
            );
            generated
                .push(model.__add_constraint(name, lhs.le(source.upper + big_m * inactive)).id());
        }
        model.indicator_constraints.borrow_mut()[self.source.index()].active = false;
        IndicatorReformulationArtifacts { source: self.source, constraints: generated }
    }
}

fn unique_constraint_name(model: &Model, base: &str) -> SmolStr {
    if model.constraint_id(base).is_none() {
        return base.into();
    }
    for suffix in 1_u64.. {
        let candidate = format!("{base}_{suffix}");
        if model.constraint_id(&candidate).is_none() {
            return candidate.into();
        }
    }
    unreachable!("u64 name suffix space exhausted")
}

impl IndicatorConstraintHandle<'_> {
    /// Clone the model and replace this active indicator with Big-M rows.
    ///
    /// # Errors
    /// Returns a reformulation error if the body, bounds, or options are invalid.
    pub fn to_reformulated_model(
        self,
        options: IndicatorReformulationOptions,
    ) -> Result<ReformulatedModel, ReformulationError> {
        self.model.to_reformulated_indicator_constraint_model(self.id, options)
    }

    /// Replace this indicator in place. Returns `None` when it is already inactive.
    ///
    /// # Errors
    /// Returns an error before mutation if the selected indicator is invalid.
    pub fn reformulate(
        self,
        options: IndicatorReformulationOptions,
    ) -> Result<Option<IndicatorReformulationArtifacts>, ReformulationError> {
        self.model.reformulate_indicator_constraint(self.id, options)
    }
}

impl Model {
    /// Clone the model and replace one indicator while preserving its source ID.
    ///
    /// # Errors
    /// Returns a reformulation error if the ID, body, bounds, or options are invalid.
    pub fn to_reformulated_indicator_constraint_model(
        &self,
        id: IndicatorConstraintId,
        options: IndicatorReformulationOptions,
    ) -> Result<ReformulatedModel, ReformulationError> {
        let plan = IndicatorPlan::one(self, id, options)?;
        let model = self.clone_preserving_ids_with_capacity(0, plan.0.len() * 2, plan.0.len() * 8);
        plan.apply(&model);
        Ok(ReformulatedModel { model })
    }

    /// Replace one indicator in place after validating its complete reformulation.
    /// Returns `None` if the source indicator is already inactive.
    ///
    /// # Errors
    /// Returns an error before mutation if the selected indicator is invalid.
    pub fn reformulate_indicator_constraint(
        &self,
        id: IndicatorConstraintId,
        options: IndicatorReformulationOptions,
    ) -> Result<Option<IndicatorReformulationArtifacts>, ReformulationError> {
        let plan = IndicatorPlan::one(self, id, options)?;
        Ok(plan.apply(self).pop())
    }

    /// Clone the model and replace every active indicator in ID order.
    ///
    /// # Errors
    /// Returns an error without producing a partial clone if any indicator is invalid.
    pub fn to_reformulated_indicator_model(
        &self,
        options: IndicatorReformulationOptions,
    ) -> Result<ReformulatedModel, ReformulationError> {
        let plan = IndicatorPlan::all(self, options)?;
        let model = self.clone_preserving_ids_with_capacity(0, plan.0.len() * 2, plan.0.len() * 8);
        plan.apply(&model);
        Ok(ReformulatedModel { model })
    }

    /// Replace every active indicator after validating the complete model plan.
    ///
    /// # Errors
    /// Returns an error without mutating the model if any indicator is invalid.
    pub fn reformulate_indicators(
        &self,
        options: IndicatorReformulationOptions,
    ) -> Result<Vec<IndicatorReformulationArtifacts>, ReformulationError> {
        let plan = IndicatorPlan::all(self, options)?;
        Ok(plan.apply(self))
    }

    /// Complete indicator reformulation history retained by this model.
    #[must_use]
    pub fn indicator_reformulations(&self) -> Ref<'_, [IndicatorReformulationArtifacts]> {
        Ref::map(self.indicator_reformulations.borrow(), Vec::as_slice)
    }
}
