//! SOS-to-MILP reformulation implementation.

use std::cell::Ref;
use std::ops::Deref;

use oximo_expr::{Expr, VarId};
use smol_str::SmolStr;
use thiserror::Error;

use crate::constraint::{ConstraintId, Relate};
use crate::domain::Domain;
use crate::model::Model;
use crate::sos::{SosConstraint, SosConstraintId, SosMember, SosType};
use crate::var::Variable;

/// Settings for an SOS-to-MILP reformulation.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SosReformulationOptions {
    fallback_big_m: Option<f64>,
}

impl SosReformulationOptions {
    /// Use `big_m` for an SOS member side whose variable bound is infinite.
    /// Finite variable bounds are always preferred.
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

/// Why an explicit model reformulation could not be constructed.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReformulationError {
    #[error("SOS constraint #{0} does not exist on this model")]
    UnknownSosConstraint(usize),
    #[error("fallback Big-M must be finite and positive, got {0}")]
    InvalidFallbackBigM(f64),
    #[error(
        "cannot reformulate SOS constraint {constraint:?}: variable {variable:?} has no finite \
         {side} bound; provide SosReformulationOptions::with_fallback_big_m(...)"
    )]
    MissingFiniteBound { constraint: SmolStr, variable: SmolStr, side: &'static str },
}

/// IDs appended while replacing one source SOS constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SosReformulationArtifacts {
    pub source: SosConstraintId,
    pub variables: Vec<VarId>,
    pub constraints: Vec<ConstraintId>,
}

/// An independent transformed model plus source-to-generated provenance.
#[derive(Debug)]
pub struct ReformulatedModel {
    model: Model,
}

impl ReformulatedModel {
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Complete SOS reformulation history carried by this transformed model.
    /// The returned guard dereferences to a slice of artifacts.
    #[must_use]
    pub fn sos_reformulations(&self) -> Ref<'_, [SosReformulationArtifacts]> {
        // The history lives on the model so a later clone-and-reformulate
        // operation maintains traceability from earlier transformations.
        Ref::map(self.model.sos_reformulations.borrow(), Vec::as_slice)
    }

    #[must_use]
    pub fn into_model(self) -> Model {
        self.model
    }
}

impl Deref for ReformulatedModel {
    type Target = Model;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl AsRef<Model> for ReformulatedModel {
    fn as_ref(&self) -> &Model {
        &self.model
    }
}

#[derive(Copy, Clone, Debug)]
struct PlannedMember {
    member_index: usize,
    member: SosMember,
    lower: f64,
    upper: f64,
}

impl PlannedMember {
    fn gate_count(self) -> usize {
        usize::from(self.lower < 0.0) + usize::from(self.upper > 0.0)
    }
}

#[derive(Debug)]
enum PlannedSosForm {
    Trivial,
    Sos1 { members: Vec<PlannedMember> },
    Sos2 { members: Vec<PlannedMember> },
}

#[derive(Debug)]
struct PlannedSos {
    source: SosConstraintId,
    form: PlannedSosForm,
}

#[derive(Debug)]
struct SosReformulationPlan {
    entries: Vec<PlannedSos>,
    additional_variables: usize,
    additional_constraints: usize,
    additional_expr_nodes: usize,
}

impl Model {
    /// Clone this model and replace one active SOS constraint with an MILP
    /// formulation. An already-inactive constraint is copied unchanged. The
    /// source model remains mutable and unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ReformulationError`] for an unknown ID, invalid fallback
    /// Big-M, or an unbounded member side without a fallback.
    pub fn to_reformulated_sos_constraint_model(
        &self,
        id: SosConstraintId,
        options: SosReformulationOptions,
    ) -> Result<ReformulatedModel, ReformulationError> {
        let plan = SosReformulationPlan::for_one(self, id, options)?;
        let model = self.clone_preserving_ids_with_capacity(
            plan.additional_variables,
            plan.additional_constraints,
            plan.additional_expr_nodes,
        );
        plan.apply(&model);
        Ok(ReformulatedModel { model })
    }

    /// Replace one active SOS constraint on this model without cloning it.
    /// Existing variable and parameter expression handles remain valid because
    /// registries are only appended and all existing IDs are preserved.
    ///
    /// Returns `Ok(None)` when the source SOS is already inactive.
    /// Bounds of all source members become immutable after reformulation because
    /// their current values are embedded in the generated Big-M rows.
    ///
    /// # Errors
    ///
    /// Returns [`ReformulationError`] before modifying the model when the ID,
    /// fallback Big-M, or required member bounds are invalid.
    pub fn reformulate_sos_constraint(
        &self,
        id: SosConstraintId,
        options: SosReformulationOptions,
    ) -> Result<Option<SosReformulationArtifacts>, ReformulationError> {
        let plan = SosReformulationPlan::for_one(self, id, options)?;
        Ok(plan.apply(self).pop())
    }

    /// Clone this model and replace every active SOS constraint with an MILP
    /// formulation, in stable [`SosConstraintId`] order. The source model
    /// remains mutable and unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ReformulationError`] when the fallback Big-M is invalid or an
    /// active member has an unbounded side without a fallback. No partial
    /// transformed model is returned.
    pub fn to_reformulated_sos_model(
        &self,
        options: SosReformulationOptions,
    ) -> Result<ReformulatedModel, ReformulationError> {
        let plan = SosReformulationPlan::for_all(self, options)?;
        let model = self.clone_preserving_ids_with_capacity(
            plan.additional_variables,
            plan.additional_constraints,
            plan.additional_expr_nodes,
        );
        plan.apply(&model);
        Ok(ReformulatedModel { model })
    }

    /// Replace every active SOS constraint on this model without cloning it.
    /// This is the memory-efficient alternative to
    /// [`Self::to_reformulated_sos_model`]
    /// when the original native-SOS form is no longer needed.
    ///
    /// # Errors
    ///
    /// Returns [`ReformulationError`] before modifying the model when the
    /// fallback Big-M or any required member bound is invalid.
    ///
    /// Reformulation history remains available through
    /// [`Self::sos_reformulations`] after the returned artifacts are dropped.
    pub fn reformulate_sos(
        &self,
        options: SosReformulationOptions,
    ) -> Result<Vec<SosReformulationArtifacts>, ReformulationError> {
        let plan = SosReformulationPlan::for_all(self, options)?;
        Ok(plan.apply(self))
    }

    /// Complete SOS reformulation history accumulated by this model.
    ///
    /// The returned guard dereferences to a slice of artifacts. This is useful
    /// after an in-place reformulation when the `Vec` returned by
    /// [`Self::reformulate_sos`] is no longer available.
    #[must_use]
    pub fn sos_reformulations(&self) -> Ref<'_, [SosReformulationArtifacts]> {
        Ref::map(self.sos_reformulations.borrow(), Vec::as_slice)
    }
}

fn validate_options(options: SosReformulationOptions) -> Result<(), ReformulationError> {
    if let Some(big_m) = options.fallback_big_m
        && (!big_m.is_finite() || big_m <= 0.0)
    {
        return Err(ReformulationError::InvalidFallbackBigM(big_m));
    }
    Ok(())
}

impl SosReformulationPlan {
    fn for_one(
        model: &Model,
        id: SosConstraintId,
        options: SosReformulationOptions,
    ) -> Result<Self, ReformulationError> {
        validate_options(options)?;
        let variables = model.variables.borrow();
        let constraints = model.sos_constraints.borrow();
        let source = constraints
            .get(id.index())
            .ok_or(ReformulationError::UnknownSosConstraint(id.index()))?;
        let entries = if source.active {
            vec![plan_one(id, source, &variables, options)?]
        } else {
            Vec::new()
        };
        Ok(Self::new(entries))
    }

    fn for_all(
        model: &Model,
        options: SosReformulationOptions,
    ) -> Result<Self, ReformulationError> {
        validate_options(options)?;
        let variables = model.variables.borrow();
        let constraints = model.sos_constraints.borrow();
        let mut entries =
            Vec::with_capacity(constraints.iter().filter(|constraint| constraint.active).count());
        for (index, source) in constraints.iter().enumerate() {
            if !source.active {
                continue;
            }
            let id = SosConstraintId(
                u32::try_from(index).expect("SOS registration guarantees IDs fit in u32"),
            );
            entries.push(plan_one(id, source, &variables, options)?);
        }
        Ok(Self::new(entries))
    }

    fn new(entries: Vec<PlannedSos>) -> Self {
        let mut additional_variables = 0;
        let mut additional_constraints = 0;
        let mut additional_expr_nodes = 0;
        for entry in &entries {
            let (variables, constraints, expr_nodes) = entry.capacity();
            additional_variables += variables;
            additional_constraints += constraints;
            additional_expr_nodes += expr_nodes;
        }
        Self { entries, additional_variables, additional_constraints, additional_expr_nodes }
    }

    fn apply(self, model: &Model) -> Vec<SosReformulationArtifacts> {
        model.variables.borrow_mut().reserve_exact(self.additional_variables);
        model.var_names.borrow_mut().reserve(self.additional_variables);
        model.constraints.borrow_mut().reserve_exact(self.additional_constraints);
        model.constraint_names.borrow_mut().reserve(self.additional_constraints);
        model.arena.borrow_mut().__reserve_nodes(self.additional_expr_nodes);

        let mut artifacts = Vec::with_capacity(self.entries.len());
        for entry in self.entries {
            artifacts.push(entry.apply(model));
        }
        if !artifacts.is_empty() {
            model.invalidate_kind();
        }
        model.sos_reformulations.borrow_mut().extend(artifacts.iter().cloned());
        artifacts
    }
}

impl PlannedSos {
    fn capacity(&self) -> (usize, usize, usize) {
        let (activation_count, members, sos2) = match &self.form {
            PlannedSosForm::Trivial => return (0, 0, 0),
            PlannedSosForm::Sos1 { members } => (members.len(), members.as_slice(), false),
            PlannedSosForm::Sos2 { members } => (members.len() - 1, members.as_slice(), true),
        };
        let gate_count: usize = members.iter().copied().map(PlannedMember::gate_count).sum();
        let gated_members = members.iter().filter(|member| member.gate_count() > 0).count();
        let adjacent_sums = if sos2 {
            members
                .iter()
                .enumerate()
                .filter(|(index, member)| {
                    *index > 0 && *index + 1 < members.len() && member.gate_count() > 0
                })
                .count()
        } else {
            0
        };
        let constraints = 1 + gate_count;
        let expr_nodes = activation_count
            + activation_count.saturating_sub(1)
            + gated_members
            + 4 * gate_count
            + adjacent_sums;
        (activation_count, constraints, expr_nodes)
    }

    fn apply(self, model: &Model) -> SosReformulationArtifacts {
        let (variable_capacity, constraint_capacity, _) = self.capacity();
        let mut generated_variables = Vec::with_capacity(variable_capacity);
        let mut generated_constraints = Vec::with_capacity(constraint_capacity);
        match self.form {
            PlannedSosForm::Trivial => {}
            PlannedSosForm::Sos1 { members } => {
                let activations = add_binary_activations(
                    model,
                    self.source,
                    members.len(),
                    "member",
                    &mut generated_variables,
                );
                add_at_most_one(model, self.source, &activations, &mut generated_constraints);
                for (member, activation) in members.iter().zip(activations.iter().copied()) {
                    add_planned_member_gates(
                        model,
                        self.source,
                        member,
                        activation,
                        &mut generated_constraints,
                    );
                }
            }
            PlannedSosForm::Sos2 { members } => {
                let intervals = add_binary_activations(
                    model,
                    self.source,
                    members.len() - 1,
                    "interval",
                    &mut generated_variables,
                );
                add_at_most_one(model, self.source, &intervals, &mut generated_constraints);
                for (index, member) in members.iter().enumerate() {
                    if member.gate_count() == 0 {
                        continue;
                    }
                    let activation = match index {
                        0 => intervals[0],
                        i if i + 1 == members.len() => intervals[i - 1],
                        i => intervals[i - 1] + intervals[i],
                    };
                    add_planned_member_gates(
                        model,
                        self.source,
                        member,
                        activation,
                        &mut generated_constraints,
                    );
                }
            }
        }
        model.sos_constraints.borrow_mut()[self.source.index()].active = false;
        SosReformulationArtifacts {
            source: self.source,
            variables: generated_variables,
            constraints: generated_constraints,
        }
    }
}

fn plan_one(
    id: SosConstraintId,
    source: &SosConstraint,
    variables: &[Variable],
    options: SosReformulationOptions,
) -> Result<PlannedSos, ReformulationError> {
    let form = match source.sos_type {
        SosType::Sos1
            if source.members.len() >= 2
                && potential_nonzero_count(variables, &source.members) >= 2 =>
        {
            let members = source
                .members
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, member)| {
                    raw_effective_bounds(&variables[member.variable.index()]) != (0.0, 0.0)
                })
                .map(|(member_index, member)| {
                    let (lower, upper) = effective_bounds(
                        &variables[member.variable.index()],
                        &source.name,
                        options,
                    )?;
                    Ok(PlannedMember { member_index, member, lower, upper })
                })
                .collect::<Result<Vec<_>, ReformulationError>>()?;
            PlannedSosForm::Sos1 { members }
        }
        SosType::Sos2 if source.members.len() >= 3 => {
            let mut ordered = source.members.clone();
            ordered.sort_by(|left, right| left.weight.total_cmp(&right.weight));
            if sos2_requires_reformulation(variables, &ordered) {
                let members = ordered
                    .into_iter()
                    .enumerate()
                    .map(|(member_index, member)| {
                        let (lower, upper) = effective_bounds(
                            &variables[member.variable.index()],
                            &source.name,
                            options,
                        )?;
                        Ok(PlannedMember { member_index, member, lower, upper })
                    })
                    .collect::<Result<Vec<_>, ReformulationError>>()?;
                PlannedSosForm::Sos2 { members }
            } else {
                PlannedSosForm::Trivial
            }
        }
        SosType::Sos1 | SosType::Sos2 => PlannedSosForm::Trivial,
    };
    Ok(PlannedSos { source: id, form })
}

fn add_binary_activations<'a>(
    model: &'a Model,
    sos_id: SosConstraintId,
    count: usize,
    label: &str,
    generated: &mut Vec<VarId>,
) -> Vec<Expr<'a>> {
    (0..count)
        .map(|index| {
            let base = format!("__oximo_sos{}_{}_{}", sos_id.index(), label, index);
            let name = unique_variable_name(model, &base);
            let expression = model.__var(name).binary().build();
            generated.push(expression.var_id().expect("new activation is a variable"));
            expression
        })
        .collect()
}

fn add_at_most_one(
    model: &Model,
    sos_id: SosConstraintId,
    activations: &[Expr<'_>],
    generated: &mut Vec<ConstraintId>,
) {
    let sum = activations
        .iter()
        .copied()
        .reduce(|left, right| left + right)
        .expect("nontrivial SOS has at least one activation");
    let name = unique_constraint_name(model, &format!("__oximo_sos{}_select", sos_id.index()));
    generated.push(model.__add_constraint(name, sum.le(1.0)));
}

fn add_planned_member_gates(
    model: &Model,
    sos_id: SosConstraintId,
    planned: &PlannedMember,
    activation: Expr<'_>,
    generated: &mut Vec<ConstraintId>,
) {
    if planned.gate_count() == 0 {
        return;
    }
    let variable = Expr::from_var(&model.arena, planned.member.variable);
    // Native variable bounds already enforce each gate while the activation is
    // one. A Big-M row is only needed to force the member toward zero while the
    // activation is zero, so sign-definite members need one row instead of two,
    // and a fixed-zero member needs none.
    if planned.lower < 0.0 {
        let lower_name = unique_constraint_name(
            model,
            &format!("__oximo_sos{}_member_{}_lower", sos_id.index(), planned.member_index),
        );
        generated.push(model.__add_constraint(lower_name, variable.ge(planned.lower * activation)));
    }
    if planned.upper > 0.0 {
        let upper_name = unique_constraint_name(
            model,
            &format!("__oximo_sos{}_member_{}_upper", sos_id.index(), planned.member_index),
        );
        generated.push(model.__add_constraint(upper_name, variable.le(planned.upper * activation)));
    }
}

fn effective_bounds(
    variable: &Variable,
    constraint_name: &SmolStr,
    options: SosReformulationOptions,
) -> Result<(f64, f64), ReformulationError> {
    let (lower, upper) = raw_effective_bounds(variable);
    let lower = finite_or_fallback(lower, -1.0, options, constraint_name, &variable.name, "lower")?;
    let upper = finite_or_fallback(upper, 1.0, options, constraint_name, &variable.name, "upper")?;
    Ok((lower, upper))
}

fn raw_effective_bounds(variable: &Variable) -> (f64, f64) {
    match variable.domain {
        Domain::SemiContinuous { threshold } | Domain::SemiInteger { threshold } => {
            (threshold.min(0.0), variable.ub.max(0.0))
        }
        Domain::Real | Domain::Integer | Domain::Binary => (variable.lb, variable.ub),
    }
}

fn potential_nonzero_count(variables: &[Variable], members: &[SosMember]) -> usize {
    members
        .iter()
        .filter(|member| raw_effective_bounds(&variables[member.variable.index()]) != (0.0, 0.0))
        .count()
}

fn sos2_requires_reformulation(variables: &[Variable], ordered_members: &[SosMember]) -> bool {
    let mut potential = ordered_members.iter().enumerate().filter_map(|(index, member)| {
        (raw_effective_bounds(&variables[member.variable.index()]) != (0.0, 0.0)).then_some(index)
    });
    let Some(first) = potential.next() else {
        return false;
    };
    let Some(second) = potential.next() else {
        return false;
    };
    potential.next().is_some() || second != first + 1
}

fn finite_or_fallback(
    bound: f64,
    sign: f64,
    options: SosReformulationOptions,
    constraint: &SmolStr,
    variable: &SmolStr,
    side: &'static str,
) -> Result<f64, ReformulationError> {
    if bound.is_finite() {
        Ok(bound)
    } else if let Some(big_m) = options.fallback_big_m {
        Ok(sign * big_m)
    } else {
        Err(ReformulationError::MissingFiniteBound {
            constraint: constraint.clone(),
            variable: variable.clone(),
            side,
        })
    }
}

fn unique_variable_name(model: &Model, base: &str) -> SmolStr {
    unique_name(base, |candidate| model.variable_id(candidate).is_some())
}

fn unique_constraint_name(model: &Model, base: &str) -> SmolStr {
    unique_name(base, |candidate| model.constraint_id(candidate).is_some())
}

fn unique_name(base: &str, exists: impl Fn(&str) -> bool) -> SmolStr {
    if !exists(base) {
        return base.into();
    }
    for suffix in 1_u64.. {
        let candidate = format!("{base}_{suffix}");
        if !exists(&candidate) {
            return candidate.into();
        }
    }
    unreachable!("u64 name suffix space exhausted")
}
