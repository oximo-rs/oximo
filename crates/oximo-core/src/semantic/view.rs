//! Borrowed source views.

use oximo_expr::{ExprId, ModelId, VarId};

use super::{
    ConstraintOrigin, FunctionRepresentationRef, FunctionValue, IndicatorFunctionData,
    IndicatorSet, SemanticConstraint, SemanticConstraintId, SemanticPair, SemanticType, SetValue,
    SpecialOrderedSet, SymmetricAffineData, VariableVectorData, VariableVectorRef,
    VectorAffineData,
};

#[derive(Debug)]
pub enum FunctionRef<'a> {
    Borrowed(&'a FunctionValue),
    Scalar(FunctionValue),
    Vector { model_id: ModelId, components: crate::SharedSlice<ExprId> },
    Symmetric { model_id: ModelId, matrix: &'a crate::SymmetricMatrix<ExprId> },
    Variables { model_id: ModelId, members: &'a [crate::SosMember] },
    Indicator { model_id: ModelId, trigger: VarId, consequent: FunctionValue },
}

impl FunctionRef<'_> {
    pub fn signature(&self) -> SemanticType {
        match self {
            Self::Borrowed(value) => value.signature(),
            Self::Scalar(value) => value.signature(),
            Self::Vector { .. } => SemanticType::of::<VectorAffineData>(),
            Self::Symmetric { .. } => SemanticType::of::<SymmetricAffineData>(),
            Self::Variables { .. } => SemanticType::of::<VariableVectorData>(),
            Self::Indicator { .. } => SemanticType::of::<IndicatorFunctionData>(),
        }
    }

    pub fn model_id(&self) -> ModelId {
        match self {
            Self::Borrowed(value) => value.model_id(),
            Self::Scalar(value) => value.model_id(),
            Self::Vector { model_id, .. }
            | Self::Symmetric { model_id, .. }
            | Self::Variables { model_id, .. }
            | Self::Indicator { model_id, .. } => *model_id,
        }
    }

    /// Borrow extension payloads or inline scalars.
    ///
    /// Use `to_owned` for other typed payloads.
    /// Synthesized vector, matrix, variable and indicator views return `None`
    /// here even when their owned `FunctionValue` can downcast to `T`. Use
    /// `representation()` for borrowed inspection, or bind `let owned = self.to_owned()`
    /// before calling `owned.downcast_ref::<T>()`.
    pub fn downcast_ref<T: super::SemanticFunction>(&self) -> Option<&T> {
        match self {
            Self::Borrowed(value) => value.downcast_ref(),
            Self::Scalar(value) => value.downcast_ref(),
            _ => None,
        }
    }

    pub fn representation(&self) -> FunctionRepresentationRef<'_> {
        match self {
            Self::Borrowed(value) => value.representation(),
            Self::Scalar(value) => value.representation(),
            Self::Vector { components, .. } => FunctionRepresentationRef::VectorAffine(components),
            Self::Symmetric { matrix, .. } => FunctionRepresentationRef::SymmetricAffine(matrix),
            Self::Variables { members, .. } => {
                FunctionRepresentationRef::Variables(VariableVectorRef::SosMembers(members))
            }
            Self::Indicator { trigger, consequent, .. } => {
                FunctionRepresentationRef::Indicator { trigger: *trigger, consequent }
            }
        }
    }

    pub fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        match self {
            Self::Borrowed(value) => value.visit_expressions(visitor),
            Self::Scalar(value) => value.visit_expressions(visitor),
            Self::Vector { components, .. } => components.iter().copied().for_each(visitor),
            Self::Symmetric { matrix, .. } => {
                matrix.upper_triangle().iter().copied().for_each(visitor);
            }
            Self::Variables { .. } => {}
            Self::Indicator { consequent, .. } => consequent.visit_expressions(visitor),
        }
    }

    pub fn to_owned(&self) -> FunctionValue {
        match self {
            Self::Borrowed(value) => (*value).clone(),
            Self::Scalar(value) => value.clone(),
            Self::Vector { model_id, components } => FunctionValue::new(VectorAffineData {
                model_id: *model_id,
                components: components.clone(),
            }),
            Self::Symmetric { model_id, matrix } => FunctionValue::new(SymmetricAffineData {
                model_id: *model_id,
                matrix: (*matrix).clone(),
            }),
            Self::Variables { model_id, members } => FunctionValue::new(VariableVectorData {
                model_id: *model_id,
                variables: members.iter().map(|m| m.variable).collect(),
            }),
            Self::Indicator { model_id, trigger, consequent } => {
                FunctionValue::new(IndicatorFunctionData {
                    model_id: *model_id,
                    trigger: *trigger,
                    consequent: consequent.clone(),
                })
            }
        }
    }
}

#[derive(Debug)]
pub enum SetRef<'a> {
    Borrowed(&'a SetValue),
    Inline(SetValue),
    Sos(&'a crate::SosConstraint),
    Indicator { active_value: bool, consequent: SetValue },
}

impl SetRef<'_> {
    pub fn signature(&self) -> SemanticType {
        match self {
            Self::Borrowed(value) => value.signature(),
            Self::Inline(value) => value.signature(),
            Self::Sos(_) => SemanticType::of::<SpecialOrderedSet>(),
            Self::Indicator { .. } => SemanticType::of::<IndicatorSet>(),
        }
    }

    pub fn scalar_bounds(&self) -> Option<(f64, f64)> {
        match self {
            Self::Borrowed(value) => value.scalar_bounds(),
            Self::Inline(value) => value.scalar_bounds(),
            _ => None,
        }
    }

    pub fn sos_kind(&self) -> Option<crate::SosType> {
        match self {
            Self::Sos(row) => Some(row.sos_type),
            Self::Borrowed(value) => value.downcast_ref::<SpecialOrderedSet>().map(|s| s.kind),
            Self::Inline(value) => value.downcast_ref::<SpecialOrderedSet>().map(|s| s.kind),
            Self::Indicator { .. } => None,
        }
    }

    pub fn to_owned(&self) -> SetValue {
        match self {
            Self::Borrowed(value) => (*value).clone(),
            Self::Inline(value) => value.clone(),
            Self::Sos(row) => SetValue::new(SpecialOrderedSet {
                kind: row.sos_type,
                weights: row.members.iter().map(|m| m.weight).collect(),
            }),
            Self::Indicator { active_value, consequent } => SetValue::new(IndicatorSet {
                active_value: *active_value,
                consequent: consequent.clone(),
            }),
        }
    }
}

#[derive(Debug)]
pub struct SemanticPairRef<'a> {
    pub function: FunctionRef<'a>,
    pub set: SetRef<'a>,
}

impl SemanticPairRef<'_> {
    pub fn to_owned(&self) -> SemanticPair {
        SemanticPair { function: self.function.to_owned(), set: self.set.to_owned() }
    }
}

#[derive(Debug)]
pub struct SemanticConstraintRef<'a> {
    pub id: SemanticConstraintId,
    pub name: &'a str,
    pub pair: SemanticPairRef<'a>,
    pub active: bool,
    pub origin: std::borrow::Cow<'a, ConstraintOrigin>,
}

impl SemanticConstraintRef<'_> {
    pub fn to_owned(&self) -> SemanticConstraint {
        SemanticConstraint {
            id: self.id,
            name: self.name.into(),
            pair: self.pair.to_owned(),
            active: self.active,
            origin: self.origin.clone().into_owned(),
        }
    }
}

impl SemanticConstraint {
    pub fn as_ref(&self) -> SemanticConstraintRef<'_> {
        SemanticConstraintRef {
            id: self.id,
            name: &self.name,
            pair: SemanticPairRef {
                function: FunctionRef::Borrowed(&self.pair.function),
                set: SetRef::Borrowed(&self.pair.set),
            },
            active: self.active,
            origin: std::borrow::Cow::Borrowed(&self.origin),
        }
    }
}
