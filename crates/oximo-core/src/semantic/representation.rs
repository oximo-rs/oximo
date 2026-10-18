//! Descriptions of owned and borrowed function representations.

use oximo_expr::{ExprClass, ExprId, VarId};

use super::{FunctionValue, SemanticPair, SemanticType};

/// Mathematical representation family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionKind {
    /// `None` means the expression's degree needs runtime classification.
    Scalar(Option<ExprClass>),
    VectorAffine,
    SymmetricAffine,
    Variables,
    Indicator,
    /// A custom type whose instance may expose a recognized representation.
    Extension,
}

impl SemanticType {
    /// Classify recognized built-in owned types.
    /// Custom types retain their exact identity and report
    /// `Extension` until an instance supplies a representation.
    pub fn function_kind(self) -> FunctionKind {
        use super::ScalarFunctionData;
        use oximo_expr::{Affine, Constant, Dynamic, Nonlinear, Quadratic};

        if self.is::<ScalarFunctionData<Affine>>() || self.is::<ScalarFunctionData<Constant>>() {
            FunctionKind::Scalar(Some(ExprClass::Linear))
        } else if self.is::<ScalarFunctionData<Quadratic>>() {
            FunctionKind::Scalar(Some(ExprClass::Quadratic))
        } else if self.is::<ScalarFunctionData<Nonlinear>>() {
            FunctionKind::Scalar(Some(ExprClass::Nonlinear))
        } else if self.is::<ScalarFunctionData<Dynamic>>()
            || self.is::<crate::AnyScalarFunctionData>()
        {
            FunctionKind::Scalar(None)
        } else if self.is::<super::VectorAffineData>() {
            FunctionKind::VectorAffine
        } else if self.is::<super::SymmetricAffineData>() {
            FunctionKind::SymmetricAffine
        } else if self.is::<super::VariableVectorData>() {
            FunctionKind::Variables
        } else if self.is::<super::IndicatorFunctionData>() {
            FunctionKind::Indicator
        } else {
            FunctionKind::Extension
        }
    }
}

/// Variable IDs borrowed from owned function data or a native SOS row.
#[derive(Clone, Copy, Debug)]
pub enum VariableVectorRef<'a> {
    Ids(&'a [VarId]),
    SosMembers(&'a [crate::SosMember]),
}

impl VariableVectorRef<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::Ids(ids) => ids.len(),
            Self::SosMembers(members) => members.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = VarId> + DoubleEndedIterator {
        (0..self.len()).map(move |index| match self {
            Self::Ids(ids) => ids[index],
            Self::SosMembers(members) => members[index].variable,
        })
    }
}

/// Borrowed function structure.
///
/// `Opaque` is reserved for functions that expose no recognized representation.
/// Extensions may expose a representation without giving up their exact Rust type.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum FunctionRepresentationRef<'a> {
    Scalar(ExprId),
    VectorAffine(&'a [ExprId]),
    SymmetricAffine(&'a crate::SymmetricMatrix<ExprId>),
    Variables(VariableVectorRef<'a>),
    Indicator { trigger: VarId, consequent: &'a FunctionValue },
    Opaque,
}

/// Recognized common views, before ownership, numeric and dimension validation.
pub(crate) enum NativeConstraintRef<'a> {
    Scalar { root: ExprId, lower: f64, upper: f64 },
    Soc { components: &'a [ExprId], dimension: usize },
    Psd { matrix: &'a crate::SymmetricMatrix<ExprId>, dimension: usize },
    Sos { variables: VariableVectorRef<'a>, set: &'a super::SpecialOrderedSet },
    Indicator { trigger: VarId, consequent: &'a FunctionValue, set: &'a super::IndicatorSet },
}

impl SemanticPair {
    /// Interpret function and set together for registration and updates.
    pub(crate) fn native_constraint(&self) -> Option<NativeConstraintRef<'_>> {
        use FunctionRepresentationRef as Function;
        Some(match self.function.representation() {
            Function::Scalar(root) => {
                let (lower, upper) = self.set.scalar_bounds()?;
                NativeConstraintRef::Scalar { root, lower, upper }
            }
            Function::VectorAffine(components) => NativeConstraintRef::Soc {
                components,
                dimension: self.set.downcast_ref::<crate::SecondOrderCone>()?.dimension,
            },
            Function::SymmetricAffine(matrix) => NativeConstraintRef::Psd {
                matrix,
                dimension: self
                    .set
                    .downcast_ref::<crate::PositiveSemidefiniteCone>()?
                    .side_dimension,
            },
            Function::Variables(variables) => NativeConstraintRef::Sos {
                variables,
                set: self.set.downcast_ref::<super::SpecialOrderedSet>()?,
            },
            Function::Indicator { trigger, consequent } => NativeConstraintRef::Indicator {
                trigger,
                consequent,
                set: self.set.downcast_ref::<super::IndicatorSet>()?,
            },
            Function::Opaque => return None,
        })
    }
}
