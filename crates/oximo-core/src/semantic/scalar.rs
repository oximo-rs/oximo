use super::{
    FunctionStorage, FunctionValue, ScalarFunctionData, SemanticPair, SetStorage, SetValue,
};
use oximo_expr::{Affine, Constant, Dynamic, ExprId, ModelId, Nonlinear, Quadratic};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarKind {
    Constant,
    Affine,
    Quadratic,
    Nonlinear,
    Dynamic,
    AnyAffine,
    AnyQuadratic,
    AnyNonlinear,
}

impl ScalarKind {
    pub(crate) fn signature(self) -> super::SemanticType {
        use super::SemanticType;
        match self {
            Self::Constant => SemanticType::of::<ScalarFunctionData<Constant>>(),
            Self::Affine => SemanticType::of::<ScalarFunctionData<Affine>>(),
            Self::Quadratic => SemanticType::of::<ScalarFunctionData<Quadratic>>(),
            Self::Nonlinear => SemanticType::of::<ScalarFunctionData<Nonlinear>>(),
            Self::Dynamic => SemanticType::of::<ScalarFunctionData<Dynamic>>(),
            Self::AnyAffine | Self::AnyQuadratic | Self::AnyNonlinear => {
                SemanticType::of::<crate::AnyScalarFunctionData>()
            }
        }
    }

    pub(crate) fn of(value: &FunctionValue) -> Option<Self> {
        value.builtin_scalar().map(|(kind, _)| kind)
    }

    pub(crate) fn function(self, model: ModelId, root: ExprId) -> FunctionValue {
        FunctionValue(match self {
            Self::AnyAffine | Self::AnyQuadratic | Self::AnyNonlinear => {
                FunctionStorage::AnyScalar(crate::AnyScalarFunctionData {
                    expression: ScalarFunctionData::new(model, root),
                    class: match self {
                        Self::AnyAffine => oximo_expr::ExprClass::Linear,
                        Self::AnyQuadratic => oximo_expr::ExprClass::Quadratic,
                        _ => oximo_expr::ExprClass::Nonlinear,
                    },
                })
            }
            Self::Constant => {
                FunctionStorage::Constant(ScalarFunctionData::<Constant>::new(model, root))
            }
            Self::Affine => FunctionStorage::Affine(ScalarFunctionData::<Affine>::new(model, root)),
            Self::Quadratic => {
                FunctionStorage::Quadratic(ScalarFunctionData::<Quadratic>::new(model, root))
            }
            Self::Nonlinear => {
                FunctionStorage::Nonlinear(ScalarFunctionData::<Nonlinear>::new(model, root))
            }
            Self::Dynamic => {
                FunctionStorage::Dynamic(ScalarFunctionData::<Dynamic>::new(model, root))
            }
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundKind {
    Less,
    Greater,
    Equal,
    Interval,
}

impl BoundKind {
    pub(crate) fn signature(self) -> super::SemanticType {
        use super::SemanticType;
        match self {
            Self::Less => SemanticType::of::<crate::LessThan>(),
            Self::Greater => SemanticType::of::<crate::GreaterThan>(),
            Self::Equal => SemanticType::of::<crate::EqualTo>(),
            Self::Interval => SemanticType::of::<crate::Interval>(),
        }
    }

    pub(crate) fn of(value: &SetValue) -> Option<Self> {
        value.builtin_bound_kind()
    }

    pub(crate) fn set(self, lower: f64, upper: f64) -> SetValue {
        SetValue(match self {
            Self::Less => SetStorage::LessThan(crate::LessThan(upper)),
            Self::Greater => SetStorage::GreaterThan(crate::GreaterThan(lower)),
            Self::Equal => SetStorage::EqualTo(crate::EqualTo(lower)),
            Self::Interval => SetStorage::Interval(crate::Interval { lower, upper }),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScalarSignature {
    pub(crate) function: ScalarKind,
    pub(crate) set: BoundKind,
}

impl ScalarSignature {
    pub(crate) fn of(pair: &SemanticPair) -> Option<Self> {
        Some(Self { function: ScalarKind::of(&pair.function)?, set: BoundKind::of(&pair.set)? })
    }
}
