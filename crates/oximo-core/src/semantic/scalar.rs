use super::{FunctionValue, ScalarFunctionData, SemanticPair, SetValue};
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
    pub(crate) fn of(value: &FunctionValue) -> Option<Self> {
        value.builtin_scalar().map(|(kind, _)| kind)
    }

    pub(crate) fn function(self, model: ModelId, root: ExprId) -> FunctionValue {
        match self {
            Self::AnyAffine | Self::AnyQuadratic | Self::AnyNonlinear => {
                FunctionValue::new(crate::AnyScalarFunctionData {
                    expression: ScalarFunctionData::new(model, root),
                    class: match self {
                        Self::AnyAffine => oximo_expr::ExprClass::Linear,
                        Self::AnyQuadratic => oximo_expr::ExprClass::Quadratic,
                        _ => oximo_expr::ExprClass::Nonlinear,
                    },
                })
            }
            Self::Constant => FunctionValue::new(ScalarFunctionData::<Constant>::new(model, root)),
            Self::Affine => FunctionValue::new(ScalarFunctionData::<Affine>::new(model, root)),
            Self::Quadratic => {
                FunctionValue::new(ScalarFunctionData::<Quadratic>::new(model, root))
            }
            Self::Nonlinear => {
                FunctionValue::new(ScalarFunctionData::<Nonlinear>::new(model, root))
            }
            Self::Dynamic => FunctionValue::new(ScalarFunctionData::<Dynamic>::new(model, root)),
        }
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
    pub(crate) fn of(value: &SetValue) -> Option<Self> {
        value.builtin_bound_kind()
    }

    pub(crate) fn set(self, lower: f64, upper: f64) -> SetValue {
        match self {
            Self::Less => SetValue::new(crate::LessThan(upper)),
            Self::Greater => SetValue::new(crate::GreaterThan(lower)),
            Self::Equal => SetValue::new(crate::EqualTo(lower)),
            Self::Interval => SetValue::new(crate::Interval { lower, upper }),
        }
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
