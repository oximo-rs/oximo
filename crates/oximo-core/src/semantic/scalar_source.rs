//! Compact scalar source capture between the DSL, worker remapping, and registration.

use std::sync::Arc;

use oximo_expr::{ExprId, ModelId};

use super::scalar::ScalarSignature;
use super::{ScalarConstraintView, SemanticPair};

#[derive(Clone, Copy, Debug)]
struct NativeSource {
    model_id: ModelId,
    root: ExprId,
    signature: ScalarSignature,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Debug)]
enum SourceStorage {
    Native(NativeSource),
    Extension(Arc<SemanticPair>),
}

#[derive(Clone, Debug)]
/// Original scalar declaration passed through custom algebraic lowering.
#[doc(hidden)]
pub struct ScalarConstraintSource(SourceStorage);

impl ScalarConstraintSource {
    /// Inspect concrete owned types before constructing erased function/set values.
    #[inline]
    pub(crate) fn capture<F: crate::Function, S: crate::ConstraintSet>(
        function: &F,
        set: &S,
    ) -> Self {
        Self::capture_unshared(function, set).unwrap_or_else(Self::from_pair)
    }

    /// Preserve non-scalar payloads without allocating an intermediate shared pair.
    #[inline(always)]
    pub(crate) fn capture_unshared<F: crate::Function, S: crate::ConstraintSet>(
        function: &F,
        set: &S,
    ) -> Result<Self, SemanticPair> {
        use super::scalar::ScalarKind;
        use std::any::Any;

        let owned = function.semantic_data();
        let any = &owned as &dyn Any;
        let scalar = {
            macro_rules! scalar {
                ($($degree:ty => $kind:ident),* $(,)?) => {
                    $(if let Some(value) = any.downcast_ref::<super::ScalarFunctionData<$degree>>() {
                        Some((ScalarKind::$kind, value.model_id, value.expression))
                    } else)* {
                        any.downcast_ref::<crate::AnyScalarFunctionData>().map(|value| {
                            let kind = match value.class {
                                oximo_expr::ExprClass::Linear => ScalarKind::AnyAffine,
                                oximo_expr::ExprClass::Quadratic => ScalarKind::AnyQuadratic,
                                oximo_expr::ExprClass::Nonlinear => ScalarKind::AnyNonlinear,
                            };
                            (kind, value.expression.model_id, value.expression.expression)
                        })
                    }
                };
            }
            scalar!(
                oximo_expr::Constant => Constant,
                oximo_expr::Affine => Affine,
                oximo_expr::Quadratic => Quadratic,
                oximo_expr::Nonlinear => Nonlinear,
                oximo_expr::Dynamic => Dynamic,
            )
        };

        let set = super::SetValue::new(set.clone());

        if let Some((lower, upper)) = set.scalar_bounds() {
            crate::function_set::validate_bounds(lower, upper);

            if let Some((function, model_id, root)) = scalar
                && let Some(set) = set.builtin_bound_kind()
            {
                return Ok(Self(SourceStorage::Native(NativeSource {
                    model_id,
                    root,
                    signature: ScalarSignature { function, set },
                    lower,
                    upper,
                })));
            }
        }

        Err(SemanticPair { function: super::FunctionValue::new(owned), set })
    }

    pub(crate) fn from_pair(pair: SemanticPair) -> Self {
        Self(SourceStorage::Extension(Arc::new(pair)))
    }

    /// Project the declared scalar pair without invoking an alternative lowering.
    #[inline]
    pub(crate) fn view(&self) -> super::ScalarConstraintView {
        match &self.0 {
            SourceStorage::Native(source) => super::ScalarConstraintView {
                lhs: source.root,
                lower: source.lower,
                upper: source.upper,
            },
            SourceStorage::Extension(pair) => {
                let Some(super::NativeConstraintRef::Scalar { root: lhs, lower, upper }) =
                    pair.native_constraint()
                else {
                    panic!(
                        "scalar registration requires a scalar representation and recognized bounds; use an explicit lowering"
                    );
                };
                super::ScalarConstraintView { lhs, lower, upper }
            }
        }
    }

    #[inline]
    pub(crate) fn validate_roots(&self, node_count: usize) {
        let mut check = |id: ExprId| {
            assert!(id.index() < node_count, "function contains an invalid expression ID");
        };
        match &self.0 {
            SourceStorage::Native(source) => check(source.root),
            SourceStorage::Extension(pair) => pair.function.visit_expressions(&mut check),
        }
    }

    #[inline]
    pub(crate) fn model_id(&self) -> ModelId {
        match &self.0 {
            SourceStorage::Native(source) => source.model_id,
            SourceStorage::Extension(pair) => pair.function.model_id(),
        }
    }

    #[inline]
    pub(crate) fn direct_signature(&self, view: ScalarConstraintView) -> Option<ScalarSignature> {
        match &self.0 {
            SourceStorage::Native(source)
                if source.root == view.lhs
                    && source.lower.to_bits() == view.lower.to_bits()
                    && source.upper.to_bits() == view.upper.to_bits() =>
            {
                Some(source.signature)
            }
            _ => None,
        }
    }

    pub fn into_pair(self) -> SemanticPair {
        match self.0 {
            SourceStorage::Native(source) => SemanticPair {
                function: source.signature.function.function(source.model_id, source.root),
                set: source.signature.set.set(source.lower, source.upper),
            },
            SourceStorage::Extension(pair) => Arc::unwrap_or_clone(pair),
        }
    }
}
