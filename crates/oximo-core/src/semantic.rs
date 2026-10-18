//! Owned mathematical functions and sets, independent of solver row storage.

pub(crate) mod scalar;
mod scalar_source;
#[doc(hidden)]
pub use scalar_source::ScalarConstraintSource;
mod representation;
mod view;

pub(crate) use representation::NativeConstraintRef;
pub use representation::{FunctionKind, FunctionRepresentationRef, VariableVectorRef};

pub use view::{FunctionRef, SemanticConstraintRef, SemanticPairRef, SetRef};

use std::any::{Any, TypeId, type_name};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use oximo_expr::{Degree, ExprId, ExprIdRemap, ModelId};
use smol_str::SmolStr;

use crate::{
    ConstraintId, IndicatorConstraintId, PsdConstraintId, SocConstraintId, SosConstraintId,
};

/// A semantic identity is tagged by row IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticConstraintId {
    Algebraic(ConstraintId),
    SecondOrderCone(SocConstraintId),
    PositiveSemidefinite(PsdConstraintId),
    SpecialOrderedSet(SosConstraintId),
    Indicator(IndicatorConstraintId),
}

impl SemanticConstraintId {
    pub fn index(self) -> usize {
        match self {
            Self::Algebraic(id) => id.index(),
            Self::SecondOrderCone(id) => id.index(),
            Self::PositiveSemidefinite(id) => id.index(),
            Self::SpecialOrderedSet(id) => id.index(),
            Self::Indicator(id) => id.index(),
        }
    }
}

/// Exact Rust identity of an owned semantic representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemanticType {
    id: TypeId,
    name: &'static str,
}

impl SemanticType {
    pub fn of<T: 'static>() -> Self {
        Self { id: TypeId::of::<T>(), name: type_name::<T>() }
    }

    pub fn is<T: 'static>(self) -> bool {
        self.id == TypeId::of::<T>()
    }

    pub fn name(self) -> &'static str {
        self.name
    }
}

/// Extension contract for an owned symbolic function.
pub trait SemanticFunction: Clone + Debug + Send + Sync + 'static {
    fn model_id(&self) -> ModelId;

    /// Collect symbolic roots when an owned list is needed.
    fn expressions(&self) -> Vec<ExprId> {
        let mut roots = Vec::new();
        self.visit_expressions(&mut |root| roots.push(root));
        roots
    }

    /// Visit every symbolic root without constructing an intermediate list.
    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId));

    /// Describe this function's structure independently.
    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::Opaque
    }

    fn remap(&mut self, remap: ExprIdRemap);

    fn rebind_model(&mut self, model_id: ModelId);
}

/// Owned scalar structure.
#[derive(Clone, Debug)]
pub struct ScalarFunctionData<D: Degree> {
    pub(crate) model_id: ModelId,
    pub(crate) expression: ExprId,
    marker: PhantomData<D>,
}

impl<D: Degree> ScalarFunctionData<D> {
    pub fn from_expression(expression: oximo_expr::Expr<'_, D>) -> Self {
        Self::new(expression.model_id(), expression.id())
    }

    pub fn expression(&self) -> ExprId {
        self.expression
    }

    pub(crate) fn new(model_id: ModelId, expression: ExprId) -> Self {
        Self { model_id, expression, marker: PhantomData }
    }
}

impl<D: Degree> SemanticFunction for ScalarFunctionData<D> {
    fn model_id(&self) -> ModelId {
        self.model_id
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model_id = model_id;
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.expression);
    }

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::Scalar(self.expression)
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.expression = remap.apply(self.expression);
    }
}

#[derive(Clone, Debug)]
pub struct VectorAffineData {
    pub model_id: ModelId,
    pub components: crate::SharedSlice<ExprId>,
}

impl SemanticFunction for VectorAffineData {
    fn model_id(&self) -> ModelId {
        self.model_id
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model_id = model_id;
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        for &root in &self.components {
            visitor(root);
        }
    }

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::VectorAffine(&self.components)
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        for id in &mut self.components {
            *id = remap.apply(*id);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SymmetricAffineData {
    pub model_id: ModelId,
    pub matrix: crate::SymmetricMatrix<ExprId>,
}

impl SemanticFunction for SymmetricAffineData {
    fn model_id(&self) -> ModelId {
        self.model_id
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model_id = model_id;
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        for &root in self.matrix.upper_triangle() {
            visitor(root);
        }
    }

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::SymmetricAffine(&self.matrix)
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.matrix.for_each_mut(|id| *id = remap.apply(*id));
    }
}

trait ErasedFunction: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn signature(&self) -> SemanticType;

    fn model_id(&self) -> ModelId;

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId));

    fn representation(&self) -> FunctionRepresentationRef<'_>;

    fn remap_mut(&mut self, remap: ExprIdRemap);

    fn rebind_mut(&mut self, model_id: ModelId);

    fn remapped(&self, remap: ExprIdRemap) -> Arc<dyn ErasedFunction>;

    fn rebound(&self, model_id: ModelId) -> Arc<dyn ErasedFunction>;
}

impl<T: SemanticFunction> ErasedFunction for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn signature(&self) -> SemanticType {
        SemanticType::of::<T>()
    }

    fn model_id(&self) -> ModelId {
        SemanticFunction::model_id(self)
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        SemanticFunction::visit_expressions(self, visitor);
    }

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        SemanticFunction::representation(self)
    }

    fn remap_mut(&mut self, remap: ExprIdRemap) {
        SemanticFunction::remap(self, remap);
    }

    fn rebind_mut(&mut self, model_id: ModelId) {
        SemanticFunction::rebind_model(self, model_id);
    }

    fn rebound(&self, model_id: ModelId) -> Arc<dyn ErasedFunction> {
        let mut value = self.clone();
        SemanticFunction::rebind_model(&mut value, model_id);
        Arc::new(value)
    }

    fn remapped(&self, remap: ExprIdRemap) -> Arc<dyn ErasedFunction> {
        let mut value = self.clone();
        SemanticFunction::remap(&mut value, remap);
        Arc::new(value)
    }
}

macro_rules! scalar_function_storage {
    ($($variant:ident($ty:ty)),* $(,)?) => {
        #[derive(Clone, Debug)]
        enum FunctionStorage {
            $($variant($ty),)*
            Shared(Arc<dyn ErasedFunction>),
        }

        impl FunctionStorage {
            #[inline(always)]
            fn new<T: SemanticFunction>(value: T) -> Self {
                let any = &value as &dyn Any;

                $(
                    if let Some(value) = any.downcast_ref::<$ty>() {
                        return Self::$variant(value.clone());
                    }
                )*

                Self::Shared(Arc::new(value))
            }

            fn erased(&self) -> &dyn ErasedFunction {
                match self {
                    $(Self::$variant(value) => value,)*
                    Self::Shared(value) => value.as_ref()
                }
            }

            fn remap(&mut self, remap: ExprIdRemap) {
                match self {
                    $(Self::$variant(value) => SemanticFunction::remap(value, remap),)*
                    Self::Shared(value) => {
                        if let Some(unique) = Arc::get_mut(value) {
                            unique.remap_mut(remap);
                        } else {
                            *value = value.remapped(remap);
                        }
                    }
                }
            }

            fn rebind_model(&mut self, id: ModelId) {
                match self {
                    $(Self::$variant(value) => SemanticFunction::rebind_model(value, id),)*
                    Self::Shared(value) => {
                        if let Some(unique) = Arc::get_mut(value) {
                            unique.rebind_mut(id);
                        } else {
                            *value = value.rebound(id);
                        }
                    }
                }
            }
        }
    };
}

scalar_function_storage!(
    Constant(ScalarFunctionData<oximo_expr::Constant>),
    Affine(ScalarFunctionData<oximo_expr::Affine>),
    Quadratic(ScalarFunctionData<oximo_expr::Quadratic>),
    Nonlinear(ScalarFunctionData<oximo_expr::Nonlinear>),
    Dynamic(ScalarFunctionData<oximo_expr::Dynamic>),
    AnyScalar(crate::AnyScalarFunctionData),
);

#[derive(Clone, Debug)]
pub struct FunctionValue(FunctionStorage);

impl FunctionValue {
    pub(crate) fn builtin_scalar(&self) -> Option<(scalar::ScalarKind, ExprId)> {
        use scalar::ScalarKind as Kind;
        let (kind, root) = match &self.0 {
            FunctionStorage::Constant(value) => (Kind::Constant, value.expression),
            FunctionStorage::Affine(value) => (Kind::Affine, value.expression),
            FunctionStorage::Quadratic(value) => (Kind::Quadratic, value.expression),
            FunctionStorage::Nonlinear(value) => (Kind::Nonlinear, value.expression),
            FunctionStorage::Dynamic(value) => (Kind::Dynamic, value.expression),
            FunctionStorage::AnyScalar(value) => (
                match value.class {
                    oximo_expr::ExprClass::Linear => Kind::AnyAffine,
                    oximo_expr::ExprClass::Quadratic => Kind::AnyQuadratic,
                    oximo_expr::ExprClass::Nonlinear => Kind::AnyNonlinear,
                },
                value.expression.expression,
            ),
            FunctionStorage::Shared(_) => return None,
        };

        Some((kind, root))
    }

    pub(crate) fn soc_terms(&self, components: &[ExprId]) -> crate::SharedSlice<ExprId> {
        self.downcast_ref::<VectorAffineData>().map_or_else(
            || components[1..].to_vec().into(),
            |data| data.components.slice(1..components.len()),
        )
    }

    #[inline]
    pub fn new<T: SemanticFunction>(value: T) -> Self {
        Self(FunctionStorage::new(value))
    }

    pub fn signature(&self) -> SemanticType {
        match &self.0 {
            FunctionStorage::Constant(_) => {
                SemanticType::of::<ScalarFunctionData<oximo_expr::Constant>>()
            }
            FunctionStorage::Affine(_) => {
                SemanticType::of::<ScalarFunctionData<oximo_expr::Affine>>()
            }
            FunctionStorage::Quadratic(_) => {
                SemanticType::of::<ScalarFunctionData<oximo_expr::Quadratic>>()
            }
            FunctionStorage::Nonlinear(_) => {
                SemanticType::of::<ScalarFunctionData<oximo_expr::Nonlinear>>()
            }
            FunctionStorage::Dynamic(_) => {
                SemanticType::of::<ScalarFunctionData<oximo_expr::Dynamic>>()
            }
            FunctionStorage::AnyScalar(_) => SemanticType::of::<crate::AnyScalarFunctionData>(),
            FunctionStorage::Shared(value) => value.signature(),
        }
    }

    pub fn downcast_ref<T: SemanticFunction>(&self) -> Option<&T> {
        self.0.erased().as_any().downcast_ref()
    }

    pub fn model_id(&self) -> ModelId {
        match &self.0 {
            FunctionStorage::Constant(value) => value.model_id,
            FunctionStorage::Affine(value) => value.model_id,
            FunctionStorage::Quadratic(value) => value.model_id,
            FunctionStorage::Nonlinear(value) => value.model_id,
            FunctionStorage::Dynamic(value) => value.model_id,
            FunctionStorage::AnyScalar(value) => value.expression.model_id,
            FunctionStorage::Shared(value) => value.model_id(),
        }
    }

    pub fn expressions(&self) -> Vec<ExprId> {
        let mut roots = Vec::new();
        self.visit_expressions(&mut |root| roots.push(root));
        roots
    }

    pub fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        if let Some((_, root)) = self.builtin_scalar() {
            visitor(root);
        } else {
            self.0.erased().visit_expressions(visitor);
        }
    }

    pub fn representation(&self) -> FunctionRepresentationRef<'_> {
        if let Some((_, root)) = self.builtin_scalar() {
            FunctionRepresentationRef::Scalar(root)
        } else {
            self.0.erased().representation()
        }
    }

    pub fn remap(&mut self, remap: ExprIdRemap) {
        self.0.remap(remap);
    }

    pub fn rebind_model(&mut self, model_id: ModelId) {
        if self.model_id() != model_id {
            self.0.rebind_model(model_id);
        }
    }
}

trait ErasedSet: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn signature(&self) -> SemanticType;
}

impl<T: crate::function_set::Set> ErasedSet for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn signature(&self) -> SemanticType {
        SemanticType::of::<T>()
    }
}

macro_rules! builtin_set_storage {
    ($($variant:ident($ty:ty)),* $(,)?) => {
        #[derive(Clone, Debug)]
        enum SetStorage {
            $($variant($ty),)*
            Shared(Arc<dyn ErasedSet>),
        }

        impl SetStorage {
            #[inline(always)]
            fn new<T: crate::function_set::Set>(value: T) -> Self {
                let any = &value as &dyn Any;

                $(
                    if let Some(value) = any.downcast_ref::<$ty>() {
                        return Self::$variant(*value);
                    }
                )*

                Self::Shared(Arc::new(value))
            }

            fn erased(&self) -> &dyn ErasedSet {
                match self {
                    $(Self::$variant(value) => value,)*
                    Self::Shared(value) => value.as_ref()
                }
            }
        }
    };
}

builtin_set_storage!(
    LessThan(crate::LessThan),
    GreaterThan(crate::GreaterThan),
    EqualTo(crate::EqualTo),
    Interval(crate::Interval),
    SecondOrderCone(crate::SecondOrderCone),
    PositiveSemidefiniteCone(crate::PositiveSemidefiniteCone),
);

#[derive(Clone, Debug)]
pub struct SetValue(SetStorage);

impl SetValue {
    #[inline]
    pub(crate) fn builtin_bound_kind(&self) -> Option<scalar::BoundKind> {
        use scalar::BoundKind;
        Some(match self.0 {
            SetStorage::LessThan(_) => BoundKind::Less,
            SetStorage::GreaterThan(_) => BoundKind::Greater,
            SetStorage::EqualTo(_) => BoundKind::Equal,
            SetStorage::Interval(_) => BoundKind::Interval,
            _ => return None,
        })
    }

    #[inline]
    pub fn new<T: crate::function_set::Set>(value: T) -> Self {
        Self(SetStorage::new(value))
    }

    pub fn signature(&self) -> SemanticType {
        self.0.erased().signature()
    }

    pub fn downcast_ref<T: crate::function_set::Set>(&self) -> Option<&T> {
        self.0.erased().as_any().downcast_ref()
    }

    /// Project built-in scalar bound sets onto interval bounds without changing set identity.
    #[inline]
    pub fn scalar_bounds(&self) -> Option<(f64, f64)> {
        match &self.0 {
            SetStorage::LessThan(s) => Some((f64::NEG_INFINITY, s.0)),
            SetStorage::GreaterThan(s) => Some((s.0, f64::INFINITY)),
            SetStorage::EqualTo(s) => Some((s.0, s.0)),
            SetStorage::Interval(s) => Some((s.lower, s.upper)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SemanticPair {
    pub function: FunctionValue,
    pub set: SetValue,
}

impl SemanticPair {
    #[inline]
    pub fn new<F: crate::Function, S: crate::function_set::Set>(function: &F, set: S) -> Self {
        Self { function: FunctionValue::new(function.semantic_data()), set: SetValue::new(set) }
    }

    pub fn remap(&mut self, remap: ExprIdRemap) {
        self.function.remap(remap);
    }
}

#[derive(Clone, Debug, Default)]
pub enum ConstraintOrigin {
    #[default]
    Declared,
    Range(SmolStr),
    Generated {
        sources: Vec<SemanticConstraintId>,
        transformation: SmolStr,
    },
}

#[derive(Clone, Debug)]
pub struct SemanticConstraint {
    pub id: SemanticConstraintId,
    pub name: SmolStr,
    pub pair: SemanticPair,
    pub active: bool,
    pub origin: ConstraintOrigin,
}

/// A scalar expression and its bounds.
#[derive(Clone, Copy, Debug)]
pub struct ScalarConstraintView {
    pub lhs: ExprId,
    pub lower: f64,
    pub upper: f64,
}

impl ScalarConstraintView {
    pub fn matches_pair(self, pair: &SemanticPair) -> bool {
        matches!(pair.function.representation(), FunctionRepresentationRef::Scalar(root) if root == self.lhs)
            && pair.set.scalar_bounds().is_some_and(|(lower, upper)| {
                lower.to_bits() == self.lower.to_bits() && upper.to_bits() == self.upper.to_bits()
            })
    }
}

/// A checked modification rejected before any semantic or solver view is changed.
#[derive(Debug, thiserror::Error)]
pub enum ConstraintUpdateError {
    #[error(transparent)]
    ModelMismatch(#[from] oximo_expr::ModelMismatchError),
    #[error(
        "cannot modify reformulated source {0:?}; change the source model before reformulating"
    )]
    ReformulatedSource(SemanticConstraintId),
    #[error(
        "cannot modify constraint {0:?} used by a GDP reformulation; change the source model before reformulating"
    )]
    GdpDependency(SemanticConstraintId),
    #[error(
        "cannot modify constraint {0:?} registered through an alternative lowering without an update mapping; rebuild it from updated semantics"
    )]
    AlternativeLowering(SemanticConstraintId),
    #[error(transparent)]
    InvalidBounds(#[from] crate::function_set::ScalarBoundsError),
    #[error("constraint dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Marker for a deliberately erased handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Erased;

#[derive(Clone, Debug)]
pub struct VariableVectorData {
    pub model_id: ModelId,
    pub variables: Vec<oximo_expr::VarId>,
}

impl SemanticFunction for VariableVectorData {
    fn model_id(&self) -> ModelId {
        self.model_id
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model_id = model_id;
    }

    fn visit_expressions(&self, _visitor: &mut dyn FnMut(ExprId)) {}

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::Variables(VariableVectorRef::Ids(&self.variables))
    }

    fn remap(&mut self, _remap: ExprIdRemap) {}
}

#[derive(Clone, Debug)]
pub struct SpecialOrderedSet {
    pub kind: crate::SosType,
    pub weights: Vec<f64>,
}

impl crate::function_set::Set for SpecialOrderedSet {}

#[derive(Clone, Debug)]
pub struct IndicatorFunctionData {
    pub model_id: ModelId,
    pub trigger: oximo_expr::VarId,
    pub consequent: FunctionValue,
}

impl SemanticFunction for IndicatorFunctionData {
    fn model_id(&self) -> ModelId {
        self.model_id
    }

    fn rebind_model(&mut self, model_id: ModelId) {
        self.model_id = model_id;
        self.consequent.rebind_model(model_id);
    }

    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        self.consequent.visit_expressions(visitor);
    }

    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::Indicator { trigger: self.trigger, consequent: &self.consequent }
    }

    fn remap(&mut self, remap: ExprIdRemap) {
        self.consequent.remap(remap);
    }
}

#[derive(Clone, Debug)]
pub struct IndicatorSet {
    pub active_value: bool,
    pub consequent: SetValue,
}

impl crate::function_set::Set for IndicatorSet {}

impl From<ConstraintId> for SemanticConstraintId {
    fn from(id: ConstraintId) -> Self {
        Self::Algebraic(id)
    }
}

impl From<SocConstraintId> for SemanticConstraintId {
    fn from(id: SocConstraintId) -> Self {
        Self::SecondOrderCone(id)
    }
}

impl From<PsdConstraintId> for SemanticConstraintId {
    fn from(id: PsdConstraintId) -> Self {
        Self::PositiveSemidefinite(id)
    }
}

impl ScalarFunctionData<oximo_expr::Affine> {
    /// Checked narrowing for a runtime root owned by this model.
    pub fn try_from_root(model: &crate::Model, root: ExprId) -> Option<Self> {
        let arena = model.arena();

        if root.index() >= arena.len()
            || oximo_expr::classify(&arena, root) != oximo_expr::ExprClass::Linear
        {
            return None;
        }

        Some(Self::new(model.id(), root))
    }
}

impl ScalarFunctionData<oximo_expr::Quadratic> {
    pub fn try_from_root(model: &crate::Model, root: ExprId) -> Option<Self> {
        let arena = model.arena();

        if root.index() >= arena.len()
            || oximo_expr::classify(&arena, root) == oximo_expr::ExprClass::Nonlinear
        {
            return None;
        }

        Some(Self::new(model.id(), root))
    }
}

impl From<SosConstraintId> for SemanticConstraintId {
    fn from(id: SosConstraintId) -> Self {
        Self::SpecialOrderedSet(id)
    }
}

impl From<IndicatorConstraintId> for SemanticConstraintId {
    fn from(id: IndicatorConstraintId) -> Self {
        Self::Indicator(id)
    }
}
