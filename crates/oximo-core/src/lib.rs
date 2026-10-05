#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

extern crate self as oximo_core;

#[doc(hidden)]
#[path = "macro_support.rs"]
pub mod __macro_support;
pub mod constraint;
pub mod display;
pub mod domain;
pub mod error;
pub mod function_set;
pub mod indexed;
pub mod indicator;
pub mod matrix;
pub mod model;
pub mod objective;
pub mod param;
pub mod prelude;
pub mod psd;
pub mod reformulation;
pub mod set;
pub mod soc;
pub mod sos;
pub mod sum;
pub mod var;

pub use constraint::{
    AlgebraicConstraint, ConstraintHandle, ConstraintId, IntoRhs, RangeConstraintHandles,
    RangeConstraintIds, Relate, Sense,
};
pub use display::{
    ConstraintDisplay, ExprDisplay, IndicatorDisplay, ObjectiveDisplay, SocDisplay, SosDisplay,
};
pub use domain::Domain;
pub use error::{Error, Result};
pub use indexed::{
    IndexedConstraint, IndexedIndicatorConstraint, IndexedParam, IndexedRangeConstraint,
    IndexedRangeIndicatorConstraint, IndexedVar,
};
pub use indicator::{
    IndicatorConstraint, IndicatorConstraintHandle, IndicatorConstraintId,
    RangeIndicatorConstraintHandles, RangeIndicatorConstraintIds,
};
pub use model::{
    ConstraintRef, IndexedVarBuilder, Model, ModelConstraints, ModelKind, display_index_key,
};
pub use objective::{Objective, ObjectiveSense};
pub use param::Parameter;
pub use reformulation::{
    IndicatorReformulationArtifacts, IndicatorReformulationOptions, ReformulatedModel,
    ReformulationError, SosReformulationArtifacts, SosReformulationOptions,
};
pub use set::{
    Axis, FromIndexKey, FromIndexKeyRef, IndexKey, IndexKeyRef, IndexTuple, KeyCat, ScalarKey, Set,
    SetIter,
};
#[doc(hidden)]
pub use soc::__detect_soc_from_quadratic;
pub use soc::{SocConstraint, SocConstraintHandle, SocConstraintId, SocForm};
pub use sos::{SosConstraint, SosConstraintHandle, SosConstraintId, SosMember, SosType};
pub use sum::SumDomain;
pub use var::{VarBuilder, Variable, var_name};

// Re-export the expression handle so downstream code does not need a separate
// `oximo-expr` import.
pub use oximo_expr::{
    AffineBuilder, Children, EvalError, Expr, ExprArena, ExprArenaCell, ExprArenaSnapshot,
    ExprArenaWriteGuard, ExprId, ExprNode, ModelId, ModelMismatchError, ParamId, UnaryOp, VarId,
    describe_nonlinear_term, dot, render_expr, render_linear_terms,
};

pub use oximo_macros::{
    constraint, indicator_constraint, max, min, objective, param, psd_constraint, set,
    soc_constraint, sos_constraint, sum, symmetric_variable, variable,
};

pub use crate::function_set::{
    AffineConstraintIr, AnyScalarFunction, Constraint, EqualTo, Function, FunctionInSet,
    GreaterThan, Interval, IntoAffineConstraint, IntoAffineFunction, IntoFunction, LessThan,
    LowerConstraint, ScalarAffineFunction, ScalarDynamicFunction, ScalarNonlinearFunction,
    ScalarQuadraticFunction, SecondOrderCone, Set as ConstraintSet, VectorAffineFunction,
};
pub use oximo_expr::{Affine, Constant, Degree, Dynamic, Nonlinear, Quadratic};

pub use matrix::{MatrixScalar, SymmetricMatrix, SymmetricMatrixError, triangle_len};
pub use psd::{
    IntoAffineMatrixEntry, IntoSymmetricAffineFunction, PositiveSemidefiniteCone, PsdConstraint,
    PsdConstraintHandle, PsdConstraintId, PsdConstraintIr, SymmetricAffineFunction,
};
