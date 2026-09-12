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
pub mod indexed;
pub mod model;
pub mod objective;
pub mod param;
pub mod prelude;
pub mod reformulation;
pub mod set;
pub mod soc;
pub mod sos;
pub mod sum;
pub mod var;

pub use constraint::{
    Constraint, ConstraintExpr, ConstraintId, IntoRhs, RangeConstraintIds, Relate, Sense,
};
pub use display::{ConstraintDisplay, ExprDisplay, ObjectiveDisplay, SocDisplay, SosDisplay};
pub use domain::Domain;
pub use error::{Error, Result};
pub use indexed::{IndexedConstraint, IndexedParam, IndexedRangeConstraint, IndexedVar};
pub use model::{
    ConstraintRef, IndexedVarBuilder, Model, ModelConstraints, ModelKind, display_index_key,
};
pub use objective::{Objective, ObjectiveSense};
pub use param::Parameter;
pub use reformulation::{
    ReformulatedModel, ReformulationError, SosReformulationArtifacts, SosReformulationOptions,
};
pub use set::{
    Axis, FromIndexKey, FromIndexKeyRef, IndexKey, IndexKeyRef, IndexTuple, KeyCat, ScalarKey, Set,
    SetIter,
};
#[doc(hidden)]
pub use soc::__detect_soc_from_quadratic;
pub use soc::{SocConstraint, SocConstraintId, SocForm};
pub use sos::{SosConstraint, SosConstraintHandle, SosConstraintId, SosMember, SosType};
pub use sum::SumDomain;
pub use var::{VarBuilder, Variable, var_name};

// Re-export the expression handle so downstream code does not need a separate
// `oximo-expr` import.
pub use oximo_expr::{
    Children, EvalError, Expr, ExprArena, ExprArenaCell, ExprArenaSnapshot, ExprArenaWriteGuard,
    ExprId, ExprNode, ParamId, UnaryOp, VarId, describe_nonlinear_term, dot, render_expr,
    render_linear_terms,
};

pub use oximo_macros::{
    constraint, max, min, objective, param, set, soc_constraint, sos_constraint, sum, variable,
};
