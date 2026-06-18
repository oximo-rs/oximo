#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

extern crate self as oximo_core;

#[doc(hidden)]
#[path = "macro_support.rs"]
pub mod __macro_support;
pub mod constraint;
pub mod domain;
pub mod error;
pub mod indexed;
pub mod model;
pub mod objective;
pub mod param;
pub mod prelude;
pub mod set;
pub mod sum;
pub mod var;

pub use constraint::{Constraint, ConstraintExpr, ConstraintId, IntoRhs, Relate, Sense};
pub use domain::Domain;
pub use error::{Error, Result};
pub use indexed::IndexedVar;
pub use model::{IndexedVarBuilder, Model, ModelKind, display_index_key};
pub use objective::{Objective, ObjectiveSense};
pub use param::Parameter;
pub use set::{Axis, FromIndexKey, IndexKey, IndexTuple, KeyCat, ScalarKey, Set, SetIter};
pub use sum::SumDomain;
#[allow(deprecated)]
pub use sum::sum_over;
pub use var::{VarBuilder, Variable};

// Re-export the expression handle so downstream code does not need a separate
// `oximo-expr` import.
pub use oximo_expr::{Expr, ExprArena, ExprId, ExprNode, ParamId, VarId, dot};

pub use oximo_macros::{constraint, objective, param, sum, variable};
