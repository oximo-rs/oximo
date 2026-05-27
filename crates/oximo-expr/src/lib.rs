#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod arena;
mod classify;
mod eval;
mod handle;
mod linear;
mod ops;
mod simplify;
mod visit;

pub use arena::{ExprArena, ExprId, ExprNode, ParamId, VarId};
pub use classify::{ExprClass, classify};
pub use eval::{EvalContext, EvalError, evaluate};
pub use handle::Expr;
pub use linear::{LinearTerms, extract_linear};
pub use ops::dot;
pub use simplify::simplify;
pub use visit::{Visitor, walk};
