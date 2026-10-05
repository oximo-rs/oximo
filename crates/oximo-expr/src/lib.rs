#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod affine_builder;
mod arena;
mod classify;
pub mod degree;
mod eval;
mod fold;
mod handle;
mod linear;
mod ops;
mod quadratic;
mod render;
mod simplify;
mod visit;

pub use affine_builder::AffineBuilder;
pub use arena::{
    Children, ExprArena, ExprArenaBatchGuard, ExprArenaCell, ExprArenaSnapshot,
    ExprArenaWriteGuard, ExprId, ExprIdRemap, ExprNode, ForkOutput, FrozenExprArena, ModelId,
    ModelMismatchError, ParamId, UnaryOp, VarId,
};
pub use classify::{ExprClass, classify};
pub use degree::{Affine, Constant, Degree, Dynamic, Nonlinear, Quadratic};
pub use eval::{EvalContext, EvalError, evaluate};
pub use handle::{ClassifiedExpr, Expr, ExprData};
pub use linear::{LinearTerms, SignedExpr, describe_nonlinear_term, extract_linear, split_linear};
pub use ops::dot;
pub use quadratic::{QuadraticTerms, extract_quadratic};
pub use render::{render_expr, render_linear_terms};
pub use simplify::simplify;
pub use visit::{Visitor, walk, walk_shared};
