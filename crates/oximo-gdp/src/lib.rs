#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

extern crate self as oximo_gdp;

mod bounds;
mod reformulate;
mod result;
mod solver;

pub use oximo_core as core;
#[doc(hidden)]
pub use oximo_core::__macro_support;
pub use oximo_core::gdp::*;
pub use oximo_core::prelude::{
    boolean_variable, disjunct, disjunct_constraint, disjunction, logical_constraint,
};
pub use oximo_core::{
    BooleanDisplay, DisjunctConstraintDisplay, DisjunctionDisplay, GdpDisplay,
    LogicalConstraintDisplay, LogicalExprDisplay,
};
pub use reformulate::*;
pub use result::{BooleanValueError, GdpResultExt};
pub use solver::{GdpSolveError, GdpSolver, GdpSolverExt};

pub mod prelude {
    pub use crate::reformulate::*;
    pub use crate::result::{BooleanValueError, GdpResultExt};
    pub use crate::solver::{GdpSolveError, GdpSolver, GdpSolverExt};
    pub use oximo_core::prelude::*;
}
