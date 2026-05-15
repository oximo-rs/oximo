//! Oximo: Rust algebraic modeling language for mathematical optimization.
//!
//! ```no_run
//! use oximo::prelude::*;
//!
//! let m = Model::new("toy");
//! let x = m.var("x").lb(0.0).build();
//! let y = m.var("y").lb(0.0).ub(4.0).build();
//! m.constraint("c1", (x + 2.0 * y).le(14.0));
//! m.minimize(3.0 * x + 4.0 * y);
//!
//! let mut solver = oximo::solvers::Highs;
//! let result = solver.solve(&m, &oximo::HighsOptions::default()).unwrap();
//! assert!(result.status.has_solution());
//! ```
#![forbid(unsafe_code)]

pub use oximo_core as core;
pub use oximo_expr as expr;
pub use oximo_solver as solver;

#[cfg(feature = "io")]
pub use oximo_io as io;

#[cfg(feature = "highs")]
pub use oximo_highs::{HighsMethod, HighsOptions, HighsPresolve};

#[cfg(feature = "gurobi")]
pub use oximo_gurobi::{GurobiOptions, GurobiPresolve};
pub mod prelude {
    //! Glob-import target. Brings the modeling and solver surface into scope.
    pub use oximo_core::prelude::*;
    pub use oximo_solver::{
        HasUniversal, Solver, SolverError, SolverResult, SolverStatus, UniversalOptions,
        UniversalOptionsExt,
    };

    #[cfg(feature = "highs")]
    pub use oximo_highs::{HighsMethod, HighsOptions, HighsPresolve};

    #[cfg(feature = "gurobi")]
    pub use oximo_gurobi::{GurobiOptions, GurobiPresolve};
}

pub mod solvers {
    //! Concrete solver backends, gated by cargo features.

    #[cfg(feature = "highs")]
    pub use oximo_highs::Highs;

    #[cfg(feature = "gurobi")]
    pub use oximo_gurobi::Gurobi;

    #[cfg(feature = "gams")]
    pub use oximo_gams::Gams;
}
