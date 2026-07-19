#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Lets the `::oximo::...` paths emitted by `oximo-macros` resolve inside this
// crate's own examples, tests, and doctests.
extern crate self as oximo;

pub use oximo_core as core;

// Runtime glue the modeling macros expand into.
#[doc(hidden)]
pub use oximo_core::__macro_support;
pub use oximo_expr as expr;
pub use oximo_solver as solver;

#[cfg(feature = "io")]
#[cfg_attr(docsrs, doc(cfg(feature = "io")))]
pub use oximo_io as io;

#[cfg(feature = "highs")]
#[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
pub use oximo_highs::{HighsMethod, HighsOptions, HighsPresolve};

#[cfg(feature = "gurobi")]
#[cfg_attr(docsrs, doc(cfg(feature = "gurobi")))]
pub use oximo_gurobi::{GurobiOptions, GurobiPersistent, GurobiPresolve};

#[cfg(feature = "gams")]
#[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
pub use oximo_gams::{GamsOptions, GamsSolver};

#[cfg(feature = "baron")]
#[cfg_attr(docsrs, doc(cfg(feature = "baron")))]
pub use oximo_baron::BaronOptions;

#[cfg(feature = "clarabel")]
#[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
pub use oximo_clarabel::{ClarabelDirectSolve, ClarabelOptions};

/// GAMS backend types: sub-solver selection and per-solver option structs.
#[cfg(feature = "gams")]
#[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
pub mod gams {
    pub use oximo_gams::{
        GamsBaronOptions, GamsCbcCuts, GamsCbcOptions, GamsCbcPresolve, GamsCplexMipEmphasis,
        GamsCplexOptions, GamsGurobiMipFocus, GamsGurobiOptions, GamsHighsOptions,
        GamsHighsPresolve, GamsHighsSolver, GamsIpoptLinearSolver, GamsIpoptMuStrategy,
        GamsIpoptOptions, GamsKnitroAlgorithm, GamsKnitroOptions, GamsMosekOptions, GamsOptions,
        GamsScipOptions, GamsSolver, GamsSolverConfig, GamsXpressOptions,
    };
}

#[cfg(feature = "pounce")]
pub mod pounce {
    pub use oximo_pounce::{
        MuStrategy, Pounce, PounceOptionValue, PounceOptions, PouncePersistent,
    };
}

pub mod prelude {
    //! Glob-import target. Brings the modeling and solver surface into scope.
    pub use oximo_core::prelude::*;
    pub use oximo_solver::{
        HasUniversal, ModelReport, PersistentSolver, PrimalStatus, SolutionPoint, Solver,
        SolverError, SolverResult, TerminationStatus, UniversalOptions, UniversalOptionsExt,
    };

    #[cfg(feature = "highs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
    pub use oximo_highs::{HighsMethod, HighsOptions, HighsPersistent, HighsPresolve};

    #[cfg(feature = "gurobi")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gurobi")))]
    pub use oximo_gurobi::{GurobiOptions, GurobiPersistent, GurobiPresolve};

    #[cfg(feature = "gams")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
    pub use oximo_gams::{GamsOptions, GamsSolver};

    #[cfg(feature = "baron")]
    #[cfg_attr(docsrs, doc(cfg(feature = "baron")))]
    pub use oximo_baron::BaronOptions;

    #[cfg(feature = "clarabel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
    pub use oximo_clarabel::{ClarabelDirectSolve, ClarabelOptions, ClarabelPersistent};

    #[cfg(feature = "pounce")]
    pub use oximo_pounce::{Pounce, PounceOptions, PouncePersistent};
}

pub mod solvers {
    //! Concrete solver backends, gated by cargo features.

    #[cfg(feature = "highs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
    pub use oximo_highs::Highs;

    #[cfg(feature = "gurobi")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gurobi")))]
    pub use oximo_gurobi::Gurobi;

    #[cfg(feature = "gams")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
    pub use oximo_gams::Gams;

    #[cfg(feature = "baron")]
    #[cfg_attr(docsrs, doc(cfg(feature = "baron")))]
    pub use oximo_baron::Baron;

    #[cfg(feature = "clarabel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
    pub use oximo_clarabel::Clarabel;

    #[cfg(feature = "pounce")]
    pub use oximo_pounce::Pounce;
}
