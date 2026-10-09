#![doc = include_str!("../README.md")]
#![doc = r"
## API map

The most commonly used modeling and solver types are re-exported at the crate
root and by [`prelude`]:

- Build models with [`Model`], [`variable!`], [`constraint!`], and [`objective!`].
- Create indexed families with [`Set`], [`IndexedVar`], and [`IndexedParam`].
- Compose algebraic expressions with [`Expr`], [`sum!`], [`min!`], and [`max!`].
- Select a backend from [`solvers`] and solve through the [`Solver`] trait.
- Inspect termination, primal, and dual information in [`SolverResult`].
- Read or write optimization formats through the `io` module when that feature is enabled.

Lower-level modeling, expression, and solver APIs remain available through
[`core`], [`expr`], and [`solver`], respectively.
"]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Lets the `::oximo::...` paths emitted by `oximo-macros` resolve inside this
// crate's own examples, tests, and doctests.
extern crate self as oximo;

pub use oximo_core as core;
pub use oximo_core::prelude::*;

// Runtime glue the modeling macros expand into.
#[doc(hidden)]
pub use oximo_core::__macro_support;
pub use oximo_expr as expr;
pub use oximo_solver as solver;
pub use oximo_solver::{
    ConstraintEvaluation, DualStatus, HasUniversal, Iis, IisReport, InfeasibilityDiagnosis,
    ModelReport, PersistentSolver, PrimalStatus, SocEvaluation, SolutionPoint, Solver, SolverError,
    SolverResult, TerminationStatus, UniversalOptions, UniversalOptionsExt, VarBoundKind,
};

#[cfg(feature = "gdp")]
pub use oximo_gdp as gdp;
#[cfg(feature = "gdp")]
pub use oximo_gdp::{
    BigM, BigMValues, BooleanValueError, GdpError, GdpMethod, GdpModelExt, GdpReformulationOptions,
    GdpResultExt, GdpSolveError, GdpSolver, GdpSolverExt,
};

#[cfg(any(feature = "scip", feature = "scip-system"))]
pub use oximo_scip::{ScipOptions, ScipPersistent, ScipSetting};

/// SCIP backend and plugin interfaces.
#[cfg(any(feature = "scip", feature = "scip-system"))]
pub mod scip {
    pub use oximo_scip::*;
}

#[cfg(feature = "io")]
#[cfg_attr(docsrs, doc(cfg(feature = "io")))]
pub use oximo_io as io;

#[cfg(feature = "highs")]
#[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
pub use oximo_highs::{HighsMethod, HighsOptions, HighsPresolve};

#[cfg(feature = "gurobi")]
#[cfg_attr(docsrs, doc(cfg(feature = "gurobi")))]
pub use oximo_gurobi::{
    Callback, CallbackLocation, CallbackMask, CbResult, GRB_METHOD_PDHG, GurobiOptions,
    GurobiPersistent, GurobiPresolve, Opcode, Status, Where,
};

#[cfg(feature = "mosek")]
#[cfg_attr(docsrs, doc(cfg(feature = "mosek")))]
pub use oximo_mosek::{MosekOptions, MosekPersistent};

#[cfg(feature = "gams")]
#[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
pub use oximo_gams::{GamsOptions, GamsSolver};

#[cfg(feature = "baron")]
#[cfg_attr(docsrs, doc(cfg(feature = "baron")))]
pub use oximo_baron::BaronOptions;

#[cfg(feature = "clarabel")]
#[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
pub use oximo_clarabel::{ClarabelDirectSolve, ClarabelOptions};

#[cfg(feature = "clarabel-sdp")]
#[cfg_attr(docsrs, doc(cfg(feature = "clarabel-sdp")))]
pub use oximo_clarabel::ClarabelChordalMerge;

#[cfg(feature = "pounce")]
#[cfg_attr(docsrs, doc(cfg(feature = "pounce")))]
pub use oximo_pounce::{
    MuStrategy, PounceAlgorithm, PounceOptionValue, PounceOptions, PounceSolverSelection,
};

/// GAMS backend types: sub-solver selection and per-solver option structs.
#[cfg(feature = "gams")]
#[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
pub mod gams {
    pub use oximo_gams::*;
}

#[cfg(feature = "pounce")]
#[cfg_attr(docsrs, doc(cfg(feature = "pounce")))]
pub mod pounce {
    //! POUNCE backend options, algorithm selection, and persistent solver types.

    pub use oximo_pounce::{
        MuStrategy, Pounce, PounceAlgorithm, PounceOptionValue, PounceOptions, PouncePersistent,
        PounceSolverSelection,
    };
}

pub mod prelude {
    //! Glob-import target. Brings the modeling and solver surface into scope.
    pub use oximo_core::prelude::*;
    #[cfg(feature = "gdp")]
    pub use oximo_gdp::{
        BigM, BigMValues, BooleanValueError, GdpError, GdpMethod, GdpModelExt,
        GdpReformulationOptions, GdpResultExt, GdpSolveError, GdpSolver, GdpSolverExt,
    };
    pub use oximo_solver::{
        ConstraintEvaluation, DualStatus, HasUniversal, Iis, IisReport, InfeasibilityDiagnosis,
        ModelReport, PersistentSolver, PrimalStatus, SocEvaluation, SolutionPoint, Solver,
        SolverError, SolverResult, TerminationStatus, UniversalOptions, UniversalOptionsExt,
        VarBoundKind,
    };

    #[cfg(feature = "highs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
    pub use oximo_highs::{HighsMethod, HighsOptions, HighsPersistent, HighsPresolve};

    #[cfg(feature = "gurobi")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gurobi")))]
    pub use oximo_gurobi::{
        Callback, CallbackLocation, CallbackMask, CbResult, GRB_METHOD_PDHG, GurobiOptions,
        GurobiPersistent, GurobiPresolve, Opcode, Status, Where,
    };

    #[cfg(feature = "mosek")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mosek")))]
    pub use oximo_mosek::{MosekOptions, MosekPersistent};

    #[cfg(feature = "gams")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gams")))]
    pub use oximo_gams::{GamsOptions, GamsSolver};

    #[cfg(feature = "baron")]
    #[cfg_attr(docsrs, doc(cfg(feature = "baron")))]
    pub use oximo_baron::BaronOptions;

    #[cfg(feature = "clarabel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
    pub use oximo_clarabel::{ClarabelDirectSolve, ClarabelOptions, ClarabelPersistent};

    #[cfg(feature = "clarabel-sdp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "clarabel-sdp")))]
    pub use oximo_clarabel::ClarabelChordalMerge;

    #[cfg(feature = "pounce")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pounce")))]
    pub use oximo_pounce::{
        Pounce, PounceAlgorithm, PounceOptions, PouncePersistent, PounceSolverSelection,
    };
}

pub mod solvers {
    //! Concrete solver backends, gated by cargo features.

    #[cfg(any(feature = "scip", feature = "scip-system"))]
    pub use oximo_scip::Scip;

    #[cfg(feature = "highs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
    pub use oximo_highs::Highs;

    #[cfg(feature = "gurobi")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gurobi")))]
    pub use oximo_gurobi::Gurobi;

    #[cfg(feature = "mosek")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mosek")))]
    pub use oximo_mosek::Mosek;

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
    #[cfg_attr(docsrs, doc(cfg(feature = "pounce")))]
    pub use oximo_pounce::Pounce;
}
