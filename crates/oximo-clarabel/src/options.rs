use oximo_solver::{HasUniversal, UniversalOptions};

// TODO: Enable Pardiso backends

/// Direct linear (KKT) solver method for Clarabel.
///
/// Selects the sparse LDL factorization backend. The built-in `qdldl` and the
/// `auto` default are always present.
#[cfg_attr(
    feature = "faer",
    doc = "The [`Faer`](Self::Faer) backend requires this crate's `faer` feature."
)]
#[cfg_attr(
    not(feature = "faer"),
    doc = "A `Faer` backend is also available behind this crate's `faer` feature."
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClarabelDirectSolve {
    /// Let Clarabel choose (`"auto"`). Clarabel default.
    Auto,
    /// Built-in QDLDL factorization (`"qdldl"`).
    Qdldl,
    /// faer sparse LDL factorization (`"faer"`). Requires the `faer` feature.
    #[cfg(feature = "faer")]
    #[cfg_attr(docsrs, doc(cfg(feature = "faer")))]
    Faer,
}

/// Declare every scalar [`ClarabelOptions`] option once. Its `Option<T>` field
/// and its `#[must_use]` builder are generated from the same entry, keeping the
/// field name identical to the matching `clarabel::solver::CoreSettings` field.
macro_rules! clarabel_options {
    ($($(#[$meta:meta])* $name:ident : $ty:ty),* $(,)?) => {
        /// Clarabel-specific solver options.
        ///
        /// A typed mirror of the tunable, non-feature-gated fields of
        /// `clarabel::solver::CoreSettings`. Every field is `Option`, `None`
        /// leaves Clarabel's own default in place, so the defaults quoted below
        /// are informational.
        ///
        /// The universal `threads` option maps to Clarabel's `max_threads`,
        /// which only affects multithreaded KKT solvers. The default `qdldl`
        /// solver is single-threaded.
        #[derive(Clone, Debug, Default)]
        pub struct ClarabelOptions {
            /// Options shared by every backend.
            pub universal: UniversalOptions,
            /// Direct linear (KKT) solver method. Clarabel default `Auto`.
            pub direct_solve_method: Option<ClarabelDirectSolve>,
            $(
                $(#[$meta])*
                pub $name: Option<$ty>,
            )*
        }

        impl ClarabelOptions {
            /// Select the direct linear (KKT) solver method.
            #[must_use]
            pub fn direct_solve_method(mut self, m: ClarabelDirectSolve) -> Self {
                self.direct_solve_method = Some(m);
                self
            }

            $(
                $(#[$meta])*
                #[must_use]
                pub fn $name(mut self, v: $ty) -> Self {
                    self.$name = Some(v);
                    self
                }
            )*
        }
    };
}

clarabel_options! {
    /// Interior-point iteration limit. Clarabel default `200`.
    max_iter: u32,
    /// Maximum interior-point step length. Clarabel default `0.99`.
    max_step_fraction: f64,
    /// Absolute duality-gap tolerance. Clarabel default `1e-8`.
    tol_gap_abs: f64,
    /// Relative duality-gap tolerance. Clarabel default `1e-8`.
    tol_gap_rel: f64,
    /// Primal/dual feasibility tolerance. Clarabel default `1e-8`.
    tol_feas: f64,
    /// Absolute infeasibility tolerance. Clarabel default `1e-8`.
    tol_infeas_abs: f64,
    /// Relative infeasibility tolerance. Clarabel default `1e-8`.
    tol_infeas_rel: f64,
    /// κ/τ tolerance. Clarabel default `1e-6`.
    tol_ktratio: f64,
    /// Reduced (low-accuracy fallback) absolute duality-gap tolerance.
    /// Clarabel default `5e-5`.
    reduced_tol_gap_abs: f64,
    /// Reduced relative duality-gap tolerance. Clarabel default `5e-5`.
    reduced_tol_gap_rel: f64,
    /// Reduced feasibility tolerance. Clarabel default `1e-4`.
    reduced_tol_feas: f64,
    /// Reduced absolute infeasibility tolerance. Clarabel default `5e-12`.
    reduced_tol_infeas_abs: f64,
    /// Reduced relative infeasibility tolerance. Clarabel default `5e-5`.
    reduced_tol_infeas_rel: f64,
    /// Reduced κ/τ tolerance. Clarabel default `1e-4`.
    reduced_tol_ktratio: f64,
    /// Enable data-equilibration pre-scaling. Clarabel default `true`.
    equilibrate_enable: bool,
    /// Maximum equilibration scaling iterations. Clarabel default `10`.
    equilibrate_max_iter: u32,
    /// Minimum equilibration scaling allowed. Clarabel default `1e-4`.
    equilibrate_min_scaling: f64,
    /// Maximum equilibration scaling allowed. Clarabel default `1e4`.
    equilibrate_max_scaling: f64,
    /// Line-search backtracking factor. Clarabel default `0.8`.
    linesearch_backtrack_step: f64,
    /// Minimum step for asymmetric cones with PrimalDual scaling.
    /// Clarabel default `1e-1`.
    min_switch_step_length: f64,
    /// Minimum step for symmetric cones / asymmetric cones with Dual scaling.
    /// Clarabel default `1e-4`.
    min_terminate_step_length: f64,
    /// Enable KKT static regularization. Clarabel default `true`.
    static_regularization_enable: bool,
    /// KKT static regularization constant. Clarabel default `1e-8`.
    static_regularization_constant: f64,
    /// Static regularization proportional to the max abs diagonal term.
    /// Clarabel default `f64::EPSILON.powi(2)`.
    static_regularization_proportional: f64,
    /// Enable KKT dynamic regularization. Clarabel default `true`.
    dynamic_regularization_enable: bool,
    /// KKT dynamic regularization threshold. Clarabel default `1e-13`.
    dynamic_regularization_eps: f64,
    /// KKT dynamic regularization shift. Clarabel default `2e-7`.
    dynamic_regularization_delta: f64,
    /// KKT direct solve with iterative refinement. Clarabel default `true`.
    iterative_refinement_enable: bool,
    /// Iterative-refinement relative tolerance. Clarabel default `1e-13`.
    iterative_refinement_reltol: f64,
    /// Iterative-refinement absolute tolerance. Clarabel default `1e-12`.
    iterative_refinement_abstol: f64,
    /// Iterative-refinement maximum iterations. Clarabel default `10`.
    iterative_refinement_max_iter: u32,
    /// Iterative-refinement stalling tolerance. Clarabel default `5.0`.
    iterative_refinement_stop_ratio: f64,
    /// Enable presolve constraint reduction. Clarabel default `true`.
    presolve_enable: bool,
    /// Drop structural zeros from sparse inputs (disables parametric updating).
    /// Clarabel default `false`.
    input_sparse_dropzeros: bool,
}

impl HasUniversal for ClarabelOptions {
    fn universal(&self) -> &UniversalOptions {
        &self.universal
    }

    fn universal_mut(&mut self) -> &mut UniversalOptions {
        &mut self.universal
    }
}
