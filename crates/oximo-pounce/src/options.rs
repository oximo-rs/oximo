use oximo_solver::{HasUniversal, UniversalOptions};

// TODO: Add the rest of the POUNCE options, when v0.9.0 is released.

/// POUNCE-specific solver options.
///
/// For more information about POUNCE's options, see the
/// [documented option reference](https://kitchingroup.cheme.cmu.edu/pounce/options.html).
///
/// Invalid option names or out-of-range values are reported by POUNCE and
/// surface as a [`SolverError::Backend`](oximo_solver::SolverError::Backend) at
/// solve time.
///
/// `UniversalOptions` mapping:
///     `time_limit` -> `max_cpu_time`,
///     `verbose` -> `print_level` 5 (else 0)
///     `threads` is ignored.
#[derive(Clone, Debug, Default)]
pub struct PounceOptions {
    pub universal: UniversalOptions,
    /// Desired convergence tolerance (`tol`).
    pub tol: Option<f64>,
    /// Iteration limit (`max_iter`).
    pub max_iter: Option<u32>,
    /// Output verbosity 0–12 (`print_level`); overrides `verbose`.
    pub print_level: Option<u32>,
    /// Barrier parameter update strategy (`mu_strategy`).
    pub mu_strategy: Option<MuStrategy>,
    /// Macro-generated typed options, kept by value kind and applied in order.
    num_opts: Vec<(&'static str, f64)>,
    int_opts: Vec<(&'static str, i32)>,
    str_opts: Vec<(&'static str, String)>,
    bool_opts: Vec<(&'static str, bool)>,
    /// Escape hatch: raw POUNCE options applied last.
    pub extra: Vec<(String, PounceOptionValue)>,
}

/// `mu_strategy` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuStrategy {
    Monotone,
    Adaptive,
}

/// A raw POUNCE option value for [`PounceOptions::extra`].
#[derive(Clone, Debug, PartialEq)]
pub enum PounceOptionValue {
    Num(f64),
    Int(i32),
    Str(String),
    Bool(bool),
}

impl From<f64> for PounceOptionValue {
    fn from(v: f64) -> Self {
        Self::Num(v)
    }
}

impl From<i32> for PounceOptionValue {
    fn from(v: i32) -> Self {
        Self::Int(v)
    }
}

impl From<&str> for PounceOptionValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
}

impl From<bool> for PounceOptionValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

// Generates one typed builder method per POUNCE option, keyed by value kind.
// The method name matches the option string.
macro_rules! pounce_options {
    ($( ($kind:ident, $method:ident, $tag:literal) ),* $(,)?) => {
        $(pounce_options!(@impl $kind, $method, $tag);)*
    };
    (@impl num, $method:ident, $tag:literal) => {
        #[doc = concat!("Set the POUNCE `", $tag, "` option.")]
        #[must_use]
        pub fn $method(mut self, v: f64) -> Self {
            self.num_opts.push(($tag, v));
            self
        }
    };
    (@impl int, $method:ident, $tag:literal) => {
        #[doc = concat!("Set the POUNCE `", $tag, "` option.")]
        #[must_use]
        pub fn $method(mut self, v: i32) -> Self {
            self.int_opts.push(($tag, v));
            self
        }
    };
    (@impl str, $method:ident, $tag:literal) => {
        #[doc = concat!("Set the POUNCE `", $tag, "` option.")]
        #[must_use]
        pub fn $method(mut self, v: impl Into<String>) -> Self {
            self.str_opts.push(($tag, v.into()));
            self
        }
    };
    (@impl bool, $method:ident, $tag:literal) => {
        #[doc = concat!("Set the POUNCE `", $tag, "` option.")]
        #[must_use]
        pub fn $method(mut self, v: bool) -> Self {
            self.bool_opts.push(($tag, v));
            self
        }
    };
}

impl PounceOptions {
    pounce_options!(
        // Barrier-parameter (μ) strategy (`mu_strategy` has a dedicated setter)
        (str, mu_oracle, "mu_oracle"),
        (num, mu_init, "mu_init"),
        (num, mu_min, "mu_min"),
        (num, mu_max, "mu_max"),
        (num, mu_max_fact, "mu_max_fact"),
        (num, mu_target, "mu_target"),
        (num, mu_linear_decrease_factor, "mu_linear_decrease_factor"),
        (num, mu_superlinear_decrease_power, "mu_superlinear_decrease_power"),
        (num, barrier_tol_factor, "barrier_tol_factor"),
        (num, sigma_max, "sigma_max"),
        (num, sigma_min, "sigma_min"),
        (str, adaptive_mu_globalization, "adaptive_mu_globalization"),
        // Quality-function oracle
        (str, quality_function_norm_type, "quality_function_norm_type"),
        (str, quality_function_centrality, "quality_function_centrality"),
        (str, quality_function_balancing_term, "quality_function_balancing_term"),
        (int, quality_function_max_section_steps, "quality_function_max_section_steps"),
        (num, quality_function_section_sigma_tol, "quality_function_section_sigma_tol"),
        (num, quality_function_section_qf_tol, "quality_function_section_qf_tol"),
        // Adaptive-μ globalization
        (num, adaptive_mu_safeguard_factor, "adaptive_mu_safeguard_factor"),
        (num, adaptive_mu_monotone_init_factor, "adaptive_mu_monotone_init_factor"),
        (bool, adaptive_mu_restore_previous_iterate, "adaptive_mu_restore_previous_iterate"),
        (int, adaptive_mu_kkterror_red_iters, "adaptive_mu_kkterror_red_iters"),
        (num, adaptive_mu_kkterror_red_fact, "adaptive_mu_kkterror_red_fact"),
        (str, adaptive_mu_kkt_norm_type, "adaptive_mu_kkt_norm_type"),
        // L1 penalty-barrier wrapper
        (bool, l1_exact_penalty_barrier, "l1_exact_penalty_barrier"),
        (bool, l1_fallback_on_restoration_failure, "l1_fallback_on_restoration_failure"),
        (num, l1_penalty_init, "l1_penalty_init"),
        (num, l1_penalty_max, "l1_penalty_max"),
        (num, l1_penalty_increase_factor, "l1_penalty_increase_factor"),
        (int, l1_penalty_max_outer_iter, "l1_penalty_max_outer_iter"),
        (num, l1_slack_tol, "l1_slack_tol"),
        (num, l1_steering_factor, "l1_steering_factor"),
        // NLP presolve
        (bool, presolve, "presolve"),
        (bool, presolve_bound_tightening, "presolve_bound_tightening"),
        (bool, presolve_redundant_constraint_removal, "presolve_redundant_constraint_removal"),
        (bool, presolve_linear_eq_reduction, "presolve_linear_eq_reduction"),
        (bool, presolve_licq_check, "presolve_licq_check"),
        (str, presolve_licq_action, "presolve_licq_action"),
        (bool, presolve_warm_z_bounds, "presolve_warm_z_bounds"),
        (num, presolve_bound_mult_init_val, "presolve_bound_mult_init_val"),
        (int, presolve_max_passes, "presolve_max_passes"),
        (int, presolve_print_level, "presolve_print_level"),
        // Feasibility-based bound tightening
        (bool, presolve_fbbt, "presolve_fbbt"),
        (num, fbbt_tol, "fbbt_tol"),
        (int, fbbt_max_iter, "fbbt_max_iter"),
        (int, fbbt_max_constraints, "fbbt_max_constraints"),
        // Auxiliary-equality preprocessing
        (bool, presolve_auxiliary, "presolve_auxiliary"),
        (str, presolve_auxiliary_coupling, "presolve_auxiliary_coupling"),
        (num, presolve_auxiliary_tol, "presolve_auxiliary_tol"),
        (int, presolve_auxiliary_max_block_dim, "presolve_auxiliary_max_block_dim"),
        (num, presolve_auxiliary_wall_time_fraction, "presolve_auxiliary_wall_time_fraction"),
        (bool, presolve_auxiliary_diagnostics, "presolve_auxiliary_diagnostics"),
        // FERAL backend (pure-Rust sparse symmetric linear solver).
        // `feral_infeasibility_scaling_retry` is registered but never read by
        // the library (the retry is implemented by the POUNCE CLI), so setting
        // it would silently do nothing — it is deliberately not exposed.
        (str, linear_solver, "linear_solver"),
        (str, feral_ordering, "feral_ordering"),
        (str, feral_scaling, "feral_scaling"),
        (num, feral_pivtol, "feral_pivtol"),
        (bool, feral_refine, "feral_refine"),
        (bool, feral_cascade_break, "feral_cascade_break"),
        (bool, feral_fma, "feral_fma"),
        (num, feral_singular_pivot_floor, "feral_singular_pivot_floor"),
    );

    #[must_use]
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = Some(tol);
        self
    }

    #[must_use]
    pub fn max_iter(mut self, n: u32) -> Self {
        self.max_iter = Some(n);
        self
    }

    #[must_use]
    pub fn print_level(mut self, level: u32) -> Self {
        self.print_level = Some(level);
        self
    }

    #[must_use]
    pub fn mu_strategy(mut self, s: MuStrategy) -> Self {
        self.mu_strategy = Some(s);
        self
    }

    /// Set a raw POUNCE option by name (the escape hatch for anything not
    /// covered by a typed setter). Applied last, so it overrides the typed
    /// options. An unknown name or invalid value fails the solve.
    #[must_use]
    pub fn set(mut self, name: impl Into<String>, value: impl Into<PounceOptionValue>) -> Self {
        self.extra.push((name.into(), value.into()));
        self
    }

    pub(crate) fn num_opts(&self) -> &[(&'static str, f64)] {
        &self.num_opts
    }

    pub(crate) fn int_opts(&self) -> &[(&'static str, i32)] {
        &self.int_opts
    }

    pub(crate) fn str_opts(&self) -> &[(&'static str, String)] {
        &self.str_opts
    }

    pub(crate) fn bool_opts(&self) -> &[(&'static str, bool)] {
        &self.bool_opts
    }
}

impl HasUniversal for PounceOptions {
    fn universal(&self) -> &UniversalOptions {
        &self.universal
    }

    fn universal_mut(&mut self) -> &mut UniversalOptions {
        &mut self.universal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_setters_push_onto_the_right_vecs() {
        let o = PounceOptions::default()
            .mu_oracle("probing")
            .mu_init(0.05)
            .presolve(true)
            .presolve_max_passes(5)
            .feral_refine(false);
        assert_eq!(o.str_opts, vec![("mu_oracle", "probing".to_owned())]);
        assert_eq!(o.num_opts, vec![("mu_init", 0.05)]);
        assert_eq!(o.int_opts, vec![("presolve_max_passes", 5)]);
        assert_eq!(o.bool_opts, vec![("presolve", true), ("feral_refine", false)]);
    }

    #[test]
    fn default_vecs_are_empty() {
        let o = PounceOptions::default();
        assert!(o.num_opts.is_empty());
        assert!(o.int_opts.is_empty());
        assert!(o.str_opts.is_empty());
        assert!(o.bool_opts.is_empty());
        assert!(o.extra.is_empty());
    }

    #[test]
    fn same_option_twice_keeps_both_entries() {
        let o = PounceOptions::default().mu_init(0.1).mu_init(0.5);
        assert_eq!(o.num_opts, vec![("mu_init", 0.1), ("mu_init", 0.5)]);
    }

    #[test]
    fn clone_preserves_all_vecs() {
        let o = PounceOptions::default().mu_init(0.1).presolve_max_passes(2).presolve(true);
        let c = o.clone();
        assert_eq!(o.num_opts, c.num_opts);
        assert_eq!(o.int_opts, c.int_opts);
        assert_eq!(o.bool_opts, c.bool_opts);
    }

    #[test]
    fn set_pushes_onto_extra_with_bool() {
        let o = PounceOptions::default().set("presolve", true).set("acceptable_tol", 1e-5);
        assert_eq!(
            o.extra,
            vec![
                ("presolve".to_owned(), PounceOptionValue::Bool(true)),
                ("acceptable_tol".to_owned(), PounceOptionValue::Num(1e-5)),
            ]
        );
    }
}
