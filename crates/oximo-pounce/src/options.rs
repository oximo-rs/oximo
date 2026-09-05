use oximo_core::ModelKind;
use oximo_solver::{HasUniversal, UniversalOptions};

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
///     `time_limit` -> `max_wall_time`,
///     `verbose` -> `print_level` 5 (else 0) and captures the iteration log
///     into [`SolverResult::raw_log`](oximo_solver::SolverResult::raw_log),
///     `threads` is ignored.
#[derive(Clone, Debug, Default, PartialEq)]
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
    /// POUNCE's general-NLP algorithm. Defaults to the interior-point method
    /// whenever structural routing selects the NLP engine.
    pub algorithm: Option<PounceAlgorithm>,
    /// Structural solver route. [`PounceSolverSelection::Auto`] is used when
    /// omitted and sends provably convex models to POUNCE's specialized
    /// convex engines.
    pub solver_selection: Option<PounceSolverSelection>,
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

/// POUNCE algorithms available through its Rust library API.
///
/// Both algorithms accept every continuous [`ModelKind`] supported by this
/// backend. `ActiveSetSqp` is a general NLP algorithm despite its QP
/// subproblems, so oximo intentionally permits it for LP, QP, QCP, and NLP
/// models. Specialized convex engines are selected separately with
/// [`PounceSolverSelection`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PounceAlgorithm {
    /// POUNCE's IPOPT-lineage primal-dual interior-point method.
    #[default]
    InteriorPoint,
    /// Active-set sequential quadratic programming.
    ActiveSetSqp,
}

/// POUNCE solver route selected after classifying the oximo model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PounceSolverSelection {
    /// Use a specialized convex engine when convexity is certified, otherwise
    /// use the general NLP engine.
    #[default]
    Auto,
    /// Always use the general NLP engine.
    Nlp,
    /// Force the convex LP interior-point route (LP models only).
    LpIpm,
    /// Force the convex QP interior-point route (LP or convex QP).
    QpIpm,
    /// Force POUNCE's direct parametric active-set QP engine.
    QpActiveSet,
    /// Force the conic interior-point route (convex LP/QP/SOCP).
    Socp,
}

impl PounceSolverSelection {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Nlp => "nlp",
            Self::LpIpm => "lp-ipm",
            Self::QpIpm => "qp-ipm",
            Self::QpActiveSet => "qp-active-set",
            Self::Socp => "socp",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "nlp" => Some(Self::Nlp),
            "lp-ipm" => Some(Self::LpIpm),
            "qp-ipm" => Some(Self::QpIpm),
            "qp-active-set" => Some(Self::QpActiveSet),
            "socp" => Some(Self::Socp),
            _ => None,
        }
    }
}

impl PounceAlgorithm {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InteriorPoint => "interior-point",
            Self::ActiveSetSqp => "active-set-sqp",
        }
    }

    pub(crate) const fn supports(self, kind: ModelKind) -> bool {
        match self {
            Self::InteriorPoint | Self::ActiveSetSqp => {
                matches!(
                    kind,
                    ModelKind::LP
                        | ModelKind::QP
                        | ModelKind::QCP
                        | ModelKind::SOCP
                        | ModelKind::NLP
                )
            }
        }
    }
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
        (int, adam_warmup_iters, "adam_warmup_iters"),
        (num, adam_warmup_learning_rate, "adam_warmup_learning_rate"),
        (num, adam_warmup_penalty, "adam_warmup_penalty"),
        (num, adaptive_mu_budget_pin_fraction, "adaptive_mu_budget_pin_fraction"),
        (int, adaptive_mu_max_free_returns, "adaptive_mu_max_free_returns"),
        (num, alpha_red_factor_min, "alpha_red_factor_min"),
        (num, bound_frac, "bound_frac"),
        (str, bound_mult_init_method, "bound_mult_init_method"),
        (num, bound_mult_init_val, "bound_mult_init_val"),
        (num, bound_push, "bound_push"),
        (num, constr_mult_init_max, "constr_mult_init_max"),
        (num, slack_bound_frac, "slack_bound_frac"),
        (num, slack_bound_push, "slack_bound_push"),
        (str, least_square_init_duals, "least_square_init_duals"),
        (str, least_square_init_primal, "least_square_init_primal"),
        (str, start_point_conditioner, "start_point_conditioner"),
        (num, start_point_perturbation, "start_point_perturbation"),
        (int, start_point_perturbation_seed, "start_point_perturbation_seed"),
        (str, warm_start_recentering, "warm_start_recentering"),
        (str, crossover, "crossover"),
        (int, crossover_max_iter, "crossover_max_iter"),
        (num, crossover_mult_tol, "crossover_mult_tol"),
        (num, crossover_primal_tol, "crossover_primal_tol"),
        (str, fd_hessian_coloring, "fd_hessian_coloring"),
        (str, fd_hessian_pattern, "fd_hessian_pattern"),
        (num, fd_hessian_reuse_tol, "fd_hessian_reuse_tol"),
        (int, limited_memory_ls_failure_restarts, "limited_memory_ls_failure_restarts"),
        (int, neg_curv_escapes, "neg_curv_escapes"),
        (int, partitioned_block_size, "partitioned_block_size"),
        (num, partitioned_curvature_cap, "partitioned_curvature_cap"),
        (str, partitioned_elements, "partitioned_elements"),
        (int, partitioned_max_element, "partitioned_max_element"),
        (str, partitioned_update_type, "partitioned_update_type"),
        (bool, feral_increase_quality, "feral_increase_quality"),
        (bool, feral_increase_quality_retry, "feral_increase_quality_retry"),
        (int, feral_refine_steps, "feral_refine_steps"),
        (num, feral_refine_target, "feral_refine_target"),
        (bool, ma57_batched_backsolve, "ma57_batched_backsolve"),
        (bool, dual_divergence_retry, "dual_divergence_retry"),
        (num, dual_divergence_retry_du_floor, "dual_divergence_retry_du_floor"),
        (num, dual_divergence_retry_step_tol, "dual_divergence_retry_step_tol"),
        (bool, infeasibility_perturbed_start_retry, "infeasibility_perturbed_start_retry"),
        (int, perturb_delta_c_max_rungs, "perturb_delta_c_max_rungs"),
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
        (str, linear_solver, "linear_solver"),
        (str, feral_ordering, "feral_ordering"),
        (str, feral_scaling, "feral_scaling"),
        (num, feral_pivtol, "feral_pivtol"),
        (bool, feral_refine, "feral_refine"),
        (bool, feral_cascade_break, "feral_cascade_break"),
        (bool, feral_fma, "feral_fma"),
        (num, feral_singular_pivot_floor, "feral_singular_pivot_floor"),
        // POUNCE convergence, restoration, scaling, and retry controls.
        (num, acceptable_progress_kappa, "acceptable_progress_kappa"),
        (num, dual_inf_scale_kappa, "dual_inf_scale_kappa"),
        (num, feral_inertia_pivot_floor, "feral_inertia_pivot_floor"),
        (bool, infeasibility_mu_strategy_retry, "infeasibility_mu_strategy_retry"),
        (num, primal_noise_floor_kappa, "primal_noise_floor_kappa"),
        (num, qp_tau_max, "qp_tau_max"),
        (int, resto_decline_deferrals, "resto_decline_deferrals"),
        (num, resto_decline_progress_ratio, "resto_decline_progress_ratio"),
        (int, sqp_qp_max_schur_updates_before_refactor, "sqp_qp_max_schur_updates_before_refactor"),
        (bool, sqp_qp_use_homotopy, "sqp_qp_use_homotopy"),
        (bool, sqp_qp_use_schur_updates, "sqp_qp_use_schur_updates"),
        (num, theta_max_adaptive_factor, "theta_max_adaptive_factor"),
        (int, theta_max_adaptive_max_raises, "theta_max_adaptive_max_raises"),
        (int, theta_max_adaptive_trigger, "theta_max_adaptive_trigger"),
        (num, theta_max_row_scale_kappa, "theta_max_row_scale_kappa"),
        // Specialized convex engine controls.
        (bool, qp_presolve, "qp_presolve"),
        (num, qp_tau, "qp_tau"),
        (num, qp_reg, "qp_reg"),
        (num, qp_infeas_tol, "qp_infeas_tol"),
        (bool, qp_hsde, "qp_hsde"),
        (bool, qp_equilibrate, "qp_equilibrate"),
        (bool, qp_crossover, "qp_crossover"),
        (int, qp_gondzio_corr, "qp_gondzio_corr"),
        (bool, feral_infeasibility_scaling_retry, "feral_infeasibility_scaling_retry"),
        // Active-set QP tuning shared by direct QP and NLP-SQP routes.
        (int, sqp_qp_max_iter, "sqp_qp_max_iter"),
        (num, sqp_qp_feas_tol, "sqp_qp_feas_tol"),
        (num, sqp_qp_opt_tol, "sqp_qp_opt_tol"),
        (num, sqp_qp_elastic_gamma, "sqp_qp_elastic_gamma"),
        (str, sqp_qp_anti_cycling, "sqp_qp_anti_cycling"),
        (bool, sqp_qp_certify_second_order, "sqp_qp_certify_second_order"),
        (num, bound_relax_factor, "bound_relax_factor"),
        (num, constr_viol_tol, "constr_viol_tol"),
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

    /// Select POUNCE's top-level algorithm.
    #[must_use]
    pub fn algorithm(mut self, algorithm: PounceAlgorithm) -> Self {
        self.algorithm = Some(algorithm);
        self
    }

    /// Select POUNCE's structural solver route.
    #[must_use]
    pub fn solver_selection(mut self, selection: PounceSolverSelection) -> Self {
        self.solver_selection = Some(selection);
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

    fn effective_value<T>(
        &self,
        name: &str,
        typed: impl Iterator<Item = (&'static str, T)>,
        from_raw: impl Fn(&PounceOptionValue) -> Option<T>,
    ) -> Option<T> {
        let mut value = typed.filter(|(key, _)| *key == name).map(|(_, value)| value).last();
        for (key, raw) in &self.extra {
            if key == name {
                value = from_raw(raw);
            }
        }
        value
    }

    pub(crate) fn effective_num(&self, name: &str) -> Option<f64> {
        self.effective_value(name, self.num_opts.iter().map(|&(key, value)| (key, value)), |raw| {
            match raw {
                PounceOptionValue::Num(value) => Some(*value),
                _ => None,
            }
        })
    }

    pub(crate) fn effective_int(&self, name: &str) -> Option<i32> {
        self.effective_value(name, self.int_opts.iter().map(|&(key, value)| (key, value)), |raw| {
            match raw {
                PounceOptionValue::Int(value) => Some(*value),
                _ => None,
            }
        })
    }

    pub(crate) fn effective_bool(&self, name: &str) -> Option<bool> {
        self.effective_value(name, self.bool_opts.iter().map(|&(key, value)| (key, value)), |raw| {
            match raw {
                PounceOptionValue::Bool(value) => Some(*value),
                PounceOptionValue::Str(value)
                    if matches!(value.as_str(), "yes" | "true" | "on") =>
                {
                    Some(true)
                }
                PounceOptionValue::Str(value)
                    if matches!(value.as_str(), "no" | "false" | "off") =>
                {
                    Some(false)
                }
                _ => None,
            }
        })
    }

    pub(crate) fn effective_string(&self, name: &str) -> Option<String> {
        self.effective_value(
            name,
            self.str_opts.iter().map(|(key, value)| (*key, value.clone())),
            |raw| match raw {
                PounceOptionValue::Str(value) => Some(value.clone()),
                _ => None,
            },
        )
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

    #[test]
    fn pounce_setters_use_the_declared_storage_kinds() {
        let o = PounceOptions::default()
            .solver_selection(PounceSolverSelection::Socp)
            .acceptable_progress_kappa(0.2)
            .resto_decline_deferrals(2)
            .infeasibility_mu_strategy_retry(false)
            .qp_tau(0.9)
            .qp_presolve(true)
            .sqp_qp_anti_cycling("bland");
        assert_eq!(o.solver_selection, Some(PounceSolverSelection::Socp));
        assert!(o.num_opts.contains(&("acceptable_progress_kappa", 0.2)));
        assert!(o.num_opts.contains(&("qp_tau", 0.9)));
        assert!(o.int_opts.contains(&("resto_decline_deferrals", 2)));
        assert!(o.bool_opts.contains(&("infeasibility_mu_strategy_retry", false)));
        assert!(o.bool_opts.contains(&("qp_presolve", true)));
        assert!(o.str_opts.contains(&("sqp_qp_anti_cycling", "bland".to_owned())));
    }

    #[test]
    fn solver_selection_parses_every_public_value() {
        for (text, expected) in [
            ("auto", PounceSolverSelection::Auto),
            ("nlp", PounceSolverSelection::Nlp),
            ("lp-ipm", PounceSolverSelection::LpIpm),
            ("qp-ipm", PounceSolverSelection::QpIpm),
            ("qp-active-set", PounceSolverSelection::QpActiveSet),
            ("socp", PounceSolverSelection::Socp),
        ] {
            assert_eq!(PounceSolverSelection::parse(text), Some(expected));
            assert_eq!(expected.as_str(), text);
        }
        assert_eq!(PounceSolverSelection::parse("unknown"), None);
    }

    #[test]
    fn algorithms_report_their_names_and_supported_model_kinds() {
        for (algorithm, name) in [
            (PounceAlgorithm::InteriorPoint, "interior-point"),
            (PounceAlgorithm::ActiveSetSqp, "active-set-sqp"),
        ] {
            assert_eq!(algorithm.as_str(), name);
            for kind in
                [ModelKind::LP, ModelKind::QP, ModelKind::QCP, ModelKind::SOCP, ModelKind::NLP]
            {
                assert!(algorithm.supports(kind), "{algorithm:?} should support {kind:?}");
            }
            assert!(!algorithm.supports(ModelKind::MILP));
        }
    }

    #[test]
    fn wrong_kind_raw_overrides_clear_typed_values() {
        let options = PounceOptions::default()
            .qp_tau(0.9)
            .set("qp_tau", true)
            .sqp_qp_max_iter(20)
            .set("sqp_qp_max_iter", "wrong")
            .qp_presolve(true)
            .set("qp_presolve", 1.0)
            .sqp_qp_anti_cycling("bland")
            .set("sqp_qp_anti_cycling", false);
        assert_eq!(options.effective_num("qp_tau"), None);
        assert_eq!(options.effective_int("sqp_qp_max_iter"), None);
        assert_eq!(options.effective_bool("qp_presolve"), None);
        assert_eq!(options.effective_string("sqp_qp_anti_cycling"), None);
    }
}
