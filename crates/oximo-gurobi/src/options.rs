use gurobi_rs::Model as GurobiModel;
use gurobi_rs::parameter::{DoubleParam, IntParam, StrParam};
use oximo_solver::{HasUniversal, UniversalOptions};

/// Gurobi presolve level. Maps to the Gurobi `Presolve` parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GurobiPresolve {
    /// Automatic (`-1`). Let Gurobi decide.
    Auto,
    /// Off (`0`).
    Off,
    /// Conservative (`1`).
    Conservative,
    /// Aggressive (`2`).
    Aggressive,
}

/// Gurobi-specific solver options.
///
/// Universal options (`time_limit`, `threads`, `verbose`) come from the embedded
/// [`UniversalOptions`] via [`UniversalOptionsExt`](oximo_solver::UniversalOptionsExt).
/// All other options are Gurobi-specific. See the
/// [Gurobi parameter reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html)
/// for semantics. Method names are the snake_case of the official parameter
/// name (e.g. `ConcurrentMIP` -> `.concurrent_mip(2)`).
#[derive(Clone, Debug, Default)]
pub struct GurobiOptions {
    pub universal: UniversalOptions,
    pub mip_gap: Option<f64>,
    pub presolve: Option<GurobiPresolve>,
    int_params: Vec<(IntParam, i32)>,
    double_params: Vec<(DoubleParam, f64)>,
    str_params: Vec<(StrParam, String)>,
}

// Generates one typed builder method per Gurobi parameter.
macro_rules! gurobi_params {
    ($( ($kind:ident, $method:ident, $variant:ident) ),* $(,)?) => {
        $(gurobi_params!(@impl $kind, $method, $variant);)*
    };
    (@impl int, $method:ident, $variant:ident) => {
        #[must_use]
        pub fn $method(mut self, v: i32) -> Self {
            self.int_params.push((IntParam::$variant, v));
            self
        }
    };
    (@impl dbl, $method:ident, $variant:ident) => {
        #[must_use]
        pub fn $method(mut self, v: f64) -> Self {
            self.double_params.push((DoubleParam::$variant, v));
            self
        }
    };
    (@impl str, $method:ident, $variant:ident) => {
        #[must_use]
        pub fn $method(mut self, v: impl Into<String>) -> Self {
            self.str_params.push((StrParam::$variant, v.into()));
            self
        }
    };
}

impl GurobiOptions {
    gurobi_params!(
        // Termination
        (int, bar_iter_limit, BarIterLimit),
        (int, solution_limit, SolutionLimit),
        (dbl, cutoff, Cutoff),
        (dbl, iteration_limit, IterationLimit),
        (dbl, node_limit, NodeLimit),
        (dbl, best_obj_stop, BestObjStop),
        (dbl, best_bd_stop, BestBdStop),
        (dbl, work_limit, WorkLimit),
        // Tolerances
        (dbl, feasibility_tol, FeasibilityTol),
        (dbl, int_feas_tol, IntFeasTol),
        (dbl, markowitz_tol, MarkowitzTol),
        (dbl, mip_gap_abs, MIPGapAbs),
        (dbl, optimality_tol, OptimalityTol),
        (dbl, psd_tol, PSDTol),
        // Simplex
        (int, method, Method),
        (dbl, perturb_value, PerturbValue),
        (dbl, obj_scale, ObjScale),
        (int, scale_flag, ScaleFlag),
        (int, simplex_pricing, SimplexPricing),
        (int, quad, Quad),
        (int, norm_adjust, NormAdjust),
        (int, sifting, Sifting),
        (int, sift_method, SiftMethod),
        // Barrier
        (dbl, bar_conv_tol, BarConvTol),
        (int, bar_correctors, BarCorrectors),
        (int, bar_homogeneous, BarHomogeneous),
        (int, bar_order, BarOrder),
        (dbl, bar_qcp_conv_tol, BarQCPConvTol),
        (int, crossover, Crossover),
        (int, crossover_basis, CrossoverBasis),
        // MIP
        (int, branch_dir, BranchDir),
        (int, degen_moves, DegenMoves),
        (int, disconnected, Disconnected),
        (dbl, heuristics, Heuristics),
        (dbl, improve_start_gap, ImproveStartGap),
        (dbl, improve_start_time, ImproveStartTime),
        (dbl, improve_start_nodes, ImproveStartNodes),
        (int, integrality_focus, IntegralityFocus),
        (int, min_rel_nodes, MinRelNodes),
        (int, mip_focus, MIPFocus),
        (str, nodefile_dir, NodefileDir),
        (dbl, nodefile_start, NodefileStart),
        (int, node_method, NodeMethod),
        (dbl, no_rel_heur_time, NoRelHeurTime),
        (dbl, no_rel_heur_work, NoRelHeurWork),
        (int, pump_passes, PumpPasses),
        (int, rins, RINS),
        (str, sol_files, SolFiles),
        (int, start_node_limit, StartNodeLimit),
        (int, sub_mip_nodes, SubMIPNodes),
        (int, symmetry, Symmetry),
        (int, var_branch, VarBranch),
        (int, solution_number, SolutionNumber),
        (int, zero_obj_nodes, ZeroObjNodes),
        // Cuts
        (int, cuts, Cuts),
        (int, clique_cuts, CliqueCuts),
        (int, cover_cuts, CoverCuts),
        (int, flow_cover_cuts, FlowCoverCuts),
        (int, flow_path_cuts, FlowPathCuts),
        (int, gub_cover_cuts, GUBCoverCuts),
        (int, implied_cuts, ImpliedCuts),
        (int, proj_implied_cuts, ProjImpliedCuts),
        (int, mip_sep_cuts, MIPSepCuts),
        (int, mir_cuts, MIRCuts),
        (int, strong_cg_cuts, StrongCGCuts),
        (int, mod_k_cuts, ModKCuts),
        (int, zero_half_cuts, ZeroHalfCuts),
        (int, network_cuts, NetworkCuts),
        (int, sub_mip_cuts, SubMIPCuts),
        (int, inf_proof_cuts, InfProofCuts),
        (int, rlt_cuts, RLTCuts),
        (int, relax_lift_cuts, RelaxLiftCuts),
        (int, bqp_cuts, BQPCuts),
        (int, psd_cuts, PSDCuts),
        (int, lift_project_cuts, LiftProjectCuts),
        (int, mixing_cuts, MixingCuts),
        (int, cut_agg_passes, CutAggPasses),
        (int, cut_passes, CutPasses),
        (int, gomory_passes, GomoryPasses),
        // Distributed / Cloud / Token server
        (str, worker_pool, WorkerPool),
        (str, worker_password, WorkerPassword),
        (str, compute_server, ComputeServer),
        (str, token_server, TokenServer),
        (str, server_password, ServerPassword),
        (int, server_timeout, ServerTimeout),
        (str, cs_router, CSRouter),
        (str, cs_group, CSGroup),
        (dbl, cs_queue_timeout, CSQueueTimeout),
        (int, cs_priority, CSPriority),
        (int, cs_idle_timeout, CSIdleTimeout),
        (int, cs_tls_insecure, CSTLSInsecure),
        (int, ts_port, TSPort),
        (str, cloud_access_id, CloudAccessID),
        (str, cloud_secret_key, CloudSecretKey),
        (str, cloud_pool, CloudPool),
        (str, cloud_host, CloudHost),
        (str, cs_manager, CSManager),
        (str, cs_auth_token, CSAuthToken),
        (str, cs_api_access_id, CSAPIAccessID),
        (str, cs_api_secret, CSAPISecret),
        (int, cs_batch_mode, CSBatchMode),
        (str, username, Username),
        (str, cs_app_name, CSAppName),
        (int, cs_client_log, CSClientLog),
        // Tuning
        (dbl, tune_time_limit, TuneTimeLimit),
        (int, tune_results, TuneResults),
        (int, tune_criterion, TuneCriterion),
        (int, tune_trials, TuneTrials),
        (int, tune_output, TuneOutput),
        (int, tune_jobs, TuneJobs),
        (dbl, tune_cleanup, TuneCleanup),
        (int, tune_metric, TuneMetric),
        (dbl, tune_target_mip_gap, TuneTargetMIPGap),
        (dbl, tune_target_time, TuneTargetTime),
        (int, tune_dynamic_jobs, TuneDynamicJobs),
        // Multi-objective / scenarios / pool
        (int, obj_number, ObjNumber),
        (int, multi_obj_method, MultiObjMethod),
        (int, multi_obj_pre, MultiObjPre),
        (int, scenario_number, ScenarioNumber),
        (int, pool_solutions, PoolSolutions),
        (dbl, pool_gap, PoolGap),
        (dbl, pool_gap_abs, PoolGapAbs),
        (int, pool_search_mode, PoolSearchMode),
        // Presolve
        (int, pre_crush, PreCrush),
        (int, pre_dep_row, PreDepRow),
        (int, pre_dual, PreDual),
        (int, pre_passes, PrePasses),
        (int, pre_q_linearize, PreQLinearize),
        (dbl, pre_sos1_big_m, PreSOS1BigM),
        (dbl, pre_sos2_big_m, PreSOS2BigM),
        (int, pre_sparsify, PreSparsify),
        (int, pre_miqcp_form, PreMIQCPForm),
        (int, pre_sos1_encoding, PreSOS1Encoding),
        (int, pre_sos2_encoding, PreSOS2Encoding),
        (dbl, feas_relax_big_m, FeasRelaxBigM),
        // General
        (int, aggregate, Aggregate),
        (int, agg_fill, AggFill),
        (int, concurrent_mip, ConcurrentMIP),
        (int, concurrent_jobs, ConcurrentJobs),
        (int, concurrent_method, ConcurrentMethod),
        (int, display_interval, DisplayInterval),
        (int, distributed_mip_jobs, DistributedMIPJobs),
        (int, dual_reductions, DualReductions),
        (int, iis_method, IISMethod),
        (int, inf_unbd_info, InfUnbdInfo),
        (int, json_sol_detail, JSONSolDetail),
        (int, lazy_constraints, LazyConstraints),
        (str, log_file, LogFile),
        (int, log_to_console, LogToConsole),
        (int, miqcp_method, MIQCPMethod),
        (int, non_convex, NonConvex),
        (int, numeric_focus, NumericFocus),
        (int, qcp_dual, QCPDual),
        (int, record, Record),
        (str, result_file, ResultFile),
        (int, seed, Seed),
        (int, update_mode, UpdateMode),
        (int, ignore_names, IgnoreNames),
        (int, start_number, StartNumber),
        (int, partition_place, PartitionPlace),
        // Piecewise-linear / nonlinear
        (int, func_pieces, FuncPieces),
        (dbl, func_piece_length, FuncPieceLength),
        (dbl, func_piece_error, FuncPieceError),
        (dbl, func_piece_ratio, FuncPieceRatio),
        (dbl, func_max_val, FuncMaxVal),
        (int, func_nonlinear, FuncNonlinear),
        (int, nlp_heur, NLPHeur),
        // Memory / resources
        (dbl, mem_limit, MemLimit),
        (dbl, soft_mem_limit, SoftMemLimit),
        // Licensing / WLS
        (int, license_id, LicenseID),
        (str, user_name, UserName),
        (str, wls_access_id, WLSAccessID),
        (str, wls_secret, WLSSecret),
        (str, wls_token, WLSToken),
        (int, wls_token_duration, WLSTokenDuration),
        (dbl, wls_token_refresh, WLSTokenRefresh),
        // Misc
        (int, network_alg, NetworkAlg),
        (int, obbt, OBBT),
        (int, solution_target, SolutionTarget),
        (int, lp_warm_start, LPWarmStart),
        (str, job_id, JobID),
        (str, dummy, Dummy),
    );

    // Gurobi 13 nonlinear-barrier, PDHG, and MIP-start parameters.
    gurobi_params!(
        (int, nl_bar_iter_limit, NLBarIterLimit),
        (dbl, nl_bar_c_feas_tol, NLBarCFeasTol),
        (dbl, nl_bar_d_feas_tol, NLBarDFeasTol),
        (dbl, nl_bar_p_feas_tol, NLBarPFeasTol),
        (dbl, pdhg_abs_tol, PDHGAbsTol),
        (dbl, pdhg_conv_tol, PDHGConvTol),
        (dbl, pdhg_rel_tol, PDHGRelTol),
        (dbl, pdhg_iter_limit, PDHGIterLimit),
        (int, pdhg_gpu, PDHGGPU),
        (int, no_rel_heur_solutions, NoRelHeurSolutions),
        (int, inherit_params, InheritParams),
        (int, fix_vars_in_indicators, FixVarsInIndicators),
        (dbl, start_time_limit, StartTimeLimit),
        (dbl, start_work_limit, StartWorkLimit),
        (dbl, improve_start_work, ImproveStartWork),
        (int, master_knapsack_cuts, MasterKnapsackCuts),
        (int, obj_pass_number, ObjPassNumber),
        (int, optimality_target, OptimalityTarget),
    );
}

impl GurobiOptions {
    #[must_use]
    pub fn mip_gap(mut self, gap: f64) -> Self {
        self.mip_gap = Some(gap);
        self
    }

    pub fn presolve(mut self, p: GurobiPresolve) -> Self {
        self.presolve = Some(p);
        self
    }

    pub(crate) fn has_non_convex(&self) -> bool {
        self.int_params.iter().any(|(param, _)| *param == IntParam::NonConvex)
    }
}

impl HasUniversal for GurobiOptions {
    fn universal(&self) -> &UniversalOptions {
        &self.universal
    }

    fn universal_mut(&mut self) -> &mut UniversalOptions {
        &mut self.universal
    }
}

/// Apply typed [`GurobiOptions`] onto a live Gurobi model.
pub(crate) fn apply(model: &mut GurobiModel, o: &GurobiOptions) -> Result<(), gurobi_rs::Error> {
    if let Some(d) = o.universal.time_limit {
        model.set_param(gurobi_rs::param::TimeLimit, d.as_secs_f64())?;
    }
    if let Some(n) = o.universal.threads {
        model.set_param(gurobi_rs::param::Threads, i32::try_from(n).unwrap_or(i32::MAX))?;
    }
    if let Some(b) = o.universal.verbose {
        model.set_param(gurobi_rs::param::OutputFlag, i32::from(b))?;
    }
    if let Some(g) = o.mip_gap {
        model.set_param(gurobi_rs::param::MIPGap, g)?;
    }
    if let Some(p) = o.presolve {
        let v = match p {
            GurobiPresolve::Off => 0,
            GurobiPresolve::Auto => -1,
            GurobiPresolve::Conservative => 1,
            GurobiPresolve::Aggressive => 2,
        };
        model.set_param(gurobi_rs::param::Presolve, v)?;
    }
    for (p, v) in &o.int_params {
        model.set_param(*p, *v)?;
    }
    for (p, v) in &o.double_params {
        model.set_param(*p, *v)?;
    }
    for (p, v) in &o.str_params {
        model.set_param(*p, v.clone())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use oximo_solver::UniversalOptionsExt;

    use super::*;

    #[test]
    fn builder_sets_universal_and_mip() {
        let o = GurobiOptions::default()
            .time_limit(Duration::from_secs(60))
            .threads(4)
            .verbose(false)
            .mip_gap(0.05)
            .presolve(GurobiPresolve::Aggressive);
        assert_eq!(o.universal.time_limit, Some(Duration::from_secs(60)));
        assert_eq!(o.universal.threads, Some(4));
        assert_eq!(o.universal.verbose, Some(false));
        assert_eq!(o.mip_gap, Some(0.05));
        assert_eq!(o.presolve, Some(GurobiPresolve::Aggressive));
        assert_eq!(o.int_params, Vec::new());
    }

    #[test]
    fn builder_pushes_gurobi_params() {
        let o = GurobiOptions::default()
            .concurrent_mip(2)
            .seed(123)
            .mip_gap_abs(1e-4)
            .log_file("solve.log");
        assert_eq!(o.int_params, vec![(IntParam::ConcurrentMIP, 2), (IntParam::Seed, 123)]);
        assert_eq!(o.double_params, vec![(DoubleParam::MIPGapAbs, 1e-4)]);
        assert_eq!(o.str_params, vec![(StrParam::LogFile, "solve.log".to_owned())]);
    }

    #[test]
    fn default_vecs_are_empty() {
        let o = GurobiOptions::default();
        assert_eq!(o.int_params, Vec::new());
        assert_eq!(o.double_params, Vec::new());
        assert_eq!(o.str_params, Vec::new());
    }

    #[test]
    fn same_param_twice_stores_both_entries() {
        // Gurobi applies params in order, the last write wins at the backend,
        // but our vec preserves the full call sequence.
        let o = GurobiOptions::default().seed(123).seed(99);
        assert_eq!(o.int_params, vec![(IntParam::Seed, 123), (IntParam::Seed, 99)]);
    }

    #[test]
    fn detects_explicit_non_convex_setting() {
        assert!(!GurobiOptions::default().has_non_convex());
        assert!(GurobiOptions::default().non_convex(0).has_non_convex());
    }

    #[test]
    fn clone_preserves_all_vecs() {
        let o = GurobiOptions::default().seed(123).feasibility_tol(1e-9).log_file("x");
        let c = o.clone();
        assert_eq!(o.int_params, c.int_params);
        assert_eq!(o.double_params, c.double_params);
        assert_eq!(o.str_params, c.str_params);
    }
}
