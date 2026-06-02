use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

use oximo_solver::{HasUniversal, UniversalOptions};

// TODO: CompIIS writes IIS to the summary file.
// oximo emits the option but doesn't parse/return the IIS (temp dir deleted).

/// BARON-specific solver options.
///
/// Universal options (`time_limit`, `threads`, `verbose`) come from the embedded
/// [`UniversalOptions`] via [`UniversalOptionsExt`](oximo_solver::UniversalOptionsExt)
/// and map to the BARON `MaxTime`, `threads`, and `PrLevel` keywords respectively.
///
/// Every other knob is a BARON option written verbatim into the `OPTIONS{ ... }`
/// block of the generated `.bar` file. Each builder method below corresponds to a
/// documented BARON option. The method name is the snake_case form of the BARON
/// keyword (e.g. `EpsR` -> `.eps_r(..)`). For any keyword not covered by a typed
/// builder, use [`BaronOptions::raw`].
///
/// `ResName`, `TimName`, `results`, and `times` are managed by the backend (it
/// needs the results and times files to parse the solution) and are not
/// user-settable, entries for those keywords passed through [`raw`] are ignored.
///
/// [`raw`]: BaronOptions::raw
#[derive(Clone, Debug, Default)]
pub struct BaronOptions {
    pub universal: UniversalOptions,
    /// Override for the `baron` executable. When `None`, `"baron"` is looked up
    /// from `PATH`.
    pub baron_path: Option<PathBuf>,
    int_opts: Vec<(&'static str, i64)>,
    dbl_opts: Vec<(&'static str, f64)>,
    str_opts: Vec<(&'static str, String)>,
    raw: Vec<(String, String)>,
}

/// BARON keywords the backend writes itself; user attempts to set these via
/// [`BaronOptions::raw`] are ignored so they cannot break solution parsing.
const MANAGED_KEYS: &[&str] = &["resname", "timname", "results", "times"];

// Generates one typed builder method per BARON option.
macro_rules! baron_opts {
    ($( ($kind:ident, $method:ident, $keyword:literal $(, $doc:literal)?) ),* $(,)?) => {
        $(baron_opts!(@impl $kind, $method, $keyword $(, $doc)?);)*
    };
    (@impl int, $method:ident, $keyword:literal $(, $doc:literal)?) => {
        $(#[doc = $doc])?
        #[doc = concat!("\n\nBARON option `", $keyword, "`.")]
        #[must_use]
        pub fn $method(mut self, v: i64) -> Self {
            self.int_opts.push(($keyword, v));
            self
        }
    };
    (@impl bool, $method:ident, $keyword:literal $(, $doc:literal)?) => {
        $(#[doc = $doc])?
        #[doc = concat!("\n\nBARON option `", $keyword, "` (written as `0`/`1`).")]
        #[must_use]
        pub fn $method(mut self, v: bool) -> Self {
            self.int_opts.push(($keyword, i64::from(v)));
            self
        }
    };
    (@impl dbl, $method:ident, $keyword:literal $(, $doc:literal)?) => {
        $(#[doc = $doc])?
        #[doc = concat!("\n\nBARON option `", $keyword, "`.")]
        #[must_use]
        pub fn $method(mut self, v: f64) -> Self {
            self.dbl_opts.push(($keyword, v));
            self
        }
    };
    (@impl str, $method:ident, $keyword:literal $(, $doc:literal)?) => {
        $(#[doc = $doc])?
        #[doc = concat!("\n\nBARON option `", $keyword, "` (written quoted).")]
        #[must_use]
        pub fn $method(mut self, v: impl Into<String>) -> Self {
            self.str_opts.push(($keyword, v.into()));
            self
        }
    };
}

impl BaronOptions {
    baron_opts!(
        // 7.1 Termination options
        // (MaxTime and threads are set via the universal `time_limit`/`threads`
        // options; use `.raw("MaxTime", "-1")` for the no-limit sentinel.)
        (dbl, eps_a, "EpsA", "Absolute termination tolerance (>= 1e-12)."),
        (dbl, eps_r, "EpsR", "Relative termination tolerance (nonnegative)."),
        (
            bool,
            delta_term,
            "DeltaTerm",
            "Terminate on insufficient progress over `DeltaT` seconds."
        ),
        (dbl, delta_t, "DeltaT", "Progress window in seconds for `DeltaTerm`."),
        (dbl, delta_a, "DeltaA", "Absolute improvement threshold for `DeltaTerm`."),
        (dbl, delta_r, "DeltaR", "Relative improvement threshold for `DeltaTerm`."),
        (dbl, cut_off, "CutOff", "Ignore solutions worse than this objective value."),
        (dbl, target, "Target", "Terminate once a solution at least as good as this is found."),
        (dbl, abs_con_feas_tol, "AbsConFeasTol", "Absolute constraint feasibility tolerance."),
        (
            dbl,
            rel_con_feas_tol,
            "RelConFeasTol",
            "Relative constraint feasibility tolerance (0..=0.1)."
        ),
        (dbl, abs_int_feas_tol, "AbsIntFeasTol", "Absolute integer feasibility tolerance."),
        (
            dbl,
            rel_int_feas_tol,
            "RelIntFeasTol",
            "Relative integer feasibility tolerance (0..=0.1)."
        ),
        (
            dbl,
            primal_cs_tol,
            "PrimalCSTol",
            "Absolute tolerance for primal complementary slackness."
        ),
        (dbl, dual_cs_tol, "DualCSTol", "Absolute tolerance for dual complementary slackness."),
        (dbl, dual_feas_tol, "DualFeasTol", "Absolute tolerance for dual feasibility."),
        (dbl, ec_tol, "ECTol", "Absolute tolerance for equilibrium-condition complementarity."),
        (dbl, box_tol, "BoxTol", "Boxes smaller than this are eliminated (>= 1e-12)."),
        (bool, first_feas, "FirstFeas", "Terminate once `NumSol` feasible solutions are found."),
        (bool, first_loc, "FirstLoc", "Terminate once a local optimum is found."),
        (int, max_iter, "MaxIter", "Branch-and-reduce iteration limit; `-1` for unlimited."),
        (bool, want_dual, "WantDual", "Return a dual solution for the best primal point."),
        (dbl, dual_budget, "DualBudget", "Extra time (s) to compute a dual solution on timeout."),
        (int, num_sol, "NumSol", "Number of feasible solutions to find; `-1` for all."),
        (
            dbl,
            isol_tol,
            "IsolTol",
            "Separation distance between distinct solutions (with `NumSol`)."
        ),
        // 7.2 Relaxation options
        (int, n_outer1, "NOuter1", "Number of outer approximators of convex univariate functions."),
        (
            int,
            n_out_per_var,
            "NOutPerVar",
            "Outer approximators per variable for convex functions."
        ),
        (int, n_out_iter, "NOutIter", "Rounds of cutting-plane generation at node relaxation."),
        (
            int,
            out_grid,
            "OutGrid",
            "Grid points per variable for convex multivariate approximators."
        ),
        // 7.3 Range reduction options
        (bool, tdo, "TDo", "Nonlinear-feasibility-based range reduction (poor man's NLPs)."),
        (bool, mdo, "MDo", "Marginals-based range reduction."),
        (bool, lbttdo, "LBTTDo", "Linear-feasibility-based range reduction (poor man's LPs)."),
        (bool, obttdo, "OBTTDo", "Optimality-based bound tightening."),
        (int, pdo, "PDo", "Probing: `-2` auto, `-1` all variables, `0` none, `n` on n variables."),
        // 7.4 Tree management options
        (
            int,
            br_var_stra,
            "BrVarStra",
            "Branching variable strategy (0 dynamic, 1 violation, 2 edge)."
        ),
        (
            int,
            br_pt_stra,
            "BrPtStra",
            "Branching point strategy (0 dynamic, 1 omega, 2 bisection, 3 mix)."
        ),
        (
            int,
            node_sel,
            "NodeSel",
            "Node selection rule (0 dynamic, 1 best-bound, 2 LIFO, 3 min-infeas)."
        ),
        // 7.5 Local search options
        (bool, do_local, "DoLocal", "Local search during upper bounding."),
        (int, num_loc, "NumLoc", "Local searches in preprocessing; `-1`/`-2` for automatic."),
        // 7.6 Output and file name options
        (
            int,
            pr_level,
            "PrLevel",
            "Print level (`0` silent). Overrides the `verbose` option when set."
        ),
        (int, pr_freq, "PrFreq", "Log output frequency in nodes."),
        (dbl, pr_time_freq, "PrTimeFreq", "Log output frequency in seconds."),
        (bool, loc_res, "LocRes", "Write detailed local-search results to the results file."),
        (str, pro_name, "ProName", "Problem name (<= 10 characters)."),
        // 7.7 Subsolver options
        (int, lp_sol, "LPSol", "LP/MIP subsolver (`-1` auto, 3 CPLEX, 8 CLP/CBC, 15 HSL LA04)."),
        (
            bool,
            allow_cplex,
            "AllowCPLEX",
            "Permit CPLEX as an LP/MIP subsolver under auto selection."
        ),
        (bool, allow_cbc, "AllowCBC", "Permit CBC as an LP/MIP subsolver under auto selection."),
        (
            bool,
            allow_hsl,
            "AllowHSL",
            "Permit HSL's LA04 as an LP/MIP subsolver under auto selection."
        ),
        (str, cplex_lib_name, "CplexLibName", "Full path to the CPLEX callable libraries."),
        (int, lp_alg, "LPAlg", "LP algorithm (0 auto, 1 primal, 2 dual, 3 barrier)."),
        (
            int,
            nlp_sol,
            "NLPSol",
            "NLP subsolver (`-1` auto, 0 none, 9 IPOPT, 10 FilterSD, 14 FilterSQP)."
        ),
        (
            bool,
            allow_filter_sd,
            "AllowFilterSD",
            "Permit FilterSD as an NLP subsolver under auto selection."
        ),
        (
            bool,
            allow_filter_sqp,
            "AllowFilterSQP",
            "Permit FilterSQP as an NLP subsolver under auto selection."
        ),
        (bool, allow_ipopt, "AllowIpopt", "Permit IPOPT as an NLP subsolver under auto selection."),
        // 7.8 Licensing options
        (str, lic_name, "LicName", "Full path to the BARON license file."),
        // 7.9 Other options
        (
            int,
            comp_iis,
            "CompIIS",
            "Search for an IIS on infeasible models (0 off .. 5 algorithms)."
        ),
        (bool, iis_int, "IISint", "Consider general integers (not binaries) as part of an IIS."),
        (
            int,
            iis_order,
            "IISorder",
            "Constraint ordering for the IIS search (`-1` auto, 1..=3, >=4 seed)."
        ),
        (bool, problem_is_convex, "ProblemIsConvex", "Assert the continuous relaxation is convex."),
        (int, seed, "seed", "Initial seed for BARON's random number generator (positive)."),
    );

    /// Override for the `baron` executable path. When unset, `"baron"` is looked
    /// up from `PATH`.
    #[must_use]
    pub fn baron_path(mut self, p: impl Into<PathBuf>) -> Self {
        self.baron_path = Some(p.into());
        self
    }

    /// Set an arbitrary BARON option by keyword, written verbatim as
    /// `keyword: value;` in the `OPTIONS{ ... }` block. Use this for any option
    /// without a dedicated builder. The `value` is emitted as-is, so use quote
    /// strings (e.g. `("LpSol", "8")`, `("NLPSol", "6")`).
    ///
    /// Keywords managed by the backend (`ResName`, `TimName`, `results`,
    /// `times`) are ignored.
    #[must_use]
    pub fn raw(mut self, keyword: impl Into<String>, value: impl Into<String>) -> Self {
        self.raw.push((keyword.into(), value.into()));
        self
    }
}

impl HasUniversal for BaronOptions {
    fn universal(&self) -> &UniversalOptions {
        &self.universal
    }

    fn universal_mut(&mut self) -> &mut UniversalOptions {
        &mut self.universal
    }
}

/// Format an `f64` for use inside a BARON `OPTIONS{}` value.
fn fmt(v: f64) -> String {
    if v == f64::INFINITY {
        return "1e51".into();
    }
    if v == f64::NEG_INFINITY {
        return "-1e51".into();
    }
    format!("{v}")
}

/// Emit the `OPTIONS{ ... }` block into `bar`.
///
/// `res_name` / `tim_name` are the (relative) result and times file names the
/// backend will read back, they are written into the block so BARON produces
/// them in the working directory.
pub fn write_options(bar: &mut String, o: &BaronOptions, res_name: &str, tim_name: &str) {
    writeln!(bar, "OPTIONS{{").unwrap();

    // Backend-managed: we need the results and times files to parse the result.
    // (`summary` is left at BARON's default so `CompIIS` can write its IIS.)
    writeln!(bar, "results: 1;").unwrap();
    writeln!(bar, "times: 1;").unwrap();
    writeln!(bar, "ResName: \"{res_name}\";").unwrap();
    writeln!(bar, "TimName: \"{tim_name}\";").unwrap();

    // Universal options.
    // Emit `-1` (no limit) when the user sets no `time_limit`,
    // matching the other oximo backends.
    match o.universal.time_limit {
        Some(d) => writeln!(bar, "MaxTime: {};", d.as_secs_f64()).unwrap(),
        None => writeln!(bar, "MaxTime: -1;").unwrap(),
    }
    if let Some(n) = o.universal.threads {
        writeln!(bar, "threads: {n};").unwrap();
    }
    if let Some(v) = o.universal.verbose {
        writeln!(bar, "PrLevel: {};", i64::from(v)).unwrap();
    }

    // Typed BARON options (a later `pr_level(..)` deliberately overrides the
    // `verbose`-derived PrLevel above, since BARON honours the last setting).
    for (k, v) in &o.int_opts {
        writeln!(bar, "{k}: {v};").unwrap();
    }
    for (k, v) in &o.dbl_opts {
        writeln!(bar, "{k}: {};", fmt(*v)).unwrap();
    }
    for (k, v) in &o.str_opts {
        writeln!(bar, "{k}: \"{v}\";").unwrap();
    }

    // Raw passthrough, skipping anything the backend manages.
    for (k, v) in &o.raw {
        if MANAGED_KEYS.contains(&k.to_ascii_lowercase().as_str()) {
            continue;
        }
        writeln!(bar, "{k}: {v};").unwrap();
    }

    writeln!(bar, "}}").unwrap();
    writeln!(bar).unwrap();
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use oximo_solver::UniversalOptionsExt;

    use super::*;

    fn render(o: &BaronOptions) -> String {
        let mut bar = String::new();
        write_options(&mut bar, o, "res.lst", "tim.lst");
        bar
    }

    #[test]
    fn builder_sets_universal_and_baron_path() {
        let o = BaronOptions::default()
            .time_limit(Duration::from_secs(45))
            .threads(3)
            .verbose(true)
            .eps_r(1e-4)
            .baron_path("/opt/baron/baron");
        assert_eq!(o.universal.time_limit, Some(Duration::from_secs(45)));
        assert_eq!(o.universal.threads, Some(3));
        assert_eq!(o.universal.verbose, Some(true));
        assert_eq!(o.dbl_opts, vec![("EpsR", 1e-4)]);
        assert_eq!(o.baron_path.as_deref(), Some(std::path::Path::new("/opt/baron/baron")));
    }

    #[test]
    fn default_emits_only_managed_block() {
        let bar = render(&BaronOptions::default());
        assert!(bar.contains("results: 1;"));
        assert!(bar.contains("times: 1;"));
        assert!(bar.contains("ResName: \"res.lst\";"));
        assert!(bar.contains("TimName: \"tim.lst\";"));
        assert!(bar.trim_start().starts_with("OPTIONS{"));
        assert!(bar.contains('}'));
    }

    #[test]
    fn write_options_emits_universal_and_typed() {
        let o = BaronOptions::default()
            .time_limit(Duration::from_secs(10))
            .threads(4)
            .eps_r(0.01)
            .eps_a(1e-6)
            .max_iter(1000)
            .first_feas(true);
        let bar = render(&o);
        assert!(bar.contains("MaxTime: 10;"), "{bar}");
        assert!(bar.contains("threads: 4;"), "{bar}");
        assert!(bar.contains("EpsR: 0.01;"), "{bar}");
        assert!(bar.contains("EpsA: 0.000001;"), "{bar}");
        assert!(bar.contains("MaxIter: 1000;"), "{bar}");
        assert!(bar.contains("FirstFeas: 1;"), "{bar}");
    }

    #[test]
    fn verbose_false_emits_prlevel_zero() {
        let bar = render(&BaronOptions::default().verbose(false));
        assert!(bar.contains("PrLevel: 0;"), "{bar}");
    }

    #[test]
    fn newly_added_options_emit_correctly() {
        let o = BaronOptions::default()
            .target(1.5)
            .comp_iis(4)
            .allow_ipopt(true)
            .allow_filter_sqp(false)
            .problem_is_convex(true)
            .seed(19_631_963)
            .pro_name("robot")
            .lic_name("/opt/baron/baronlice.txt");
        let bar = render(&o);
        assert!(bar.contains("Target: 1.5;"), "{bar}");
        assert!(bar.contains("CompIIS: 4;"), "{bar}");
        assert!(bar.contains("AllowIpopt: 1;"), "{bar}");
        assert!(bar.contains("AllowFilterSQP: 0;"), "{bar}");
        assert!(bar.contains("ProblemIsConvex: 1;"), "{bar}");
        assert!(bar.contains("seed: 19631963;"), "{bar}");
        assert!(bar.contains("ProName: \"robot\";"), "{bar}");
        assert!(bar.contains("LicName: \"/opt/baron/baronlice.txt\";"), "{bar}");
    }

    #[test]
    fn default_disables_baron_time_cap() {
        // No time_limit => MaxTime: -1 so BARON does not silently stop.
        assert!(render(&BaronOptions::default()).contains("MaxTime: -1;"));
        let bar = render(&BaronOptions::default().time_limit(Duration::from_secs(60)));
        assert!(bar.contains("MaxTime: 60;"), "{bar}");
    }

    #[test]
    fn summary_not_forced() {
        // CompIIS writes its IIS to the summary file, so we must not suppress it.
        assert!(!render(&BaronOptions::default()).contains("summary"));
    }

    #[test]
    fn raw_passthrough_and_managed_filtered() {
        let o = BaronOptions::default().raw("LBTTDo", "1").raw("ResName", "\"hack.lst\";"); // managed: must be ignored
        let bar = render(&o);
        assert!(bar.contains("LBTTDo: 1;"), "{bar}");
        assert!(!bar.contains("hack.lst"), "managed key must be filtered:\n{bar}");
    }
}
