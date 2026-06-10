//! Per-solver typed option structs and [`GamsSolverConfig`].
// TODO: Add more SolverConfig variants for other solvers when we add NLP/MINLP support

use std::fmt::Write as FmtWrite;

use oximo_core::ModelKind;

use crate::options::GamsSolver;
use crate::translate::gams_solve_type;

// - Config enum

/// Selects a GAMS sub-solver and optionally carries typed options written to
/// a `<solver>.opt` file before invoking GAMS.
///
/// Use [`GamsSolverConfig::Named`] to select a solver by name with no extra
/// options.
///
/// A [`From<GamsSolver>`] impl allows passing a bare `GamsSolver` wherever
/// a `GamsSolverConfig` is expected.
///
/// References:
/// - "GAMS Solver Manuals," GAMS Development Corporation.
///   <https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES> (accessed May 14, 2026).
#[derive(Clone, Debug)]
pub enum GamsSolverConfig {
    Baron(GamsBaronOptions),
    Cbc(GamsCbcOptions),
    Cplex(GamsCplexOptions),
    Gurobi(GamsGurobiOptions),
    Highs(GamsHighsOptions),
    Ipopt(GamsIpoptOptions),
    Knitro(GamsKnitroOptions),
    Mosek(GamsMosekOptions),
    Scip(GamsScipOptions),
    Xpress(GamsXpressOptions),
    /// Any solver selectable by name with no typed option file.
    Named(GamsSolver),
    /// A solver selected by name with raw option-file lines written verbatim to
    /// `<solver>.opt`. Use for options oximo has no typed field for.
    Raw(GamsSolver, Vec<String>),
}

impl GamsSolverConfig {
    /// GAMS solver keyword for `option {LP|MIP} = ...;`.
    #[must_use]
    pub fn gams_name(&self) -> &str {
        match self {
            Self::Baron(_) => "BARON",
            Self::Cbc(_) => "CBC",
            Self::Cplex(_) => "CPLEX",
            Self::Gurobi(_) => "GUROBI",
            Self::Highs(_) => "HIGHS",
            Self::Ipopt(_) => "IPOPT",
            Self::Knitro(_) => "KNITRO",
            Self::Mosek(_) => "MOSEK",
            Self::Scip(_) => "SCIP",
            Self::Xpress(_) => "XPRESS",
            Self::Named(s) | Self::Raw(s, _) => s.name(),
        }
    }

    /// Write options to `buf`. Returns `true` if anything was written.
    /// When `true`, the caller should write `buf` to `<solver_lowercase>.opt`
    /// in the GAMS working directory and set `model.optfile = 1`.
    #[must_use]
    pub fn write_opt_file(&self, buf: &mut String) -> bool {
        match self {
            Self::Baron(o) => o.write(buf),
            Self::Cbc(o) => o.write(buf),
            Self::Cplex(o) => o.write(buf),
            Self::Gurobi(o) => o.write(buf),
            Self::Highs(o) => o.write(buf),
            Self::Ipopt(o) => o.write(buf),
            Self::Knitro(o) => o.write(buf),
            Self::Mosek(o) => o.write(buf),
            Self::Scip(o) => o.write(buf),
            Self::Xpress(o) => o.write(buf),
            Self::Named(_) => false,
            Self::Raw(_, lines) => {
                for line in lines {
                    writeln!(buf, "{line}").unwrap();
                }
                !lines.is_empty()
            }
        }
    }

    /// Whether this solver can handle `kind` under oximo's GAMS translation,
    /// which emits `QP` as a `QCP` solve and `MIQP` as a `MIQCP` solve.
    ///
    /// [`GamsSolver::Custom`] and any unrecognized name return `true`: their
    /// capabilities are unknown, so they are left for GAMS to accept or reject.
    #[must_use]
    pub fn supports(&self, kind: ModelKind) -> bool {
        solver_supports_type(self.gams_name(), gams_solve_type(kind))
    }
}

/// GAMS solve types a named solver supports, restricted to the six oximo can
/// emit: `LP` / `MIP` / `NLP` / `MINLP` / `QCP` / `MIQCP`. `None` means the
/// name is unrecognized and cannot be validated.
///
/// Transcribed from the GAMS solver/model-type matrix (other model types
/// (`MCP`, `MPEC`, `CNS`, `DNLP`, `EMP`, stochastic) are omitted because oximo
/// never emits them):
/// - "GAMS Solver Manuals," GAMS Development Corporation.
///   <https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES> (accessed May 14, 2026).
fn supported_solve_types(gams_name: &str) -> Option<&'static [&'static str]> {
    Some(match gams_name {
        "ALPHAECP" | "DICOPT" | "SBB" | "SHOT" => &["MINLP", "MIQCP"],
        "CONOPT" | "CONOPT3" | "CONOPT4" | "IPOPT" | "MINOS" | "SNOPT" => &["LP", "NLP", "QCP"],
        "DECIS" | "SOPLEX" | "QUADMINOS" => &["LP"],
        "CBC" | "GLPK" | "HIGHS" => &["LP", "MIP"],
        "ODHCPLEX" => &["MIP", "MIQCP"],
        "COPT" | "CPLEX" => &["LP", "MIP", "QCP", "MIQCP"],
        "ANTIGONE" => &["NLP", "MINLP", "QCP", "MIQCP"],
        "KNITRO" => &["LP", "NLP", "MINLP", "QCP", "MIQCP"],
        "SCIP" => &["MIP", "NLP", "MINLP", "QCP", "MIQCP"],
        "BARON" | "GUROBI" | "GUSS" | "KESTREL" | "LINDO" | "LINDOGLOBAL" | "MOSEK" | "XPRESS" => {
            &["LP", "MIP", "NLP", "MINLP", "QCP", "MIQCP"]
        }
        // JAMS (EMP), MILES (MCP), NLPEC (MCP/MPEC), PATH (MCP/MPEC/CNS),
        // RESHOP (EMP) support none of the model types oximo emits.
        "JAMS" | "MILES" | "NLPEC" | "PATH" | "RESHOP" => &[],
        _ => return None,
    })
}

/// Whether the GAMS solver named `gams_name` supports `gams_type`
/// (`"LP"` / `"MIP"` / `"NLP"` / `"MINLP"` / `"QCP"` / `"MIQCP"`). Unrecognized
/// names return `true`.
pub(crate) fn solver_supports_type(gams_name: &str, gams_type: &str) -> bool {
    supported_solve_types(gams_name).is_none_or(|types| types.contains(&gams_type))
}

impl From<GamsSolver> for GamsSolverConfig {
    fn from(s: GamsSolver) -> Self {
        Self::Named(s)
    }
}

// - Helpers

fn kv(buf: &mut String, key: &str, val: impl std::fmt::Display) {
    writeln!(buf, "{key} {val}").unwrap();
}

fn kv_eq(buf: &mut String, key: &str, val: impl std::fmt::Display) {
    writeln!(buf, "{key} = {val}").unwrap();
}

/// Append verbatim option-file lines. Returns `true` if any were written.
fn write_raw(buf: &mut String, raw: &[String]) -> bool {
    for line in raw {
        writeln!(buf, "{line}").unwrap();
    }
    !raw.is_empty()
}

// - BARON

/// Options for the BARON global solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_BARON.html>
#[derive(Clone, Debug, Default)]
pub struct GamsBaronOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    /// Max wall-clock time in seconds (`MaxTime`)
    pub max_time: Option<f64>,
    /// Max branch-and-reduce iterations (`MaxIter`)
    pub max_iter: Option<i64>,
    /// Relative optimality gap (`EpsR`)
    pub eps_r: Option<f64>,
    /// Absolute optimality gap (`EpsA`)
    pub eps_a: Option<f64>,
    /// Absolute constraint feasibility tolerance (`AbsConFeasTol`)
    pub abs_con_feas_tol: Option<f64>,
    /// Absolute integrality tolerance (`AbsIntFeasTol`)
    pub abs_int_feas_tol: Option<f64>,
    /// Threads for MIP subproblems (`Threads`)
    pub threads: Option<u32>,
    /// Local searches in preprocessing (`NumLoc`)
    pub num_loc: Option<i32>,
    /// Number of feasible solutions to find (`NumSol`)
    pub num_sol: Option<i32>,
    /// Objective cutoff (`CutOff`)
    pub cut_off: Option<f64>,
}

impl GamsBaronOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("MaxTime", self.max_time);
        w!("MaxIter", self.max_iter);
        w!("EpsR", self.eps_r);
        w!("EpsA", self.eps_a);
        w!("AbsConFeasTol", self.abs_con_feas_tol);
        w!("AbsIntFeasTol", self.abs_int_feas_tol);
        w!("Threads", self.threads);
        w!("NumLoc", self.num_loc);
        w!("NumSol", self.num_sol);
        w!("CutOff", self.cut_off);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - CBC

/// `presolve` setting for CBC.
#[derive(Clone, Debug)]
pub enum GamsCbcPresolve {
    On,
    Off,
    More,
}

/// `cuts` setting for CBC.
#[derive(Clone, Debug)]
pub enum GamsCbcCuts {
    Off,
    On,
    Root,
    IfMove,
    ForceOn,
}

/// Options for the CBC LP/MIP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_CBC.html>
#[derive(Clone, Debug, Default)]
pub struct GamsCbcOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    pub threads: Option<u32>,
    /// Relative MIP gap (`optcr`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`optca`)
    pub mip_abs_gap: Option<f64>,
    /// Max branch-and-bound nodes (`nodlim`)
    pub node_limit: Option<u64>,
    pub presolve: Option<GamsCbcPresolve>,
    pub cuts: Option<GamsCbcCuts>,
    /// Enable MIP heuristics (`heuristics`)
    pub heuristics: Option<bool>,
}

impl GamsCbcOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("threads", self.threads);
        w!("optcr", self.mip_rel_gap);
        w!("optca", self.mip_abs_gap);
        w!("nodlim", self.node_limit);
        if let Some(p) = &self.presolve {
            kv(
                buf,
                "presolve",
                match p {
                    GamsCbcPresolve::On => "on",
                    GamsCbcPresolve::Off => "off",
                    GamsCbcPresolve::More => "more",
                },
            );
            n += 1;
        }
        if let Some(c) = &self.cuts {
            kv(
                buf,
                "cuts",
                match c {
                    GamsCbcCuts::Off => "off",
                    GamsCbcCuts::On => "on",
                    GamsCbcCuts::Root => "root",
                    GamsCbcCuts::IfMove => "ifmove",
                    GamsCbcCuts::ForceOn => "forceOn",
                },
            );
            n += 1;
        }
        if let Some(h) = self.heuristics {
            kv(buf, "heuristics", i32::from(h));
            n += 1;
        }
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - CPLEX

/// `mipemphasis` strategy for CPLEX.
#[derive(Clone, Debug)]
pub enum GamsCplexMipEmphasis {
    /// Balanced: feasibility and optimality (0)
    Balanced,
    /// Emphasize feasibility (1)
    Feasibility,
    /// Emphasize proving optimality (2)
    Optimality,
    /// Emphasize best bound (3)
    BestBound,
    /// Emphasize hidden feasibility (4)
    HiddenFeasibility,
}

/// Options for the CPLEX LP/MIP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_CPLEX.html>
#[derive(Clone, Debug, Default)]
pub struct GamsCplexOptions {
    /// Extra option-file lines written verbatim (options without a typed field),
    pub raw: Vec<String>,
    pub threads: Option<u32>,
    /// Relative MIP gap (`epgap`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`epagap`)
    pub mip_abs_gap: Option<f64>,
    /// Max B&B nodes (`nodelim`)
    pub node_limit: Option<u64>,
    /// Limit on integer solutions found (`intsollim`)
    pub int_sol_limit: Option<u32>,
    /// Enable presolve (`preind`)
    pub presolve: Option<bool>,
    /// MIP solution tactics (`mipemphasis`)
    pub mip_emphasis: Option<GamsCplexMipEmphasis>,
    /// Node selection (`nodesel`)
    pub node_select: Option<i32>,
    /// Variable selection (`varsel`)
    pub var_select: Option<i32>,
    /// Integrality tolerance (`epint`)
    pub int_tol: Option<f64>,
    /// Feasibility tolerance (`eprhs`)
    pub feasibility_tol: Option<f64>,
    /// Optimality tolerance (`epopt`)
    pub optimality_tol: Option<f64>,
    /// LP algorithm (`lpmethod`)
    pub lp_method: Option<i32>,
}

impl GamsCplexOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("threads", self.threads);
        w!("epgap", self.mip_rel_gap);
        w!("epagap", self.mip_abs_gap);
        w!("nodelim", self.node_limit);
        w!("intsollim", self.int_sol_limit);
        if let Some(pre) = self.presolve {
            kv(buf, "preind", i32::from(pre));
            n += 1;
        }
        if let Some(e) = &self.mip_emphasis {
            kv(
                buf,
                "mipemphasis",
                match e {
                    GamsCplexMipEmphasis::Balanced => 0,
                    GamsCplexMipEmphasis::Feasibility => 1,
                    GamsCplexMipEmphasis::Optimality => 2,
                    GamsCplexMipEmphasis::BestBound => 3,
                    GamsCplexMipEmphasis::HiddenFeasibility => 4,
                },
            );
            n += 1;
        }
        w!("nodesel", self.node_select);
        w!("varsel", self.var_select);
        w!("epint", self.int_tol);
        w!("eprhs", self.feasibility_tol);
        w!("epopt", self.optimality_tol);
        w!("lpmethod", self.lp_method);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - GUROBI

/// `mipfocus` strategy for Gurobi.
#[derive(Clone, Debug)]
pub enum GamsGurobiMipFocus {
    /// Balanced (0)
    Balanced,
    /// Emphasize feasible solutions (1)
    Feasibility,
    /// Emphasize proving optimality (2)
    Optimality,
    /// Emphasize improving best bound (3)
    BestBound,
}

/// Options for the Gurobi LP/MIP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_GUROBI.html>
#[derive(Clone, Debug, Default)]
pub struct GamsGurobiOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    pub threads: Option<u32>,
    /// Relative MIP gap (`mipgap`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`mipgapabs`)
    pub mip_abs_gap: Option<f64>,
    /// Max nodes (`nodelimit`).
    pub node_limit: Option<u64>,
    /// Presolve level (`presolve`)
    pub presolve: Option<i32>,
    /// Cut generation (`cuts`)
    pub cuts: Option<i32>,
    /// MIP heuristics effort (`heuristics`)
    pub heuristics: Option<f64>,
    /// Algorithm (`method`)
    pub method: Option<i32>,
    /// MIP solution focus (`mipfocus`)
    pub mip_focus: Option<GamsGurobiMipFocus>,
    /// Primal feasibility tolerance (`feasibilitytol`)
    pub feasibility_tol: Option<f64>,
    /// Integer feasibility tolerance (`intfeastol`)
    pub int_feas_tol: Option<f64>,
    /// Dual feasibility tolerance (`optimalitytol`)
    pub optimality_tol: Option<f64>,
}

impl GamsGurobiOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("threads", self.threads);
        w!("mipgap", self.mip_rel_gap);
        w!("mipgapabs", self.mip_abs_gap);
        w!("nodelimit", self.node_limit);
        w!("presolve", self.presolve);
        w!("cuts", self.cuts);
        w!("heuristics", self.heuristics);
        w!("method", self.method);
        if let Some(f) = &self.mip_focus {
            kv(
                buf,
                "mipfocus",
                match f {
                    GamsGurobiMipFocus::Balanced => 0,
                    GamsGurobiMipFocus::Feasibility => 1,
                    GamsGurobiMipFocus::Optimality => 2,
                    GamsGurobiMipFocus::BestBound => 3,
                },
            );
            n += 1;
        }
        w!("feasibilitytol", self.feasibility_tol);
        w!("intfeastol", self.int_feas_tol);
        w!("optimalitytol", self.optimality_tol);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - HiGHS

/// `presolve` setting for HiGHS.
#[derive(Clone, Debug)]
pub enum GamsHighsPresolve {
    On,
    Off,
    Choose,
}

/// LP algorithm for HiGHS.
#[derive(Clone, Debug)]
pub enum GamsHighsSolver {
    Simplex,
    Ipm,
    Ipx,
    Pdlp,
    Choose,
}

/// Options for the HiGHS LP/MIP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_HIGHS.html>
#[derive(Clone, Debug, Default)]
pub struct GamsHighsOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    pub threads: Option<u32>,
    /// Relative MIP gap (`mip_rel_gap`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`mip_abs_gap`)
    pub mip_abs_gap: Option<f64>,
    /// Max nodes (`nodlim`)
    pub node_limit: Option<u64>,
    pub presolve: Option<GamsHighsPresolve>,
    pub solver: Option<GamsHighsSolver>,
    pub primal_feasibility_tol: Option<f64>,
    pub dual_feasibility_tol: Option<f64>,
    pub optimality_tol: Option<f64>,
}

impl GamsHighsOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv_eq(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("threads", self.threads);
        w!("mip_rel_gap", self.mip_rel_gap);
        w!("mip_abs_gap", self.mip_abs_gap);
        w!("nodlim", self.node_limit);
        if let Some(p) = &self.presolve {
            kv_eq(
                buf,
                "presolve",
                match p {
                    GamsHighsPresolve::On => "on",
                    GamsHighsPresolve::Off => "off",
                    GamsHighsPresolve::Choose => "choose",
                },
            );
            n += 1;
        }
        if let Some(s) = &self.solver {
            kv_eq(
                buf,
                "solver",
                match s {
                    GamsHighsSolver::Simplex => "simplex",
                    GamsHighsSolver::Ipm => "ipm",
                    GamsHighsSolver::Ipx => "ipx",
                    GamsHighsSolver::Pdlp => "pdlp",
                    GamsHighsSolver::Choose => "choose",
                },
            );
            n += 1;
        }
        w!("primal_feasibility_tolerance", self.primal_feasibility_tol);
        w!("dual_feasibility_tolerance", self.dual_feasibility_tol);
        w!("optimality_tolerance", self.optimality_tol);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - IPOPT

/// Linear solver for IPOPT.
#[derive(Clone, Debug)]
pub enum GamsIpoptLinearSolver {
    Mumps,
    Ma27,
    Ma57,
    Ma86,
    Ma97,
    PardisoMkl,
}

/// Barrier parameter update strategy for IPOPT.
#[derive(Clone, Debug)]
pub enum GamsIpoptMuStrategy {
    Monotone,
    Adaptive,
}

/// Options for the IPOPT NLP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_IPOPT.html>
#[derive(Clone, Debug, Default)]
pub struct GamsIpoptOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    /// Max iterations (`max_iter`)
    pub max_iter: Option<u32>,
    /// Primary optimality tolerance (`tol`)
    pub tol: Option<f64>,
    /// Constraint violation tolerance (`constr_viol_tol`)
    pub constr_viol_tol: Option<f64>,
    /// Dual infeasibility tolerance (`dual_inf_tol`)
    pub dual_inf_tol: Option<f64>,
    /// Complementarity tolerance (`compl_inf_tol`)
    pub compl_inf_tol: Option<f64>,
    /// Relaxed convergence tolerance (`acceptable_tol`)
    pub acceptable_tol: Option<f64>,
    pub linear_solver: Option<GamsIpoptLinearSolver>,
    /// Print level 0–12 (`print_level`)
    pub print_level: Option<u32>,
    pub mu_strategy: Option<GamsIpoptMuStrategy>,
}

impl GamsIpoptOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("max_iter", self.max_iter);
        w!("tol", self.tol);
        w!("constr_viol_tol", self.constr_viol_tol);
        w!("dual_inf_tol", self.dual_inf_tol);
        w!("compl_inf_tol", self.compl_inf_tol);
        w!("acceptable_tol", self.acceptable_tol);
        if let Some(ls) = &self.linear_solver {
            kv(
                buf,
                "linear_solver",
                match ls {
                    GamsIpoptLinearSolver::Mumps => "mumps",
                    GamsIpoptLinearSolver::Ma27 => "ma27",
                    GamsIpoptLinearSolver::Ma57 => "ma57",
                    GamsIpoptLinearSolver::Ma86 => "ma86",
                    GamsIpoptLinearSolver::Ma97 => "ma97",
                    GamsIpoptLinearSolver::PardisoMkl => "pardisomkl",
                },
            );
            n += 1;
        }
        w!("print_level", self.print_level);
        if let Some(mu) = &self.mu_strategy {
            kv(
                buf,
                "mu_strategy",
                match mu {
                    GamsIpoptMuStrategy::Monotone => "monotone",
                    GamsIpoptMuStrategy::Adaptive => "adaptive",
                },
            );
            n += 1;
        }
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - KNITRO

/// NLP algorithm for KNITRO.
#[derive(Clone, Debug)]
pub enum GamsKnitroAlgorithm {
    /// Automatic (0)
    Auto,
    /// Interior-point / Direct (1)
    InteriorDirect,
    /// Interior-point / CG (2)
    InteriorCg,
    /// Active-set (3)
    ActiveSet,
    /// SQP (4)
    Sqp,
}

/// Options for the KNITRO NLP/MIP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_KNITRO.html>
#[derive(Clone, Debug, Default)]
pub struct GamsKnitroOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    pub algorithm: Option<GamsKnitroAlgorithm>,
    /// Max iterations (`maxit`)
    pub max_iter: Option<u32>,
    /// Relative KKT optimality tolerance (`opttol`)
    pub opt_tol: Option<f64>,
    /// Absolute KKT optimality tolerance (`opttol_abs`)
    pub opt_tol_abs: Option<f64>,
    /// Relative feasibility tolerance (`feastol`)
    pub feas_tol: Option<f64>,
    /// Absolute feasibility tolerance (`feastol_abs`)
    pub feas_tol_abs: Option<f64>,
    pub threads: Option<u32>,
    /// Max B&B nodes (`mip_maxnodes`)
    pub mip_max_nodes: Option<u64>,
    /// Relative MIP gap (`mip_opt_gap_rel`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`mip_opt_gap_abs`)
    pub mip_abs_gap: Option<f64>,
}

impl GamsKnitroOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        if let Some(alg) = &self.algorithm {
            kv(
                buf,
                "nlp_algorithm",
                match alg {
                    GamsKnitroAlgorithm::Auto => 0,
                    GamsKnitroAlgorithm::InteriorDirect => 1,
                    GamsKnitroAlgorithm::InteriorCg => 2,
                    GamsKnitroAlgorithm::ActiveSet => 3,
                    GamsKnitroAlgorithm::Sqp => 4,
                },
            );
            n += 1;
        }
        w!("maxit", self.max_iter);
        w!("opttol", self.opt_tol);
        w!("opttol_abs", self.opt_tol_abs);
        w!("feastol", self.feas_tol);
        w!("feastol_abs", self.feas_tol_abs);
        w!("threads", self.threads);
        w!("mip_maxnodes", self.mip_max_nodes);
        w!("mip_opt_gap_rel", self.mip_rel_gap);
        w!("mip_opt_gap_abs", self.mip_abs_gap);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - MOSEK

/// Options for the MOSEK LP/MIP/NLP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_MOSEK.html>
#[derive(Clone, Debug, Default)]
pub struct GamsMosekOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    /// Threads (`MSK_IPAR_NUM_THREADS`)
    pub threads: Option<u32>,
    /// Relative MIP gap (`MSK_DPAR_MIO_TOL_REL_GAP`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`MSK_DPAR_MIO_TOL_ABS_GAP`)
    pub mip_abs_gap: Option<f64>,
    /// Max relaxations in B&B (`MSK_IPAR_MIO_MAX_NUM_RELAXS`)
    pub max_relaxations: Option<i64>,
    /// Max branches (`MSK_IPAR_MIO_MAX_NUM_BRANCHES`)
    pub max_branches: Option<i64>,
    /// Primal feasibility tolerance (`MSK_DPAR_INTPNT_TOL_PFEAS`)
    pub primal_feas_tol: Option<f64>,
    /// Dual feasibility tolerance (`MSK_DPAR_INTPNT_TOL_DFEAS`)
    pub dual_feas_tol: Option<f64>,
    /// MIO feasibility tolerance (`MSK_DPAR_MIO_TOL_FEAS`)
    pub mio_feas_tol: Option<f64>,
    /// Integer relaxation tolerance (`MSK_DPAR_MIO_TOL_ABS_RELAX_INT`)
    pub int_relax_tol: Option<f64>,
}

impl GamsMosekOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("MSK_IPAR_NUM_THREADS", self.threads);
        w!("MSK_DPAR_MIO_TOL_REL_GAP", self.mip_rel_gap);
        w!("MSK_DPAR_MIO_TOL_ABS_GAP", self.mip_abs_gap);
        w!("MSK_IPAR_MIO_MAX_NUM_RELAXS", self.max_relaxations);
        w!("MSK_IPAR_MIO_MAX_NUM_BRANCHES", self.max_branches);
        w!("MSK_DPAR_INTPNT_TOL_PFEAS", self.primal_feas_tol);
        w!("MSK_DPAR_INTPNT_TOL_DFEAS", self.dual_feas_tol);
        w!("MSK_DPAR_MIO_TOL_FEAS", self.mio_feas_tol);
        w!("MSK_DPAR_MIO_TOL_ABS_RELAX_INT", self.int_relax_tol);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - SCIP

/// Options for the SCIP LP/MIP/NLP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_SCIP.html>
#[derive(Clone, Debug, Default)]
pub struct GamsScipOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    /// Max nodes (`limits/nodes`)
    pub node_limit: Option<i64>,
    /// Relative MIP gap (`limits/gap`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`limits/gapabs`)
    pub mip_abs_gap: Option<f64>,
    /// Stop after N feasible solutions (`limits/solutions`)
    pub sol_limit: Option<u32>,
    /// Primal feasibility tolerance (`numerics/feastol`)
    pub feas_tol: Option<f64>,
    /// Dual feasibility tolerance (`numerics/dualfeastol`)
    pub dual_feas_tol: Option<f64>,
    /// Max presolve rounds (`presolving/maxrounds`)
    pub presolve_rounds: Option<i32>,
    /// Separation rounds at root (`separating/maxroundsroot`)
    pub sep_rounds_root: Option<u32>,
}

impl GamsScipOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv_eq(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("limits/nodes", self.node_limit);
        w!("limits/gap", self.mip_rel_gap);
        w!("limits/gapabs", self.mip_abs_gap);
        w!("limits/solutions", self.sol_limit);
        w!("numerics/feastol", self.feas_tol);
        w!("numerics/dualfeastol", self.dual_feas_tol);
        w!("presolving/maxrounds", self.presolve_rounds);
        w!("separating/maxroundsroot", self.sep_rounds_root);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

// - XPRESS

/// Options for the XPRESS LP/MIP solver.
///
/// Reference: <https://www.gams.com/latest/docs/S_XPRESS.html>
#[derive(Clone, Debug, Default)]
pub struct GamsXpressOptions {
    /// Extra option-file lines written verbatim (options without a typed field).
    pub raw: Vec<String>,
    pub threads: Option<u32>,
    /// Relative MIP gap (`mipRelStop`)
    pub mip_rel_gap: Option<f64>,
    /// Absolute MIP gap (`mipAbsStop`)
    pub mip_abs_gap: Option<f64>,
    /// Max nodes (`maxNode`)
    pub node_limit: Option<u64>,
    /// Enable presolve (`presolve`)
    pub presolve: Option<bool>,
    /// Cut strategy (`cutStrategy`)
    pub cut_strategy: Option<i32>,
    /// Primal feasibility tolerance (`feasTol`)
    pub feas_tol: Option<f64>,
    /// Dual optimality tolerance (`optimalityTol`)
    pub optimality_tol: Option<f64>,
    /// MIP integrality tolerance (`mipTol`)
    pub mip_tol: Option<f64>,
    /// LP algorithm (`defaultAlg`)
    pub lp_algorithm: Option<i32>,
}

impl GamsXpressOptions {
    fn write(&self, buf: &mut String) -> bool {
        let mut n = 0usize;
        macro_rules! w {
            ($key:expr, $opt:expr) => {
                if let Some(v) = $opt {
                    kv(buf, $key, v);
                    n += 1;
                }
            };
        }
        w!("threads", self.threads);
        w!("mipRelStop", self.mip_rel_gap);
        w!("mipAbsStop", self.mip_abs_gap);
        w!("maxNode", self.node_limit);
        if let Some(p) = self.presolve {
            kv(buf, "presolve", i32::from(p));
            n += 1;
        }
        w!("cutStrategy", self.cut_strategy);
        w!("feasTol", self.feas_tol);
        w!("optimalityTol", self.optimality_tol);
        w!("mipTol", self.mip_tol);
        w!("defaultAlg", self.lp_algorithm);
        let extra = write_raw(buf, &self.raw);
        n > 0 || extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::GamsSolver;

    #[test]
    fn baron_writes_space_separated() {
        let cfg = GamsSolverConfig::Baron(GamsBaronOptions {
            threads: Some(4),
            eps_r: Some(0.01),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("Threads 4\n"), "got: {buf}");
        assert!(buf.contains("EpsR 0.01\n"), "got: {buf}");
    }

    #[test]
    fn highs_writes_eq_separated() {
        let cfg = GamsSolverConfig::Highs(GamsHighsOptions {
            mip_rel_gap: Some(0.05),
            threads: Some(2),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("mip_rel_gap = 0.05\n"), "got: {buf}");
        assert!(buf.contains("threads = 2\n"), "got: {buf}");
    }

    #[test]
    fn highs_presolve_and_solver_enum() {
        let cfg = GamsSolverConfig::Highs(GamsHighsOptions {
            presolve: Some(GamsHighsPresolve::Off),
            solver: Some(GamsHighsSolver::Simplex),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("presolve = off\n"), "got: {buf}");
        assert!(buf.contains("solver = simplex\n"), "got: {buf}");
    }

    #[test]
    fn scip_writes_eq_separated() {
        let cfg = GamsSolverConfig::Scip(GamsScipOptions {
            mip_rel_gap: Some(0.01),
            node_limit: Some(1000),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("limits/gap = 0.01\n"), "got: {buf}");
        assert!(buf.contains("limits/nodes = 1000\n"), "got: {buf}");
    }

    #[test]
    fn gurobi_writes_space_separated() {
        let cfg = GamsSolverConfig::Gurobi(GamsGurobiOptions {
            threads: Some(8),
            mip_focus: Some(GamsGurobiMipFocus::Feasibility),
            presolve: Some(2),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("threads 8\n"), "got: {buf}");
        assert!(buf.contains("mipfocus 1\n"), "got: {buf}");
        assert!(buf.contains("presolve 2\n"), "got: {buf}");
    }

    #[test]
    fn cplex_bool_as_int_and_emphasis() {
        let cfg = GamsSolverConfig::Cplex(GamsCplexOptions {
            presolve: Some(false),
            mip_emphasis: Some(GamsCplexMipEmphasis::Feasibility),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("preind 0\n"), "got: {buf}");
        assert!(buf.contains("mipemphasis 1\n"), "got: {buf}");
    }

    #[test]
    fn cbc_enum_options() {
        let cfg = GamsSolverConfig::Cbc(GamsCbcOptions {
            presolve: Some(GamsCbcPresolve::On),
            cuts: Some(GamsCbcCuts::Root),
            heuristics: Some(true),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("presolve on\n"), "got: {buf}");
        assert!(buf.contains("cuts root\n"), "got: {buf}");
        assert!(buf.contains("heuristics 1\n"), "got: {buf}");
    }

    #[test]
    fn ipopt_string_options() {
        let cfg = GamsSolverConfig::Ipopt(GamsIpoptOptions {
            linear_solver: Some(GamsIpoptLinearSolver::Ma57),
            mu_strategy: Some(GamsIpoptMuStrategy::Adaptive),
            max_iter: Some(500),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("linear_solver ma57\n"), "got: {buf}");
        assert!(buf.contains("mu_strategy adaptive\n"), "got: {buf}");
        assert!(buf.contains("max_iter 500\n"), "got: {buf}");
    }

    #[test]
    fn knitro_algorithm_enum() {
        let cfg = GamsSolverConfig::Knitro(GamsKnitroOptions {
            algorithm: Some(GamsKnitroAlgorithm::Sqp),
            mip_rel_gap: Some(0.001),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("nlp_algorithm 4\n"), "got: {buf}");
        assert!(buf.contains("mip_opt_gap_rel 0.001\n"), "got: {buf}");
    }

    #[test]
    fn mosek_long_key_names() {
        let cfg = GamsSolverConfig::Mosek(GamsMosekOptions {
            threads: Some(2),
            mip_rel_gap: Some(1e-4),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("MSK_IPAR_NUM_THREADS 2\n"), "got: {buf}");
        assert!(buf.contains("MSK_DPAR_MIO_TOL_REL_GAP 0.0001\n"), "got: {buf}");
    }

    #[test]
    fn xpress_bool_presolve_and_gap() {
        let cfg = GamsSolverConfig::Xpress(GamsXpressOptions {
            presolve: Some(false),
            mip_rel_gap: Some(0.02),
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("presolve 0\n"), "got: {buf}");
        assert!(buf.contains("mipRelStop 0.02\n"), "got: {buf}");
    }

    #[test]
    fn empty_options_writes_nothing() {
        let cfg = GamsSolverConfig::Baron(GamsBaronOptions::default());
        let mut buf = String::new();
        assert!(!cfg.write_opt_file(&mut buf));
        assert!(buf.is_empty());
    }

    #[test]
    fn named_writes_nothing() {
        let cfg = GamsSolverConfig::Named(GamsSolver::Baron);
        let mut buf = String::new();
        assert!(!cfg.write_opt_file(&mut buf));
        assert!(buf.is_empty());
    }

    #[test]
    fn raw_lines_are_written_verbatim() {
        // Per-struct `raw` is appended after the typed options.
        let cfg = GamsSolverConfig::Cplex(GamsCplexOptions {
            mip_rel_gap: Some(0.01),
            raw: vec!["solnpool out.gdx".into(), "solnpoolpop 2".into()],
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("epgap 0.01"), "typed option missing:\n{buf}");
        assert!(buf.contains("solnpool out.gdx"), "raw line missing:\n{buf}");
        assert!(buf.contains("solnpoolpop 2"), "raw line missing:\n{buf}");

        // `raw` alone still triggers the option file.
        let cfg = GamsSolverConfig::Gurobi(GamsGurobiOptions {
            raw: vec!["solnpool out.gdx".into()],
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("solnpool out.gdx"), "raw line missing:\n{buf}");

        // The `Raw` variant writes the same verbatim lines for a named solver.
        let cfg = GamsSolverConfig::Raw(GamsSolver::Xpress, vec!["miptol 1e-6".into()]);
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert_eq!(buf, "miptol 1e-6\n");
    }

    #[test]
    fn gams_name_matches_variant() {
        assert_eq!(GamsSolverConfig::Baron(GamsBaronOptions::default()).gams_name(), "BARON");
        assert_eq!(GamsSolverConfig::Cbc(GamsCbcOptions::default()).gams_name(), "CBC");
        assert_eq!(GamsSolverConfig::Cplex(GamsCplexOptions::default()).gams_name(), "CPLEX");
        assert_eq!(GamsSolverConfig::Gurobi(GamsGurobiOptions::default()).gams_name(), "GUROBI");
        assert_eq!(GamsSolverConfig::Highs(GamsHighsOptions::default()).gams_name(), "HIGHS");
        assert_eq!(GamsSolverConfig::Ipopt(GamsIpoptOptions::default()).gams_name(), "IPOPT");
        assert_eq!(GamsSolverConfig::Knitro(GamsKnitroOptions::default()).gams_name(), "KNITRO");
        assert_eq!(GamsSolverConfig::Mosek(GamsMosekOptions::default()).gams_name(), "MOSEK");
        assert_eq!(GamsSolverConfig::Scip(GamsScipOptions::default()).gams_name(), "SCIP");
        assert_eq!(GamsSolverConfig::Xpress(GamsXpressOptions::default()).gams_name(), "XPRESS");
        assert_eq!(GamsSolverConfig::Named(GamsSolver::Cplex).gams_name(), "CPLEX");
        assert_eq!(
            GamsSolverConfig::Named(GamsSolver::Custom("MYMIP".into())).gams_name(),
            "MYMIP"
        );
    }

    #[test]
    fn from_gams_solver_becomes_named() {
        let cfg: GamsSolverConfig = GamsSolver::Gurobi.into();
        assert!(matches!(cfg, GamsSolverConfig::Named(GamsSolver::Gurobi)));
        assert_eq!(cfg.gams_name(), "GUROBI");
    }

    #[test]
    fn supports_matches_solver_capabilities() {
        // CPLEX: LP, MIP, QCP, MIQCP. QP/MIQP route through QCP/MIQCP, so the
        // quadratic kinds pass while the general (MI)NLP kinds fail.
        let cplex = GamsSolverConfig::Cplex(GamsCplexOptions::default());
        assert!(cplex.supports(ModelKind::LP));
        assert!(cplex.supports(ModelKind::MILP));
        assert!(cplex.supports(ModelKind::QP), "QP routes through QCP");
        assert!(cplex.supports(ModelKind::MIQP), "MIQP routes through MIQCP");
        assert!(!cplex.supports(ModelKind::NLP));
        assert!(!cplex.supports(ModelKind::MINLP));

        // IPOPT: LP, NLP, QCP. LP, NLP, and QP pass, the integer kinds fail.
        let ipopt = GamsSolverConfig::Ipopt(GamsIpoptOptions::default());
        assert!(ipopt.supports(ModelKind::LP));
        assert!(ipopt.supports(ModelKind::NLP));
        assert!(ipopt.supports(ModelKind::QP), "QP routes through QCP");
        assert!(!ipopt.supports(ModelKind::MIQP), "MIQP routes through MIQCP");
        assert!(!ipopt.supports(ModelKind::MINLP));

        // BARON handles all six oximo solve types.
        let baron = GamsSolverConfig::Named(GamsSolver::Baron);
        for k in [
            ModelKind::LP,
            ModelKind::MILP,
            ModelKind::QP,
            ModelKind::MIQP,
            ModelKind::NLP,
            ModelKind::MINLP,
        ] {
            assert!(baron.supports(k), "BARON should support {k:?}");
        }

        // HiGHS: LP/MIP only, no quadratic or nonlinear support THROUGH GAMS.
        let highs = GamsSolverConfig::Highs(GamsHighsOptions::default());
        assert!(highs.supports(ModelKind::LP));
        assert!(!highs.supports(ModelKind::QP));
        assert!(!highs.supports(ModelKind::NLP));
    }

    #[test]
    fn supports_is_permissive_for_unknown_names() {
        let custom = GamsSolverConfig::Named(GamsSolver::Custom("MYSOLVER".into()));
        assert!(custom.supports(ModelKind::MINLP));
        assert!(custom.supports(ModelKind::LP));
    }
}
