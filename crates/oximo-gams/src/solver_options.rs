//! Per-solver typed option structs and [`GamsSolverConfig`].
//!
//! The option structs are generated at build time from the checked-in
//! `option-snapshots/<solver>.txt` files (see `build.rs`), which are scraped
//! from the GAMS solver docs.
//!
//! Each solver gets a `Gams<Name>Options` struct with one builder setter
//! per documented option. The setter name is the snake_case of the option
//! and the exact GAMS key is written to the `.opt` file.
//!
//! For any option without a generated setter, push a verbatim line onto
//! the public `raw` field.
//!
//! Reference: "GAMS Solver Manuals," GAMS Development Corporation.
//! <https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES>

use oximo_core::ModelKind;

use crate::options::GamsSolver;
use crate::translate::gams_solve_type;

/// Generates one setter for a scraped option, typed by the kind GAMS declares.
///
/// `any` means the solver's manual publishes no declared type, so the setter
/// stays permissive. Booleans are written as `1`/`0`.
macro_rules! gams_setter {
    (int, $method:ident, $key:literal) => {
        #[doc = concat!("Set GAMS option `", $key, "` (integer).")]
        #[must_use]
        pub fn $method(mut self, v: i64) -> Self {
            self.opts.push(($key, v.to_string()));
            self
        }
    };
    (dbl, $method:ident, $key:literal) => {
        #[doc = concat!("Set GAMS option `", $key, "` (real).")]
        #[must_use]
        pub fn $method(mut self, v: f64) -> Self {
            self.opts.push(($key, v.to_string()));
            self
        }
    };
    (bool, $method:ident, $key:literal) => {
        #[doc = concat!("Set GAMS option `", $key, "` (boolean, written as `1`/`0`).")]
        #[must_use]
        pub fn $method(mut self, v: bool) -> Self {
            self.opts.push(($key, if v { "1" } else { "0" }.to_owned()));
            self
        }
    };
    (str, $method:ident, $key:literal) => {
        #[doc = concat!("Set GAMS option `", $key, "` (string).")]
        #[must_use]
        pub fn $method(mut self, v: impl ::std::fmt::Display) -> Self {
            self.opts.push(($key, v.to_string()));
            self
        }
    };
    (any, $method:ident, $key:literal) => {
        #[doc = concat!("Set GAMS option `", $key, "`. This solver's manual \
            declares no type for it, so any `Display` value is accepted.")]
        #[must_use]
        pub fn $method(mut self, v: impl ::std::fmt::Display) -> Self {
            self.opts.push(($key, v.to_string()));
            self
        }
    };
}

/// Generates a solver's option struct, including a public `raw` passthrough,
/// the private key/value store, one setter per option, and `render`
/// (writes the `.opt` body using `$sep` between key and value: `" "` or `" = "`).
macro_rules! gams_params {
    ($(#[$meta:meta])* $struct:ident, $sep:literal,
     options: [ $( ($kind:ident, $method:ident, $key:literal) ),* $(,)? ],
     enums: [ $( ($ename:ident, $emethod:ident, $ekey:literal,
                  [ $( $variant:ident = $value:literal ),* $(,)? ]) ),* $(,)? ] $(,)?) => {
        $(
            #[doc = concat!("Allowed values for GAMS option `", $ekey, "`.")]
            #[derive(Clone, Copy, Debug, Eq, PartialEq)]
            pub enum $ename {
                $( #[doc = concat!("`", $value, "`")] $variant, )*
            }

            impl $ename {
                /// The exact token written to the option file.
                #[must_use]
                pub fn as_str(self) -> &'static str {
                    match self { $( Self::$variant => $value, )* }
                }
            }
        )*

        $(#[$meta])*
        #[derive(Clone, Debug, Default)]
        pub struct $struct {
            /// Extra option-file lines written verbatim, for options without a
            /// generated setter (or values a setter can't express).
            pub raw: Vec<String>,
            /// Key/value store behind the generated setters. Public (but hidden)
            /// only so `Struct { raw, ..Default::default() }` construction works.
            /// Prefer the setters.
            #[doc(hidden)]
            pub opts: Vec<(&'static str, String)>,
        }

        #[allow(clippy::pedantic)]
        impl $struct {
            $( gams_setter!($kind, $method, $key); )*

            $(
                #[doc = concat!("Set GAMS option `", $ekey, "`, restricted to its \
                    documented values.")]
                #[must_use]
                pub fn $emethod(mut self, v: $ename) -> Self {
                    self.opts.push(($ekey, v.as_str().to_owned()));
                    self
                }
            )*

            /// Write the `.opt` file body into `buf`. Returns `true` if anything
            /// was written.
            fn render(&self, buf: &mut String) -> bool {
                use ::std::fmt::Write as _;
                for (k, v) in &self.opts {
                    let _ = writeln!(buf, concat!("{}", $sep, "{}"), k, v);
                }
                for line in &self.raw {
                    let _ = writeln!(buf, "{line}");
                }
                !self.opts.is_empty() || !self.raw.is_empty()
            }
        }
    };
}

// The generated structs, the `GamsSolverConfig` enum, its `gams_name`/
// `write_opt_file` dispatch, and `From<GamsSolver>`.
include!(concat!(env!("OUT_DIR"), "/gams_generated.rs"));

impl GamsSolverConfig {
    /// Whether this solver can handle `kind` under oximo's GAMS translation,
    /// which emits `QP` as a `QCP` solve and `MIQP` as a `MIQCP` solve.
    ///
    /// [`GamsSolver::Custom`] and any unrecognized name return `true`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::GamsSolver;

    #[test]
    fn setter_and_render_space_separated() {
        let cfg = GamsSolverConfig::Gurobi(GamsGurobiOptions::default().threads(8).mipgap(0.01));
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("threads 8\n"), "got: {buf}");
        assert!(buf.contains("mipgap 0.01\n"), "got: {buf}");
    }

    #[test]
    fn highs_and_scip_render_eq_separated() {
        let mut buf = String::new();
        assert!(
            GamsSolverConfig::Highs(GamsHighsOptions::default().mip_rel_gap(0.05))
                .write_opt_file(&mut buf)
        );
        assert!(buf.contains("mip_rel_gap = 0.05\n"), "got: {buf}");

        let mut buf = String::new();
        assert!(
            GamsSolverConfig::Scip(GamsScipOptions::default().limits_gap(0.01))
                .write_opt_file(&mut buf)
        );
        assert!(buf.contains("limits/gap = 0.01\n"), "got: {buf}");
    }

    #[test]
    fn untyped_setter_accepts_any_display_value() {
        // HiGHS publishes no declared types, so its setters stay permissive and
        // take floats, ints or strings alike.
        //
        // Note these are raw GAMS option keys. Time limits, thread counts and MIP
        // gaps are better set once on `GamsOptions` via the universal options.
        let cfg = GamsSolverConfig::Highs(
            GamsHighsOptions::default()
                .mip_rel_gap(1e-6)
                .simplex_max_concurrency(4)
                .solution_file("sol.txt"),
        );
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("mip_rel_gap = 0.000001\n"), "got: {buf}");
        assert!(buf.contains("simplex_max_concurrency = 4\n"), "got: {buf}");
        assert!(buf.contains("solution_file = sol.txt\n"), "got: {buf}");
    }

    #[test]
    fn time_limit_comes_from_the_universal_layer_as_a_duration() {
        use oximo_solver::UniversalOptionsExt;
        let opts = crate::GamsOptions::default()
            .time_limit(std::time::Duration::from_secs(3600))
            .solver(GamsSolverConfig::Highs(GamsHighsOptions::default()));
        let mut gms = String::new();
        crate::options::write_options(&mut gms, &opts, "LP");
        assert!(gms.contains("option ResLim = 3600;"), "got: {gms}");
    }

    #[test]
    fn enumerated_option_takes_a_generated_enum() {
        let cfg =
            GamsSolverConfig::Highs(GamsHighsOptions::default().solver(GamsHighsSolver::Simplex));
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("solver = simplex\n"), "got: {buf}");
        assert_eq!(GamsHighsSolver::Ipm.as_str(), "ipm");
    }

    #[test]
    fn declared_types_produce_typed_setters() {
        let cfg = GamsSolverConfig::Cplex(
            GamsCplexOptions::default().preind(false).threads(4).epgap(0.01),
        );
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("preind 0\n"), "bool renders as 0/1: {buf}");
        assert!(buf.contains("threads 4\n"), "got: {buf}");
        assert!(buf.contains("epgap 0.01\n"), "got: {buf}");
    }

    #[test]
    fn raw_lines_written_after_typed() {
        let cfg = GamsSolverConfig::Cplex(GamsCplexOptions {
            raw: vec!["solnpool out.gdx".into(), "solnpoolpop 2".into()],
            ..Default::default()
        });
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert!(buf.contains("solnpool out.gdx\n"), "got: {buf}");
        assert!(buf.contains("solnpoolpop 2\n"), "got: {buf}");
    }

    #[test]
    fn empty_options_write_nothing() {
        let mut buf = String::new();
        assert!(!GamsSolverConfig::Baron(GamsBaronOptions::default()).write_opt_file(&mut buf));
        assert!(buf.is_empty());
    }

    #[test]
    fn named_writes_nothing_but_raw_variant_does() {
        let mut buf = String::new();
        assert!(!GamsSolverConfig::Named(GamsSolver::Baron).write_opt_file(&mut buf));
        assert!(buf.is_empty());

        let cfg = GamsSolverConfig::Raw(GamsSolver::Xpress, vec!["miptol 1e-6".into()]);
        let mut buf = String::new();
        assert!(cfg.write_opt_file(&mut buf));
        assert_eq!(buf, "miptol 1e-6\n");
    }

    #[test]
    fn gams_name_matches_variant() {
        assert_eq!(GamsSolverConfig::Baron(GamsBaronOptions::default()).gams_name(), "BARON");
        assert_eq!(GamsSolverConfig::Cplex(GamsCplexOptions::default()).gams_name(), "CPLEX");
        assert_eq!(GamsSolverConfig::Scip(GamsScipOptions::default()).gams_name(), "SCIP");
        assert_eq!(
            GamsSolverConfig::Odhcplex(GamsOdhcplexOptions::default()).gams_name(),
            "ODHCPLEX"
        );
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
        // CPLEX: LP, MIP, QCP, MIQCP. QP/MIQP route through QCP/MIQCP.
        let cplex = GamsSolverConfig::Cplex(GamsCplexOptions::default());
        assert!(cplex.supports(ModelKind::LP));
        assert!(cplex.supports(ModelKind::MILP));
        assert!(cplex.supports(ModelKind::QP), "QP routes through QCP");
        assert!(cplex.supports(ModelKind::MIQP), "MIQP routes through MIQCP");
        assert!(!cplex.supports(ModelKind::NLP));
        assert!(!cplex.supports(ModelKind::MINLP));

        // IPOPT: LP, NLP, QCP.
        let ipopt = GamsSolverConfig::Ipopt(GamsIpoptOptions::default());
        assert!(ipopt.supports(ModelKind::LP));
        assert!(ipopt.supports(ModelKind::NLP));
        assert!(ipopt.supports(ModelKind::QP), "QP routes through QCP");
        assert!(!ipopt.supports(ModelKind::MIQP));
        assert!(!ipopt.supports(ModelKind::MINLP));

        // HiGHS under GAMS is LP/MIP only.
        let highs = GamsSolverConfig::Highs(GamsHighsOptions::default());
        assert!(highs.supports(ModelKind::LP));
        assert!(!highs.supports(ModelKind::QP));
        assert!(!highs.supports(ModelKind::NLP));
    }

    #[test]
    fn every_snapshot_option_is_generated() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/option-snapshots");
        let mut seen = 0;
        for entry in std::fs::read_dir(dir).expect("option-snapshots dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read snapshot");
            let mut lines = text.lines();
            let name = lines.next().unwrap_or("").trim_start_matches("//").trim().to_string();
            let on_disk = lines.filter(|l| l.trim_start().starts_with('(')).count();

            let (_, _, generated) = GENERATED_SOLVERS
                .iter()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| panic!("snapshot {name} produced no generated solver"));
            assert_eq!(
                *generated, on_disk,
                "{name}: generated {generated} setters but snapshot has {on_disk} options"
            );
            seen += 1;
        }
        assert_eq!(seen, GENERATED_SOLVERS.len(), "generated solvers without a snapshot file");
        assert!(seen >= 24, "expected at least 24 solvers, found {seen}");
    }

    #[test]
    fn separator_matches_documented_solver_family() {
        for (name, sep, _) in GENERATED_SOLVERS {
            let expected = match *name {
                "SCIP" | "SOPLEX" | "SHOT" | "HIGHS" => " = ",
                _ => " ",
            };
            assert_eq!(sep, &expected, "{name} has separator {sep:?}, expected {expected:?}");
        }
    }

    #[test]
    fn soplex_and_shot_render_eq_separated() {
        let mut buf = String::new();
        assert!(
            GamsSolverConfig::Soplex(GamsSoplexOptions::default().real_feastol(1e-5))
                .write_opt_file(&mut buf)
        );
        assert!(buf.contains("real:feastol = 0.00001\n"), "got: {buf}");

        let mut buf = String::new();
        assert!(
            GamsSolverConfig::Shot(GamsShotOptions::default().dual_mip_solver(2))
                .write_opt_file(&mut buf)
        );
        assert!(buf.contains("Dual.MIP.Solver = 2\n"), "got: {buf}");
    }

    #[test]
    fn supports_is_permissive_for_unknown_names() {
        let custom = GamsSolverConfig::Named(GamsSolver::Custom("MYSOLVER".into()));
        assert!(custom.supports(ModelKind::MINLP));
        assert!(custom.supports(ModelKind::LP));
    }
}
