use highs::{HighsOptionValue, Model as HighsModel};
use oximo_solver::{HasUniversal, SolverError, UniversalOptions};

/// HiGHS-specific solver options.
#[derive(Clone, Debug, Default)]
pub struct HighsOptions {
    pub universal: UniversalOptions,
    pub mip_gap: Option<f64>,
    pub presolve: Option<HighsPresolve>,
    pub method: Option<HighsMethod>,
    pub parallel: Option<bool>,
}

/// HiGHS presolve options.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HighsPresolve {
    Off,
    On,
    Auto,
}

/// HiGHS LP / root-relaxation algorithm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HighsMethod {
    /// Let HiGHS pick.
    Choose,
    Simplex,
    /// Interior-point method.
    Ipm,
    /// First-order primal-dual LP solver.
    PdLp,
}

impl HighsOptions {
    #[must_use]
    pub fn mip_gap(mut self, gap: f64) -> Self {
        self.mip_gap = Some(gap);
        self
    }

    #[must_use]
    pub fn presolve(mut self, p: HighsPresolve) -> Self {
        self.presolve = Some(p);
        self
    }

    #[must_use]
    pub fn method(mut self, m: HighsMethod) -> Self {
        self.method = Some(m);
        self
    }

    #[must_use]
    pub fn parallel(mut self, on: bool) -> Self {
        self.parallel = Some(on);
        self
    }
}

impl HasUniversal for HighsOptions {
    fn universal(&self) -> &UniversalOptions {
        &self.universal
    }

    fn universal_mut(&mut self) -> &mut UniversalOptions {
        &mut self.universal
    }
}

/// Set a single HiGHS option through the non-panicking `try_set_option`,
/// mapping any failure to a [`SolverError::Backend`].
fn set<V: HighsOptionValue>(
    model: &mut HighsModel,
    name: &str,
    value: V,
) -> Result<(), SolverError> {
    model
        .try_set_option(name, value)
        .map_err(|e| SolverError::Backend(format!("HiGHS option {name:?}: {e}")))
}

/// Apply typed [`HighsOptions`] onto a live HiGHS model.
///
/// # Errors
///
/// Returns [`SolverError::Backend`] if HiGHS rejects an option name or value.
pub(crate) fn apply(model: &mut HighsModel, o: &HighsOptions) -> Result<(), SolverError> {
    if let Some(d) = o.universal.time_limit {
        set(model, "time_limit", d.as_secs_f64())?;
    }
    if let Some(n) = o.universal.threads {
        set(model, "threads", i32::try_from(n).unwrap_or(i32::MAX))?;
    }
    if let Some(b) = o.universal.verbose {
        set(model, "output_flag", b)?;
        set(model, "log_to_console", b)?;
    }
    if let Some(g) = o.mip_gap {
        set(model, "mip_rel_gap", g)?;
    }
    if let Some(p) = o.presolve {
        set(model, "presolve", presolve_str(p))?;
    }
    if let Some(m) = o.method {
        set(model, "solver", method_str(m))?;
    }
    if let Some(p) = o.parallel {
        set(model, "parallel", if p { "on" } else { "off" })?;
    }
    Ok(())
}

fn presolve_str(p: HighsPresolve) -> &'static str {
    match p {
        HighsPresolve::Off => "off",
        HighsPresolve::On => "on",
        HighsPresolve::Auto => "choose",
    }
}

fn method_str(m: HighsMethod) -> &'static str {
    match m {
        HighsMethod::Choose => "choose",
        HighsMethod::Simplex => "simplex",
        HighsMethod::Ipm => "ipm",
        HighsMethod::PdLp => "pdlp",
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use highs::{RowProblem, Sense as HighsSense};
    use oximo_solver::UniversalOptionsExt;

    use super::*;

    fn empty_highs_model() -> HighsModel {
        RowProblem::default().optimise(HighsSense::Minimise)
    }

    #[test]
    fn builder_sets_all_fields() {
        let o = HighsOptions::default()
            .time_limit(Duration::from_secs(30))
            .threads(8)
            .verbose(true)
            .mip_gap(0.01)
            .presolve(HighsPresolve::Off)
            .method(HighsMethod::Ipm)
            .parallel(true);
        assert_eq!(o.universal.time_limit, Some(Duration::from_secs(30)));
        assert_eq!(o.universal.threads, Some(8));
        assert_eq!(o.universal.verbose, Some(true));
        assert_eq!(o.mip_gap, Some(0.01));
        assert_eq!(o.presolve, Some(HighsPresolve::Off));
        assert_eq!(o.method, Some(HighsMethod::Ipm));
        assert_eq!(o.parallel, Some(true));
    }

    #[test]
    fn apply_default_succeeds() {
        let mut m = empty_highs_model();
        apply(&mut m, &HighsOptions::default()).unwrap();
    }

    #[test]
    fn apply_all_options_succeeds() {
        let mut m = empty_highs_model();
        let o = HighsOptions::default()
            .time_limit(Duration::from_secs(10))
            .threads(1)
            .verbose(false)
            .mip_gap(0.01)
            .presolve(HighsPresolve::Off)
            .method(HighsMethod::Simplex)
            .parallel(false);
        apply(&mut m, &o).unwrap();
    }

    #[test]
    fn apply_every_method_variant() {
        for method in
            [HighsMethod::Choose, HighsMethod::Simplex, HighsMethod::Ipm, HighsMethod::PdLp]
        {
            let mut m = empty_highs_model();
            apply(&mut m, &HighsOptions::default().method(method)).unwrap();
        }
    }

    #[test]
    fn apply_every_presolve_variant() {
        for presolve in [HighsPresolve::Off, HighsPresolve::On, HighsPresolve::Auto] {
            let mut m = empty_highs_model();
            apply(&mut m, &HighsOptions::default().presolve(presolve)).unwrap();
        }
    }
}
