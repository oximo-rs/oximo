use highs::Model as HighsModel;
use oximo_solver::{HasMip, HasUniversal, MipOptions, Presolve, UniversalOptions};

/// HiGHS-specific solver options.
///
/// Universal options (`time_limit`, `threads`, `verbose`) come from the embedded
/// [`UniversalOptions`] via [`UniversalOptionsExt`](oximo_solver::UniversalOptionsExt).
/// LP/MILP options (`mip_gap`, `presolve`) come from the embedded [`MipOptions`]
/// via [`MipOptionsExt`](oximo_solver::MipOptionsExt). HiGHS-only options
/// (`method`, `parallel`) live as their own fields.
#[derive(Clone, Debug, Default)]
pub struct HighsOptions {
    pub universal: UniversalOptions,
    pub mip: MipOptions,
    /// LP / MILP algorithm choice. Maps to the HiGHS `solver` option.
    pub method: Option<HighsMethod>,
    /// Enable HiGHS parallel solving. Maps to the HiGHS `parallel` option.
    pub parallel: Option<bool>,
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

impl HasMip for HighsOptions {
    fn mip(&self) -> &MipOptions {
        &self.mip
    }

    fn mip_mut(&mut self) -> &mut MipOptions {
        &mut self.mip
    }
}

/// Apply typed [`HighsOptions`] onto a live HiGHS model.
pub(crate) fn apply(model: &mut HighsModel, o: &HighsOptions) {
    if let Some(d) = o.universal.time_limit {
        model.set_option("time_limit", d.as_secs_f64());
    }
    if let Some(n) = o.universal.threads {
        model.set_option("threads", i32::try_from(n).unwrap_or(i32::MAX));
    }
    if let Some(b) = o.universal.verbose {
        model.set_option("output_flag", b);
        model.set_option("log_to_console", b);
    }
    if let Some(g) = o.mip.mip_gap {
        model.set_option("mip_rel_gap", g);
    }
    if let Some(p) = o.mip.presolve {
        model.set_option("presolve", presolve_str(p));
    }
    if let Some(m) = o.method {
        model.set_option("solver", method_str(m));
    }
    if let Some(p) = o.parallel {
        model.set_option("parallel", if p { "on" } else { "off" });
    }
}

fn presolve_str(p: Presolve) -> &'static str {
    match p {
        Presolve::Off => "off",
        Presolve::On => "on",
        Presolve::Auto => "choose",
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
    use oximo_solver::{MipOptionsExt, Presolve, UniversalOptionsExt};

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
            .presolve(Presolve::Off)
            .method(HighsMethod::Ipm)
            .parallel(true);
        assert_eq!(o.universal.time_limit, Some(Duration::from_secs(30)));
        assert_eq!(o.universal.threads, Some(8));
        assert_eq!(o.universal.verbose, Some(true));
        assert_eq!(o.mip.mip_gap, Some(0.01));
        assert_eq!(o.mip.presolve, Some(Presolve::Off));
        assert_eq!(o.method, Some(HighsMethod::Ipm));
        assert_eq!(o.parallel, Some(true));
    }

    #[test]
    fn apply_default_does_not_panic() {
        let mut m = empty_highs_model();
        apply(&mut m, &HighsOptions::default());
    }

    #[test]
    fn apply_all_options_does_not_panic() {
        let mut m = empty_highs_model();
        let o = HighsOptions::default()
            .time_limit(Duration::from_secs(10))
            .threads(1)
            .verbose(false)
            .mip_gap(0.01)
            .presolve(Presolve::Off)
            .method(HighsMethod::Simplex)
            .parallel(false);
        apply(&mut m, &o);
    }

    #[test]
    fn apply_every_method_variant() {
        for method in
            [HighsMethod::Choose, HighsMethod::Simplex, HighsMethod::Ipm, HighsMethod::PdLp]
        {
            let mut m = empty_highs_model();
            apply(&mut m, &HighsOptions::default().method(method));
        }
    }

    #[test]
    fn apply_every_presolve_variant() {
        for presolve in [Presolve::Off, Presolve::On, Presolve::Auto] {
            let mut m = empty_highs_model();
            apply(&mut m, &HighsOptions::default().presolve(presolve));
        }
    }
}
