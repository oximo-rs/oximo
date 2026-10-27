//! Human-readable reformulation summaries and model-aware details.

use std::fmt;

use super::{IndicatorReformulationArtifacts, SosReformulationArtifacts};
use crate::{ConstraintId, Model};

/// Displays a reformulation's generated variables and rows using model names.
#[derive(Debug)]
pub struct ReformulationDisplay<'a> {
    model: &'a Model,
    report: Report<'a>,
}

#[derive(Debug)]
enum Report<'a> {
    Sos(&'a SosReformulationArtifacts),
    Indicator(&'a IndicatorReformulationArtifacts),
    #[cfg(feature = "gdp")]
    Gdp(&'a crate::gdp::GdpReformulationReport),
}

impl<'a> ReformulationDisplay<'a> {
    pub(crate) const fn sos(report: &'a SosReformulationArtifacts, model: &'a Model) -> Self {
        Self { model, report: Report::Sos(report) }
    }

    pub(crate) const fn indicator(
        report: &'a IndicatorReformulationArtifacts,
        model: &'a Model,
    ) -> Self {
        Self { model, report: Report::Indicator(report) }
    }

    #[cfg(feature = "gdp")]
    pub(crate) const fn gdp(
        report: &'a crate::gdp::GdpReformulationReport,
        model: &'a Model,
    ) -> Self {
        Self { model, report: Report::Gdp(report) }
    }
}

impl fmt::Display for SosReformulationArtifacts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SOS reformulation #{}: {} generated variable{}, {} generated constraint{}",
            self.source.index(),
            self.variables.len(),
            plural(self.variables.len()),
            self.constraints.len(),
            plural(self.constraints.len())
        )
    }
}

impl fmt::Display for IndicatorReformulationArtifacts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Indicator reformulation #{}: {} generated constraint{}",
            self.source.index(),
            self.constraints.len(),
            plural(self.constraints.len())
        )
    }
}

#[cfg(feature = "gdp")]
impl fmt::Display for crate::gdp::GdpReformulationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GDP reformulation: {} generated variable{}, {} generated constraint{}, {} Hull {}",
            self.generated_variables.len(),
            plural(self.generated_variables.len()),
            self.generated_rows.len(),
            plural(self.generated_rows.len()),
            self.hull_variables.len(),
            if self.hull_variables.len() == 1 { "copy" } else { "copies" }
        )
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

impl fmt::Display for ReformulationDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model = self.model;
        match self.report {
            Report::Sos(report) => {
                writeln!(f, "{report}")?;
                writeln!(f, "source: {}", model.display_sos(report.source))?;
                write_variables(f, model, report.generated_variables(model))?;
                write_rows(f, model, report.constraints.iter().map(|id| id.raw_id()))
            }
            Report::Indicator(report) => {
                writeln!(f, "{report}")?;
                writeln!(f, "source: {}", model.display_indicator(report.source))?;
                write_rows(f, model, report.constraints.iter().map(|id| id.raw_id()))
            }
            #[cfg(feature = "gdp")]
            Report::Gdp(report) => {
                writeln!(f, "{report}")?;
                write_variables(f, model, report.generated_variables(model))?;
                write_rows(
                    f,
                    model,
                    report.generated_rows.clone().map(|index| {
                        ConstraintId(u32::try_from(index).expect("constraint ID fits u32"))
                    }),
                )
            }
        }
    }
}

fn write_variables<'model>(
    f: &mut fmt::Formatter<'_>,
    model: &'model Model,
    variables: impl ExactSizeIterator<Item = std::cell::Ref<'model, crate::Variable>>,
) -> fmt::Result {
    if variables.len() == 0 {
        return Ok(());
    }
    writeln!(f, "vars")?;
    let arena = model.arena();
    for variable in variables {
        let (lower, upper) = variable.resolved_bounds(&arena);
        writeln!(f, "  {}: {}, [{lower}, {upper}]", variable.name, variable.domain)?;
    }
    Ok(())
}

fn write_rows(
    f: &mut fmt::Formatter<'_>,
    model: &Model,
    rows: impl ExactSizeIterator<Item = ConstraintId>,
) -> fmt::Result {
    if rows.len() == 0 {
        return Ok(());
    }
    writeln!(f, "s.t.")?;
    for id in rows {
        writeln!(f, "  {}", model.display_row(id))?;
    }
    Ok(())
}
