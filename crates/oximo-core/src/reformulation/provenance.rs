//! Shared provenance of registered solver-independent reformulations.

use std::ops::Range;

use crate::{GeneratedVarId, Model, ReformulationVarId, VarId};

/// A snapshot of generated variable and algebraic-row membership.
///
/// Capture once per planning or projection pass.
#[derive(Clone, Debug)]
pub struct ReformulationProvenance {
    captured_variables: usize,
    variables: Vec<Range<usize>>,
    rows: Vec<Range<usize>>,
}

impl ReformulationProvenance {
    pub fn capture(model: &Model) -> Self {
        let mut variables = Vec::new();
        let mut rows = Vec::new();
        #[cfg(feature = "gdp")]
        for report in model.gdp_reformulations().iter() {
            variables.push(report.generated_variables.clone());
            rows.push(report.generated_rows.clone());
        }

        for report in model.sos_reformulations().iter() {
            append_indices(&mut variables, report.variables.iter().map(|id| id.index()));
            append_indices(&mut rows, report.constraints.iter().map(|id| id.index()));
        }

        for report in model.indicator_reformulations().iter() {
            append_indices(&mut rows, report.constraints.iter().map(|id| id.index()));
        }

        normalize_ranges(&mut variables);
        normalize_ranges(&mut rows);

        Self { captured_variables: model.num_variables(), variables, rows }
    }

    /// Sorted, disjoint generated variable-index ranges.
    pub fn variable_ranges(&self) -> &[Range<usize>] {
        &self.variables
    }

    /// Sorted, disjoint generated algebraic-row-index ranges.
    pub fn row_ranges(&self) -> &[Range<usize>] {
        &self.rows
    }

    /// Classify a variable in this registry or an appended planning coordinate.
    pub fn variable_origin(&self, id: VarId) -> ReformulationVarId {
        let index = id.index();
        let range = self.variables.partition_point(|range| range.end <= index);
        if index >= self.captured_variables
            || self.variables.get(range).is_some_and(|range| range.contains(&index))
        {
            ReformulationVarId::Generated(GeneratedVarId::from_raw(id))
        } else {
            ReformulationVarId::Source(id)
        }
    }
}

fn append_indices(ranges: &mut Vec<Range<usize>>, indices: impl Iterator<Item = usize>) {
    for index in indices {
        if let Some(last) = ranges.last_mut().filter(|last| last.end == index) {
            last.end = index + 1;
        } else {
            ranges.push(index..index + 1);
        }
    }
}

fn normalize_ranges(ranges: &mut Vec<Range<usize>>) {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut retained = 0;
    for i in 0..ranges.len() {
        let range = ranges[i].clone();
        if range.is_empty() {
            continue;
        }
        if retained > 0 && range.start <= ranges[retained - 1].end {
            ranges[retained - 1].end = ranges[retained - 1].end.max(range.end);
        } else {
            ranges[retained] = range;
            retained += 1;
        }
    }
    ranges.truncate(retained);
}
