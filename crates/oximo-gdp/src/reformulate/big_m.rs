//! Conditional numeric rows and disjunction selection.

use super::plan::{Linear, PlanBuilder};
use super::{BigMValues, GdpError, GdpMethod, GdpReformulationOptions};
use crate::bounds::{
    BoundError, BoundScopes, Bounder, Bounds, ExpressionAnalysis, PropagationScratch,
};
use oximo_core::gdp::{
    BigMOrigin, BigMSide, BooleanId, BooleanRecord, BoundSide, DisjunctConstraintId, DisjunctionId,
    DisjunctionKind, GdpBigMTerm, GdpDisjunctionArtifacts, GdpMethodKind, GdpRowArtifacts,
    GdpSnapshot,
};
use oximo_core::{ExprArena, ExprId, Model, VarId};
use oximo_expr::{SignedExpr, split_linear};
use rustc_hash::FxHashMap;
use smol_str::SmolStr;

pub(super) fn validate_m(value: Option<f64>) -> Result<(), GdpError> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(GdpError::InvalidBigM(value));
    }

    Ok(())
}

pub(super) fn method_m(method: GdpMethod) -> Option<f64> {
    match method {
        GdpMethod::BigM(m) => m.fallback,
    }
}

fn side_m(
    required: f64,
    explicit: Option<f64>,
    local: Option<f64>,
    global: Option<f64>,
    name: &SmolStr,
    side: BoundSide,
) -> Result<BigMSide, GdpError> {
    if let Some(value) = explicit {
        return Ok(BigMSide { value, origin: BigMOrigin::Explicit });
    }

    if required.is_finite() || required == f64::NEG_INFINITY {
        return Ok(BigMSide { value: required.max(0.0), origin: BigMOrigin::Estimated });
    }

    if let Some(value) = local {
        return Ok(BigMSide { value, origin: BigMOrigin::DisjunctionFallback });
    }

    if let Some(value) = global {
        return Ok(BigMSide { value, origin: BigMOrigin::GlobalFallback });
    }

    Err(GdpError::MissingBigM { constraint: name.clone(), side })
}

fn checked_big_m_bound(
    bound: f64,
    shift: f64,
    name: &SmolStr,
    side: BoundSide,
) -> Result<f64, GdpError> {
    let shifted = bound + shift;

    if !bound.is_finite() || !shifted.is_finite() {
        return Err(GdpError::NumericOverflow);
    }

    // When the indicator is one, the solver cancels this shift against its
    // coefficient. Finite arithmetic can still erase a small active bound.
    // Require recovery within 1e-9 relative error.
    if ((shifted - shift) - bound).abs() > 1e-9 * bound.abs() {
        return Err(GdpError::PrecisionLoss { constraint: name.clone(), side });
    }

    Ok(shifted)
}

/// Shift a normalized affine body and bound together, including any existing
/// coefficient of the row's indicator. At indicator = 1, both shifts must cancel
/// without changing the original coefficient or effective active bound.
fn checked_big_m_row(
    mut body: Linear,
    indicator: VarId,
    bound: f64,
    shift: f64,
    name: &SmolStr,
    side: BoundSide,
) -> Result<(Linear, f64), GdpError> {
    let shifted_bound = checked_big_m_bound(bound, shift, name, side)?;

    if let Some((_, coefficient)) = body.terms.iter_mut().find(|(v, _)| *v == indicator) {
        checked_coefficient(coefficient, bound, shifted_bound, shift, name, side)?;
    } else {
        body.terms.push((indicator, shift));
    }

    Ok((body, shifted_bound))
}

fn checked_coefficient(
    coefficient: &mut f64,
    bound: f64,
    shifted_bound: f64,
    shift: f64,
    name: &SmolStr,
    side: BoundSide,
) -> Result<(), GdpError> {
    let shifted_coefficient = checked_big_m_bound(*coefficient, shift, name, side)?;

    // Individually recoverable large values can still lose their small
    // difference. Check the active row after substituting indicator = 1.
    let active_bound = bound - *coefficient;
    let recovered_bound = shifted_bound - shifted_coefficient;

    if active_bound.is_finite()
        && (recovered_bound - active_bound).abs() > 1e-9 * active_bound.abs()
    {
        return Err(GdpError::PrecisionLoss { constraint: name.clone(), side });
    }
    *coefficient = shifted_coefficient;

    Ok(())
}

fn hierarchical_method(
    options: &GdpReformulationOptions,
    methods: &FxHashMap<DisjunctionId, GdpMethod>,
    boolean: &BooleanRecord,
) -> bool {
    let method =
        boolean.owner.and_then(|id| methods.get(&id)).copied().unwrap_or(options.default_method);
    match method {
        GdpMethod::BigM(settings) => options.bound_tightening && settings.hierarchical_tightening(),
    }
}

/// Split the global M into telescoping child/ancestor amounts.
/// Each context includes every ancestor above it.
/// The amounts sum to at least the global M when the entire branch is inactive.
/// Round differences outward as well.
///
/// Perez, H.D., Grossmann, I.E. Extensions to generalized disjunctive programming:
/// hierarchical structures and first-order logic. Optim Eng 25, 959–998 (2024).
/// Theorem 2, equations (8a)-(12).
/// https://doi.org/10.1007/s11081-023-09831-x
fn hierarchical_terms(
    indicator: BooleanId,
    contexts: &[(BooleanId, Bounds)],
    lower: f64,
    upper: f64,
    lower_m: Option<BigMSide>,
    upper_m: Option<BigMSide>,
) -> Vec<GdpBigMTerm> {
    if contexts.is_empty() {
        return Vec::new();
    }

    let mut child =
        GdpBigMTerm { indicator, lower: lower_m.map(|m| m.value), upper: upper_m.map(|m| m.value) };
    let mut terms = Vec::with_capacity(contexts.len() + 1);
    let mut improved = false;

    for &(parent, body) in contexts {
        let mut split = |previous: &mut Option<f64>, required: f64| {
            previous.as_mut().map(|previous| {
                let local = previous.min(round_up(required).max(0.0));
                let difference = round_up(*previous - local);
                improved |= difference > 0.0;
                *previous = local;
                difference
            })
        };

        terms.push(GdpBigMTerm {
            indicator: parent,
            lower: split(&mut child.lower, lower - body.lower),
            upper: split(&mut child.upper, body.upper - upper),
        });
    }

    if !improved {
        return Vec::new();
    }

    terms.insert(0, child);
    terms
}

fn checked_hierarchical_row(
    mut body: Linear,
    mut bound: f64,
    terms: &[GdpBigMTerm],
    data: &GdpSnapshot,
    diagnostic: (&SmolStr, BoundSide),
    slots: &mut Vec<usize>,
) -> Result<(Linear, f64), GdpError> {
    let (name, side) = diagnostic;

    // Locate each indicator once. The same indices serve both individual
    // shifts and the complete active-bound check, in the original term order.
    slots.clear();
    body.terms.reserve(terms.len());

    for term in terms {
        let binary = data.booleans[term.indicator.index()].binary;
        let slot = body.terms.iter().position(|(variable, _)| *variable == binary);
        slots.push(slot.unwrap_or_else(|| {
            body.terms.push((binary, 0.0));
            body.terms.len() - 1
        }));
    }

    let active_bound = |body: &Linear, bound: f64| {
        // Sum before subtracting to avoid intermediate cancellation when
        // substituting the indicators one at a time.
        bound - slots.iter().map(|&slot| body.terms[slot].1).sum::<f64>()
    };

    let original = active_bound(&body, bound);

    for (term, &slot) in terms.iter().zip(slots.iter()) {
        let shift = match side {
            BoundSide::Lower => -term.lower.expect("lower M"),
            BoundSide::Upper => term.upper.expect("upper M"),
        };

        let shifted_bound = checked_big_m_bound(bound, shift, name, side)?;
        checked_coefficient(&mut body.terms[slot].1, bound, shifted_bound, shift, name, side)?;
        bound = shifted_bound;
    }

    let recovered = active_bound(&body, bound);

    if original.is_finite() && (recovered - original).abs() > 1e-9 * original.abs() {
        return Err(GdpError::PrecisionLoss { constraint: name.clone(), side });
    }

    Ok((body, bound))
}

fn parent_scopes(
    plan: &mut PlanBuilder,
    bounder: &Bounder<'_>,
    data: &GdpSnapshot,
    options: &GdpReformulationOptions,
    methods: &FxHashMap<DisjunctionId, GdpMethod>,
    row_start: usize,
) -> Option<BoundScopes> {
    // Build only ancestor contexts needed by pending numeric rows.
    // We keep sparse deltas.
    let parents: Vec<_> = data.rows[row_start..]
        .iter()
        .filter_map(|row| {
            let boolean = &data.booleans[row.indicator.index()];
            boolean.parent.filter(|_| hierarchical_method(options, methods, boolean))
        })
        .collect();
    if parents.is_empty() {
        None
    } else {
        let mut first = vec![usize::MAX; data.booleans.len()];
        let mut next = vec![usize::MAX; data.rows.len()];

        for (index, row) in data.rows.iter().enumerate().rev() {
            next[index] = first[row.indicator.index()];
            first[row.indicator.index()] = index;
        }

        let mut scopes = BoundScopes::new(bounder, data.booleans.len());
        let mut stack = Vec::new();
        let mut scratch = PropagationScratch::default();

        for parent in parents {
            let mut ancestor = Some(parent);

            while let Some(id) = ancestor {
                if scopes.contains(id) {
                    break;
                }
                stack.push(id);
                ancestor = data.booleans[id.index()].parent;
            }

            while let Some(id) = stack.pop() {
                let boolean = &data.booleans[id.index()];
                let mut local = bounder.scoped_reusing(&scopes, boolean.parent, &mut scratch);
                local.assume_true(boolean.binary);
                local.tighten_rows_reusing(
                    std::iter::successors(
                        (first[id.index()] != usize::MAX).then_some(first[id.index()]),
                        |&index| (next[index] != usize::MAX).then_some(next[index]),
                    )
                    .map(|index| {
                        let row = &data.rows[index];
                        (row.lhs, row.lower, row.upper)
                    }),
                    None,
                    &mut scratch,
                );

                for &root in local.supporting_roots() {
                    plan.dependencies(bounder.arena(), root);
                }

                let bounds = local.into_context(&mut scratch);
                scopes.insert(id, boolean.parent, bounds);
            }
        }
        Some(scopes)
    }
}

fn row_m(
    row: &oximo_core::gdp::DisjunctConstraintRecord,
    body: Bounds,
    boolean: &BooleanRecord,
    options: &GdpReformulationOptions,
    methods: &FxHashMap<DisjunctionId, GdpMethod>,
    explicit: BigMValues,
) -> Result<(Option<BigMSide>, Option<BigMSide>), GdpError> {
    let local = boolean.owner.and_then(|id| methods.get(&id).copied()).and_then(method_m);
    let global = options.fallback.or_else(|| method_m(options.default_method));
    let lower_m = if row.lower.is_finite() {
        Some(side_m(
            round_up(row.lower - body.lower),
            explicit.lower,
            local,
            global,
            &row.name,
            BoundSide::Lower,
        )?)
    } else {
        None
    };

    let upper_m = if row.upper.is_finite() {
        Some(side_m(
            round_up(body.upper - row.upper),
            explicit.upper,
            local,
            global,
            &row.name,
            BoundSide::Upper,
        )?)
    } else {
        None
    };

    Ok((lower_m, upper_m))
}

fn estimate_contexts(
    contexts: &mut Vec<(BooleanId, Bounds)>,
    local: &mut Bounder<'_>,
    data: &GdpSnapshot,
    row: &oximo_core::gdp::DisjunctConstraintRecord,
    body: Bounds,
    affine: &Linear,
    has_residual: bool,
) {
    let mut ancestor = data.booleans[row.indicator.index()].parent;

    while let Some(id) = ancestor {
        contexts.push((id, body));
        ancestor = data.booleans[id.index()].parent;
    }

    contexts.reverse();

    for (parent, bounds) in contexts.iter_mut() {
        local.set_scope(*parent);

        // Global domain validation already succeeded.
        // If a tighter numerical enclosure cannot be computed,
        // keep the global M.
        let estimate = if has_residual {
            local.analyze(row.lhs).map(|analysis| analysis.bounds)
        } else {
            local.affine_box(affine.constant, &affine.terms)
        };

        if let Ok(estimate) = estimate {
            *bounds = estimate;
        }
    }
}

fn analyze_source(
    bounder: &mut Bounder<'_>,
    row: &oximo_core::gdp::DisjunctConstraintRecord,
) -> Result<ExpressionAnalysis, GdpError> {
    bounder.analyze(row.lhs).map_err(|error| match error {
        BoundError::Domain(operation) => {
            GdpError::UnsafeDomain { constraint: row.name.clone(), operation }
        }
        BoundError::InvalidNumber => GdpError::InvalidExpression(row.name.clone()),
    })
}

fn row_body(
    arena: &ExprArena,
    lhs: ExprId,
    analysis: ExpressionAnalysis,
    slots: &mut FxHashMap<VarId, usize>,
) -> Result<(Bounds, Linear, Vec<SignedExpr>), GdpError> {
    // The full source was domain-validated.
    // Extract affine addends even when other summands are nonlinear,
    // and merge repeated source variables before checking cancellation.
    let (terms, residual) =
        analysis.affine.map_or_else(|| split_linear(arena, lhs), |terms| (terms, Vec::new()));
    let mut linear = Linear { constant: terms.constant, terms: terms.coeffs.into_owned() };
    linear.normalize(slots)?;

    Ok((analysis.bounds, linear, residual))
}

impl PlanBuilder {
    pub(super) fn plan_big_m(
        &mut self,
        model: &Model,
        options: &GdpReformulationOptions,
        data: &GdpSnapshot,
        row_start: usize,
        methods: &FxHashMap<DisjunctionId, GdpMethod>,
        overrides: &FxHashMap<DisjunctConstraintId, BigMValues>,
    ) -> Result<(), GdpError> {
        if row_start == data.rows.len() {
            return Ok(());
        }

        let vars = model.variables();
        let arena = model.arena();

        for row in &data.rows[row_start..] {
            self.dependencies(&arena, row.lhs);
        }

        let mut bounder = Bounder::new(&arena, &vars);

        if options.bound_tightening {
            bounder.tighten_global(model.constraints().algebraic(), &self.locked_variables);
        }

        for &root in bounder.supporting_roots() {
            self.dependencies(&arena, root);
        }

        let scopes = parent_scopes(self, &bounder, data, options, methods, row_start);
        let mut local_bounder = scopes.as_ref().map(|scopes| bounder.scoped(scopes, None));
        let mut contexts = Vec::new();
        let mut coefficient_slots = FxHashMap::default();
        let mut indicator_slots = Vec::new();

        for (index, row) in data.rows.iter().enumerate().skip(row_start) {
            let source = DisjunctConstraintId(index.try_into().map_err(|_| GdpError::Capacity)?);
            let analysis = analyze_source(&mut bounder, row)?;
            let (body, mut body_linear, residual) =
                row_body(&arena, row.lhs, analysis, &mut coefficient_slots)?;
            let body_source = (!residual.is_empty()).then_some(residual.as_slice());

            // Cancel the source constant against the bound before applying M.
            let boolean = &data.booleans[row.indicator.index()];
            let (lower_m, upper_m) = row_m(
                row,
                body,
                boolean,
                options,
                methods,
                overrides.get(&source).copied().unwrap_or_default(),
            )?;

            contexts.clear();

            if hierarchical_method(options, methods, boolean)
                && let Some(local) = &mut local_bounder
            {
                estimate_contexts(
                    &mut contexts,
                    local,
                    data,
                    row,
                    body,
                    &body_linear,
                    !residual.is_empty(),
                );
            }

            let hierarchical_m = hierarchical_terms(
                row.indicator,
                &contexts,
                row.lower,
                row.upper,
                lower_m,
                upper_m,
            );
            let body_constant = std::mem::take(&mut body_linear.constant);
            let mut constraints = Vec::new();

            for (side, amount, bound) in [
                (BoundSide::Lower, lower_m, row.lower - body_constant),
                (BoundSide::Upper, upper_m, row.upper - body_constant),
            ] {
                let Some(amount) = amount else { continue };

                let affine = if side == BoundSide::Lower && upper_m.is_some() {
                    body_linear.clone()
                } else {
                    std::mem::take(&mut body_linear)
                };

                let (affine, bound) = if hierarchical_m.is_empty() {
                    let shift = if side == BoundSide::Lower { -amount.value } else { amount.value };
                    checked_big_m_row(affine, boolean.binary, bound, shift, &row.name, side)?
                } else {
                    checked_hierarchical_row(
                        affine,
                        bound,
                        &hierarchical_m,
                        data,
                        (&row.name, side),
                        &mut indicator_slots,
                    )?
                };

                let (lower, upper) = if side == BoundSide::Lower {
                    (bound, f64::INFINITY)
                } else {
                    (f64::NEG_INFINITY, bound)
                };

                constraints.push(self.numeric_row(model, body_source, affine, lower, upper)?);
            }

            self.report.rows.push(GdpRowArtifacts {
                source,
                constraints,
                lower_m,
                upper_m,
                hierarchical_m,
            });
        }

        Ok(())
    }

    pub(super) fn plan_disjunctions(
        &mut self,
        model: &Model,
        data: &GdpSnapshot,
        disjunction_start: usize,
    ) -> Result<(), GdpError> {
        for (i, d) in data.disjunctions.iter().enumerate().skip(disjunction_start) {
            let start = self.row_count();
            let parent = d.parent.map_or_else(
                || Linear::constant(1.0),
                |p| Linear::variable(data.booleans[p.index()].binary),
            );
            let mut sum = Linear::default();

            for b in &d.branches {
                let value = Linear::variable(data.booleans[b.index()].binary);
                sum = sum.add(value.clone());

                if d.parent.is_some() && d.kind == DisjunctionKind::AtLeastOne {
                    self.row(
                        model,
                        None,
                        value.add(parent.clone().scaled(-1.0)),
                        f64::NEG_INFINITY,
                        0.0,
                    )?;
                }
            }

            self.row(
                model,
                None,
                sum.add(parent.scaled(-1.0)),
                0.0,
                if d.kind == DisjunctionKind::ExactlyOne { 0.0 } else { f64::INFINITY },
            )?;
            let ids = self.row_ids(start)?;
            self.report.disjunctions.push(GdpDisjunctionArtifacts {
                source: DisjunctionId(u32::try_from(i).map_err(|_| GdpError::Capacity)?),
                constraints: ids,
                method: GdpMethodKind::BigM,
            });
        }

        Ok(())
    }
}

fn round_up(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 { value.next_up() } else { value }
}
