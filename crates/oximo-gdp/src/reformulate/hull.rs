//! Inside-out Hull with local child auxiliaries and scoped aggregation equations.
//!
//! - Hector D. Perez and Ignacio E. Grossmann (2024), "Extensions to generalized
//!   disjunctive programming: hierarchical structures and first-order logic",
//!   Optimization and Engineering 25, 959–998, section 3.3.
//!   <https://doi.org/10.1007/s11081-023-09831-x>
//! - Kevin C. Furman, Nicolas W. Sawaya, and Ignacio E. Grossmann (2020),
//!   "A computationally useful algebraic representation of nonlinear disjunctive
//!   convex sets using the perspective function", Computational Optimization and
//!   Applications 76, 589–614.
//!   <https://doi.org/10.1007/s10589-020-00176-0>

use super::big_m::{BigMRowAnalysis, analyze_big_m_row, checked_big_m_row, row_m};
use super::plan::{Linear, PlanBuilder};
use super::{
    BigMValues, GdpError, GdpHullDomainContext, GdpHullScope, GdpMethod, GdpReformulationOptions,
    GdpSizeLimits, Hull, HullInclusivePolicy,
};
use crate::bounds::{BoundError, Bounder, Bounds};
use oximo_core::gdp::{
    BooleanId, BoundSide, DisjunctConstraintId, DisjunctConstraintRecord, DisjunctionId,
    DisjunctionKind, GdpDisjunctionArtifacts, GdpHullActivation, GdpHullBranchSumMapping,
    GdpHullPatternMapping, GdpHullVariableMapping, GdpMethodKind, GdpRowArtifacts, GdpSnapshot,
};
use oximo_core::{
    Domain, ExprArena, ExprId, ExprNode, GeneratedConstraintId, GeneratedVarId, Model, VarId,
};
use oximo_expr::{
    EvalContext, LinearTerms, ParamId, SignedExpr, evaluate, extract_linear, split_linear,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smol_str::SmolStr;
use std::sync::Arc;

/// All private-arena appends pass through a checked node budget.
struct PlanningArena {
    inner: ExprArena,
    limit: Option<usize>,
    // Reused by row construction, DAG rewrites and mixed Big-M analysis.
    coefficient_slots: FxHashMap<VarId, usize>,
    // Normalization removes zero terms, so its slot indices cannot be used
    // for recovery. Rebuild values here while retaining the map's capacity.
    recovery_coefficients: FxHashMap<VarId, f64>,
}

impl std::ops::Deref for PlanningArena {
    type Target = ExprArena;

    fn deref(&self) -> &ExprArena {
        &self.inner
    }
}

impl PlanningArena {
    fn new(source: &ExprArena, limit: Option<usize>) -> Result<Self, GdpError> {
        if let Some(limit) = limit
            && source.len() > limit
        {
            return Err(GdpError::PlanningLimit {
                resource: "private expression nodes",
                required: source.len(),
                limit,
            });
        }

        Ok(Self {
            inner: source.__clone_with_additional_capacity(0),
            limit,
            coefficient_slots: FxHashMap::default(),
            recovery_coefficients: FxHashMap::default(),
        })
    }

    fn push(&mut self, node: ExprNode) -> Result<ExprId, GdpError> {
        let required = self.inner.len().checked_add(1).ok_or(GdpError::Capacity)?;
        u32::try_from(required).map_err(|_| GdpError::Capacity)?;

        if let Some(limit) = self.limit
            && required > limit
        {
            return Err(GdpError::PlanningLimit {
                resource: "private expression nodes",
                required,
                limit,
            });
        }

        Ok(self.inner.push(node))
    }

    fn affine_body(&mut self, mut linear: Linear) -> Result<RowBody, GdpError> {
        linear.normalize(&mut self.coefficient_slots)?;

        Ok(RowBody::normalized_affine(linear))
    }

    fn constant(&mut self, value: f64) -> Result<ExprId, GdpError> {
        self.push(ExprNode::Const(value))
    }

    fn var(&mut self, variable: VarId) -> Result<ExprId, GdpError> {
        self.push(ExprNode::Var(variable))
    }

    fn linear(&mut self, coeffs: Vec<(VarId, f64)>, constant: f64) -> Result<ExprId, GdpError> {
        self.push(ExprNode::Linear { coeffs, constant })
    }
}

pub(super) fn validate_method(method: GdpMethod) -> Result<(), GdpError> {
    if let GdpMethod::Hull(settings) = method {
        if !settings.epsilon.is_finite() || settings.epsilon <= 0.0 || settings.epsilon >= 1.0 {
            return Err(GdpError::InvalidHullEpsilon(settings.epsilon));
        }

        if matches!(settings.inclusive_policy, HullInclusivePolicy::Enumerate { max_patterns: 0 }) {
            return Err(GdpError::HullPatternLimit { disjunction: "settings".into(), limit: 0 });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Origin {
    Numeric(usize),
    Disjunction(usize),
    Bounds(usize),
}

/// Whether a coordinate is physical or local to a Boolean activation subtree.
#[derive(Clone, Copy, Debug)]
enum CoordinateScope {
    Physical,
    ActivationLocal { owner: BooleanId },
}

#[derive(Clone, Copy, Debug)]
enum RowKind {
    Source(DisjunctConstraintId),
    CopyBound,
    Selection,
    PatternExclusion,
    Aggregation { source: VarId, coordinates: CoordinateScope },
    Reformulated,
}

impl RowKind {
    fn source_constraint(self) -> Option<DisjunctConstraintId> {
        match self {
            Self::Source(source) => Some(source),
            _ => None,
        }
    }
}

/// A local row holds without another ancestor reformulation when its
/// activation is zero.
#[derive(Clone, Copy, Debug)]
enum RowEnforcement {
    Conditional,
    Homogeneous { activation: GdpHullActivation },
    Perspective { activation: GdpHullActivation },
}

impl RowEnforcement {
    fn local_activation(self) -> Option<GdpHullActivation> {
        match self {
            Self::Conditional => None,
            Self::Homogeneous { activation } | Self::Perspective { activation } => Some(activation),
        }
    }
}

#[derive(Clone, Debug)]
struct AffineBody {
    linear: Linear,
    support: Arc<[VarId]>,
}

#[derive(Clone, Debug)]
enum RowBody {
    Affine(Arc<AffineBody>),
    Expression(ExprId),
}

impl RowBody {
    /// The caller has normalized the coefficients and checked finite numbers.
    fn normalized_affine(linear: Linear) -> Self {
        let mut support: Vec<_> = linear.terms.iter().map(|&(v, _)| v).collect();
        support.sort_unstable_by_key(|v| v.index());

        Self::Affine(Arc::new(AffineBody { linear, support: support.into() }))
    }

    fn split<'a>(&'a self, arena: &'a ExprArena) -> (LinearTerms<'a>, Vec<SignedExpr>) {
        match self {
            Self::Affine(body) => {
                (LinearTerms::borrowed(&body.linear.terms, body.linear.constant), Vec::new())
            }
            Self::Expression(root) => split_linear(arena, *root),
        }
    }
}

#[derive(Clone, Debug)]
struct Row {
    body: RowBody,
    lower: f64,
    upper: f64,
    name: SmolStr,
    origins: Vec<Origin>,
    kind: RowKind,
    enforcement: RowEnforcement,
    /// Roundoff allowance from source coefficients, propagated unchanged through
    /// transformations so large reference/M coefficients cannot inflate it.
    roundoff: f64,
}

struct PreparedSource {
    body: RowBody,
    offsets: Vec<f64>,
    roundoff: f64,
}

impl PreparedSource {
    fn row(
        &self,
        source: &DisjunctConstraintRecord,
        index: usize,
        start: usize,
    ) -> Result<Row, GdpError> {
        let (lower, upper) = if self.offsets.is_empty() {
            (source.lower, source.upper)
        } else {
            (
                substitute_bound(
                    source.lower,
                    &self.offsets,
                    self.roundoff,
                    (&source.name, BoundSide::Lower),
                )?,
                substitute_bound(
                    source.upper,
                    &self.offsets,
                    self.roundoff,
                    (&source.name, BoundSide::Upper),
                )?,
            )
        };

        Ok(Row {
            body: self.body.clone(),
            lower,
            upper,
            roundoff: self.roundoff,
            name: source.name.clone(),
            origins: vec![Origin::Numeric(index - start)],
            kind: RowKind::Source(DisjunctConstraintId(u32::try_from(index).expect("row ID"))),
            enforcement: RowEnforcement::Conditional,
        })
    }
}

/// Coordinates used by each branch and the branches using each coordinate.
struct HullSupport {
    coordinates: Vec<VarId>,
    branches: Vec<Vec<VarId>>,
    using: FxHashMap<VarId, Vec<usize>>,
}

impl HullSupport {
    fn copy_count(&self, combined: &FxHashMap<VarId, Linear>) -> Result<usize, GdpError> {
        self.coordinates.iter().try_fold(0_usize, |count, variable| {
            let copies = if combined.contains_key(variable) {
                self.using[variable].len() + 1
            } else {
                self.branches.len()
            };

            count.checked_add(copies).ok_or(GdpError::Capacity)
        })
    }
}

struct Substitution {
    activation: GdpHullActivation,
    scope: GdpHullScope,
    guard: Linear,
    denominator: Option<ExprId>,
    copies: FxHashMap<VarId, VarId>,
    nonlinear_map: FxHashMap<VarId, ExprId>,
    nonlinear_cache: FxHashMap<ExprId, ExprId>,
    perspectives: FxHashMap<ExprId, Perspective>,
    references: Arc<FxHashMap<VarId, f64>>,
}

#[derive(Clone)]
struct Perspective {
    body: ExprId,
    correction: Linear,
}

struct ReferenceContext<'a>(&'a FxHashMap<VarId, f64>);

impl EvalContext for ReferenceContext<'_> {
    fn var(&self, v: VarId) -> Option<f64> {
        Some(self.0.get(&v).copied().unwrap_or(0.0))
    }

    fn param(&self, _: ParamId) -> Option<f64> {
        None
    }
}

struct Compiler<'a> {
    model: &'a Model,
    options: &'a GdpReformulationOptions,
    data: &'a GdpSnapshot,
    plan: &'a mut PlanBuilder,
    arena: PlanningArena,
    boxes: Vec<Bounds>,
    references: FxHashMap<VarId, f64>,
    binary_booleans: FxHashMap<VarId, BooleanId>,
    local_origins: FxHashMap<VarId, BooleanId>,
    provenance: oximo_core::ReformulationProvenance,
    enter: Vec<usize>,
    leave: Vec<usize>,
    enumeration_above: Vec<bool>,
    methods: &'a FxHashMap<DisjunctionId, GdpMethod>,
    overrides: &'a FxHashMap<DisjunctConstraintId, BigMValues>,
    row_reports: Vec<GdpRowArtifacts>,
    disjunction_reports: Vec<GdpDisjunctionArtifacts>,
    row_start: usize,
    disjunction_start: usize,
    domain_checks: Vec<(ExprId, SmolStr, GdpHullScope)>,
    pure_big_m: FxHashSet<BooleanId>,
    supports: FxHashMap<ExprId, Arc<[VarId]>>,
    prepared_sources: FxHashMap<(BooleanId, ExprId), PreparedSource>,
    activation_nodes: FxHashMap<GdpHullActivation, ExprId>,
    branch_sum_weights: FxHashMap<VarId, DisjunctionId>,
    has_children: Vec<bool>,
}

impl PlanBuilder {
    #[expect(clippy::too_many_arguments, reason = "shared atomic planning inputs")]
    pub(super) fn plan_hull(
        &mut self,
        model: &Model,
        options: &GdpReformulationOptions,
        data: &GdpSnapshot,
        row_start: usize,
        disjunction_start: usize,
        methods: &FxHashMap<DisjunctionId, GdpMethod>,
        overrides: &FxHashMap<DisjunctConstraintId, BigMValues>,
    ) -> Result<(), GdpError> {
        if row_start == data.rows.len() && disjunction_start == data.disjunctions.len() {
            return Ok(());
        }

        let source_arena = model.arena();
        let pure_big_m = big_m_components(data, options, methods);

        if !pure_big_m.is_empty() {
            self.plan_big_m(
                model,
                options,
                data,
                row_start,
                methods,
                overrides,
                Some(&pure_big_m),
            )?;
        }

        let boxes = effective_boxes(model, self, options, data, row_start)?;
        let references = options
            .reference_points
            .iter()
            .map(|(&(.., v), &a)| (v.expect("validated reference handle"), a))
            .collect();
        let mut compiler = Compiler {
            model,
            options,
            data,
            plan: self,
            // One private copy for the entire transformation, we never do one copy per scope.
            arena: PlanningArena::new(&source_arena, options.planning_limits.expression_nodes)?,
            boxes,
            references,
            binary_booleans: data
                .booleans
                .iter()
                .enumerate()
                .map(|(i, b)| (b.binary, BooleanId(u32::try_from(i).expect("Boolean ID"))))
                .collect(),
            local_origins: FxHashMap::default(),
            provenance: oximo_core::ReformulationProvenance::capture(model),
            enter: vec![0; data.booleans.len()],
            leave: vec![0; data.booleans.len()],
            enumeration_above: vec![false; data.booleans.len()],
            methods,
            overrides,
            row_start,
            disjunction_start,
            domain_checks: Vec::new(),
            pure_big_m,
            supports: FxHashMap::default(),
            prepared_sources: FxHashMap::default(),
            activation_nodes: FxHashMap::default(),
            branch_sum_weights: FxHashMap::default(),
            has_children: vec![false; data.booleans.len()],
            row_reports: data
                .rows
                .iter()
                .enumerate()
                .skip(row_start)
                .map(|(i, _)| GdpRowArtifacts {
                    source: DisjunctConstraintId(u32::try_from(i).expect("row ID")),
                    constraints: Vec::new(),
                    lower_m: None,
                    upper_m: None,
                    hierarchical_m: Vec::new(),
                })
                .collect(),
            disjunction_reports: data
                .disjunctions
                .iter()
                .enumerate()
                .skip(disjunction_start)
                .map(|(i, _)| GdpDisjunctionArtifacts {
                    source: DisjunctionId(u32::try_from(i).expect("disjunction ID")),
                    constraints: Vec::new(),
                    method: GdpMethodKind::Hull,
                })
                .collect(),
        };
        compiler.compile()?;

        compiler.plan.report.planning.private_expression_nodes = compiler.arena.len();
        let mut nodes = compiler.arena.inner.__into_nodes();
        nodes.drain(..source_arena.len());
        compiler.plan.generated_nodes = nodes;

        compiler.plan.report.rows.extend(
            compiler
                .row_reports
                .into_iter()
                .filter(|r| !compiler.pure_big_m.contains(&data.rows[r.source.index()].indicator)),
        );
        compiler.plan.report.rows.sort_unstable_by_key(|r| r.source.index());

        compiler.plan.report.disjunctions.extend(compiler.disjunction_reports);

        Ok(())
    }
}

fn effective_boxes(
    model: &Model,
    plan: &mut PlanBuilder,
    options: &GdpReformulationOptions,
    data: &GdpSnapshot,
    row_start: usize,
) -> Result<Vec<Bounds>, GdpError> {
    let arena = model.arena();

    for row in &data.rows[row_start..] {
        plan.dependencies(&arena, row.lhs);
    }

    let source = model.resolved_variables();
    let mut bounder = Bounder::new(&arena, &source);
    if options.bound_tightening {
        bounder.tighten_global(model.constraints().algebraic(), &plan.locked_variables);
    }

    for &root in bounder.supporting_roots() {
        plan.dependencies(&arena, root);
    }
    plan.locked_constraints.extend(bounder.supporting_constraints());

    source
        .iter()
        .map(|v| bounder.variable(v.id).map_err(|_| GdpError::InvalidExpression(v.name.clone())))
        .collect()
}

impl Compiler<'_> {
    fn method(&self, index: usize) -> GdpMethod {
        self.methods
            .get(&DisjunctionId(u32::try_from(index).expect("disjunction ID")))
            .copied()
            .unwrap_or(self.options.default_method)
    }

    fn patterns(&self, index: usize, settings: Hull) -> Result<usize, GdpError> {
        let d = &self.data.disjunctions[index];
        let HullInclusivePolicy::Enumerate { max_patterns } = settings.inclusive_policy else {
            return Err(GdpError::InclusiveHull(d.name.clone()));
        };
        let count = u32::try_from(d.branches.len())
            .ok()
            .and_then(|n| 1_usize.checked_shl(n))
            .and_then(|n| n.checked_sub(1));
        count.filter(|&n| n <= max_patterns).ok_or_else(|| GdpError::HullPatternLimit {
            disjunction: d.name.clone(),
            limit: max_patterns,
        })
    }

    fn compile(&mut self) -> Result<(), GdpError> {
        let mut blocks = self.source_blocks()?;
        let mut order: Vec<_> = (self.disjunction_start..self.data.disjunctions.len()).collect();
        order.sort_unstable_by_key(|&i| {
            self.data.disjunctions[i]
                .branches
                .iter()
                .map(|b| self.leave[b.index()])
                .max()
                .unwrap_or(0)
        });
        for i in order {
            let d = &self.data.disjunctions[i];
            let parent = d.parent;
            let branches = d.branches.clone();
            let method = self.method(i);
            let pattern_start = self.plan.report.hull_patterns.len();

            let result = match method {
                GdpMethod::Hull(h) if d.kind == DisjunctionKind::AtLeastOne => {
                    self.enumerate(i, &branches, &mut blocks, h)?
                }
                GdpMethod::Hull(h) => {
                    let inputs = branches
                        .iter()
                        .map(|&b| {
                            (
                                Some(b),
                                Linear::variable(self.data.booleans[b.index()].binary),
                                std::mem::take(&mut blocks[b.index()]),
                            )
                        })
                        .collect();
                    self.hull(Some(i), inputs, h, None)?
                }
                GdpMethod::BigM(_) => {
                    let mut result = Vec::new();
                    for &b in &branches {
                        for row in std::mem::take(&mut blocks[b.index()]) {
                            result.extend(self.big_m(row, b)?);
                        }
                    }
                    result
                }
            };

            self.deliver(result, parent, &mut blocks)?;
            self.scoped_selection(i, parent, pattern_start, &mut blocks)?;
        }

        // Unowned Booleans represent standalone conditional blocks.
        for (i, b) in self.data.booleans.iter().enumerate() {
            if b.owner.is_some() || blocks[i].is_empty() {
                continue;
            }

            let boolean = BooleanId(u32::try_from(i).expect("Boolean ID"));
            let rows = std::mem::take(&mut blocks[i]);
            let result = match self.options.default_method {
                GdpMethod::BigM(_) => {
                    let mut out = Vec::new();
                    for row in rows {
                        out.extend(self.big_m(row, boolean)?);
                    }
                    out
                }
                GdpMethod::Hull(h) => {
                    let guard = Linear::variable(b.binary);
                    self.hull(
                        None,
                        vec![(Some(boolean), guard.clone(), rows), (None, guard.not(), Vec::new())],
                        h,
                        None,
                    )?
                }
            };

            self.deliver(result, None, &mut blocks)?;
        }

        let mut bounder = Bounder::from_boxes(&self.arena, &self.boxes);

        for (root, name, scope) in &self.domain_checks {
            bounder.analyze(*root).map_err(|e| match e {
                BoundError::Domain { operation, arguments } => GdpError::UnsafeHullDomain {
                    constraint: name.clone(),
                    operation,
                    arguments,
                    scope: *scope,
                    scope_name: self.scope_name(*scope),
                    context: GdpHullDomainContext::GeneratedBox,
                },
                BoundError::InvalidNumber => GdpError::InvalidExpression(name.clone()),
            })?;
        }

        Ok(())
    }

    /// An enumerating ancestor copies descendant selectors and pattern weights.
    /// Their selection equations must hold within each ancestor pattern, as well
    /// as on the recombined original variables. In exclusive trees they stay local
    /// and the ordinary global selection equations already provide these links.
    fn scoped_selection(
        &mut self,
        index: usize,
        parent: Option<BooleanId>,
        pattern_start: usize,
        blocks: &mut [Vec<Row>],
    ) -> Result<(), GdpError> {
        let Some(parent) = parent.filter(|p| self.enumeration_above[p.index()]) else {
            return Ok(());
        };
        let d = &self.data.disjunctions[index];
        let sum = d.branches.iter().fold(Linear::constant(-1.0), |sum, b| {
            sum.add(Linear::variable(self.data.booleans[b.index()].binary))
        });
        let mut equations = vec![(sum, d.kind == DisjunctionKind::ExactlyOne)];
        let patterns = &self.plan.report.hull_patterns[pattern_start..];
        if !patterns.is_empty() {
            let sum = patterns.iter().fold(Linear::constant(-1.0), |sum, pattern| {
                sum.add(Linear::variable(pattern.weight.raw_id()))
            });
            equations.push((sum, true));

            for &b in &d.branches {
                let sum = patterns.iter().filter(|pattern| pattern.selected.contains(&b)).fold(
                    Linear::variable(self.data.booleans[b.index()].binary).scaled(-1.0),
                    |sum, pattern| sum.add(Linear::variable(pattern.weight.raw_id())),
                );
                equations.push((sum, true));
            }
        }

        for (body, equality) in equations {
            self.plan.record_row_expansions(1)?;
            blocks[parent.index()].push(Row {
                body: self.arena.affine_body(body)?,
                lower: 0.0,
                upper: if equality { 0.0 } else { f64::INFINITY },
                name: "Hull scoped selection".into(),
                origins: vec![Origin::Disjunction(index - self.disjunction_start)],
                kind: RowKind::Selection,
                enforcement: RowEnforcement::Conditional,
                roundoff: 32.0 * f64::EPSILON,
            });
        }

        Ok(())
    }

    fn source_blocks(&mut self) -> Result<Vec<Vec<Row>>, GdpError> {
        // Check exponential sizes before allocating any pattern data.
        for i in self.disjunction_start..self.data.disjunctions.len() {
            if let GdpMethod::Hull(h) = self.method(i)
                && self.data.disjunctions[i].kind == DisjunctionKind::AtLeastOne
            {
                self.patterns(i, h)?;
            }
        }

        let mut children = vec![Vec::new(); self.data.booleans.len()];

        for (i, b) in self.data.booleans.iter().enumerate() {
            if let Some(p) = b.parent {
                children[p.index()].push(i);
            }
        }

        for (i, children) in children.iter().enumerate() {
            self.has_children[i] = !children.is_empty();
        }

        let mut clock = 0;
        let mut stack: Vec<_> = self
            .data
            .booleans
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, b)| b.parent.is_none())
            .map(|(i, _)| (i, false, false))
            .collect();

        while let Some((i, visited, inherited)) = stack.pop() {
            if visited {
                self.leave[i] = clock;
                clock += 1;
                continue;
            }

            let enumerating = self.data.booleans[i].owner.is_some_and(|d| {
                self.data.disjunctions[d.index()].kind == DisjunctionKind::AtLeastOne
                    && matches!(self.method(d.index()), GdpMethod::Hull(_))
            });
            self.enumeration_above[i] = inherited || enumerating;
            self.enter[i] = clock;
            clock += 1;
            stack.push((i, true, inherited));
            stack.extend(children[i].iter().rev().map(|&c| (c, false, inherited || enumerating)));
        }

        let mut blocks: Vec<Vec<Row>> = vec![Vec::new(); self.data.booleans.len()];
        let mut source_caches: FxHashMap<BooleanId, FxHashMap<ExprId, ExprId>> =
            FxHashMap::default();
        for i in self.row_start..self.data.rows.len() {
            let source = &self.data.rows[i];
            if self.pure_big_m.contains(&source.indicator) {
                continue;
            }

            self.plan.record_row_expansions(1)?;
            let cache = source_caches.entry(source.indicator).or_default();
            let row = self.source_row(i, cache)?;
            blocks[source.indicator.index()].push(row);
        }

        Ok(blocks)
    }

    fn selector_is_active(&self, variable: VarId, branch: BooleanId) -> bool {
        self.binary_booleans.get(&variable).is_some_and(|b| {
            self.enter[b.index()] <= self.enter[branch.index()]
                && self.leave[branch.index()] <= self.leave[b.index()]
        })
    }

    fn source_row(
        &mut self,
        index: usize,
        cache: &mut FxHashMap<ExprId, ExprId>,
    ) -> Result<Row, GdpError> {
        let source = &self.data.rows[index];
        let key = (source.indicator, source.lhs);
        if let Some(prepared) = self.prepared_sources.get(&key) {
            return prepared.row(source, index, self.row_start);
        }

        // Split before selector substitution: rewriting the whole affine body
        // first can round away a small constant next to a large selector term.
        let (terms, residual) = split_linear(&self.arena, source.lhs);
        let nonlinear_source = !residual.is_empty();
        let constant = terms.constant;
        let coefficients = terms.coeffs.into_owned();
        let mut affine = Linear::default();
        let mut offsets = vec![constant];

        for (variable, coefficient) in coefficients {
            if self.selector_is_active(variable, source.indicator) {
                offsets.push(coefficient);
            } else {
                affine.terms.push((variable, coefficient));
            }
        }

        let residual =
            self.source_residuals(source.indicator, residual, &mut affine, &mut offsets, cache)?;
        affine.normalize(&mut self.arena.coefficient_slots)?;
        let roundoff = coefficient_roundoff(
            affine.terms.iter().map(|&(_, c)| c).chain(residual.iter().map(|_| 1.0)),
        );
        if offsets.len() == 1 {
            affine.constant = constant;
            offsets.clear();
        }

        let body = if residual.is_empty() {
            RowBody::normalized_affine(affine)
        } else {
            RowBody::Expression(residual_body(&mut self.arena, affine, &residual)?)
        };
        let prepared = PreparedSource { body, offsets, roundoff };
        let row = prepared.row(source, index, self.row_start)?;
        if nonlinear_source {
            self.prepared_sources.insert(key, prepared);
        }

        Ok(row)
    }

    fn source_residuals(
        &mut self,
        branch: BooleanId,
        residual: Vec<SignedExpr>,
        affine: &mut Linear,
        offsets: &mut Vec<f64>,
        cache: &mut FxHashMap<ExprId, ExprId>,
    ) -> Result<Vec<SignedExpr>, GdpError> {
        let mut map = FxHashMap::default();

        for term in &residual {
            for &v in self.expression_support(term.id).iter() {
                if self.selector_is_active(v, branch) && !map.contains_key(&v) {
                    map.insert(v, self.arena.constant(1.0)?);
                }
            }
        }

        let mut nonlinear = Vec::new();
        for mut term in residual {
            term.id = rewrite(&mut self.arena, term.id, &map, cache)?;
            if let Some(terms) = extract_linear(&self.arena, term.id) {
                // Selector substitution can turn a nonlinear term into an
                // affine one.
                let sign = if term.neg { -1.0 } else { 1.0 };

                if terms.constant != 0.0 {
                    offsets.push(sign * terms.constant);
                }

                affine.terms.extend(terms.coeffs.iter().map(|&(v, c)| (v, sign * c)));
            } else {
                nonlinear.push(term);
            }
        }

        Ok(nonlinear)
    }

    fn local_owner(&self, variable: VarId) -> Option<BooleanId> {
        self.local_origins.get(&variable).or_else(|| self.binary_booleans.get(&variable)).copied()
    }

    fn coordinate_scope(
        &self,
        variable: VarId,
        locals: Option<&FxHashSet<VarId>>,
    ) -> CoordinateScope {
        if locals.is_some_and(|locals| locals.contains(&variable)) {
            CoordinateScope::ActivationLocal {
                owner: self.local_owner(variable).expect("local coordinate owner"),
            }
        } else {
            CoordinateScope::Physical
        }
    }

    fn row_is_local_to(&self, row: &Row, branch: Option<BooleanId>) -> bool {
        match row.enforcement.local_activation() {
            Some(GdpHullActivation::Indicator { binary, .. }) => self.is_local(binary, branch),
            Some(GdpHullActivation::Pattern { weight, .. }) => {
                self.is_local(weight.raw_id(), branch)
            }
            // A complementary on/off branch belongs to a standalone block.
            _ => false,
        }
    }

    fn is_local(&self, variable: VarId, branch: Option<BooleanId>) -> bool {
        let Some(branch) = branch else {
            return false;
        };
        self.local_owner(variable).is_some_and(|b| {
            self.enter[branch.index()] <= self.enter[b.index()]
                && self.leave[b.index()] <= self.leave[branch.index()]
        })
    }

    fn deliver(
        &mut self,
        rows: Vec<Row>,
        parent: Option<BooleanId>,
        blocks: &mut [Vec<Row>],
    ) -> Result<(), GdpError> {
        for row in rows {
            match parent {
                Some(parent)
                    if !self.row_is_local_to(&row, Some(parent))
                        || self.enumeration_above[parent.index()] =>
                {
                    blocks[parent.index()].push(row);
                }
                _ => self.emit(row)?,
            }
        }

        Ok(())
    }

    fn expression_support(&mut self, root: ExprId) -> Arc<[VarId]> {
        self.supports.entry(root).or_insert_with(|| variables(&self.arena, root).into()).clone()
    }

    fn row_support(&mut self, row: &Row) -> Arc<[VarId]> {
        match &row.body {
            RowBody::Affine(body) => Arc::clone(&body.support),
            RowBody::Expression(root) => self.expression_support(*root),
        }
    }

    fn activation_node(
        &mut self,
        kind: GdpHullActivation,
        guard: &Linear,
    ) -> Result<ExprId, GdpError> {
        if let Some(&node) = self.activation_nodes.get(&kind) {
            return Ok(node);
        }

        let node = linear(&mut self.arena, guard.clone())?;

        self.activation_nodes.insert(kind, node);
        Ok(node)
    }

    fn emit(&mut self, row: Row) -> Result<(), GdpError> {
        let id = match row.body {
            RowBody::Affine(body) => self.plan.numeric_row(
                self.model,
                None,
                Arc::unwrap_or_clone(body).linear,
                row.lower,
                row.upper,
            )?,
            RowBody::Expression(root) => self.plan.row(
                self.model,
                Some(&[SignedExpr { id: root, neg: false }]),
                Linear::default(),
                row.lower,
                row.upper,
            )?,
        };

        let generated = GeneratedConstraintId::from_raw(id);
        for origin in row.origins {
            match origin {
                Origin::Numeric(i) => self.row_reports[i].constraints.push(generated),
                Origin::Disjunction(i) => self.disjunction_reports[i].constraints.push(generated),
                Origin::Bounds(i) => self.plan.report.hull_variables[i].constraints.push(generated),
            }
        }

        Ok(())
    }

    fn auxiliary(
        &mut self,
        lower: f64,
        upper: f64,
        local: Option<BooleanId>,
    ) -> Result<VarId, GdpError> {
        let id = self.plan.variable(self.model, Domain::Real, lower, upper)?;
        debug_assert_eq!(id.index(), self.boxes.len());
        self.boxes.push(Bounds { lower, upper });

        if let Some(b) = local {
            self.local_origins.insert(id, b);
        }
        self.plan.locked_variables.insert(id);

        Ok(id)
    }

    fn copy(
        &mut self,
        source: VarId,
        disjunction: Option<usize>,
        branch: Option<BooleanId>,
        guard: &Linear,
        reference: f64,
        rows: &mut Vec<Row>,
    ) -> Result<VarId, GdpError> {
        self.plan.check_hull_copies(1)?;
        self.plan.record_row_expansions(2)?;

        let interval = self.boxes[source.index()];
        if !interval.lower.is_finite() || !interval.upper.is_finite() {
            return Err(GdpError::MissingHullBounds { variable: self.variable_name(source) });
        }

        let bounds = [interval.lower, interval.upper];
        let lower = bounds[0] - reference;
        let upper = bounds[1] - reference;

        if reference != 0.0 {
            for (bound, shifted) in [(bounds[0], lower), (bounds[1], upper)] {
                if shifted.is_finite() && ((shifted + reference) - bound).abs() > 1e-9 * bound.abs()
                {
                    return Err(GdpError::HullReferencePrecisionLoss {
                        variable: self.variable_name(source),
                        reference,
                    });
                }
            }
        }

        let copy = self.auxiliary(lower.min(0.0), upper.max(0.0), branch)?;

        let activation_kind = self.activation_kind(disjunction, guard);
        let activation = self.activation_node(activation_kind, guard)?;
        let indicator = match activation_kind {
            GdpHullActivation::Indicator { indicator, .. } => Some(indicator),
            _ => None,
        };

        let mapping = self.plan.report.hull_variables.len();
        self.plan.report.hull_variables.push(GdpHullVariableMapping {
            source: self.provenance.variable_origin(source),
            generated: GeneratedVarId::from_raw(copy),
            disjunction: disjunction
                .map(|i| DisjunctionId(u32::try_from(i).expect("disjunction ID"))),
            indicator,
            activation,
            activation_kind,
            reference,
            bounds,
            constraints: Vec::new(),
        });

        let mut origins = vec![Origin::Bounds(mapping)];
        if let Some(i) = disjunction {
            origins.push(Origin::Disjunction(i - self.disjunction_start));
        }

        for (coefficient, lo, hi) in [(lower, 0.0, f64::INFINITY), (upper, f64::NEG_INFINITY, 0.0)]
        {
            let body = self
                .arena
                .affine_body(Linear::variable(copy).add(guard.clone().scaled(-coefficient)))?;
            rows.push(Row {
                body,
                lower: lo,
                upper: hi,
                name: "Hull variable bound".into(),
                origins: origins.clone(),
                kind: RowKind::CopyBound,
                enforcement: RowEnforcement::Homogeneous { activation: activation_kind },
                roundoff: 32.0 * f64::EPSILON,
            });
        }

        Ok(copy)
    }

    fn variable_name(&self, variable: VarId) -> SmolStr {
        self.model
            .variables()
            .get(variable.index())
            .map_or_else(|| format!("Hull copy {}", variable.index()).into(), |v| v.name.clone())
    }

    fn scope_name(&self, scope: GdpHullScope) -> SmolStr {
        match scope {
            GdpHullScope::Disjunction(id) => self.data.disjunctions[id.index()].name.clone(),
            GdpHullScope::Standalone(id) => self.data.booleans[id.index()].name.clone(),
        }
    }

    fn activation_kind(&self, disjunction: Option<usize>, guard: &Linear) -> GdpHullActivation {
        let (variable, coefficient) = guard.terms[0];
        if let Some(&indicator) = self.binary_booleans.get(&variable) {
            if coefficient < 0.0 {
                GdpHullActivation::Complement { indicator, binary: variable }
            } else {
                GdpHullActivation::Indicator { indicator, binary: variable }
            }
        } else if let Some(&disjunction) = self.branch_sum_weights.get(&variable) {
            GdpHullActivation::BranchSum { disjunction, weight: GeneratedVarId::from_raw(variable) }
        } else {
            GdpHullActivation::Pattern {
                disjunction: DisjunctionId(
                    u32::try_from(disjunction.expect("pattern scope")).expect("disjunction ID"),
                ),
                weight: GeneratedVarId::from_raw(variable),
            }
        }
    }

    /// Identical source boxes can be combined only outside nested scopes.
    fn can_combine_unused(&self, disjunction: Option<usize>) -> bool {
        disjunction.is_some_and(|i| {
            let group = &self.data.disjunctions[i];
            group.parent.is_none()
                && group.kind == DisjunctionKind::ExactlyOne
                && group.branches.iter().all(|b| !self.has_children[b.index()])
        })
    }

    fn unused_branch_guards(
        &mut self,
        disjunction: Option<usize>,
        inputs: &[(Option<BooleanId>, Linear, Vec<Row>)],
        support: &HullSupport,
    ) -> Result<FxHashMap<VarId, Linear>, GdpError> {
        let mut unused_guards = FxHashMap::default();
        let mut shared_activations: FxHashMap<Vec<BooleanId>, Linear> = FxHashMap::default();

        if self.can_combine_unused(disjunction) && inputs.len() >= 3 {
            for &variable in &support.coordinates {
                let using = &support.using[&variable];
                if inputs.len() - using.len() < 2 {
                    continue;
                }

                let used: Vec<_> =
                    using.iter().map(|&i| inputs[i].0.expect("flat branch")).collect();
                let guard = if let Some(guard) = shared_activations.get(&used) {
                    guard.clone()
                } else {
                    let guard = if used.len() == 1 {
                        Linear::variable(self.data.booleans[used[0].index()].binary).not()
                    } else {
                        let weight = self.auxiliary(0.0, 1.0, None)?;
                        let source = DisjunctionId(
                            u32::try_from(disjunction.expect("flat group"))
                                .expect("disjunction ID"),
                        );

                        self.branch_sum_weights.insert(weight, source);
                        let equation = used.iter().fold(
                            Linear::variable(weight).add(Linear::constant(-1.0)),
                            |row, b| {
                                row.add(Linear::variable(self.data.booleans[b.index()].binary))
                            },
                        );

                        self.plan.record_row_expansions(1)?;
                        let body = self.arena.affine_body(equation)?;
                        self.emit(Row {
                            body,
                            lower: 0.0,
                            upper: 0.0,
                            name: "Hull unused branch activation".into(),
                            origins: vec![Origin::Disjunction(
                                source.index() - self.disjunction_start,
                            )],
                            kind: RowKind::Selection,
                            enforcement: RowEnforcement::Conditional,
                            roundoff: 32.0 * f64::EPSILON,
                        })?;

                        self.plan.report.hull_branch_sums.push(GdpHullBranchSumMapping {
                            source,
                            excluded: used.clone(),
                            weight: GeneratedVarId::from_raw(weight),
                        });

                        Linear::variable(weight)
                    };

                    shared_activations.insert(used, guard.clone());
                    guard
                };

                unused_guards.insert(variable, guard);
            }
        }

        Ok(unused_guards)
    }

    fn branch_supports(
        &mut self,
        inputs: &[(Option<BooleanId>, Linear, Vec<Row>)],
        disaggregate_local: bool,
    ) -> HullSupport {
        let mut using: FxHashMap<VarId, Vec<usize>> = FxHashMap::default();
        let mut branches = Vec::with_capacity(inputs.len());

        for (index, (branch, _, rows)) in inputs.iter().enumerate() {
            let mut used = FxHashSet::default();

            for row in rows {
                if self.row_is_local_to(row, *branch) && !disaggregate_local {
                    continue;
                }

                used.extend(
                    self.row_support(row)
                        .iter()
                        .copied()
                        .filter(|&v| disaggregate_local || !self.is_local(v, *branch)),
                );
            }

            let mut used: Vec<_> = used.into_iter().collect();
            used.sort_unstable_by_key(|v| v.index());

            for &variable in &used {
                using.entry(variable).or_default().push(index);
            }
            branches.push(used);
        }

        let mut coordinates: Vec<_> = using.keys().copied().collect();
        coordinates.sort_unstable_by_key(|v| v.index());

        HullSupport { coordinates, branches, using }
    }

    /// Copies and bounds first, then perspectives.
    fn hull(
        &mut self,
        disjunction: Option<usize>,
        inputs: Vec<(Option<BooleanId>, Linear, Vec<Row>)>,
        settings: Hull,
        local_coordinates: Option<&FxHashSet<VarId>>,
    ) -> Result<Vec<Row>, GdpError> {
        let disaggregate_local = local_coordinates.is_some();
        let support = self.branch_supports(&inputs, disaggregate_local);
        let unused_guards = if disaggregate_local {
            FxHashMap::default()
        } else {
            self.unused_branch_guards(disjunction, &inputs, &support)?
        };

        // Local selectors and child auxiliaries vanish with the enclosing group.
        // Keep their coordinates unshifted so unconditional recombination also
        // vanishes.
        let references: FxHashMap<_, _> = support
            .coordinates
            .iter()
            .filter(|&&v| {
                matches!(self.coordinate_scope(v, local_coordinates), CoordinateScope::Physical)
            })
            .filter_map(|v| self.references.get(v).map(|&a| (*v, a)))
            .collect();
        let references = Arc::new(references);
        self.plan.check_hull_copies(support.copy_count(&unused_guards)?)?;

        let mut output = Vec::new();
        let mut aggregates: FxHashMap<VarId, Linear> =
            support.coordinates.iter().map(|&v| (v, Linear::default())).collect();
        let full_support: Vec<_> = support
            .coordinates
            .iter()
            .copied()
            .filter(|v| !unused_guards.contains_key(v))
            .collect();

        for ((branch, guard, rows), mut used) in inputs.into_iter().zip(support.branches) {
            used.retain(|v| unused_guards.contains_key(v));
            used.extend_from_slice(&full_support);
            used.sort_unstable_by_key(|v| v.index());

            let mut copies = FxHashMap::default();
            for v in used {
                let reference = references.get(&v).copied().unwrap_or(0.0);
                let copy = self.copy(v, disjunction, branch, &guard, reference, &mut output)?;
                aggregates.get_mut(&v).expect("support").terms.push((copy, 1.0));
                copies.insert(v, copy);
            }

            let scope = disjunction.map_or_else(
                || GdpHullScope::Standalone(self.binary_booleans[&guard.terms[0].0]),
                |i| {
                    GdpHullScope::Disjunction(DisjunctionId(
                        u32::try_from(i).expect("disjunction ID"),
                    ))
                },
            );
            let mut substitution = Substitution {
                activation: self.activation_kind(disjunction, &guard),
                scope,
                guard,
                denominator: None,
                copies,
                nonlinear_map: FxHashMap::default(),
                nonlinear_cache: FxHashMap::default(),
                perspectives: FxHashMap::default(),
                references: Arc::clone(&references),
            };

            for row in rows {
                if self.row_is_local_to(&row, branch) && !disaggregate_local {
                    output.push(row);
                    continue;
                }

                output.extend(self.hull_row(row, settings, &mut substitution)?);
            }
        }

        for &v in &support.coordinates {
            let Some(guard) = unused_guards.get(&v) else { continue };

            let reference = references.get(&v).copied().unwrap_or(0.0);
            let copy = self.copy(v, disjunction, None, guard, reference, &mut output)?;
            aggregates.get_mut(&v).expect("support").terms.push((copy, 1.0));
        }

        for v in support.coordinates {
            self.plan.record_row_expansions(1)?;
            let a = references.get(&v).copied().unwrap_or(0.0);
            let body = self.arena.affine_body(
                aggregates
                    .remove(&v)
                    .expect("support")
                    .add(Linear::variable(v).scaled(-1.0))
                    .add(Linear::constant(a)),
            )?;

            output.push(Row {
                body,
                lower: 0.0,
                upper: 0.0,
                name: "Hull aggregation".into(),
                origins: disjunction
                    .map(|i| vec![Origin::Disjunction(i - self.disjunction_start)])
                    .unwrap_or_default(),
                kind: RowKind::Aggregation {
                    source: v,
                    coordinates: self.coordinate_scope(v, local_coordinates),
                },
                enforcement: RowEnforcement::Conditional,
                roundoff: 32.0 * f64::EPSILON,
            });
        }

        Ok(output)
    }

    fn hull_row(
        &mut self,
        row: Row,
        settings: Hull,
        substitution: &mut Substitution,
    ) -> Result<Vec<Row>, GdpError> {
        let mut output = Vec::new();
        let (terms, residual) = row.body.split(&self.arena);
        let terms = terms.into_owned();
        let mut transformed = Linear::default();
        transformed = transformed.add(substitution.guard.clone().scaled(terms.constant));

        for &(v, c) in terms.coeffs.iter() {
            if let Some(&copy) = substitution.copies.get(&v) {
                transformed.terms.push((copy, c));
                transformed = transformed.add(
                    substitution
                        .guard
                        .clone()
                        .scaled(c * substitution.references.get(&v).copied().unwrap_or(0.0)),
                );
            } else {
                transformed.terms.push((v, c));
            }
        }

        let mut parts = Vec::with_capacity(residual.len());

        for term in residual {
            let part = self.perspective(term.id, &row.name, settings, substitution)?;
            transformed = transformed.add(if term.neg {
                part.correction.scaled(-1.0)
            } else {
                part.correction
            });
            parts.push(SignedExpr { id: part.body, neg: term.neg });
        }

        for (bound, lower, upper) in sides(row.lower, row.upper) {
            self.plan.record_row_expansions(1)?;
            let affine = transformed.clone().add(substitution.guard.clone().scaled(-bound));
            let side = if lower.is_finite() { BoundSide::Lower } else { BoundSide::Upper };
            let bounded = checked_hull_body(
                &mut self.arena,
                affine,
                &parts,
                &terms,
                substitution,
                bound,
                (&row.name, side, row.roundoff),
            )?;
            if let RowBody::Expression(root) = bounded {
                self.domain_checks.push((root, row.name.clone(), substitution.scope));
            }

            output.push(Row {
                body: bounded,
                lower,
                upper,
                name: row.name.clone(),
                origins: row.origins.clone(),
                kind: RowKind::Reformulated,
                enforcement: if parts.is_empty() {
                    RowEnforcement::Homogeneous { activation: substitution.activation }
                } else {
                    RowEnforcement::Perspective { activation: substitution.activation }
                },
                roundoff: row.roundoff,
            });
        }

        Ok(output)
    }

    fn perspective(
        &mut self,
        root: ExprId,
        name: &SmolStr,
        settings: Hull,
        substitution: &mut Substitution,
    ) -> Result<Perspective, GdpError> {
        if let Some(part) = substitution.perspectives.get(&root) {
            return Ok(part.clone());
        }

        let denominator = if let Some(node) = substitution.denominator {
            node
        } else {
            let node = linear(
                &mut self.arena,
                Linear::constant(settings.epsilon)
                    .add(substitution.guard.clone().scaled(1.0 - settings.epsilon)),
            )?;
            substitution.denominator = Some(node);
            node
        };

        for &v in self.expression_support(root).iter() {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                substitution.nonlinear_map.entry(v)
            {
                let copy = substitution.copies.get(&v).copied().unwrap_or(v);
                let a = if substitution.copies.contains_key(&v) {
                    substitution.references.get(&v).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let numerator = self.arena.var(copy)?;
                let quotient = self.arena.push(ExprNode::Div(numerator, denominator))?;
                entry.insert(add_constant(&mut self.arena, quotient, a)?);
            }
        }

        let value = evaluate(&self.arena, root, &ReferenceContext(&substitution.references))
            .map_err(|_| GdpError::InvalidExpression(name.clone()))?;

        if !value.is_finite() {
            let reference_boxes: Vec<_> = self
                .boxes
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let variable = VarId(u32::try_from(i).expect("variable ID"));
                    Bounds::point(substitution.references.get(&variable).copied().unwrap_or(0.0))
                })
                .collect();
            let error = Bounder::from_boxes(&self.arena, &reference_boxes).analyze(root).err();
            return Err(match error {
                Some(BoundError::Domain { operation, arguments }) => GdpError::UnsafeHullDomain {
                    constraint: name.clone(),
                    operation,
                    arguments,
                    scope: substitution.scope,
                    scope_name: self.scope_name(substitution.scope),
                    context: GdpHullDomainContext::Reference,
                },
                _ => GdpError::InvalidExpression(name.clone()),
            });
        }

        let rewritten = rewrite(
            &mut self.arena,
            root,
            &substitution.nonlinear_map,
            &mut substitution.nonlinear_cache,
        )?;
        let product =
            self.arena.push(ExprNode::Mul([denominator, rewritten].into_iter().collect()))?;
        let correction = substitution.guard.clone().not().scaled(-settings.epsilon * value);
        let part = Perspective { body: product, correction };
        substitution.perspectives.insert(root, part.clone());

        Ok(part)
    }

    fn big_m(&mut self, row: Row, branch: BooleanId) -> Result<Vec<Row>, GdpError> {
        if self.row_is_local_to(&row, Some(branch)) {
            return Ok(vec![row]);
        }

        let BigMRowAnalysis { bounds, mut affine, residual } = match &row.body {
            RowBody::Affine(body) => super::big_m::analyze_big_m_affine(
                &Bounder::from_boxes(&self.arena, &self.boxes),
                &body.linear,
                &row.name,
            )?,
            RowBody::Expression(root) => analyze_big_m_row(
                &mut Bounder::from_boxes(&self.arena.inner, &self.boxes),
                *root,
                &row.name,
                &mut self.arena.coefficient_slots,
            )?,
        };

        let explicit = row
            .kind
            .source_constraint()
            .and_then(|id| self.overrides.get(&id).copied())
            .unwrap_or_default();
        let amounts = row_m(
            Bounds { lower: row.lower, upper: row.upper },
            bounds,
            &row.name,
            &self.data.booleans[branch.index()],
            self.options,
            self.methods,
            explicit,
        )?;

        if let Some(id) = row.kind.source_constraint() {
            let report = &mut self.row_reports[id.index() - self.row_start];
            report.lower_m = amounts[0];
            report.upper_m = amounts[1];
        }

        let constant = std::mem::take(&mut affine.constant);
        let mut output = Vec::new();
        for (amount, bound, side) in
            [(amounts[0], row.lower, BoundSide::Lower), (amounts[1], row.upper, BoundSide::Upper)]
        {
            let Some(amount) = amount else {
                continue;
            };

            self.plan.record_row_expansions(1)?;
            let shift = if side == BoundSide::Lower { -amount.value } else { amount.value };
            let (body, shifted) = checked_big_m_row(
                affine.clone(),
                self.data.booleans[branch.index()].binary,
                bound - constant,
                shift,
                &row.name,
                side,
            )?;

            let affine = body.add(Linear::constant(-shifted));
            let body = if residual.is_empty() {
                self.arena.affine_body(affine)?
            } else {
                let affine_node = linear(&mut self.arena, affine)?;
                let mut parts = vec![affine_node];
                for term in &residual {
                    parts.push(if term.neg {
                        self.arena.push(ExprNode::Unary(oximo_core::UnaryOp::Neg, term.id))?
                    } else {
                        term.id
                    });
                }
                RowBody::Expression(self.arena.push(ExprNode::Add(parts.into_iter().collect()))?)
            };

            output.push(Row {
                body,
                lower: if side == BoundSide::Lower { 0.0 } else { f64::NEG_INFINITY },
                upper: if side == BoundSide::Upper { 0.0 } else { f64::INFINITY },
                name: row.name.clone(),
                origins: row.origins.clone(),
                kind: RowKind::Reformulated,
                enforcement: RowEnforcement::Conditional,
                roundoff: row.roundoff,
            });
        }

        Ok(output)
    }

    fn enumerate(
        &mut self,
        index: usize,
        branches: &[BooleanId],
        blocks: &mut [Vec<Row>],
        settings: Hull,
    ) -> Result<Vec<Row>, GdpError> {
        let count = self.patterns(index, settings)?;
        self.plan.check_size(count, branches.len() + 1, 0)?;

        let mut inputs = Vec::with_capacity(count);
        let mut sum = Linear::default();
        let mut selected_sums = vec![Linear::default(); branches.len()];
        let mut locals = FxHashSet::default();
        let check_expansion = self.options.size_limits != GdpSizeLimits::default();
        let mut group_support = FxHashSet::default();
        let mut absent_locals = Vec::with_capacity(branches.len());

        for &branch in branches {
            let mut local = FxHashSet::default();

            for row in &blocks[branch.index()] {
                let support = self.row_support(row);

                if check_expansion {
                    group_support.extend(support.iter().copied());
                }

                local.extend(support.iter().copied().filter(|&v| self.is_local(v, Some(branch))));
            }

            locals.extend(local.iter().copied());
            let mut local: Vec<_> = local.into_iter().collect();
            local.sort_unstable_by_key(|v| v.index());
            absent_locals.push(local);
        }
        if check_expansion {
            // Every pattern copies the union support, including absent locals.
            let copies = count.checked_mul(group_support.len()).ok_or(GdpError::Capacity)?;
            self.plan.check_hull_copies(copies)?;
        }

        for mask in 1..=count {
            let selected: Vec<_> = branches
                .iter()
                .enumerate()
                .filter(|(j, _)| mask & (1 << j) != 0)
                .map(|(_, &b)| b)
                .collect();
            let weight = self.auxiliary(0.0, 1.0, selected.first().copied())?;
            let guard = Linear::variable(weight);
            sum = sum.add(guard.clone());

            let mut rows = Vec::new();
            for (j, &b) in branches.iter().enumerate() {
                if mask & (1 << j) != 0 {
                    selected_sums[j] = selected_sums[j].clone().add(guard.clone());
                    self.plan.record_row_expansions(blocks[b.index()].len())?;
                    rows.extend(blocks[b.index()].iter().cloned());
                } else {
                    // Descendant selectors and child copies must be zero in a
                    // pattern where their branch is not selected.
                    self.plan.record_row_expansions(absent_locals[j].len())?;
                    for &v in &absent_locals[j] {
                        rows.push(Row {
                            body: self.arena.affine_body(Linear::variable(v))?,
                            lower: 0.0,
                            upper: 0.0,
                            name: "inactive pattern local".into(),
                            origins: Vec::new(),
                            kind: RowKind::PatternExclusion,
                            enforcement: RowEnforcement::Conditional,
                            roundoff: 32.0 * f64::EPSILON,
                        });
                    }
                }
            }
            self.plan.report.hull_patterns.push(GdpHullPatternMapping {
                source: DisjunctionId(u32::try_from(index).expect("disjunction ID")),
                selected,
                weight: GeneratedVarId::from_raw(weight),
            });

            inputs.push((
                selected_first(self.plan.report.hull_patterns.last().expect("pattern")),
                guard,
                rows,
            ));
        }

        for &b in branches {
            blocks[b.index()].clear();
        }

        let parent = self.data.disjunctions[index].parent;
        let parent_guard = parent.map_or_else(
            || Linear::constant(1.0),
            |b| Linear::variable(self.data.booleans[b.index()].binary),
        );
        let origin = Origin::Disjunction(index - self.disjunction_start);
        for expression in std::iter::once(sum.add(parent_guard.scaled(-1.0))).chain(
            selected_sums.into_iter().zip(branches).map(|(s, b)| {
                s.add(Linear::variable(self.data.booleans[b.index()].binary).scaled(-1.0))
            }),
        ) {
            self.plan.record_row_expansions(1)?;
            let body = self.arena.affine_body(expression)?;
            self.emit(Row {
                body,
                lower: 0.0,
                upper: 0.0,
                name: "Hull pattern selection".into(),
                origins: vec![origin],
                kind: RowKind::Selection,
                enforcement: RowEnforcement::Conditional,
                roundoff: 32.0 * f64::EPSILON,
            })?;
        }

        let result = self.hull(Some(index), inputs, settings, Some(&locals))?;
        self.emit_local_recombinations(result)
    }

    fn emit_local_recombinations(&mut self, rows: Vec<Row>) -> Result<Vec<Row>, GdpError> {
        let mut pending = Vec::new();

        for row in rows {
            if let RowKind::Aggregation {
                source,
                coordinates: CoordinateScope::ActivationLocal { owner },
            } = row.kind
            {
                debug_assert!(self.is_local(source, Some(owner)));
                self.emit(row)?;
            } else {
                pending.push(row);
            }
        }

        Ok(pending)
    }
}

/// Entire Boolean subtrees containing only Big-M use its existing estimator,
/// hierarchy tightening, and provenance, even beside independent Hull groups.
fn big_m_components(
    data: &GdpSnapshot,
    options: &GdpReformulationOptions,
    methods: &FxHashMap<DisjunctionId, GdpMethod>,
) -> FxHashSet<BooleanId> {
    let mut children = vec![Vec::new(); data.booleans.len()];

    for (i, b) in data.booleans.iter().enumerate() {
        if let Some(p) = b.parent {
            children[p.index()].push(i);
        }
    }

    let mut roots = vec![0; data.booleans.len()];
    let mut mixed = vec![false; data.booleans.len()];
    let mut stack: Vec<_> = data
        .booleans
        .iter()
        .enumerate()
        .filter(|(_, b)| b.parent.is_none())
        .map(|(i, _)| (i, i))
        .collect();

    while let Some((i, root)) = stack.pop() {
        roots[i] = root;
        let method = data.booleans[i]
            .owner
            .and_then(|d| methods.get(&d))
            .copied()
            .unwrap_or(options.default_method);
        mixed[root] |= matches!(method, GdpMethod::Hull(_));
        stack.extend(children[i].iter().map(|&c| (c, root)));
    }

    roots
        .into_iter()
        .enumerate()
        .filter(|(_, root)| !mixed[*root])
        .map(|(i, _)| BooleanId(u32::try_from(i).expect("Boolean ID")))
        .collect()
}

#[expect(clippy::float_cmp, reason = "only exactly equal interval endpoints define an equality")]
fn sides(lower: f64, upper: f64) -> Vec<(f64, f64, f64)> {
    if lower == upper && lower.is_finite() {
        return vec![(lower, 0.0, 0.0)];
    }

    let mut result = Vec::with_capacity(2);

    if lower.is_finite() {
        result.push((lower, 0.0, f64::INFINITY));
    }

    if upper.is_finite() {
        result.push((upper, f64::NEG_INFINITY, 0.0));
    }

    result
}

fn selected_first(pattern: &GdpHullPatternMapping) -> Option<BooleanId> {
    pattern.selected.first().copied()
}

/// Normalize the complete affine part, including nonlinear perspective
/// corrections, before checking and committing exactly that representation.
fn checked_hull_body(
    arena: &mut PlanningArena,
    mut generated: Linear,
    residual: &[SignedExpr],
    source: &LinearTerms<'_>,
    substitution: &Substitution,
    bound: f64,
    diagnostic: (&SmolStr, BoundSide, f64),
) -> Result<RowBody, GdpError> {
    generated.normalize(&mut arena.coefficient_slots)?;
    check_active_row(
        &LinearTerms::borrowed(&generated.terms, generated.constant),
        source,
        substitution,
        bound,
        diagnostic,
        &mut arena.recovery_coefficients,
    )?;

    if residual.is_empty() {
        return Ok(RowBody::normalized_affine(generated));
    }

    Ok(RowBody::Expression(residual_body(arena, generated, residual)?))
}

fn residual_body(
    arena: &mut PlanningArena,
    affine: Linear,
    residual: &[SignedExpr],
) -> Result<ExprId, GdpError> {
    if affine.terms.is_empty() && affine.constant == 0.0 && residual.len() == 1 {
        let term = residual[0];
        return if term.neg {
            arena.push(ExprNode::Unary(oximo_core::UnaryOp::Neg, term.id))
        } else {
            Ok(term.id)
        };
    }

    let affine = linear(arena, affine)?;
    let mut parts = vec![affine];

    for term in residual {
        parts.push(if term.neg {
            arena.push(ExprNode::Unary(oximo_core::UnaryOp::Neg, term.id))?
        } else {
            term.id
        });
    }

    arena.push(ExprNode::Add(parts.into_iter().collect()))
}

/// A few ulps on the retained source coefficients, independent of selector
/// offsets, reference magnitudes and later generated coefficients.
fn coefficient_roundoff(coefficients: impl Iterator<Item = f64>) -> f64 {
    let mut scale: f64 = 0.0;
    let mut count = 0.0;

    for coefficient in coefficients {
        scale = scale.max(coefficient.abs());
        count += 1.0;
    }

    (32.0 * f64::EPSILON * scale) * count
}

fn recovery_tolerance(effective_bound: f64, roundoff: f64) -> f64 {
    if effective_bound == 0.0 { roundoff } else { 1e-9 * effective_bound.abs() }
}

/// Move the original constant and active selectors to each finite bound in one
/// compensated sum.
/// Validate against those original components before storing the rounded bound.
fn substitute_bound(
    bound: f64,
    offsets: &[f64],
    roundoff: f64,
    (name, side): (&SmolStr, BoundSide),
) -> Result<f64, GdpError> {
    if !bound.is_finite() {
        return Ok(bound);
    }

    let mut sum = RecoverySum::default();
    sum.add(bound)?;

    for &offset in offsets {
        sum.add(-offset)?;
    }

    let effective = sum.value()?;
    sum.add(-effective)?;
    let error = sum.value()?.abs();
    let tolerance = recovery_tolerance(effective, roundoff);

    if error > tolerance {
        return Err(GdpError::HullRowPrecisionLoss {
            constraint: name.clone(),
            side,
            error,
            tolerance,
        });
    }

    Ok(effective)
}

/// Undo every copy shift at activation one.
fn check_active_row(
    generated: &LinearTerms<'_>,
    source: &LinearTerms<'_>,
    substitution: &Substitution,
    bound: f64,
    (name, side, roundoff): (&SmolStr, BoundSide, f64),
    coefficients: &mut FxHashMap<VarId, f64>,
) -> Result<(), GdpError> {
    let failure = |error, tolerance| GdpError::HullRowPrecisionLoss {
        constraint: name.clone(),
        side,
        error,
        tolerance,
    };

    coefficients.clear();
    coefficients.extend(generated.coeffs.iter().copied());

    debug_assert_eq!(substitution.guard.terms.len(), 1);
    let (indicator, coefficient) = substitution.guard.terms[0];
    let active = (1.0 - substitution.guard.constant) / coefficient;

    let mut expected = RecoverySum::default();
    expected.add(bound)?;
    expected.add(-source.constant)?;

    let mut difference = RecoverySum::default();
    difference.add(-generated.constant)?;
    difference.add_product(-coefficients.get(&indicator).copied().unwrap_or(0.0), active)?;
    difference.add(-bound)?;
    difference.add(source.constant)?;

    for &(variable, source_coefficient) in source.coeffs.iter() {
        let target = substitution.copies.get(&variable).copied().unwrap_or(variable);
        let generated_coefficient = coefficients.get(&target).copied().unwrap_or(0.0);

        if target != indicator {
            let error = (generated_coefficient - source_coefficient).abs();
            let tolerance = 1e-9 * source_coefficient.abs();

            if error > tolerance {
                return Err(failure(error, tolerance));
            }

            if let Some(&reference) = substitution.references.get(&variable) {
                difference.add_product(generated_coefficient, reference)?;
            }
        }

        if variable == indicator {
            expected.add_product(-source_coefficient, active)?;
            difference.add_product(source_coefficient, active)?;

            if target != indicator {
                difference.add_product(-generated_coefficient, active)?;
            }
        }
    }

    let error = difference.value()?.abs();
    let tolerance = recovery_tolerance(expected.value()?, roundoff);

    if error > tolerance {
        return Err(failure(error, tolerance));
    }

    Ok(())
}

#[derive(Default)]
struct RecoverySum {
    sum: f64,
    correction: f64,
}

impl RecoverySum {
    fn add(&mut self, value: f64) -> Result<(), GdpError> {
        let next = self.sum + value;

        if !value.is_finite() || !next.is_finite() {
            return Err(GdpError::NumericOverflow);
        }
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;

        Ok(())
    }

    fn add_product(&mut self, coefficient: f64, value: f64) -> Result<(), GdpError> {
        let product = coefficient * value;
        self.add(product)?;

        // Recover multiplication roundoff too.
        self.add(coefficient.mul_add(value, -product))
    }

    fn value(&self) -> Result<f64, GdpError> {
        let value = self.sum + self.correction;

        if value.is_finite() { Ok(value) } else { Err(GdpError::NumericOverflow) }
    }
}

fn linear(arena: &mut PlanningArena, mut expression: Linear) -> Result<ExprId, GdpError> {
    expression.normalize(&mut arena.coefficient_slots)?;
    u32::try_from(arena.len()).map_err(|_| GdpError::Capacity)?;
    arena.linear(expression.terms, expression.constant)
}

fn add_constant(
    arena: &mut PlanningArena,
    value: ExprId,
    constant: f64,
) -> Result<ExprId, GdpError> {
    if constant == 0.0 {
        Ok(value)
    } else {
        let c = arena.constant(constant)?;
        arena.push(ExprNode::Add([value, c].into_iter().collect()))
    }
}

fn variables(arena: &ExprArena, root: ExprId) -> Vec<VarId> {
    let mut seen = FxHashSet::default();
    let mut variables = FxHashSet::default();
    let mut stack = vec![root];

    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }

        match arena.get(id) {
            ExprNode::Var(v) => {
                variables.insert(*v);
            }
            ExprNode::Linear { coeffs, .. } => {
                variables.extend(coeffs.iter().filter(|(_, c)| *c != 0.0).map(|(v, _)| *v));
            }
            ExprNode::Add(cs) | ExprNode::Mul(cs) | ExprNode::Min(cs) | ExprNode::Max(cs) => {
                stack.extend(cs);
            }
            ExprNode::Unary(_, c) => stack.push(*c),
            ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            _ => {}
        }
    }

    let mut result: Vec<_> = variables.into_iter().collect();
    result.sort_unstable_by_key(|v| v.index());

    result
}

/// Iterative memoized substitution. Parameters are frozen at the locked snapshot.
fn rewrite(
    arena: &mut PlanningArena,
    root: ExprId,
    map: &FxHashMap<VarId, ExprId>,
    cache: &mut FxHashMap<ExprId, ExprId>,
) -> Result<ExprId, GdpError> {
    let mut stack = vec![(root, false)];

    while let Some((id, visited)) = stack.pop() {
        if cache.contains_key(&id) {
            continue;
        }

        let node = arena.get(id).clone();
        if !visited {
            stack.push((id, true));
            match &node {
                ExprNode::Add(cs) | ExprNode::Mul(cs) | ExprNode::Min(cs) | ExprNode::Max(cs) => {
                    stack.extend(cs.iter().map(|&c| (c, false)));
                }
                ExprNode::Unary(_, c) => stack.push((*c, false)),
                ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                    stack.push((*a, false));
                    stack.push((*b, false));
                }
                _ => {}
            }

            continue;
        }

        u32::try_from(arena.len()).map_err(|_| GdpError::Capacity)?;
        let child = |c| cache[&c];
        let unchanged = match &node {
            ExprNode::Linear { coeffs, .. } => !coeffs.iter().any(|(v, _)| map.contains_key(v)),
            ExprNode::Add(cs) | ExprNode::Mul(cs) | ExprNode::Min(cs) | ExprNode::Max(cs) => {
                cs.iter().all(|&c| child(c) == c)
            }
            ExprNode::Unary(_, c) => child(*c) == *c,
            ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => {
                child(*a) == *a && child(*b) == *b
            }
            _ => false,
        };

        if unchanged {
            cache.insert(id, id);
            continue;
        }

        let rewritten = match node {
            ExprNode::Var(v) => map.get(&v).copied().unwrap_or(id),
            ExprNode::Param(p) => arena.constant(arena.param_value(p))?,
            ExprNode::Const(_) => id,
            ExprNode::Linear { coeffs, constant } => {
                let mut affine = Linear::constant(constant);
                let mut residual = Vec::new();

                for (v, c) in coeffs {
                    if c == 0.0 {
                        continue;
                    }

                    if let Some(&substitution) = map.get(&v) {
                        if let Some(terms) = extract_linear(arena, substitution) {
                            affine = affine.add(
                                Linear {
                                    constant: terms.constant,
                                    terms: terms.coeffs.into_owned(),
                                }
                                .scaled(c),
                            );
                        } else {
                            let coefficient = arena.constant(c)?;
                            residual.push(arena.push(ExprNode::Mul(
                                [coefficient, substitution].into_iter().collect(),
                            ))?);
                        }
                    } else {
                        affine.terms.push((v, c));
                    }
                }

                let affine = linear(arena, affine)?;

                if residual.is_empty() {
                    affine
                } else {
                    residual.push(affine);
                    arena.push(ExprNode::Add(residual.into_iter().collect()))?
                }
            }
            ExprNode::Add(cs) => arena.push(ExprNode::Add(cs.into_iter().map(child).collect()))?,
            ExprNode::Mul(cs) => arena.push(ExprNode::Mul(cs.into_iter().map(child).collect()))?,
            ExprNode::Min(cs) => arena.push(ExprNode::Min(cs.into_iter().map(child).collect()))?,
            ExprNode::Max(cs) => arena.push(ExprNode::Max(cs.into_iter().map(child).collect()))?,
            ExprNode::Unary(op, c) => arena.push(ExprNode::Unary(op, child(c)))?,
            ExprNode::Pow(a, b) => arena.push(ExprNode::Pow(child(a), child(b)))?,
            ExprNode::Div(a, b) => arena.push(ExprNode::Div(child(a), child(b)))?,
            ExprNode::Atan2(a, b) => arena.push(ExprNode::Atan2(child(a), child(b)))?,
        };

        cache.insert(id, rewritten);
    }

    Ok(cache[&root])
}
