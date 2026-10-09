//! GDP modeling records and model-bound logical handles.
//!
//! Reformulation algorithms live in `oximo-gdp`.

use std::cell::Ref;
use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr, BitXor, Index, Not};
use std::sync::Arc;

use oximo_expr::{Affine, Expr, ExprArenaCell, ExprId, ModelId, ParamId, VarId};
use rustc_hash::{FxHashMap, FxHashSet};
use smol_str::SmolStr;

use crate::constraint::{ConstraintId, IntoRhs, Relate};
use crate::function_set::{AlgebraicConstraintIr, FunctionDegree, IntoFunction};
use crate::{FromIndexKey, IndexKey, Model, ReformulatedModel, Set, display_index_key};

macro_rules! gdp_id {
    ($name:ident) => {
        #[doc = concat!("Stable numeric ID for a GDP ", stringify!($name), ".")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub u32);
        impl $name {
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

gdp_id!(BooleanId);
gdp_id!(DisjunctionId);
gdp_id!(DisjunctConstraintId);
gdp_id!(LogicalConstraintId);

/// A logical decision with an explicit algebraic binary counterpart.
#[derive(Clone, Copy, Debug)]
pub struct BooleanHandle<'a> {
    model: &'a Model,
    id: BooleanId,
}

/// A disjunct is the group of conditional rows attached to a Boolean decision.
pub type DisjunctHandle<'a> = BooleanHandle<'a>;

impl<'a> BooleanHandle<'a> {
    pub const fn id(self) -> BooleanId {
        self.id
    }

    pub fn model_id(self) -> ModelId {
        self.model.id()
    }

    pub const fn indicator(self) -> Self {
        self
    }

    pub fn binary(self) -> Expr<'a, Affine> {
        self.model.variable_handle(self.model.gdp.borrow().booleans[self.id.index()].binary)
    }

    pub fn context(self) -> DisjunctContext<'a> {
        DisjunctContext { model: self.model, indicator: self.id }
    }

    pub fn implies(self, other: impl Into<LogicalExpr>) -> LogicalExpr {
        implies(self, other)
    }

    pub fn equivalent_to(self, other: impl Into<LogicalExpr>) -> LogicalExpr {
        iff(self, other)
    }
}

impl From<BooleanHandle<'_>> for BooleanId {
    fn from(value: BooleanHandle<'_>) -> Self {
        value.id
    }
}

impl std::fmt::Display for BooleanHandle<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.model.gdp.borrow().booleans[self.id.index()].name)
    }
}

/// Owned logical expression. Leaves retain model provenance.
#[derive(Debug)]
pub enum LogicalExpr {
    Literal(bool),
    Boolean { id: BooleanId, model_id: ModelId },
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
    Xor(Box<Self>, Box<Self>),
    Implies(Box<Self>, Box<Self>),
    Iff(Box<Self>, Box<Self>),
    Cardinality { kind: Cardinality, count: usize, terms: Vec<Self> },
}

impl Clone for LogicalExpr {
    fn clone(&self) -> Self {
        let mut stack = vec![(self, false)];
        let mut values = Vec::new();

        while let Some((expression, visited)) = stack.pop() {
            if !visited {
                stack.push((expression, true));

                match expression {
                    Self::Not(child) => stack.push((child, false)),
                    Self::And(children)
                    | Self::Or(children)
                    | Self::Cardinality { terms: children, .. } => {
                        stack.extend(children.iter().rev().map(|child| (child, false)));
                    }
                    Self::Xor(a, b) | Self::Implies(a, b) | Self::Iff(a, b) => {
                        stack.push((b, false));
                        stack.push((a, false));
                    }
                    Self::Literal(_) | Self::Boolean { .. } => {}
                }
                continue;
            }

            let clone = match expression {
                Self::Literal(value) => Self::Literal(*value),
                Self::Boolean { id, model_id } => Self::Boolean { id: *id, model_id: *model_id },
                Self::Not(_) => Self::Not(Box::new(values.pop().expect("one child"))),
                Self::And(children) => Self::And(values.split_off(values.len() - children.len())),
                Self::Or(children) => Self::Or(values.split_off(values.len() - children.len())),
                Self::Cardinality { kind, count, terms } => Self::Cardinality {
                    kind: *kind,
                    count: *count,
                    terms: values.split_off(values.len() - terms.len()),
                },
                Self::Xor(_, _) | Self::Implies(_, _) | Self::Iff(_, _) => {
                    let b = Box::new(values.pop().expect("second child"));
                    let a = Box::new(values.pop().expect("first child"));

                    match expression {
                        Self::Xor(_, _) => Self::Xor(a, b),
                        Self::Implies(_, _) => Self::Implies(a, b),
                        Self::Iff(_, _) => Self::Iff(a, b),
                        _ => unreachable!(),
                    }
                }
            };

            values.push(clone);
        }

        values.pop().expect("logical root")
    }
}

impl LogicalExpr {
    fn take_children(&mut self, pending: &mut Vec<Self>) {
        // Leave boxed leaves in place, since their ordinary destruction is shallow.
        let take_box = |child: &mut Box<Self>, pending: &mut Vec<Self>| {
            if !matches!(child.as_ref(), Self::Literal(_) | Self::Boolean { .. }) {
                pending.push(std::mem::replace(child.as_mut(), Self::Literal(false)));
            }
        };

        match self {
            Self::Not(child) => take_box(child, pending),
            Self::And(children)
            | Self::Or(children)
            | Self::Cardinality { terms: children, .. } => pending.append(children),
            Self::Xor(a, b) | Self::Implies(a, b) | Self::Iff(a, b) => {
                take_box(a, pending);
                take_box(b, pending);
            }
            Self::Literal(_) | Self::Boolean { .. } => {}
        }
    }
}

impl Drop for LogicalExpr {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        self.take_children(&mut pending);

        while let Some(mut expression) = pending.pop() {
            expression.take_children(&mut pending);

            // Its compound descendants are now on the worklist.
        }
    }
}

/// Cardinality predicate over logical expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cardinality {
    Exactly,
    AtMost,
    AtLeast,
}

impl From<bool> for LogicalExpr {
    fn from(value: bool) -> Self {
        Self::Literal(value)
    }
}

impl From<BooleanHandle<'_>> for LogicalExpr {
    fn from(value: BooleanHandle<'_>) -> Self {
        Self::Boolean { id: value.id, model_id: value.model.id() }
    }
}

impl From<&BooleanHandle<'_>> for LogicalExpr {
    fn from(value: &BooleanHandle<'_>) -> Self {
        (*value).into()
    }
}

impl LogicalExpr {
    pub fn implies(self, other: impl Into<Self>) -> Self {
        implies(self, other)
    }

    pub fn equivalent_to(self, other: impl Into<Self>) -> Self {
        iff(self, other)
    }
}

macro_rules! logical_ops {
    ($ty:ty) => {
        impl Not for $ty {
            type Output = LogicalExpr;

            fn not(self) -> Self::Output {
                LogicalExpr::Not(Box::new(self.into()))
            }
        }

        impl<R: Into<LogicalExpr>> BitAnd<R> for $ty {
            type Output = LogicalExpr;

            fn bitand(self, rhs: R) -> Self::Output {
                LogicalExpr::And(vec![self.into(), rhs.into()])
            }
        }

        impl<R: Into<LogicalExpr>> BitOr<R> for $ty {
            type Output = LogicalExpr;

            fn bitor(self, rhs: R) -> Self::Output {
                LogicalExpr::Or(vec![self.into(), rhs.into()])
            }
        }

        impl<R: Into<LogicalExpr>> BitXor<R> for $ty {
            type Output = LogicalExpr;

            fn bitxor(self, rhs: R) -> Self::Output {
                LogicalExpr::Xor(Box::new(self.into()), Box::new(rhs.into()))
            }
        }
    };
}
logical_ops!(BooleanHandle<'_>);
logical_ops!(LogicalExpr);

macro_rules! bool_left_ops {
    ($ty:ty) => {
        impl BitAnd<$ty> for bool {
            type Output = LogicalExpr;

            fn bitand(self, rhs: $ty) -> LogicalExpr {
                LogicalExpr::from(self) & rhs
            }
        }

        impl BitOr<$ty> for bool {
            type Output = LogicalExpr;

            fn bitor(self, rhs: $ty) -> LogicalExpr {
                LogicalExpr::from(self) | rhs
            }
        }

        impl BitXor<$ty> for bool {
            type Output = LogicalExpr;

            fn bitxor(self, rhs: $ty) -> LogicalExpr {
                LogicalExpr::from(self) ^ rhs
            }
        }
    };
}
bool_left_ops!(BooleanHandle<'_>);
bool_left_ops!(LogicalExpr);

pub fn implies(a: impl Into<LogicalExpr>, b: impl Into<LogicalExpr>) -> LogicalExpr {
    LogicalExpr::Implies(Box::new(a.into()), Box::new(b.into()))
}

pub fn iff(a: impl Into<LogicalExpr>, b: impl Into<LogicalExpr>) -> LogicalExpr {
    LogicalExpr::Iff(Box::new(a.into()), Box::new(b.into()))
}

pub fn logical_and<T: Into<LogicalExpr>>(terms: impl IntoIterator<Item = T>) -> LogicalExpr {
    LogicalExpr::And(terms.into_iter().map(Into::into).collect())
}

pub fn logical_or<T: Into<LogicalExpr>>(terms: impl IntoIterator<Item = T>) -> LogicalExpr {
    LogicalExpr::Or(terms.into_iter().map(Into::into).collect())
}

fn cardinality<T: Into<LogicalExpr>>(
    kind: Cardinality,
    count: usize,
    terms: impl IntoIterator<Item = T>,
) -> LogicalExpr {
    LogicalExpr::Cardinality { kind, count, terms: terms.into_iter().map(Into::into).collect() }
}

pub fn exactly<T: Into<LogicalExpr>>(
    count: usize,
    terms: impl IntoIterator<Item = T>,
) -> LogicalExpr {
    cardinality(Cardinality::Exactly, count, terms)
}

pub fn at_most<T: Into<LogicalExpr>>(
    count: usize,
    terms: impl IntoIterator<Item = T>,
) -> LogicalExpr {
    cardinality(Cardinality::AtMost, count, terms)
}

pub fn at_least<T: Into<LogicalExpr>>(
    count: usize,
    terms: impl IntoIterator<Item = T>,
) -> LogicalExpr {
    cardinality(Cardinality::AtLeast, count, terms)
}

/// Disjunction selector. Exactly one branch is selected by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DisjunctionKind {
    #[default]
    ExactlyOne,
    AtLeastOne,
}

macro_rules! gdp_handle {
    ($name:ident, $id:ident) => {
        #[doc = concat!("Model-bound ", stringify!($id), " handle.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            id: $id,
            model_id: ModelId,
        }

        impl $name {
            pub const fn id(self) -> $id {
                self.id
            }

            pub const fn model_id(self) -> ModelId {
                self.model_id
            }
        }

        impl From<$name> for $id {
            fn from(value: $name) -> Self {
                value.id
            }
        }
    };
}
gdp_handle!(DisjunctionHandle, DisjunctionId);
gdp_handle!(DisjunctConstraintHandle, DisjunctConstraintId);
gdp_handle!(LogicalConstraintHandle, LogicalConstraintId);

/// One or two conditional rows returned by a range declaration.
#[derive(Clone, Copy, Debug)]
pub enum GdpRangeHandles {
    Interval(DisjunctConstraintHandle),
    Split { lower: DisjunctConstraintHandle, upper: DisjunctConstraintHandle },
}

/// Ordered indexed GDP family with typed keys.
#[derive(Clone, Debug)]
pub struct GdpFamily<K, T> {
    entries: Vec<(IndexKey, T)>,
    lookup: FxHashMap<IndexKey, usize>,
    marker: PhantomData<fn() -> K>,
}

impl<K, T> GdpFamily<K, T> {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_ref(&self, key: impl Into<IndexKey>) -> Option<&T> {
        self.lookup.get(&key.into()).map(|&i| &self.entries[i].1)
    }
}

impl<K: FromIndexKey, T> GdpFamily<K, T> {
    pub fn iter_ref(&self) -> impl Iterator<Item = (K, &T)> {
        self.entries.iter().map(|(k, v)| (K::from_index_key(k), v))
    }
}

impl<K, T: Copy> GdpFamily<K, T> {
    pub fn get(&self, key: impl Into<IndexKey>) -> Option<T> {
        self.get_ref(key).copied()
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = T> + DoubleEndedIterator + '_ {
        self.entries.iter().map(|(_, value)| *value)
    }
}

impl<K: FromIndexKey, T: Copy> GdpFamily<K, T> {
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (K, T)> + DoubleEndedIterator + '_ {
        self.entries.iter().map(|(key, value)| (K::from_index_key(key), *value))
    }
}

/// Whether a disjunct can still receive conditional rows or nested declarations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisjunctState {
    /// Declarations may be added before reformulation consumes this disjunct.
    Open,
    /// Reformulation consumed this disjunct. Further declarations are rejected.
    Sealed,
}

/// Lifecycle of a GDP source component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReformulationState {
    /// No solver rows have been generated for this source yet.
    Pending,
    /// The source was consumed by a successful reformulation.
    Reformulated,
}

impl<K, T, Q: Into<IndexKey>> Index<Q> for GdpFamily<K, T> {
    type Output = T;

    fn index(&self, index: Q) -> &T {
        self.get_ref(index).expect("GDP family key not found")
    }
}

/// Source logical decision and its algebraic counterpart.
#[derive(Clone, Debug)]
pub struct BooleanRecord {
    pub name: SmolStr,
    pub binary: VarId,
    pub owner: Option<DisjunctionId>,
    pub parent: Option<BooleanId>,
    state: DisjunctState,
}

/// Conditional algebraic row, separate from solver rows.
#[derive(Clone, Debug)]
pub struct DisjunctConstraintRecord {
    pub name: SmolStr,
    pub indicator: BooleanId,
    pub lhs: ExprId,
    pub lower: f64,
    pub upper: f64,
    state: ReformulationState,
}

/// Disjunction source record.
#[derive(Clone, Debug)]
pub struct DisjunctionRecord {
    pub name: SmolStr,
    pub branches: Vec<BooleanId>,
    pub parent: Option<BooleanId>,
    pub kind: DisjunctionKind,
    state: ReformulationState,
}

/// Logical proposition source record.
#[derive(Clone, Debug)]
pub struct LogicalConstraintRecord {
    pub name: SmolStr,
    pub expression: LogicalExpr,
    pub parent: Option<BooleanId>,
    state: ReformulationState,
}

impl BooleanRecord {
    pub const fn state(&self) -> DisjunctState {
        self.state
    }
}

macro_rules! source_state {
    ($($record:ident),+ $(,)?) => { $(impl $record {
        pub const fn state(&self) -> ReformulationState { self.state }
    })+ };
}
source_state!(DisjunctConstraintRecord, DisjunctionRecord, LogicalConstraintRecord);

/// Side of an interval used by a Big-M diagnostic or override.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundSide {
    Lower,
    Upper,
}

impl std::fmt::Display for BoundSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
        })
    }
}

/// Reformulation method identity, independent of its configuration.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdpMethodKind {
    BigM,
}

/// Logical decision and its algebraic binary counterpart in a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdpBooleanMapping {
    pub source: BooleanId,
    pub binary: VarId,
}

/// Generated selection rows and the method used for a source disjunction.
#[derive(Clone, Debug)]
pub struct GdpDisjunctionArtifacts {
    pub source: DisjunctionId,
    pub constraints: Vec<ConstraintId>,
    pub method: GdpMethodKind,
}

/// Rows newly emitted for an assertion, including shared predicate definitions.
#[derive(Clone, Debug)]
pub struct GdpLogicalArtifacts {
    pub source: LogicalConstraintId,
    pub constraints: Vec<ConstraintId>,
}

/// Origin of the nonnegative M used for one side of a conditional row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BigMOrigin {
    Explicit,
    Estimated,
    DisjunctionFallback,
    GlobalFallback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BigMSide {
    pub value: f64,
    pub origin: BigMOrigin,
}

#[derive(Clone, Debug)]
pub struct GdpBigMTerm {
    pub indicator: BooleanId,
    /// Nonnegative relaxation amount multiplying `1 - indicator` on this side.
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct GdpRowArtifacts {
    pub source: DisjunctConstraintId,
    pub constraints: Vec<ConstraintId>,
    pub lower_m: Option<BigMSide>,
    pub upper_m: Option<BigMSide>,
    /// Actual child and ancestor relaxation amounts when a side was
    /// tightened hierarchically.
    pub hierarchical_m: Vec<GdpBigMTerm>,
}

/// Source-to-generated provenance for a single successful transformation call.
#[derive(Clone, Debug, Default)]
pub struct GdpReformulationReport(Arc<GdpReformulationReportData>);

impl std::ops::Deref for GdpReformulationReport {
    type Target = GdpReformulationReportData;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for GdpReformulationReport {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}

impl From<GdpReformulationReportData> for GdpReformulationReport {
    fn from(data: GdpReformulationReportData) -> Self {
        Self(Arc::new(data))
    }
}

impl GdpReformulationReport {
    /// Consume the report into independently owned artifacts, moving the data
    /// when uniquely held or copying it when other reports still share it.
    pub fn into_owned(self) -> GdpReformulationReportData {
        Arc::unwrap_or_clone(self.0)
    }
}

/// Owned artifact vectors behind a [`GdpReformulationReport`]. Field access and
/// iteration work through the report. Use `into_owned` to move individual fields.
#[derive(Clone, Debug, Default)]
pub struct GdpReformulationReportData {
    /// All registered Boolean decisions, including previously transformed ones.
    pub booleans: Vec<GdpBooleanMapping>,
    /// Auxiliary variables created by this call. Boolean counterparts are in `booleans`.
    pub variables: Vec<VarId>,
    pub rows: Vec<GdpRowArtifacts>,
    pub disjunctions: Vec<GdpDisjunctionArtifacts>,
    /// Rows newly emitted for each assertion.
    pub logical_constraints: Vec<GdpLogicalArtifacts>,
}

/// Owned copy of GDP source records. Later model changes do not affect it.
#[derive(Clone, Debug, Default)]
pub struct GdpSnapshot {
    pub booleans: Vec<BooleanRecord>,
    pub rows: Vec<DisjunctConstraintRecord>,
    pub disjunctions: Vec<DisjunctionRecord>,
    pub logical_constraints: Vec<LogicalConstraintRecord>,
}

/// Borrowed inspection of GDP source records. Holds an immutable registry borrow
/// until dropped. Use [`Model::gdp_snapshot`] to retain an independent copy.
#[derive(Debug)]
pub struct GdpView<'a> {
    data: Ref<'a, GdpSnapshot>,
}

impl std::ops::Deref for GdpView<'_> {
    type Target = GdpSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Named cursors into append-only source registries, shared with planners.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GdpPendingStarts {
    pub rows: usize,
    pub disjunctions: usize,
    pub logical_constraints: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GdpData {
    snapshot: GdpSnapshot,
    reports: Vec<GdpReformulationReport>,
    locked_variables: FxHashSet<VarId>,
    pending_starts: GdpPendingStarts,
    names: FxHashSet<SmolStr>,
    auto_sequences: FxHashMap<(Option<BooleanId>, SmolStr), u64>,
}

impl std::ops::Deref for GdpData {
    type Target = GdpSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.snapshot
    }
}

impl GdpData {
    fn register_boolean(&mut self, name: SmolStr, binary: VarId) {
        self.names.insert(name.clone());
        self.snapshot.booleans.push(BooleanRecord {
            name,
            binary,
            owner: None,
            parent: None,
            state: DisjunctState::Open,
        });
    }

    fn register_disjunction(
        &mut self,
        name: SmolStr,
        branches: &[BooleanHandle<'_>],
        kind: DisjunctionKind,
        parent: Option<BooleanId>,
    ) -> DisjunctionId {
        assert!(!self.names.contains(&name), "GDP name {name:?} already registered");

        if let Some(p) = parent {
            assert!(
                self.snapshot.booleans[p.index()].state == DisjunctState::Open,
                "cannot edit a transformed disjunct"
            );
        }

        for branch in branches {
            let record = &self.snapshot.booleans[branch.id.index()];
            assert!(record.owner.is_none(), "a disjunct can belong to only one disjunction");
            assert!(record.state == DisjunctState::Open, "cannot reuse a transformed disjunct");
            let mut ancestor = parent;

            while let Some(p) = ancestor {
                assert_ne!(p, branch.id, "cyclic GDP nesting");
                ancestor = self.snapshot.booleans[p.index()].parent;
            }
        }

        let id = DisjunctionId(
            u32::try_from(self.snapshot.disjunctions.len()).expect("disjunction ID overflow"),
        );

        for branch in branches {
            self.snapshot.booleans[branch.id.index()].owner = Some(id);
            self.snapshot.booleans[branch.id.index()].parent = parent;
        }

        self.names.insert(name.clone());
        self.snapshot.disjunctions.push(DisjunctionRecord {
            name,
            branches: branches.iter().map(|b| b.id).collect(),
            kind,
            parent,
            state: ReformulationState::Pending,
        });
        id
    }

    fn register_logic(
        &mut self,
        name: SmolStr,
        expression: LogicalExpr,
        parent: Option<BooleanId>,
    ) -> LogicalConstraintId {
        assert!(!self.names.contains(&name), "GDP name {name:?} already registered");

        if let Some(p) = parent {
            assert!(
                self.snapshot.booleans[p.index()].state == DisjunctState::Open,
                "cannot edit a transformed disjunct"
            );
        }

        let id = LogicalConstraintId(
            u32::try_from(self.snapshot.logical_constraints.len())
                .expect("logical constraint ID overflow"),
        );
        self.names.insert(name.clone());
        self.snapshot.logical_constraints.push(LogicalConstraintRecord {
            name,
            expression,
            parent,
            state: ReformulationState::Pending,
        });
        id
    }

    fn register_row(
        &mut self,
        name: SmolStr,
        indicator: BooleanId,
        c: AlgebraicConstraintIr<'_>,
    ) -> DisjunctConstraintId {
        assert!(
            self.snapshot.booleans[indicator.index()].state == DisjunctState::Open,
            "cannot edit a transformed disjunct"
        );
        assert!(!self.names.contains(&name), "GDP name {name:?} already registered");
        let id = DisjunctConstraintId(
            u32::try_from(self.snapshot.rows.len()).expect("conditional row ID overflow"),
        );
        self.names.insert(name.clone());
        self.snapshot.rows.push(DisjunctConstraintRecord {
            name,
            indicator,
            lhs: c.lhs.id,
            lower: c.lower,
            upper: c.upper,
            state: ReformulationState::Pending,
        });
        id
    }

    fn finish(&mut self, report: GdpReformulationReport, variables: Vec<VarId>) {
        for row in &report.rows {
            let record = &mut self.snapshot.rows[row.source.index()];
            record.state = ReformulationState::Reformulated;
            self.snapshot.booleans[record.indicator.index()].state = DisjunctState::Sealed;
        }

        for artifacts in &report.disjunctions {
            let record = &mut self.snapshot.disjunctions[artifacts.source.index()];
            record.state = ReformulationState::Reformulated;

            for &boolean in &record.branches {
                self.snapshot.booleans[boolean.index()].state = DisjunctState::Sealed;
            }

            if let Some(parent) = record.parent {
                self.snapshot.booleans[parent.index()].state = DisjunctState::Sealed;
            }
        }

        for artifacts in &report.logical_constraints {
            let record = &mut self.snapshot.logical_constraints[artifacts.source.index()];
            record.state = ReformulationState::Reformulated;

            if let Some(parent) = record.parent {
                self.snapshot.booleans[parent.index()].state = DisjunctState::Sealed;
            }
        }

        self.locked_variables.extend(variables);
        self.pending_starts = GdpPendingStarts {
            rows: self.rows.len(),
            disjunctions: self.disjunctions.len(),
            logical_constraints: self.logical_constraints.len(),
        };

        if !report.rows.is_empty()
            || !report.disjunctions.is_empty()
            || !report.logical_constraints.is_empty()
        {
            self.reports.push(report);
        }
    }

    #[inline]
    pub(crate) fn has_pending(&self) -> bool {
        // Check the cheap incremental assertion path first.
        self.logical_constraints.len() != self.pending_starts.logical_constraints
            || self.rows.len() != self.pending_starts.rows
            || self.disjunctions.len() != self.pending_starts.disjunctions
    }

    pub(crate) fn locks_variable(&self, id: VarId) -> bool {
        self.locked_variables.contains(&id)
    }
}

macro_rules! component_lookup {
    ($id_method:ident, $named:ident, $rebind:ident, $field:ident, $id:ident, $handle:ident) => {
        impl Model {
            pub fn $id_method(&self, name: &str) -> Option<$id> {
                self.gdp
                    .borrow()
                    .$field
                    .iter()
                    .position(|r| r.name == name)
                    .and_then(|i| u32::try_from(i).ok())
                    .map($id)
            }

            pub fn $named(&self, name: &str) -> Option<$handle> {
                self.$id_method(name).and_then(|id| self.$rebind(id))
            }

            /// Bind a source ID to this model, including an independent clone.
            /// Numeric IDs are local to a model. This does not validate provenance.
            pub fn $rebind(&self, id: $id) -> Option<$handle> {
                (id.index() < self.gdp.borrow().$field.len())
                    .then_some($handle { id, model_id: self.id() })
            }
        }
    };
}
component_lookup!(
    disjunct_constraint_id,
    disjunct_constraint_handle,
    disjunct_constraint_handle_from_id,
    rows,
    DisjunctConstraintId,
    DisjunctConstraintHandle
);
component_lookup!(
    disjunction_id,
    disjunction_handle,
    disjunction_handle_from_id,
    disjunctions,
    DisjunctionId,
    DisjunctionHandle
);
component_lookup!(
    logical_constraint_id,
    logical_constraint_handle,
    logical_constraint_handle_from_id,
    logical_constraints,
    LogicalConstraintId,
    LogicalConstraintHandle
);

fn check_logic(model: &Model, expression: &LogicalExpr) {
    let mut stack = vec![expression];

    while let Some(e) = stack.pop() {
        match e {
            LogicalExpr::Boolean { id, model_id } => {
                assert_eq!(
                    *model_id,
                    model.id(),
                    "logical expression belongs to a different model"
                );
                assert!(id.index() < model.gdp.borrow().booleans.len(), "unknown Boolean ID");
            }
            LogicalExpr::Not(e) => stack.push(e),
            LogicalExpr::And(es)
            | LogicalExpr::Or(es)
            | LogicalExpr::Cardinality { terms: es, .. } => stack.extend(es),
            LogicalExpr::Xor(a, b) | LogicalExpr::Implies(a, b) | LogicalExpr::Iff(a, b) => {
                stack.push(a);
                stack.push(b);
            }
            LogicalExpr::Literal(_) => {}
        }
    }
}

impl Model {
    /// Inspect the logical and conditional source representation.
    pub fn gdp(&self) -> GdpView<'_> {
        GdpView { data: Ref::map(self.gdp.borrow(), |d| &d.snapshot) }
    }

    /// Take an owned snapshot without retaining a borrow of the model registry.
    pub fn gdp_snapshot(&self) -> GdpSnapshot {
        self.gdp.borrow().snapshot.clone()
    }

    // TODO: Add public add_disjunct, add_disjunction and add_logical_constraint
    // methods on Model and DisjunctContext.
    // We can back them by the macro registration paths.

    pub fn boolean_id(&self, name: &str) -> Option<BooleanId> {
        self.gdp
            .borrow()
            .booleans
            .iter()
            .position(|b| b.name == name)
            .and_then(|i| u32::try_from(i).ok())
            .map(BooleanId)
    }

    pub(crate) fn rebind_gdp_logic_identity(&self) {
        for record in &mut self.gdp.borrow_mut().snapshot.logical_constraints {
            let mut stack = vec![&mut record.expression];

            while let Some(e) = stack.pop() {
                match e {
                    LogicalExpr::Boolean { model_id, .. } => *model_id = self.id(),
                    LogicalExpr::Not(e) => stack.push(e),
                    LogicalExpr::And(es)
                    | LogicalExpr::Or(es)
                    | LogicalExpr::Cardinality { terms: es, .. } => stack.extend(es),
                    LogicalExpr::Xor(a, b)
                    | LogicalExpr::Implies(a, b)
                    | LogicalExpr::Iff(a, b) => {
                        stack.push(a);
                        stack.push(b);
                    }
                    LogicalExpr::Literal(_) => {}
                }
            }
        }
    }

    /// Register a logical decision and its algebraic binary counterpart.
    ///
    /// # Panics
    /// Panics for duplicate GDP names or numeric ID exhaustion.
    pub fn add_boolean(&self, name: impl Into<SmolStr>) -> BooleanHandle<'_> {
        let name = name.into();
        assert!(!self.gdp.borrow().names.contains(&name), "GDP name {name:?} already registered");
        let index = u32::try_from(self.gdp.borrow().booleans.len()).expect("Boolean ID overflow");
        let binary_name = self.__gdp_unique_variable_name(&format!("__oximo_gdp_boolean{index}"));
        let binary = self.__var(binary_name).binary().build().var_id().expect("binary variable");
        self.gdp.borrow_mut().register_boolean(name, binary);
        BooleanHandle { model: self, id: BooleanId(index) }
    }

    /// Rebind a numeric Boolean ID to this model, including an independent clone.
    ///
    /// # Panics
    /// Panics if the ID is not registered on this model.
    pub fn boolean_handle(&self, id: BooleanId) -> BooleanHandle<'_> {
        assert!(id.index() < self.gdp.borrow().booleans.len(), "unknown Boolean ID");
        BooleanHandle { model: self, id }
    }

    pub fn boolean_handle_by_name(&self, name: &str) -> Option<BooleanHandle<'_>> {
        self.boolean_id(name).and_then(|id| self.boolean_handle_from_id(id))
    }

    /// Bind a model-local numeric ID without panicking for an unknown ID.
    pub fn boolean_handle_from_id(&self, id: BooleanId) -> Option<BooleanHandle<'_>> {
        (id.index() < self.gdp.borrow().booleans.len()).then_some(BooleanHandle { model: self, id })
    }

    #[doc(hidden)]
    pub fn __gdp_context<'a>(&'a self, indicator: BooleanHandle<'a>) -> DisjunctContext<'a> {
        assert_eq!(indicator.model_id(), self.id(), "foreign GDP indicator");
        indicator.context()
    }

    #[doc(hidden)]
    pub fn __gdp_disjunct<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        rule: impl FnOnce(DisjunctContext<'a>),
    ) -> BooleanHandle<'a> {
        let indicator = self.add_boolean(name);
        rule(indicator.context());
        indicator
    }

    #[doc(hidden)]
    pub fn __gdp_disjunction<'a>(
        &'a self,
        name: impl Into<SmolStr>,
        branches: impl IntoIterator<Item = BooleanHandle<'a>>,
        kind: DisjunctionKind,
    ) -> DisjunctionHandle {
        self.register_disjunction(name.into(), branches.into_iter().collect(), kind, None)
    }

    fn register_disjunction(
        &self,
        name: SmolStr,
        branches: Vec<BooleanHandle<'_>>,
        kind: DisjunctionKind,
        parent: Option<BooleanId>,
    ) -> DisjunctionHandle {
        assert!(!branches.is_empty(), "a disjunction must have at least one branch");
        let mut seen = FxHashSet::default();

        for branch in &branches {
            assert_eq!(branch.model_id(), self.id(), "foreign disjunction branch");
            assert!(seen.insert(branch.id), "duplicate disjunction branch");
        }

        let id = self.gdp.borrow_mut().register_disjunction(name, &branches, kind, parent);
        DisjunctionHandle { id, model_id: self.id() }
    }

    #[doc(hidden)]
    pub fn __gdp_logic(
        &self,
        name: impl Into<SmolStr>,
        expression: impl Into<LogicalExpr>,
    ) -> LogicalConstraintHandle {
        self.register_logic(name.into(), expression.into(), None)
    }

    fn register_logic(
        &self,
        name: SmolStr,
        expression: LogicalExpr,
        parent: Option<BooleanId>,
    ) -> LogicalConstraintHandle {
        check_logic(self, &expression);
        let id = self.gdp.borrow_mut().register_logic(name, expression, parent);
        LogicalConstraintHandle { id, model_id: self.id() }
    }

    #[doc(hidden)]
    pub fn __gdp_family_over<K: FromIndexKey, T>(
        &self,
        prefix: &str,
        set: &Set<K>,
        mut rule: impl FnMut(K, String) -> T,
    ) -> GdpFamily<K, T> {
        let entries: Vec<_> = set
            .iter()
            .map(|key| {
                let name = format!("{prefix}[{}]", display_index_key(&key));
                let value = rule(K::from_index_key(&key), name);
                (key, value)
            })
            .collect();
        let lookup = entries.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();
        GdpFamily { entries, lookup, marker: PhantomData }
    }

    #[doc(hidden)]
    pub fn __gdp_pending_starts(&self) -> GdpPendingStarts {
        self.gdp.borrow().pending_starts
    }

    #[doc(hidden)]
    pub fn __gdp_auto_name(&self, prefix: &str) -> String {
        self.next_gdp_auto_name(prefix, None)
    }

    fn next_gdp_auto_name(&self, prefix: &str, parent: Option<BooleanId>) -> String {
        let mut data = self.gdp.borrow_mut();
        let key = (parent, SmolStr::new(prefix));
        loop {
            let sequence = data.auto_sequences.entry(key.clone()).or_default();
            let name = format!("_{prefix}{sequence}");
            *sequence = sequence.checked_add(1).expect("GDP name space exhausted");
            let scoped_name = parent.map_or_else(
                || name.clone(),
                |id| format!("{}::{name}", data.snapshot.booleans[id.index()].name),
            );

            if !data.names.contains(scoped_name.as_str()) {
                return name;
            }
        }
    }

    pub fn gdp_reformulations(&self) -> Ref<'_, [GdpReformulationReport]> {
        Ref::map(self.gdp.borrow(), |d| d.reports.as_slice())
    }

    #[doc(hidden)]
    pub fn __gdp_clone(&self) -> Model {
        self.clone_preserving_ids_with_capacity(0, 0, 0)
    }

    #[doc(hidden)]
    pub fn __gdp_reserve(&self, variables: usize, constraints: usize, nodes: usize) {
        self.variables.borrow_mut().reserve(variables);
        self.var_names.borrow_mut().reserve(variables);
        self.constraints.borrow_mut().reserve(constraints);
        self.constraint_names.borrow_mut().reserve(constraints);
        self.arena.borrow_mut().__reserve_nodes(nodes);
    }

    #[doc(hidden)]
    pub fn __gdp_finish(
        &self,
        report: GdpReformulationReport,
        variables: Vec<VarId>,
        parameters: Vec<ParamId>,
    ) {
        self.gdp.borrow_mut().finish(report, variables);
        self.arena.borrow_mut().__lock_parameters(parameters);
    }

    #[doc(hidden)]
    pub fn __gdp_unique_variable_name(&self, base: &str) -> String {
        crate::reformulation::helpers::unique_variable_name(self, base).to_string()
    }
}

impl ReformulatedModel {
    /// Complete GDP reformulation history carried by this transformed model.
    #[must_use]
    pub fn gdp_reformulations(&self) -> Ref<'_, [GdpReformulationReport]> {
        self.model().gdp_reformulations()
    }
}

/// Constraint receiver for a disjunct block.
/// This does not dereference to `Model`, so variables and
/// objectives cannot accidentally become conditional.
#[derive(Clone, Copy, Debug)]
pub struct DisjunctContext<'a> {
    model: &'a Model,
    indicator: BooleanId,
}

impl<'a> DisjunctContext<'a> {
    fn qualify(&self, name: &str) -> String {
        format!("{}::{name}", self.model.gdp.borrow().booleans[self.indicator.index()].name)
    }

    #[doc(hidden)]
    pub fn __sum_context(&self) -> &'a ExprArenaCell {
        self.model.__sum_context()
    }

    /// Register a conditional scalar row, with names scoped to this disjunct.
    ///
    /// # Panics
    /// Panics for foreign expressions, duplicate names, invalid bounds, ID
    /// exhaustion, or edits to an already transformed disjunct.
    pub fn add_constraint(
        &self,
        name: impl Into<SmolStr>,
        c: impl Into<AlgebraicConstraintIr<'a>>,
    ) -> DisjunctConstraintHandle {
        let c = c.into();
        assert_eq!(c.lhs.model_id(), self.model.id(), "foreign conditional expression");
        let name = SmolStr::new(self.qualify(&name.into()));
        let id = self.model.gdp.borrow_mut().register_row(name, self.indicator, c);
        DisjunctConstraintHandle { id, model_id: self.model.id() }
    }

    #[doc(hidden)]
    pub fn __add_constraint_auto(
        &self,
        c: impl Into<AlgebraicConstraintIr<'a>>,
    ) -> DisjunctConstraintHandle {
        self.add_constraint(self.__gdp_auto_name("c"), c)
    }

    #[doc(hidden)]
    pub fn __add_range<D: FunctionDegree, B1: IntoRhs<'a, D>, B2: IntoRhs<'a, D>>(
        &self,
        name: &str,
        mid: Expr<'a, D>,
        lo: B1,
        hi: B2,
    ) -> GdpRangeHandles {
        if let (Some(lower), Some(upper)) = (lo.const_bound(), hi.const_bound()) {
            GdpRangeHandles::Interval(self.add_constraint(
                name,
                crate::Constraint::new(mid.into_function(), crate::Interval { lower, upper }),
            ))
        } else {
            GdpRangeHandles::Split {
                lower: self.add_constraint(format!("{name}_lo"), mid.ge(lo)),
                upper: self.add_constraint(format!("{name}_hi"), mid.le(hi)),
            }
        }
    }

    #[doc(hidden)]
    pub fn __add_range_auto<D: FunctionDegree, B1: IntoRhs<'a, D>, B2: IntoRhs<'a, D>>(
        &self,
        mid: Expr<'a, D>,
        lo: B1,
        hi: B2,
    ) -> GdpRangeHandles {
        if lo.const_bound().is_some() && hi.const_bound().is_some() {
            self.__add_range(&self.__gdp_auto_name("c"), mid, lo, hi)
        } else {
            GdpRangeHandles::Split {
                lower: self.__add_constraint_auto(mid.ge(lo)),
                upper: self.__add_constraint_auto(mid.le(hi)),
            }
        }
    }

    #[doc(hidden)]
    pub fn __add_constraints_over<K: FromIndexKey, C: Into<AlgebraicConstraintIr<'a>>>(
        &self,
        prefix: &str,
        set: &Set<K>,
        rule: impl Fn(K) -> C,
    ) -> GdpFamily<K, DisjunctConstraintHandle> {
        self.model.__gdp_family_over(prefix, set, |k, name| self.add_constraint(name, rule(k)))
    }

    #[doc(hidden)]
    pub fn __add_range_constraints_over<
        K: FromIndexKey,
        D: FunctionDegree,
        B1: IntoRhs<'a, D>,
        B2: IntoRhs<'a, D>,
    >(
        &self,
        prefix: &str,
        set: &Set<K>,
        rule: impl Fn(K) -> (Expr<'a, D>, B1, B2),
    ) -> GdpFamily<K, GdpRangeHandles> {
        self.model.__gdp_family_over(prefix, set, |k, name| {
            let (mid, lo, hi) = rule(k);
            self.__add_range(&name, mid, lo, hi)
        })
    }

    #[doc(hidden)]
    pub fn __gdp_family_over<K: FromIndexKey, T>(
        &self,
        prefix: &str,
        set: &Set<K>,
        rule: impl FnMut(K, String) -> T,
    ) -> GdpFamily<K, T> {
        self.model.__gdp_family_over(prefix, set, rule)
    }

    #[doc(hidden)]
    pub fn __gdp_auto_name(&self, prefix: &str) -> String {
        self.model.next_gdp_auto_name(prefix, Some(self.indicator))
    }

    /// Register a logical decision with a name scoped to this disjunct.
    ///
    /// # Panics
    /// Panics for duplicate names, ID exhaustion, or a transformed context.
    pub fn add_boolean(&self, name: impl Into<SmolStr>) -> BooleanHandle<'a> {
        assert!(
            self.model.gdp.borrow().booleans[self.indicator.index()].state == DisjunctState::Open,
            "cannot edit a transformed disjunct"
        );
        self.model.add_boolean(self.qualify(&name.into()))
    }

    #[doc(hidden)]
    pub fn __gdp_disjunct(
        &self,
        name: impl Into<SmolStr>,
        rule: impl FnOnce(DisjunctContext<'a>),
    ) -> BooleanHandle<'a> {
        let indicator = self.add_boolean(name);
        rule(indicator.context());
        indicator
    }

    #[doc(hidden)]
    pub fn __gdp_disjunction(
        &self,
        name: impl Into<SmolStr>,
        branches: impl IntoIterator<Item = BooleanHandle<'a>>,
        kind: DisjunctionKind,
    ) -> DisjunctionHandle {
        self.model.register_disjunction(
            self.qualify(&name.into()).into(),
            branches.into_iter().collect(),
            kind,
            Some(self.indicator),
        )
    }

    #[doc(hidden)]
    pub fn __gdp_logic(
        &self,
        name: impl Into<SmolStr>,
        expression: impl Into<LogicalExpr>,
    ) -> LogicalConstraintHandle {
        self.model.register_logic(
            self.qualify(&name.into()).into(),
            expression.into(),
            Some(self.indicator),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_clone_preserves_variants_child_order_and_provenance() {
        let model = Model::new("logical cloning");
        let a = model.add_boolean("a");
        let b = model.add_boolean("b");
        let expression = logical_and([
            a.into(),
            true.into(),
            !b,
            a ^ b,
            implies(a, !b),
            iff(b, a),
            logical_or([b, a]),
            logical_and(std::iter::empty::<bool>()),
            logical_or(std::iter::empty::<bool>()),
            exactly(2, [a, a, b]),
            at_most(1, [a, b]),
            at_least(0, [b, a]),
        ]);
        assert_eq!(format!("{:?}", expression.clone()), format!("{expression:?}"));
    }

    #[test]
    fn mixed_deep_logical_trees_clone_and_drop_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(|| {
                let model = Model::new("mixed deep logic");
                let a = model.add_boolean("a");
                let b = model.add_boolean("b");
                let mut expression: LogicalExpr = a.into();

                for index in 0..20_000 {
                    expression = match index % 7 {
                        0 => !expression,
                        1 => logical_and([expression, a.into()]),
                        2 => logical_or([b.into(), expression]),
                        3 => expression ^ a,
                        4 => implies(a, expression),
                        5 => iff(expression, b),
                        _ => exactly(index % 3, [expression, a.into(), false.into()]),
                    };
                }

                let mut copy = expression.clone();
                copy.clone_from(&expression);
                drop(expression);
                drop(copy);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
