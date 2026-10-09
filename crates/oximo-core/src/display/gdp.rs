//! Model-aware rendering of GDP sources and logical expressions.

use std::fmt;

use oximo_expr::{ModelId, render_expr};
use smol_str::SmolStr;

use super::fmt_num;
use crate::Model;
use crate::gdp::{
    BooleanId, Cardinality, DisjunctConstraintId, DisjunctState, DisjunctionId, DisjunctionKind,
    LogicalConstraintId, LogicalExpr, ReformulationState,
};
use crate::var::var_name;

/// Displays a Boolean decision, its binary counterpart, parent, and state.
#[derive(Debug)]
pub struct BooleanDisplay<'a> {
    model: &'a Model,
    id: BooleanId,
}

/// Displays a named conditional algebraic row with its indicator and state.
#[derive(Debug)]
pub struct DisjunctConstraintDisplay<'a> {
    model: &'a Model,
    id: DisjunctConstraintId,
}

/// Displays a disjunction's selection rule, branches, parent, and state.
#[derive(Debug)]
pub struct DisjunctionDisplay<'a> {
    model: &'a Model,
    id: DisjunctionId,
}

/// Displays a named logical assertion with its parent guard and state.
#[derive(Debug)]
pub struct LogicalConstraintDisplay<'a> {
    model: &'a Model,
    id: LogicalConstraintId,
}

/// Displays a logical expression using this model's Boolean names.
/// Formatting rejects leaves belonging to another model.
#[derive(Debug)]
pub struct LogicalExprDisplay<'a> {
    model: &'a Model,
    expression: &'a LogicalExpr,
}

/// Displays the model's GDP source representation, including consumed sources.
#[derive(Debug)]
pub struct GdpDisplay<'a> {
    model: &'a Model,
}

impl Model {
    /// Display a Boolean decision, resolving its binary counterpart and parent.
    #[must_use]
    pub fn display_boolean(&self, id: impl Into<BooleanId>) -> BooleanDisplay<'_> {
        BooleanDisplay { model: self, id: id.into() }
    }

    /// Display a conditional row, resolving expressions and Boolean names.
    #[must_use]
    pub fn display_disjunct_constraint(
        &self,
        id: impl Into<DisjunctConstraintId>,
    ) -> DisjunctConstraintDisplay<'_> {
        DisjunctConstraintDisplay { model: self, id: id.into() }
    }

    /// Display a disjunction's named branches and optional parent guard.
    #[must_use]
    pub fn display_disjunction(&self, id: impl Into<DisjunctionId>) -> DisjunctionDisplay<'_> {
        DisjunctionDisplay { model: self, id: id.into() }
    }

    /// Display a logical assertion using the names of its Boolean decisions.
    #[must_use]
    pub fn display_logical_constraint(
        &self,
        id: impl Into<LogicalConstraintId>,
    ) -> LogicalConstraintDisplay<'_> {
        LogicalConstraintDisplay { model: self, id: id.into() }
    }

    /// Display a logical expression using Boolean names instead of numeric IDs.
    #[must_use]
    pub fn display_logical_expr<'a>(
        &'a self,
        expression: &'a LogicalExpr,
    ) -> LogicalExprDisplay<'a> {
        LogicalExprDisplay { model: self, expression }
    }

    /// Display all GDP sources, with lifecycle states and resolved names.
    /// An empty GDP registry produces no output.
    #[must_use]
    pub fn display_gdp(&self) -> GdpDisplay<'_> {
        GdpDisplay { model: self }
    }
}

fn state_suffix(state: ReformulationState) -> &'static str {
    match state {
        ReformulationState::Pending => " (pending)",
        ReformulationState::Reformulated => " (reformulated)",
    }
}

impl fmt::Display for BooleanDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.model.gdp.borrow();
        let vars = self.model.variables.borrow();
        let boolean = &data.booleans[self.id.index()];
        write!(f, "{}: Boolean ({})", boolean.name, var_name(&vars, boolean.binary))?;

        if let Some(parent) = boolean.parent {
            write!(f, ", parent = {}", data.booleans[parent.index()].name)?;
        }
        f.write_str(match boolean.state() {
            DisjunctState::Open => " (open)",
            DisjunctState::Sealed => " (sealed)",
        })
    }
}

impl fmt::Display for DisjunctConstraintDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.model.gdp.borrow();
        let arena = self.model.arena.borrow();
        let vars = self.model.variables.borrow();
        let row = &data.rows[self.id.index()];
        let expression = render_expr(&arena, row.lhs, &|v| var_name(&vars, v));
        write!(f, "{}: {} -> ", row.name, data.booleans[row.indicator.index()].name)?;

        match (row.lower.is_finite(), row.upper.is_finite()) {
            (true, true) if row.lower.total_cmp(&row.upper).is_eq() => {
                write!(f, "{expression} = {}", fmt_num(row.lower))?;
            }
            (true, true) => {
                write!(f, "{} <= {expression} <= {}", fmt_num(row.lower), fmt_num(row.upper))?;
            }
            (true, false) => write!(f, "{expression} >= {}", fmt_num(row.lower))?,
            (false, true) => write!(f, "{expression} <= {}", fmt_num(row.upper))?,
            (false, false) => write!(f, "{expression} free")?,
        }
        f.write_str(state_suffix(row.state()))
    }
}

impl fmt::Display for DisjunctionDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.model.gdp.borrow();
        let disjunction = &data.disjunctions[self.id.index()];
        write!(f, "{}: ", disjunction.name)?;

        if let Some(parent) = disjunction.parent {
            write!(f, "{} -> ", data.booleans[parent.index()].name)?;
        }

        let kind = match disjunction.kind {
            DisjunctionKind::ExactlyOne => "exactly_one",
            DisjunctionKind::AtLeastOne => "at_least_one",
        };
        write!(f, "{kind} [")?;

        for (index, branch) in disjunction.branches.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            f.write_str(&data.booleans[branch.index()].name)?;
        }
        f.write_str("]")?;
        f.write_str(state_suffix(disjunction.state()))
    }
}

impl fmt::Display for LogicalConstraintDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.model.gdp.borrow();
        let constraint = &data.logical_constraints[self.id.index()];
        write!(f, "{}: ", constraint.name)?;

        if let Some(parent) = constraint.parent {
            write!(f, "{} -> ", data.booleans[parent.index()].name)?;
        }
        write!(f, "{}", self.model.display_logical_expr(&constraint.expression))?;
        f.write_str(state_suffix(constraint.state()))
    }
}

impl fmt::Display for LogicalExprDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.model.gdp.borrow();
        write_logical_expr(f, self.expression, |id, model_id| {
            assert_eq!(
                model_id,
                self.model.id(),
                "logical expression belongs to a different model"
            );
            data.booleans[id.index()].name.clone()
        })
    }
}

impl fmt::Display for LogicalExpr {
    /// Format without a model, identifying Boolean leaves as `boolean[index]`.
    /// Use [`Model::display_logical_expr`] to resolve Boolean names.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_logical_expr(f, self, |id, _| format!("boolean[{}]", id.index()).into())
    }
}

enum Action<'a> {
    Expression(&'a LogicalExpr),
    Text(&'static str),
}

fn push_joined<'a>(stack: &mut Vec<Action<'a>>, terms: &'a [LogicalExpr], separator: &'static str) {
    for (index, term) in terms.iter().enumerate().rev() {
        stack.push(Action::Expression(term));

        if index > 0 {
            stack.push(Action::Text(separator));
        }
    }
}

fn write_logical_expr(
    f: &mut fmt::Formatter<'_>,
    expression: &LogicalExpr,
    resolve: impl Fn(BooleanId, ModelId) -> SmolStr,
) -> fmt::Result {
    let mut stack = vec![Action::Expression(expression)];

    while let Some(action) = stack.pop() {
        let expression = match action {
            Action::Text(text) => {
                f.write_str(text)?;
                continue;
            }
            Action::Expression(expression) => expression,
        };

        match expression {
            LogicalExpr::Literal(value) => write!(f, "{value}")?,
            LogicalExpr::Boolean { id, model_id } => f.write_str(&resolve(*id, *model_id))?,
            LogicalExpr::Not(child) => {
                f.write_str("!")?;
                stack.push(Action::Expression(child));
            }
            LogicalExpr::And(terms) | LogicalExpr::Or(terms) => {
                let is_and = matches!(expression, LogicalExpr::And(_));

                if terms.is_empty() {
                    write!(f, "{is_and}")?;
                } else {
                    f.write_str("(")?;
                    stack.push(Action::Text(")"));
                    push_joined(&mut stack, terms, if is_and { " & " } else { " | " });
                }
            }
            LogicalExpr::Xor(left, right)
            | LogicalExpr::Implies(left, right)
            | LogicalExpr::Iff(left, right) => {
                let separator = match expression {
                    LogicalExpr::Xor(_, _) => " ^ ",
                    LogicalExpr::Implies(_, _) => " -> ",
                    LogicalExpr::Iff(_, _) => " <-> ",
                    _ => unreachable!(),
                };
                f.write_str("(")?;
                stack.push(Action::Text(")"));
                stack.push(Action::Expression(right));
                stack.push(Action::Text(separator));
                stack.push(Action::Expression(left));
            }
            LogicalExpr::Cardinality { kind, count, terms } => {
                let name = match kind {
                    Cardinality::Exactly => "exactly",
                    Cardinality::AtMost => "at_most",
                    Cardinality::AtLeast => "at_least",
                };
                write!(f, "{name}({count}, [")?;
                stack.push(Action::Text("])"));
                push_joined(&mut stack, terms, ", ");
            }
        }
    }

    Ok(())
}

impl fmt::Display for GdpDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.model.gdp.borrow();

        if data.booleans.is_empty()
            && data.rows.is_empty()
            && data.disjunctions.is_empty()
            && data.logical_constraints.is_empty()
        {
            return Ok(());
        }
        writeln!(f, "gdp")?;

        for index in 0..data.booleans.len() {
            let id = BooleanId(u32::try_from(index).expect("Boolean count fits u32"));
            writeln!(f, "  {}", self.model.display_boolean(id))?;
        }

        for index in 0..data.rows.len() {
            let id = DisjunctConstraintId(u32::try_from(index).expect("GDP row count fits u32"));
            writeln!(f, "  {}", self.model.display_disjunct_constraint(id))?;
        }

        for index in 0..data.disjunctions.len() {
            let id = DisjunctionId(u32::try_from(index).expect("disjunction count fits u32"));
            writeln!(f, "  {}", self.model.display_disjunction(id))?;
        }

        for index in 0..data.logical_constraints.len() {
            let id = LogicalConstraintId(u32::try_from(index).expect("logical count fits u32"));
            writeln!(f, "  {}", self.model.display_logical_constraint(id))?;
        }

        Ok(())
    }
}
