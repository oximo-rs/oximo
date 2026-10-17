use std::fmt;

use oximo_expr::ExprId;

/// Whether the model minimizes or maximizes its objective.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

impl fmt::Display for ObjectiveSense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Minimize => "minimize",
            Self::Maximize => "maximize",
        })
    }
}

/// Algebraic view of [`SemanticObjective`].
#[derive(Clone, Debug)]
pub struct Objective {
    /// Root node of the objective expression in the model's [`oximo_expr::ExprArena`].
    pub expr: ExprId,
    pub sense: ObjectiveSense,
}

/// A scalar-valued mathematical function that can be optimized.
pub trait ObjectiveFunction: crate::Function {}

/// Authoritative objective semantics, registered together with a scalar expression view.
#[derive(Clone, Debug)]
pub struct SemanticObjective {
    pub function: crate::FunctionValue,
    pub sense: ObjectiveSense,
}

pub trait IntoObjectiveExpression<'a> {
    type Degree: crate::function_set::FunctionDegree;

    fn into_objective_expression(
        self,
        arena: &'a oximo_expr::ExprArenaCell,
    ) -> oximo_expr::Expr<'a, Self::Degree>;
}

impl<'a, D: crate::function_set::FunctionDegree> IntoObjectiveExpression<'a>
    for oximo_expr::Expr<'a, D>
{
    type Degree = D;

    fn into_objective_expression(self, _arena: &'a oximo_expr::ExprArenaCell) -> Self {
        self
    }
}

impl<'a> IntoObjectiveExpression<'a> for f64 {
    type Degree = oximo_expr::Constant;

    fn into_objective_expression(
        self,
        arena: &'a oximo_expr::ExprArenaCell,
    ) -> oximo_expr::Expr<'a, oximo_expr::Constant> {
        oximo_expr::Expr::constant(arena, self)
    }
}

impl<'a> IntoObjectiveExpression<'a> for i32 {
    type Degree = oximo_expr::Constant;

    fn into_objective_expression(
        self,
        arena: &'a oximo_expr::ExprArenaCell,
    ) -> oximo_expr::Expr<'a, oximo_expr::Constant> {
        oximo_expr::Expr::constant(arena, f64::from(self))
    }
}

#[cfg(test)]
mod tests {
    use super::ObjectiveSense;

    #[test]
    fn display_uses_user_facing_ascii_labels() {
        let labels = [ObjectiveSense::Minimize.to_string(), ObjectiveSense::Maximize.to_string()];
        assert_eq!(labels, ["minimize", "maximize"]);
        assert!(labels.iter().all(|label| label.is_ascii()));
    }
}
