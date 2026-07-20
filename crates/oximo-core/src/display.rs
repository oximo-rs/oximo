//! Human-readable rendering of models, constraints, objectives, and expressions.
//!
//! Everything here is a lazy adapter holding `&Model`. The model's `RefCell`s
//! are borrowed at format time, so do not format one while holding a mutable
//! borrow of the arena or registries.

use std::fmt;

use oximo_expr::{Expr, ExprId, render_expr};

use crate::constraint::{ConstraintId, Sense};
use crate::domain::Domain;
use crate::model::Model;
use crate::objective::ObjectiveSense;
use crate::soc::SocConstraintId;
use crate::var::{Variable, var_name};

/// Compact `f64` rendering (shortest round-trip).
fn fmt_num(v: f64) -> String {
    if v == 0.0 { "0".to_string() } else { format!("{v}") }
}

/// Displays one expression as infix algebra, e.g. `3 x + 4 y - z`.
/// Built by [`Model::display_expr`]/[`Model::display_expr_id`].
#[derive(Debug)]
pub struct ExprDisplay<'a> {
    model: &'a Model,
    id: ExprId,
}

impl fmt::Display for ExprDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arena = self.model.arena.borrow();
        let vars = self.model.variables.borrow();
        f.write_str(&render_expr(&arena, self.id, &|v| var_name(&vars, v)))
    }
}

/// Displays one constraint as `name: lhs <= rhs` (or a range / equality).
/// Built by [`Model::display_constraint`].
#[derive(Debug)]
pub struct ConstraintDisplay<'a> {
    model: &'a Model,
    id: ConstraintId,
}

impl fmt::Display for ConstraintDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arena = self.model.arena.borrow();
        let vars = self.model.variables.borrow();
        let constraints = self.model.constraints.borrow();
        let c = &constraints[self.id.index()];
        let resolve = |v| var_name(&vars, v);
        let expr = render_expr(&arena, c.lhs, &resolve);
        write!(f, "{}: ", c.name)?;
        match c.as_single() {
            Some((Sense::Le, rhs)) => write!(f, "{expr} <= {}", fmt_num(rhs))?,
            Some((Sense::Ge, rhs)) => write!(f, "{expr} >= {}", fmt_num(rhs))?,
            Some((Sense::Eq, rhs)) => write!(f, "{expr} = {}", fmt_num(rhs))?,
            None if c.is_range() => {
                write!(f, "{} <= {expr} <= {}", fmt_num(c.lower), fmt_num(c.upper))?;
            }
            None => write!(f, "{expr} free")?,
        }
        if !c.active {
            f.write_str(" (inactive)")?;
        }
        Ok(())
    }
}

/// Displays the objective as `min <expr>`/`max <expr>`/`feasibility`.
/// Built by [`Model::display_objective`].
#[derive(Debug)]
pub struct ObjectiveDisplay<'a> {
    model: &'a Model,
}

impl fmt::Display for ObjectiveDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.model.is_feasibility() {
            return f.write_str("feasibility");
        }
        // Copy `(sense, expr)` out so the objective borrow ends before the
        // expression rendering takes its own borrows.
        let obj = self.model.objective.borrow().as_ref().map(|o| (o.sense, o.expr));
        let Some((sense, expr)) = obj else {
            return f.write_str("(no objective)");
        };
        let word = match sense {
            ObjectiveSense::Minimize => "min",
            ObjectiveSense::Maximize => "max",
        };
        write!(f, "{word} {}", ExprDisplay { model: self.model, id: expr })
    }
}

/// Displays one second-order cone constraint as `name: ||t1, t2|| <= bound`.
/// Built by [`Model::display_soc`].
#[derive(Debug)]
pub struct SocDisplay<'a> {
    model: &'a Model,
    id: SocConstraintId,
}

impl fmt::Display for SocDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arena = self.model.arena.borrow();
        let vars = self.model.variables.borrow();
        let socs = self.model.soc_constraints.borrow();
        let c = &socs[self.id.index()];
        let resolve = |v| var_name(&vars, v);
        write!(f, "{}: ||", c.name)?;
        for (i, t) in c.terms.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(&render_expr(&arena, *t, &resolve))?;
        }
        write!(f, "|| <= {}", render_expr(&arena, c.bound, &resolve))?;
        if !c.active {
            f.write_str(" (inactive)")?;
        }
        Ok(())
    }
}

/// One `vars` section line: bounds relation plus a domain suffix,
/// e.g. `0 <= y <= 1, binary`.
fn write_var_line(f: &mut fmt::Formatter<'_>, v: &Variable) -> fmt::Result {
    match (v.lb.is_finite(), v.ub.is_finite()) {
        (true, true) if v.lb.total_cmp(&v.ub).is_eq() => {
            write!(f, "{} = {}", v.name, fmt_num(v.lb))?;
        }
        (true, true) => write!(f, "{} <= {} <= {}", fmt_num(v.lb), v.name, fmt_num(v.ub))?,
        (true, false) => write!(f, "{} >= {}", v.name, fmt_num(v.lb))?,
        (false, true) => write!(f, "{} <= {}", v.name, fmt_num(v.ub))?,
        (false, false) => write!(f, "{} free", v.name)?,
    }
    match v.domain {
        Domain::Real => Ok(()),
        Domain::Integer => f.write_str(", integer"),
        Domain::Binary => f.write_str(", binary"),
        Domain::SemiContinuous { threshold } => {
            write!(f, ", semicontinuous(threshold={})", fmt_num(threshold))
        }
        Domain::SemiInteger { threshold } => {
            write!(f, ", semiinteger(threshold={})", fmt_num(threshold))
        }
    }
}

impl Model {
    /// Display adapter for an expression handle, resolving variable names
    /// against this model.
    #[must_use]
    pub fn display_expr(&self, e: Expr<'_>) -> ExprDisplay<'_> {
        self.display_expr_id(e.id)
    }

    /// Display adapter for a raw [`ExprId`] (as stored in constraints and the
    /// objective).
    #[must_use]
    pub fn display_expr_id(&self, id: ExprId) -> ExprDisplay<'_> {
        ExprDisplay { model: self, id }
    }

    /// Display adapter for one algebraic constraint.
    #[must_use]
    pub fn display_constraint(&self, id: ConstraintId) -> ConstraintDisplay<'_> {
        ConstraintDisplay { model: self, id }
    }

    /// Display adapter for the objective (`min <expr>`/`max <expr>`/
    /// `feasibility`/`(no objective)`).
    #[must_use]
    pub fn display_objective(&self) -> ObjectiveDisplay<'_> {
        ObjectiveDisplay { model: self }
    }

    /// Display adapter for one second-order cone constraint.
    #[must_use]
    pub fn display_soc(&self, id: SocConstraintId) -> SocDisplay<'_> {
        SocDisplay { model: self, id }
    }
}

/// Pretty-print the whole model as readable algebra:
///
/// ```text
/// Model 'diet' (LP)
/// min 3 x + 4 y
/// s.t.
///   c1: x + 2 y <= 14
///   c2: 3 x - y >= 0
/// vars
///   x >= 0
///   y >= 0
/// ```
///
/// Sections (`s.t.`, `vars`, `params`) are omitted when empty. The objective
/// line is omitted when no objective was declared.
impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model '{}' ({:?})", self.name, self.kind())?;
        if self.is_feasibility() || self.objective.borrow().is_some() {
            writeln!(f, "{}", self.display_objective())?;
        }
        let n_constraints = self.constraints.borrow().len();
        let n_socs = self.soc_constraints.borrow().len();
        if n_constraints + n_socs > 0 {
            let n_constraints = u32::try_from(n_constraints).expect("constraint count fits u32");
            let n_socs = u32::try_from(n_socs).expect("soc count fits u32");
            writeln!(f, "s.t.")?;
            for i in 0..n_constraints {
                writeln!(f, "  {}", self.display_constraint(ConstraintId(i)))?;
            }
            for i in 0..n_socs {
                writeln!(f, "  {}", self.display_soc(SocConstraintId(i)))?;
            }
        }
        {
            let vars = self.variables.borrow();
            if !vars.is_empty() {
                writeln!(f, "vars")?;
                for v in vars.iter() {
                    f.write_str("  ")?;
                    write_var_line(f, v)?;
                    writeln!(f)?;
                }
            }
        }
        let params = self.parameters.borrow();
        if !params.is_empty() {
            let arena = self.arena.borrow();
            writeln!(f, "params")?;
            for p in params.iter() {
                writeln!(f, "  {} = {}", p.name, fmt_num(arena.param_value(p.id)))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Relate;

    #[test]
    fn full_model_snapshot() {
        let m = Model::new("diet");
        let x = m.__var("x").lb(0.0).build();
        let y = m.__var("y").lb(0.0).build();
        m.__add_constraint("c1", (x + 2.0 * y).le(14.0));
        m.__add_constraint("c2", (3.0 * x - y).ge(0.0));
        m.__minimize(3.0 * x + 4.0 * y);

        let expected = "Model 'diet' (LP)\n\
                        min 3 x + 4 y\n\
                        s.t.\n\
                        \x20 c1: x + 2 y <= 14\n\
                        \x20 c2: 3 x - y >= 0\n\
                        vars\n\
                        \x20 x >= 0\n\
                        \x20 y >= 0\n";
        assert_eq!(format!("{m}"), expected);
    }

    #[test]
    fn constraint_variants_render() {
        let m = Model::new("t");
        let x = m.__var("x").build();
        let eq = m.__add_constraint("e", x.eq(5.0));
        assert_eq!(m.display_constraint(eq).to_string(), "e: x = 5");

        m.__add_range("r", x, 2.0, 10.0);
        let r = m.constraint_id("r").unwrap();
        assert_eq!(m.display_constraint(r).to_string(), "r: 2 <= x <= 10");

        let ge = m.__add_constraint("g", x.ge(1.0));
        m.constraints.borrow_mut()[ge.index()].active = false;
        assert_eq!(m.display_constraint(ge).to_string(), "g: x >= 1 (inactive)");
    }

    #[test]
    fn objective_and_feasibility() {
        let m = Model::new("t");
        let x = m.__var("x").build();
        assert_eq!(m.display_objective().to_string(), "(no objective)");
        m.__maximize(x);
        assert_eq!(m.display_objective().to_string(), "max x");

        let f = Model::new("f");
        f.__feasibility();
        assert_eq!(f.display_objective().to_string(), "feasibility");
        assert!(format!("{f}").contains("feasibility\n"));
    }

    #[test]
    fn var_lines_cover_domains_and_bounds() {
        let m = Model::new("t");
        m.__var("a").bounds(0.0, 5.0).build();
        m.__var("b").binary().build();
        m.__var("c").integer().build();
        m.__var("d").build();
        m.__var("e").ub(3.0).build();
        m.__var("g").fix(2.0).build();
        m.__var("h").lb(0.0).domain(crate::Domain::SemiContinuous { threshold: 2.0 }).build();

        let out = format!("{m}");
        assert!(out.contains("  0 <= a <= 5\n"), "{out}");
        assert!(out.contains("  0 <= b <= 1, binary\n"), "{out}");
        assert!(out.contains("  c free, integer\n"), "{out}");
        assert!(out.contains("  d free\n"), "{out}");
        assert!(out.contains("  e <= 3\n"), "{out}");
        assert!(out.contains("  g = 2\n"), "{out}");
        assert!(out.contains("  h >= 0, semicontinuous(threshold=2)\n"), "{out}");
    }

    #[test]
    fn params_section_shows_current_values() {
        let m = Model::new("t");
        let x = m.__var("x").build();
        let price = m.__param("price", 4.0);
        m.__minimize(price * x);
        let out = format!("{m}");
        assert!(out.contains("min 4 x\n"), "{out}");
        assert!(out.contains("params\n  price = 4\n"), "{out}");

        price.set_param_value(7.5);
        let out = format!("{m}");
        assert!(out.contains("min 7.5 x\n"), "{out}");
        assert!(out.contains("params\n  price = 7.5\n"), "{out}");
    }

    #[test]
    fn soc_row_renders_in_model_display() {
        let m = Model::new("t");
        let x = m.__var("x").build();
        let y = m.__var("y").build();
        let t = m.__var("t").lb(0.0).build();
        let id = m.add_soc_constraint("q1", [x, y], t);
        assert_eq!(m.display_soc(id).to_string(), "q1: ||x, y|| <= t");
        assert!(format!("{m}").contains("  q1: ||x, y|| <= t\n"));
    }

    #[test]
    fn display_expr_renders_nonlinear() {
        let m = Model::new("t");
        let x = m.__var("x").build();
        let y = m.__var("y").build();
        let e = x * y - y;
        assert_eq!(m.display_expr(e).to_string(), "-y + x * y");
    }
}
