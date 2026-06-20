use std::cell::RefCell;

use crate::arena::{ExprArena, ExprId, ExprNode, ParamId, VarId};

/// Lightweight handle to a node in an [`ExprArena`].
///
/// Carries a borrow of the arena (wrapped in `RefCell` so operator overloads
/// can push new nodes during arithmetic). `Expr` is `Copy`, so users freely
/// reuse a variable handle in many constraints.
#[derive(Copy, Clone)]
pub struct Expr<'a> {
    pub id: ExprId,
    pub arena: &'a RefCell<ExprArena>,
}

impl std::fmt::Debug for Expr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expr").field("id", &self.id).finish()
    }
}

impl<'a> Expr<'a> {
    #[inline]
    pub fn new(id: ExprId, arena: &'a RefCell<ExprArena>) -> Self {
        Self { id, arena }
    }

    pub fn constant(arena: &'a RefCell<ExprArena>, v: f64) -> Self {
        let id = arena.borrow_mut().constant(v);
        Self::new(id, arena)
    }

    pub fn from_var(arena: &'a RefCell<ExprArena>, v: VarId) -> Self {
        let id = arena.borrow_mut().var(v);
        Self::new(id, arena)
    }

    /// If this handle is a bare variable, return its [`VarId`].
    /// `None` for compound expressions (sums, products, constants, ...).
    pub fn var_id(self) -> Option<VarId> {
        match self.arena.borrow().get(self.id) {
            ExprNode::Var(id) => Some(*id),
            _ => None,
        }
    }

    /// If this handle is a bare parameter, return its [`ParamId`].
    /// `None` for compound expressions.
    pub fn param_id(self) -> Option<ParamId> {
        match self.arena.borrow().get(self.id) {
            ExprNode::Param(id) => Some(*id),
            _ => None,
        }
    }

    /// Re-bind the parameter this handle references to `value`. Takes effect on
    /// the next extraction/evaluation, which read the value straight from the
    /// arena.
    ///
    /// # Panics
    /// Panics if this handle is not a bare parameter (see [`Self::param_id`]).
    pub fn set_param_value(self, value: f64) {
        let id = self.param_id().expect("set_param_value expects a bare parameter handle");
        self.arena.borrow_mut().set_param_value(id, value);
    }

    pub fn pow(self, exponent: Self) -> Self {
        let id = self.arena.borrow_mut().push(ExprNode::Pow(self.id, exponent.id));
        Self::new(id, self.arena)
    }

    pub fn powi(self, n: i32) -> Self {
        let id = {
            let mut a = self.arena.borrow_mut();
            let exp_id = a.constant(f64::from(n));
            a.push(ExprNode::Pow(self.id, exp_id))
        };
        Self::new(id, self.arena)
    }

    pub fn powf(self, n: f64) -> Self {
        let id = {
            let mut a = self.arena.borrow_mut();
            let exp_id = a.constant(n);
            a.push(ExprNode::Pow(self.id, exp_id))
        };
        Self::new(id, self.arena)
    }

    pub fn sin(self) -> Self {
        let id = self.arena.borrow_mut().push(ExprNode::Sin(self.id));
        Self::new(id, self.arena)
    }

    pub fn cos(self) -> Self {
        let id = self.arena.borrow_mut().push(ExprNode::Cos(self.id));
        Self::new(id, self.arena)
    }

    pub fn exp(self) -> Self {
        let id = self.arena.borrow_mut().push(ExprNode::Exp(self.id));
        Self::new(id, self.arena)
    }

    pub fn log(self) -> Self {
        let id = self.arena.borrow_mut().push(ExprNode::Log(self.id));
        Self::new(id, self.arena)
    }

    pub fn abs(self) -> Self {
        let id = self.arena.borrow_mut().push(ExprNode::Abs(self.id));
        Self::new(id, self.arena)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::Expr;
    use crate::arena::ExprArena;

    #[test]
    fn set_param_value_rebinds_through_handle() {
        let arena = RefCell::new(ExprArena::new());
        let pid = arena.borrow_mut().new_param(0.05);
        let node = arena.borrow_mut().param(pid);
        let p = Expr::new(node, &arena);

        p.set_param_value(0.2);
        assert!((arena.borrow().param_value(pid) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "bare parameter handle")]
    fn set_param_value_panics_on_non_param() {
        let arena = RefCell::new(ExprArena::new());
        let c = Expr::constant(&arena, 1.0);
        c.set_param_value(3.0);
    }
}
