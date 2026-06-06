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
