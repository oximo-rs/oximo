use smallvec::SmallVec;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

impl ExprId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

impl VarId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(pub u32);

impl ParamId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

pub type Children = SmallVec<[ExprId; 4]>;

/// Here we use a linear fast-path: `sum(coeff * var) + constant`.
/// Built by the operator overloads when all children are linear,
/// so LP/MILP construction never walks an `Add(Mul(Const, Var), ...)` tree.

#[derive(Clone, Debug)]
pub enum ExprNode {
    Const(f64),
    Var(VarId),
    Param(ParamId),
    Add(Children),
    Mul(Children),
    Neg(ExprId),
    Pow(ExprId, ExprId),
    Div(ExprId, ExprId),
    Sin(ExprId),
    Cos(ExprId),
    Exp(ExprId),
    Log(ExprId),
    Abs(ExprId),
    Linear { coeffs: Vec<(VarId, f64)>, constant: f64 },
}

#[derive(Clone, Debug, Default)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
    param_values: Vec<f64>,
}

impl ExprArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Vec::with_capacity(cap), param_values: Vec::new() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// # Panics
    ///
    /// Panics if the number of expressions exceeds `u32::MAX` (expression arena overflow).
    pub fn push(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(u32::try_from(self.nodes.len()).expect("expression arena overflow"));
        self.nodes.push(node);
        id
    }

    #[inline]
    pub fn get(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, id: ExprId) -> &mut ExprNode {
        &mut self.nodes[id.index()]
    }

    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }

    pub fn constant(&mut self, v: f64) -> ExprId {
        self.push(ExprNode::Const(v))
    }

    pub fn var(&mut self, v: VarId) -> ExprId {
        self.push(ExprNode::Var(v))
    }

    pub fn param(&mut self, p: ParamId) -> ExprId {
        self.push(ExprNode::Param(p))
    }

    /// Allocate a fresh parameter initialized to `value`, returning its
    /// [`ParamId`]. Push a [`ExprNode::Param`] with [`Self::param`] to reference
    /// it inside an expression.
    ///
    /// # Panics
    ///
    /// Panics if the number of parameters exceeds `u32::MAX`.
    pub fn new_param(&mut self, value: f64) -> ParamId {
        let id = ParamId(u32::try_from(self.param_values.len()).expect("parameter arena overflow"));
        self.param_values.push(value);
        id
    }

    #[inline]
    pub fn num_params(&self) -> usize {
        self.param_values.len()
    }

    /// Current value bound to parameter `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` was not allocated by [`Self::new_param`] on this arena.
    #[inline]
    pub fn param_value(&self, p: ParamId) -> f64 {
        self.param_values[p.index()]
    }

    /// Look up the value of `p`, returning `None` if `p` is out of range.
    #[inline]
    pub fn try_param_value(&self, p: ParamId) -> Option<f64> {
        self.param_values.get(p.index()).copied()
    }

    /// Re-bind parameter `p` to `value`. Takes effect on the next extraction or
    /// evaluation.
    ///
    /// # Panics
    ///
    /// Panics if `p` was not allocated by [`Self::new_param`] on this arena.
    #[inline]
    pub fn set_param_value(&mut self, p: ParamId, value: f64) {
        self.param_values[p.index()] = value;
    }

    pub fn linear(&mut self, coeffs: Vec<(VarId, f64)>, constant: f64) -> ExprId {
        self.push(ExprNode::Linear { coeffs, constant })
    }
}
