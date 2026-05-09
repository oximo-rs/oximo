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
    Sin(ExprId),
    Cos(ExprId),
    Exp(ExprId),
    Log(ExprId),
    Linear { coeffs: Vec<(VarId, f64)>, constant: f64 },
}

#[derive(Clone, Debug, Default)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
}

impl ExprArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Vec::with_capacity(cap) }
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

    pub fn linear(&mut self, coeffs: Vec<(VarId, f64)>, constant: f64) -> ExprId {
        self.push(ExprNode::Linear { coeffs, constant })
    }
}
