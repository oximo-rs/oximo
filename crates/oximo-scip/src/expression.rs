//! Only transformations documented by SCIP/russcip's nonlinear manuals.
use oximo_expr::{ExprArena, ExprId, ExprNode, UnaryOp, VarId};
use oximo_solver::SolverError;
use russcip::{Expr, Variable};
use rustc_hash::FxHashMap;

#[derive(Clone, Debug, PartialEq)]
enum Op {
    Constant(f64),
    Variable(VarId),
    Linear(Vec<(VarId, f64)>, f64),
    Add(Vec<usize>),
    Mul(Vec<usize>),
    Unary(UnaryOp, usize),
    Pow(usize, f64),
    Div(usize, usize),
}

/// A flat postorder expression snapshot.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Formula(Vec<Op>);

fn unsupported(operator: &'static str) -> SolverError {
    SolverError::UnsupportedNonlinearOperator { backend: "SCIP", operator }
}

fn validate_node(node: &ExprNode) -> Result<(), SolverError> {
    match node {
        ExprNode::Unary(op, _)
            if !crate::supports_operator(oximo_solver::ExpressionOperator::Unary(*op)) =>
        {
            Err(unsupported(op.name()))
        }
        ExprNode::Atan2(_, _) => Err(unsupported("atan2")),
        ExprNode::Min(_) => Err(unsupported("min")),
        ExprNode::Max(_) => Err(unsupported("max")),
        _ => Ok(()),
    }
}

impl Formula {
    pub(crate) fn prepare(
        arena: &ExprArena,
        root: ExprId,
        nvars: usize,
    ) -> Result<Self, SolverError> {
        let mut indices = FxHashMap::default();
        let mut ops = Vec::new();
        let mut stack = vec![(root, false)];
        while let Some((id, visited)) = stack.pop() {
            if indices.contains_key(&id) {
                continue;
            }
            // Validate before constant folding, so unsupported operators do
            // not disappear merely because their current arguments are const.
            validate_node(arena.get(id))?;
            let children: Vec<ExprId> = match arena.get(id) {
                ExprNode::Add(v) | ExprNode::Mul(v) | ExprNode::Min(v) | ExprNode::Max(v) => {
                    v.to_vec()
                }
                ExprNode::Unary(_, v) => vec![*v],
                ExprNode::Pow(a, b) | ExprNode::Div(a, b) | ExprNode::Atan2(a, b) => vec![*a, *b],
                _ => vec![],
            };
            if !visited {
                stack.push((id, true));
                stack.extend(children.iter().rev().map(|&c| (c, false)));
                continue;
            }
            let args: Vec<usize> = children.iter().map(|c| indices[c]).collect();
            let constants: Option<Vec<f64>> = args
                .iter()
                .map(|&i| match ops[i] {
                    Op::Constant(v) => Some(v),
                    _ => None,
                })
                .collect();
            let constant = match (arena.get(id), constants) {
                (ExprNode::Const(v), _) => Some(*v),
                (ExprNode::Param(p), _) => Some(arena.param_value(*p)),
                (ExprNode::Linear { coeffs, constant }, _) if coeffs.is_empty() => Some(*constant),
                (ExprNode::Add(_), Some(v)) => Some(v.iter().sum()),
                (ExprNode::Mul(_), Some(v)) => Some(v.iter().product()),
                (ExprNode::Unary(op, _), Some(v)) => Some(op.apply(v[0])),
                (ExprNode::Pow(_, _), Some(v)) => Some(v[0].powf(v[1])),
                (ExprNode::Div(_, _), Some(v)) => Some(v[0] / v[1]),
                _ => None,
            };
            let op = if let Some(v) = constant {
                crate::finite(v)?;
                Op::Constant(v)
            } else {
                match arena.get(id) {
                    ExprNode::Var(v) => {
                        check_var(*v, nvars)?;
                        Op::Variable(*v)
                    }
                    ExprNode::Linear { coeffs, constant } => {
                        crate::finite(*constant)?;
                        for &(v, c) in coeffs {
                            check_var(v, nvars)?;
                            crate::finite(c)?;
                        }
                        Op::Linear(coeffs.clone(), *constant)
                    }
                    ExprNode::Add(_) => Op::Add(args),
                    ExprNode::Mul(_) => Op::Mul(args),
                    ExprNode::Unary(op, _) => {
                        if !crate::supports_operator(oximo_solver::ExpressionOperator::Unary(*op)) {
                            return Err(unsupported(op.name()));
                        }
                        Op::Unary(*op, args[0])
                    }
                    ExprNode::Pow(_, _) => match ops[args[1]] {
                        Op::Constant(e) => Op::Pow(args[0], e),
                        _ => return Err(unsupported("variable-exponent pow")),
                    },
                    ExprNode::Div(_, _) => Op::Div(args[0], args[1]),
                    ExprNode::Atan2(_, _) | ExprNode::Min(_) | ExprNode::Max(_) => {
                        unreachable!("validated operator")
                    }
                    ExprNode::Const(_) | ExprNode::Param(_) => unreachable!(),
                }
            };
            indices.insert(id, ops.len());
            ops.push(op);
        }
        Ok(Self(ops))
    }

    pub(crate) fn native(&self, vars: &[Variable]) -> Expr {
        let mut values: Vec<Expr> = Vec::with_capacity(self.0.len());
        for op in &self.0 {
            let e = match op {
                Op::Constant(v) => Expr::constant(*v),
                Op::Variable(v) => Expr::var(&vars[v.index()]),
                Op::Linear(cs, k) => {
                    Expr::sum_weighted(cs.iter().map(|(v, c)| (*c, Expr::var(&vars[v.index()]))))
                        + *k
                }
                Op::Add(cs) => Expr::sum(cs.iter().map(|&i| values[i].clone())),
                Op::Mul(cs) => Expr::product(cs.iter().map(|&i| values[i].clone())),
                Op::Unary(op, i) => {
                    let v = values[*i].clone();
                    match op {
                        UnaryOp::Neg => -v,
                        UnaryOp::Abs => Expr::abs(v),
                        UnaryOp::Sqrt => Expr::pow(v, 0.5),
                        UnaryOp::Exp => Expr::exp(v),
                        UnaryOp::Log => Expr::log(v),
                        UnaryOp::Sin => Expr::sin(v),
                        UnaryOp::Cos => Expr::cos(v),
                        _ => unreachable!("validated operator"),
                    }
                }
                Op::Pow(i, e) => Expr::pow(values[*i].clone(), *e),
                Op::Div(a, b) => values[*a].clone() / values[*b].clone(),
            };
            values.push(e);
        }
        values.pop().expect("root expression")
    }
}

fn check_var(v: VarId, nvars: usize) -> Result<(), SolverError> {
    if v.index() >= nvars {
        Err(crate::backend("expression refers to an unknown variable"))
    } else {
        Ok(())
    }
}
