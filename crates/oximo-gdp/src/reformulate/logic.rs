//! Logical assertion and shared predicate compilation.

use super::GdpError;
use super::plan::{Linear, PlanBuilder};
use oximo_core::Model;
use oximo_core::gdp::{
    BooleanId, Cardinality, GdpLogicalArtifacts, GdpSnapshot, LogicalConstraintId, LogicalExpr,
};
use rustc_hash::FxHashMap;

#[derive(Debug, Hash, PartialEq, Eq)]
enum LogicKey {
    Not(usize),
    And(Vec<usize>),
    Or(Vec<usize>),
    Xor(usize, usize),
    Implies(usize, usize),
    Iff(usize, usize),
    Cardinality(Cardinality, usize, Vec<usize>),
}

fn connective_terms(root: &LogicalExpr) -> Vec<&LogicalExpr> {
    let mut terms = Vec::new();
    let mut stack = vec![root];

    while let Some(term) = stack.pop() {
        match (root, term) {
            (LogicalExpr::And(_), LogicalExpr::And(children))
            | (LogicalExpr::Or(_), LogicalExpr::Or(children)) => {
                stack.extend(children.iter().rev());
            }
            _ => terms.push(term),
        }
    }
    terms
}

#[derive(Debug)]
pub(super) struct LogicCompiler {
    cache: FxHashMap<LogicKey, (usize, Linear)>,
    next_key: usize,
}

impl LogicCompiler {
    pub(super) fn new(next_key: usize) -> Self {
        Self { cache: FxHashMap::default(), next_key }
    }
}

impl PlanBuilder {
    pub(super) fn plan_logic(
        &mut self,
        model: &Model,
        data: &GdpSnapshot,
        logical_start: usize,
    ) -> Result<(), GdpError> {
        for (i, l) in data.logical_constraints.iter().enumerate().skip(logical_start) {
            let start = self.row_count();
            self.assert_logic(model, data, &l.expression, l.parent)?;
            let ids = self.row_ids(start)?;
            self.report.logical_constraints.push(GdpLogicalArtifacts {
                source: LogicalConstraintId(u32::try_from(i).map_err(|_| GdpError::Capacity)?),
                constraints: ids,
            });
        }

        Ok(())
    }

    fn and(&mut self, model: &Model, values: Vec<Linear>) -> Result<Linear, GdpError> {
        if values.is_empty() {
            return Ok(Linear::constant(1.0));
        }

        if values.len() == 1 {
            return Ok(values.into_iter().next().expect("one term"));
        }

        let z = self.auxiliary(model)?;
        let count = f64::from(u32::try_from(values.len()).map_err(|_| GdpError::Capacity)?);
        let mut sum = Linear::constant(0.0);

        for value in values {
            self.row(
                model,
                None,
                z.clone().add(value.clone().scaled(-1.0)),
                f64::NEG_INFINITY,
                0.0,
            )?;
            sum = sum.add(value);
        }

        self.row(model, None, z.clone().add(sum.scaled(-1.0)), 1.0 - count, f64::INFINITY)?;

        Ok(z)
    }

    fn or(&mut self, model: &Model, values: Vec<Linear>) -> Result<Linear, GdpError> {
        Ok(self.and(model, values.into_iter().map(Linear::not).collect())?.not())
    }

    fn xor(&mut self, model: &Model, a: Linear, b: Linear) -> Result<Linear, GdpError> {
        let z = self.auxiliary(model)?;
        self.row(
            model,
            None,
            z.clone().add(a.clone().scaled(-1.0)).add(b.clone()),
            0.0,
            f64::INFINITY,
        )?;
        self.row(
            model,
            None,
            z.clone().add(b.clone().scaled(-1.0)).add(a.clone()),
            0.0,
            f64::INFINITY,
        )?;
        self.row(
            model,
            None,
            z.clone().add(a.clone().scaled(-1.0)).add(b.clone().scaled(-1.0)),
            f64::NEG_INFINITY,
            0.0,
        )?;
        self.row(model, None, z.clone().add(a).add(b), f64::NEG_INFINITY, 2.0)?;

        Ok(z)
    }

    fn cardinality(
        &mut self,
        model: &Model,
        kind: Cardinality,
        count: usize,
        values: Vec<Linear>,
    ) -> Result<Linear, GdpError> {
        let n = values.len();

        if count > n {
            return Ok(Linear::constant(f64::from(kind == Cardinality::AtMost)));
        }

        if kind == Cardinality::AtLeast && count == 0 || kind == Cardinality::AtMost && count == n {
            return Ok(Linear::constant(1.0));
        }

        if kind == Cardinality::Exactly {
            let lo = self.cardinality(model, Cardinality::AtLeast, count, values.clone())?;
            let hi = self.cardinality(model, Cardinality::AtMost, count, values)?;
            return self.and(model, vec![lo, hi]);
        }

        let z = self.auxiliary(model)?;
        let sum = values.into_iter().fold(Linear::default(), Linear::add);
        let (n, k) = (
            f64::from(u32::try_from(n).map_err(|_| GdpError::Capacity)?),
            f64::from(u32::try_from(count).map_err(|_| GdpError::Capacity)?),
        );

        match kind {
            Cardinality::AtMost => {
                self.row(
                    model,
                    None,
                    sum.clone().add(z.clone().scaled(n - k)),
                    f64::NEG_INFINITY,
                    n,
                )?;
                self.row(model, None, sum.add(z.clone().scaled(k + 1.0)), k + 1.0, f64::INFINITY)?;
            }
            Cardinality::AtLeast => {
                self.row(model, None, sum.clone().add(z.clone().scaled(-k)), 0.0, f64::INFINITY)?;
                self.row(
                    model,
                    None,
                    sum.add(z.clone().scaled(-(n - k + 1.0))),
                    f64::NEG_INFINITY,
                    k - 1.0,
                )?;
            }
            Cardinality::Exactly => unreachable!(),
        }

        Ok(z)
    }

    // An asserted predicate only needs the implication from its context.
    pub(super) fn assert_logic(
        &mut self,
        model: &Model,
        data: &GdpSnapshot,
        root: &LogicalExpr,
        parent_id: Option<BooleanId>,
    ) -> Result<(), GdpError> {
        let parent = parent_id.map_or_else(
            || Linear::constant(1.0),
            |p| Linear::variable(data.booleans[p.index()].binary),
        );

        if let LogicalExpr::And(_) = root
            && parent_id.is_some()
        {
            // Each child must dominate a fractional parent as well.
            for term in connective_terms(root) {
                let value = self.logic(model, data, term)?;
                self.row(model, None, value.add(parent.clone().scaled(-1.0)), 0.0, f64::INFINITY)?;
            }

            return Ok(());
        }

        let predicate = match root {
            LogicalExpr::And(_) => {
                let terms = connective_terms(root);
                Some((Cardinality::AtLeast, terms.len(), terms))
            }
            LogicalExpr::Or(_) => Some((Cardinality::AtLeast, 1, connective_terms(root))),
            LogicalExpr::Cardinality { kind, count, terms } => {
                Some((*kind, *count, terms.iter().collect()))
            }
            _ => None,
        };

        if let Some((kind, count, terms)) = predicate {
            self.assert_cardinality(model, data, kind, count, &terms, parent_id)?;
        } else if let LogicalExpr::Implies(a, b) | LogicalExpr::Iff(a, b) | LogicalExpr::Xor(a, b) =
            root
        {
            let a = self.logic(model, data, a)?;
            let b = self.logic(model, data, b)?;

            match root {
                LogicalExpr::Implies(_, _) => {
                    self.row(
                        model,
                        None,
                        b.add(a.scaled(-1.0)).add(parent.scaled(-1.0)),
                        -1.0,
                        f64::INFINITY,
                    )?;
                }
                LogicalExpr::Iff(_, _) if parent_id.is_none() => {
                    self.row(model, None, a.add(b.scaled(-1.0)), 0.0, 0.0)?;
                }
                LogicalExpr::Xor(_, _) if parent_id.is_none() => {
                    self.row(model, None, a.add(b), 1.0, 1.0)?;
                }
                LogicalExpr::Iff(_, _) => {
                    let difference = a.add(b.scaled(-1.0));
                    self.row(
                        model,
                        None,
                        difference.clone().add(parent.clone()),
                        f64::NEG_INFINITY,
                        1.0,
                    )?;
                    self.row(
                        model,
                        None,
                        difference.scaled(-1.0).add(parent),
                        f64::NEG_INFINITY,
                        1.0,
                    )?;
                }
                LogicalExpr::Xor(_, _) => {
                    let sum = a.add(b);
                    self.row(
                        model,
                        None,
                        sum.clone().add(parent.clone().scaled(-1.0)),
                        0.0,
                        f64::INFINITY,
                    )?;
                    self.row(model, None, sum.add(parent), f64::NEG_INFINITY, 2.0)?;
                }
                _ => unreachable!(),
            }
        } else {
            let value = self.logic(model, data, root)?;
            self.row(model, None, value.add(parent.scaled(-1.0)), 0.0, f64::INFINITY)?;
        }

        Ok(())
    }

    fn assert_cardinality(
        &mut self,
        model: &Model,
        data: &GdpSnapshot,
        kind: Cardinality,
        count: usize,
        terms: &[&LogicalExpr],
        parent_id: Option<BooleanId>,
    ) -> Result<(), GdpError> {
        let n = terms.len();

        if kind == Cardinality::AtMost && count >= n || kind == Cardinality::AtLeast && count == 0 {
            return Ok(());
        }

        let parent = parent_id.map_or_else(
            || Linear::constant(1.0),
            |p| Linear::variable(data.booleans[p.index()].binary),
        );

        if count > n {
            self.row(model, None, parent, f64::NEG_INFINITY, 0.0)?;
            return Ok(());
        }

        let (n, k) = (
            f64::from(u32::try_from(n).map_err(|_| GdpError::Capacity)?),
            f64::from(u32::try_from(count).map_err(|_| GdpError::Capacity)?),
        );
        let mut sum = Linear::default();

        for term in terms {
            sum = sum.add(self.logic(model, data, term)?);
        }

        if parent_id.is_none() {
            self.row(
                model,
                None,
                sum,
                if kind == Cardinality::AtMost { f64::NEG_INFINITY } else { k },
                if kind == Cardinality::AtLeast { f64::INFINITY } else { k },
            )?;
        } else {
            if kind != Cardinality::AtMost {
                self.row(
                    model,
                    None,
                    sum.clone().add(parent.clone().scaled(-k)),
                    0.0,
                    f64::INFINITY,
                )?;
            }

            if kind != Cardinality::AtLeast {
                self.row(model, None, sum.add(parent.scaled(n - k)), f64::NEG_INFINITY, n)?;
            }
        }

        Ok(())
    }

    fn logic(
        &mut self,
        model: &Model,
        data: &GdpSnapshot,
        root: &LogicalExpr,
    ) -> Result<Linear, GdpError> {
        let mut stack = vec![(root, None)];
        let mut values: Vec<(usize, Linear)> = Vec::new();

        while let Some((e, count)) = stack.pop() {
            let Some(count) = count else {
                let children: Vec<_> = match e {
                    LogicalExpr::And(_) | LogicalExpr::Or(_) => connective_terms(e),
                    LogicalExpr::Cardinality { terms, .. } => terms.iter().collect(),
                    LogicalExpr::Not(e) => vec![e.as_ref()],
                    LogicalExpr::Xor(a, b)
                    | LogicalExpr::Implies(a, b)
                    | LogicalExpr::Iff(a, b) => vec![a.as_ref(), b.as_ref()],
                    LogicalExpr::Literal(b) => {
                        values.push((usize::from(*b), Linear::constant(f64::from(*b))));
                        continue;
                    }
                    LogicalExpr::Boolean { id, .. } => {
                        values.push((
                            id.index() + 2,
                            Linear::variable(data.booleans[id.index()].binary),
                        ));
                        continue;
                    }
                };
                stack.push((e, Some(children.len())));
                stack.extend(children.into_iter().rev().map(|child| (child, None)));
                continue;
            };

            let children = values.split_off(values.len() - count);
            let keys = || children.iter().map(|(key, _)| *key).collect();
            let key = match e {
                LogicalExpr::Not(_) => LogicKey::Not(children[0].0),
                LogicalExpr::And(_) => LogicKey::And(keys()),
                LogicalExpr::Or(_) => LogicKey::Or(keys()),
                LogicalExpr::Cardinality { kind, count, .. } => {
                    LogicKey::Cardinality(*kind, *count, keys())
                }
                LogicalExpr::Xor(_, _) => LogicKey::Xor(children[0].0, children[1].0),
                LogicalExpr::Implies(_, _) => LogicKey::Implies(children[0].0, children[1].0),
                LogicalExpr::Iff(_, _) => LogicKey::Iff(children[0].0, children[1].0),
                _ => unreachable!("leaves handled before traversal"),
            };

            if let Some(value) = self.logic.cache.get(&key) {
                values.push(value.clone());
                continue;
            }

            let mut children: Vec<_> = children.into_iter().map(|(_, value)| value).collect();
            let value = match e {
                LogicalExpr::Not(_) => children.pop().expect("one child").not(),
                LogicalExpr::And(_) => self.and(model, children)?,
                LogicalExpr::Or(_) => self.or(model, children)?,
                LogicalExpr::Cardinality { kind, count, .. } => {
                    self.cardinality(model, *kind, *count, children)?
                }
                LogicalExpr::Xor(_, _) | LogicalExpr::Implies(_, _) | LogicalExpr::Iff(_, _) => {
                    let b = children.pop().expect("second child");
                    let a = children.pop().expect("first child");

                    match e {
                        LogicalExpr::Xor(_, _) => self.xor(model, a, b)?,
                        LogicalExpr::Implies(_, _) => self.or(model, vec![a.not(), b])?,
                        LogicalExpr::Iff(_, _) => self.xor(model, a, b)?.not(),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };

            let token = self.logic.next_key;
            self.logic.next_key += 1;
            self.logic.cache.insert(key, (token, value.clone()));
            values.push((token, value));
        }

        Ok(values.pop().expect("logical root").1)
    }
}
