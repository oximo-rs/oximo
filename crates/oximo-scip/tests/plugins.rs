#![expect(clippy::many_single_char_names, reason = "mathematical test models")]
use oximo_core::prelude::*;
use oximo_scip::*;
use oximo_solver::{PersistentSolver, Solver, SolverError};
use std::{cell::Cell, cmp::Ordering, rc::Rc};

fn options() -> ScipOptions {
    ScipOptions::default()
        .presolving(ScipSetting::Off)
        .separating(ScipSetting::Off)
        .heuristics(ScipSetting::Off)
}
fn model() -> (Model, VarId, VarId) {
    let m = Model::new("plugin");
    variable!(m, x, Bin);
    variable!(m, y, Bin);
    constraint!(m, cap, x + y <= 1.5);
    objective!(m, Max, x + y);
    let ids = (x.var_id().unwrap(), y.var_id().unwrap());
    (m, ids.0, ids.1)
}
fn settings(name: &str) -> ScipPluginOptions {
    let mut s = ScipPluginOptions::new(name);
    s.priority = 1_000_000;
    s
}

struct Events(Rc<Cell<usize>>, VarId);
impl ScipEventHandler for Events {
    fn execute(&mut self, ctx: &mut ScipContext<'_>, _: EventMask) -> Result<(), SolverError> {
        self.0.set(self.0.get() + 1);
        assert!(ctx.variable(self.1).is_some());
        Ok(())
    }
}
#[test]
fn events_and_factory_rebuilds() {
    let (m, x, _) = model();
    let count = Rc::new(Cell::new(0));
    let calls = count.clone();
    let builds = Rc::new(Cell::new(0));
    let b = builds.clone();
    let solver = Scip::new().with_plugins(move |r| {
        b.set(b.get() + 1);
        r.event_handler(settings("events"), EventMask::NODE_FOCUSED, Events(calls.clone(), x))
    });
    let mut resident = solver.persistent();
    resident.solve(&m, &options()).unwrap();
    resident.solve(&m, &options()).unwrap();
    assert!(count.get() > 0);
    assert_eq!(builds.get(), 2);
    assert_eq!(resident.build_count(), 2);
}

struct PanicEvent;
impl ScipEventHandler for PanicEvent {
    fn execute(&mut self, _: &mut ScipContext<'_>, _: EventMask) -> Result<(), SolverError> {
        panic!("test callback failure")
    }
}
#[test]
fn callback_panic_never_crosses_c_boundary() {
    let (m, _, _) = model();
    let mut s = Scip::new()
        .with_plugins(|r| r.event_handler(settings("panic"), EventMask::NODE_FOCUSED, PanicEvent));
    let e = s.solve(&m, &options()).unwrap_err();
    assert!(e.to_string().contains("test callback failure"));
    assert!(Scip::new().solve(&m, &options()).unwrap().has_solution());
}

struct Cut {
    vars: [VarId; 2],
    calls: Rc<Cell<usize>>,
}
impl ScipSeparator for Cut {
    fn execute(
        &mut self,
        c: &mut ScipSeparationContext<'_>,
    ) -> Result<SeparationResult, SolverError> {
        self.calls.set(self.calls.get() + 1);
        if self.vars.iter().map(|&v| c.context.value(v).unwrap()).sum::<f64>() > 1.01 {
            let cutoff = c.add_cut(&ScipLinearRow {
                name: "rounding".into(),
                coefficients: self.vars.iter().map(|&v| (v.into(), 1.0)).collect(),
                lower: f64::NEG_INFINITY,
                upper: 1.0,
                local: false,
            })?;
            Ok(if cutoff { SeparationResult::Cutoff } else { SeparationResult::Separated })
        } else {
            Ok(SeparationResult::DidNotFind)
        }
    }
}
#[test]
fn separator_adds_an_oximo_id_cut() {
    let (m, x, y) = model();
    let calls = Rc::new(Cell::new(0));
    let c = calls.clone();
    let mut s = Scip::new().with_plugins(move |r| {
        r.separator(settings("rounding"), Cut { vars: [x, y], calls: c.clone() })
    });
    let out = s.solve(&m, &options()).unwrap();
    assert_eq!(out.objective(), Some(1.0));
    assert!(calls.get() > 0);
}

struct Branch(Rc<Cell<usize>>, [VarId; 2]);
impl ScipBranchRule for Branch {
    fn execute(
        &mut self,
        context: &mut ScipContext<'_>,
        c: &[ScipBranchCandidate],
    ) -> Result<ScipBranchingResult, SolverError> {
        self.0.set(self.0.get() + 1);
        assert!(c.iter().all(|c| c.variable.is_some()));
        for &id in &self.1 {
            let native = context.variable(id).unwrap();
            assert_eq!(context.identify(&native), Some(ScipVarId::Original(id)));
        }
        Ok(ScipBranchingResult::BranchOn(0))
    }
}
#[test]
fn custom_branching_sees_original_ids() {
    let (m, x, y) = model();
    let calls = Rc::new(Cell::new(0));
    let c = calls.clone();
    let mut s = Scip::new()
        .with_plugins(move |r| r.branch_rule(settings("branch"), Branch(c.clone(), [x, y])));
    s.solve(
        &m,
        &options().int_param("propagating/maxrounds", 0).int_param("propagating/maxroundsroot", 0),
    )
    .unwrap();
    assert!(calls.get() > 0);
}

struct Heur {
    var: VarId,
    calls: Rc<Cell<usize>>,
}
impl ScipHeuristic for Heur {
    fn execute(
        &mut self,
        c: &mut ScipContext<'_>,
        _: HeurTiming,
        _: bool,
    ) -> Result<HeurResult, SolverError> {
        self.calls.set(self.calls.get() + 1);
        Ok(if c.submit_solution(&[(self.var.into(), 1.0)])? {
            HeurResult::FoundSol
        } else {
            HeurResult::DidNotRun
        })
    }
}
#[test]
fn heuristic_submits_original_values() {
    let (m, x, _) = model();
    let calls = Rc::new(Cell::new(0));
    let c = calls.clone();
    let mut s = Scip::new().with_plugins(move |r| {
        r.heuristic(settings("candidate"), Heur { var: x, calls: c.clone() })
    });
    s.solve(&m, &options()).unwrap();
    assert!(calls.get() > 0);
}

struct Lazy([VarId; 2]);
impl ScipConstraintHandler for Lazy {
    fn check(
        &mut self,
        _: &mut ScipContext<'_>,
        values: &rustc_hash::FxHashMap<VarId, f64>,
    ) -> Result<bool, SolverError> {
        Ok(self.0.iter().map(|v| values[v]).sum::<f64>() < 0.5)
    }
    fn enforce(
        &mut self,
        c: &mut ScipConstraintContext<'_>,
    ) -> Result<ConshdlrResult, SolverError> {
        if self.0.iter().map(|&v| c.context.value(v).unwrap()).sum::<f64>() > 0.5 {
            c.add_constraint(&ScipLinearRow {
                name: "lazy_zero".into(),
                coefficients: self.0.iter().map(|&v| (v.into(), 1.0)).collect(),
                lower: f64::NEG_INFINITY,
                upper: 0.0,
                local: false,
            })?;
            Ok(ConshdlrResult::ConsAdded)
        } else {
            Ok(ConshdlrResult::Feasible)
        }
    }
}
#[test]
fn lazy_constraints_change_the_feasible_set() {
    let (m, x, y) = model();
    let mut s =
        Scip::new().with_plugins(move |r| r.constraint_handler(settings("lazy"), Lazy([x, y])));
    assert_eq!(s.solve(&m, &options()).unwrap().objective(), Some(0.0));
}

struct Select(Rc<Cell<usize>>);
impl ScipNodeSelector for Select {
    fn select(&mut self, c: &mut ScipContext<'_>) -> Result<Option<Node>, SolverError> {
        self.0.set(self.0.get() + 1);
        Ok(c.native().best_node())
    }
    fn compare(&mut self, _: &Node, _: &Node) -> Result<Ordering, SolverError> {
        Ok(Ordering::Equal)
    }
}
#[test]
fn node_selector_is_invoked() {
    let (m, _, _) = model();
    let calls = Rc::new(Cell::new(0));
    let c = calls.clone();
    let mut s =
        Scip::new().with_plugins(move |r| r.node_selector(settings("select"), Select(c.clone())));
    s.solve(&m, &options()).unwrap();
    assert!(calls.get() > 0);
}

struct Price {
    row: ConstraintId,
    added: bool,
}
impl ScipPricer for Price {
    fn generate_columns(
        &mut self,
        c: &mut ScipPricingContext<'_>,
    ) -> Result<PricerResult, SolverError> {
        if self.added {
            Ok(PricerResult { state: PricerResultState::NoColumns, lower_bound: None })
        } else {
            assert!(c.price(self.row).is_some());
            c.add_column(&ScipColumn {
                name: "cheap".into(),
                lower: 0.0,
                upper: f64::INFINITY,
                objective: 1.0,
                integer: false,
                coefficients: vec![(self.row, 1.0)],
            })?;
            self.added = true;
            Ok(PricerResult { state: PricerResultState::FoundColumns, lower_bound: None })
        }
    }
}
#[test]
fn priced_columns_have_separate_ids_and_objective_contributions() {
    let m = Model::new("pricing");
    variable!(m, x >= 0.0);
    let row = constraint!(m, demand, x >= 1.0);
    let row_id = row.row_id();
    objective!(m, Min, 10.0 * x);
    let mut s = Scip::new().with_plugins(move |r| {
        r.pricer(settings("price"), &[row_id], Price { row: row_id, added: false })
    });
    let out = s.solve_detailed(&m, &options()).unwrap();
    assert!((out.result.objective().unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(out.result.primal().unwrap().len(), 1);
    assert!((out.generated_solutions[0][&ScipGeneratedVarId(0)] - 1.0).abs() < 1e-6);
}

#[test]
fn concurrent_custom_plugins_are_rejected() {
    let (m, _, _) = model();
    let mut s = Scip::new().with_plugins(|_| Ok(()));
    assert!(s.solve(&m, &options().concurrent(true)).is_err());
}
