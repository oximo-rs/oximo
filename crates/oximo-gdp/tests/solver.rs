use oximo_gdp::prelude::*;
use oximo_solver::{Solver, SolverError, SolverResult};

#[derive(Debug, Default)]
struct RecordingSolver {
    calls: usize,
}

impl Solver for RecordingSolver {
    type Options = bool;

    fn name(&self) -> &str {
        "recording"
    }

    fn supports(&self, _kind: ModelKind) -> bool {
        true
    }

    fn supports_model_extra(&self, _model: &Model) -> bool {
        true
    }

    fn solve(&mut self, model: &Model, fail: &bool) -> Result<SolverResult, SolverError> {
        self.calls += 1;
        model.ensure_gdp_reformulated()?;
        if *fail {
            return Err(SolverError::Backend("requested failure".into()));
        }
        Ok(SolverResult { model_id: model.id(), ..SolverResult::default() })
    }
}

#[test]
fn backend_hook_cannot_accept_unreformulated_gdp() {
    let model = Model::new("GDP capabilities");
    boolean_variable!(model, active);
    logical_constraint!(model, active);
    let solver = RecordingSolver::default();

    assert!(!solver.supports_model(&model));
    model.reformulate_gdp(BigM::default()).unwrap();
    assert!(solver.supports_model(&model));
}

#[test]
fn solve_reformulates_in_place_and_processes_only_new_components_on_reuse() {
    let model = Model::new("solve GDP");
    boolean_variable!(model, active);
    logical_constraint!(model, active);
    let mut solver = RecordingSolver::default().with_gdp(BigM::default());

    let result = solver.solve(&model, &false).unwrap();
    assert_eq!(result.model_id(), model.id());
    assert!(!model.has_unreformulated_gdp());
    assert_eq!(model.gdp_reformulations().len(), 1);
    let row_count = model.constraints().algebraic().len();

    solver.solve(&model, &false).unwrap();
    assert_eq!(model.constraints().algebraic().len(), row_count);
    assert_eq!(model.gdp_reformulations().len(), 1);

    logical_constraint!(model, !active);
    solver.solve(&model, &false).unwrap();
    assert!(!model.has_unreformulated_gdp());
    assert_eq!(model.gdp_reformulations().len(), 2);
    assert_eq!(solver.into_inner().calls, 3);
}

#[test]
fn missing_m_leaves_model_unchanged_and_does_not_call_backend() {
    let model = Model::new("invalid GDP");
    variable!(model, x);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 2.0);
    let source = model.to_string();
    let variable_count = model.num_variables();
    let mut solver = RecordingSolver::default().with_gdp(BigM::default());

    assert!(matches!(
        solver.solve(&model, &false),
        Err(GdpSolveError::Reformulation(GdpError::MissingBigM { .. }))
    ));
    assert_eq!(model.to_string(), source);
    assert_eq!(model.num_variables(), variable_count);
    assert!(model.constraints().algebraic().is_empty());
    assert!(model.gdp_reformulations().is_empty());
    assert_eq!(solver.into_inner().calls, 0);
}

#[test]
fn foreign_override_is_rejected_before_calling_backend() {
    let other = Model::new("foreign");
    boolean_variable!(other, selected);
    let foreign = disjunction!(other, [selected]);
    let model = Model::new("target");
    boolean_variable!(model, active);
    logical_constraint!(model, active);
    let source = model.to_string();
    let options =
        GdpReformulationOptions::new(BigM::default()).with_method(foreign, BigM::default());
    let mut solver = RecordingSolver::default().with_gdp(options);

    assert!(matches!(
        solver.solve(&model, &false),
        Err(GdpSolveError::Reformulation(GdpError::ForeignHandle("disjunction")))
    ));
    assert_eq!(model.to_string(), source);
    assert!(model.constraints().algebraic().is_empty());
    assert_eq!(solver.into_inner().calls, 0);
}

#[test]
fn backend_failure_remains_distinct_and_retains_successful_reformulation() {
    let model = Model::new("backend failure");
    boolean_variable!(model, active);
    logical_constraint!(model, active);
    let mut solver = RecordingSolver::default().with_gdp(GdpMethod::from(BigM::default()));

    assert!(matches!(
        solver.solve(&model, &true),
        Err(GdpSolveError::Solver(SolverError::Backend(message))) if message == "requested failure"
    ));
    assert!(!model.has_unreformulated_gdp());
    assert_eq!(model.gdp_reformulations().len(), 1);
    assert_eq!(solver.into_inner().calls, 1);
}

#[test]
fn ordinary_models_pass_through_without_a_gdp_transformation() {
    let model = Model::new("ordinary");
    variable!(model, 0.0 <= x <= 1.0);
    objective!(model, Max, x);
    let mut solver = RecordingSolver::default().with_gdp(BigM::default());
    assert_eq!(solver.solve(&model, &false).unwrap().model_id(), model.id());
    assert!(model.gdp_reformulations().is_empty());
    assert_eq!(solver.into_inner().calls, 1);
}
