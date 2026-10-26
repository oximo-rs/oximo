use oximo_gdp::core::VarId;
use oximo_gdp::prelude::*;
use oximo_solver::{
    DualStatus, PrimalStatus, Solver, SolverError, SolverResult, TerminationStatus,
};

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

/// Returns a full primal point so projection has something to filter.
#[derive(Debug, Default)]
struct PrimalSolver {
    calls: usize,
}

impl Solver for PrimalSolver {
    type Options = ();

    fn name(&self) -> &str {
        "primal"
    }

    fn supports(&self, _kind: ModelKind) -> bool {
        true
    }

    fn solve(&mut self, model: &Model, _opts: &()) -> Result<SolverResult, SolverError> {
        self.calls += 1;
        model.ensure_gdp_reformulated()?;
        let mut result = SolverResult::for_model(model);
        result.termination = TerminationStatus::Optimal;
        result.primal_status = PrimalStatus::OptimalPoint;
        let primal = (0..model.num_variables())
            .map(|index| (VarId(u32::try_from(index).expect("variable index")), 1.0))
            .collect();
        result.add_solution(primal, Some(0.0));
        Ok(result)
    }
}

#[derive(Debug)]
struct DualSolver;

impl Solver for DualSolver {
    type Options = ();

    fn name(&self) -> &str {
        "dual"
    }

    fn supports(&self, _kind: ModelKind) -> bool {
        true
    }

    fn solve(&mut self, model: &Model, opts: &()) -> Result<SolverResult, SolverError> {
        let mut result = PrimalSolver::default().solve(model, opts)?;
        result.dual_status = DualStatus::FeasiblePoint;
        result.dual = model
            .constraints()
            .algebraic()
            .iter()
            .enumerate()
            .map(|(i, _)| (oximo_gdp::core::ConstraintId(u32::try_from(i).unwrap()), 1.0))
            .collect();
        result.semantic_dual.extend(
            result
                .dual
                .keys()
                .map(|&row| (oximo_core::SemanticConstraintId::Algebraic(row), Some(7.0))),
        );
        Ok(result)
    }
}

#[test]
fn mixed_soc_and_gdp_counts_only_algebraic_source_rows() {
    for cones in [1, 2] {
        let model = Model::new("mixed SOC and GDP projection");
        variable!(model, 0.0 <= x <= 10.0);
        variable!(model, 0.0 <= t <= 10.0);
        constraint!(model, source_row, x <= 8.0);
        for i in 0..cones {
            model.add_soc_constraint(format!("cone{i}"), [x], t);
        }
        boolean_variable!(model, active);
        logical_constraint!(model, active);
        let source_row = model.constraint_id("source_row").unwrap();
        let mut solver = DualSolver.with_gdp(BigM::default());
        for _ in 0..3 {
            let result = solver.solve(&model, &()).unwrap();
            assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [source_row]);
        }
    }
}

#[test]
fn standalone_hull_drops_every_generated_row_on_reuse_and_fresh_wrappers() {
    let model = Model::new("standalone Hull projection");

    variable!(model, -2.0 <= x <= 10.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 3.0);
    let source_variables = model.num_variables();
    let mut solver = DualSolver.with_gdp(Hull::default());

    for _ in 0..3 {
        assert!(solver.solve(&model, &()).unwrap().dual.is_empty());
    }

    let reports = model.gdp_reformulations();
    let report = &reports[0];
    assert_eq!(report.generated_variables, source_variables..model.num_variables());
    assert_eq!(report.generated_rows, 0..model.constraints().algebraic().len());
    assert!(!report.generated_rows.is_empty());

    drop(reports);
    assert!(DualSolver.with_gdp(Hull::default()).solve(&model, &()).unwrap().dual.is_empty());

    let copy = model.to_reformulated_gdp_model(Hull::default()).unwrap();
    assert!(DualSolver.with_gdp(Hull::default()).solve(&copy, &()).unwrap().dual.is_empty());
}

#[test]
fn interleaved_source_rows_survive_multiple_gdp_generations_and_wrapper_reuse() {
    let model = Model::new("interleaved source rows");

    variable!(model, -2.0 <= x <= 10.0);
    boolean_variable!(model, active);
    logical_constraint!(model, active);
    let mut solver = DualSolver.with_gdp(BigM::default());
    solver.solve(&model, &()).unwrap();
    constraint!(model, first_source, x <= 4.0);

    let first = model.constraint_id("first_source").unwrap();
    for pass in 0..3 {
        if pass == 1 {
            logical_constraint!(model, !active);
        }
        let result = solver.solve(&model, &()).unwrap();
        assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [first]);
    }

    constraint!(model, second_source, x >= -1.0);
    let second = model.constraint_id("second_source").unwrap();
    for result in [
        solver.solve(&model, &()).unwrap(),
        DualSolver.with_gdp(BigM::default()).solve(&model, &()).unwrap(),
    ] {
        assert_eq!(result.dual.len(), 2);
        assert!(result.dual.contains_key(&first));
        assert!(result.dual.contains_key(&second));
    }
}

#[derive(Debug, Default)]
struct AppendingSolver {
    appended: bool,
}

impl Solver for AppendingSolver {
    type Options = ();

    fn name(&self) -> &str {
        "appending"
    }

    fn supports(&self, _kind: ModelKind) -> bool {
        true
    }

    fn solve(&mut self, model: &Model, opts: &()) -> Result<SolverResult, SolverError> {
        let result = DualSolver.solve(model, opts)?;
        if !self.appended {
            let later = model.__var("callback_variable").bounds(0.0, 1.0).build();
            model.add_constraint("callback_row", later.le(1.0));
            self.appended = true;
        }
        Ok(result)
    }
}

#[test]
fn backend_additions_enter_the_next_source_snapshot_and_parameters_stay_solve_local() {
    let model = Model::new("callback snapshot");

    variable!(model, 0.0 <= x <= 10.0);
    param!(model, cost = 2.0);
    boolean_variable!(model, active);
    logical_constraint!(model, active);
    let mut solver = AppendingSolver::default().with_gdp(BigM::default());
    let first = solver.solve(&model, &()).unwrap();
    let later = model.variable_id("callback_variable").unwrap();
    assert!(!first.primal().unwrap().contains_key(&later));

    model.set_param(cost, 4.0).unwrap();
    let second = solver.solve(&model, &()).unwrap();
    assert!(second.primal().unwrap().contains_key(&later));
    assert!(second.dual.contains_key(&model.constraint_id("callback_row").unwrap()));

    let third = solver.solve(&model, &()).unwrap();
    assert_eq!(first.value_of(cost * x).unwrap(), Some(2.0));
    assert_eq!(second.value_of(cost * x).unwrap(), Some(4.0));
    assert_eq!(third.value_of(cost * x).unwrap(), Some(4.0));
}

#[test]
fn solve_projects_results_and_keeps_hull_activations_queryable() {
    let model = Model::new("projected GDP solve");

    variable!(model, -2.0 <= x <= 10.0);
    boolean_variable!(model, a);
    boolean_variable!(model, b);
    boolean_variable!(model, c);
    disjunct_constraint!(model, a, x >= 1.0);
    disjunct_constraint!(model, b, x <= 3.0);
    disjunct_constraint!(model, c, x == 2.0);
    disjunction!(model, [a, b, c], AtLeastOne);

    let source_variables = model.num_variables();
    let options = GdpReformulationOptions::new(
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 7 }),
    );
    let mut solver = PrimalSolver::default().with_gdp(options);

    let result = solver.solve(&model, &()).unwrap();
    assert_eq!(result.model_id(), model.id());

    let (retained, handles) = {
        let reports = model.gdp_reformulations();
        let report = &reports[0];
        let retained: rustc_hash::FxHashSet<VarId> = report
            .hull_variables
            .iter()
            .map(|copy| copy.activation_kind.variable().raw_id())
            .chain(report.hull_patterns.iter().map(|pattern| pattern.weight.raw_id()))
            .chain(report.hull_branch_sums.iter().map(|sum| sum.weight.raw_id()))
            .filter(|id| id.index() >= source_variables)
            .collect();
        (retained, report.hull_activations().collect::<Vec<_>>())
    };

    assert!(!retained.is_empty(), "inclusive Hull patterns generate activation weights");

    let assert_projected = |result: &SolverResult| {
        let primal = result.primal().expect("projected primal");
        assert_eq!(primal.len(), source_variables + retained.len());
        assert!(
            primal.keys().all(|id| id.index() < source_variables || retained.contains(id)),
            "generated variables without an activation contract must be filtered"
        );
        for id in &retained {
            assert_eq!(primal.get(id), Some(&1.0), "retained activation {id:?} lost its value");
        }
        for &handle in &handles {
            assert!(result.hull_activation_value_of(handle).unwrap().is_some());
        }
        assert_eq!(result.boolean_value_of(a).unwrap(), Some(true));
    };
    assert_projected(&result);

    // A second solve with no pending GDP still projects and retains activations.
    let result = solver.solve(&model, &()).unwrap();
    assert_eq!(solver.into_inner().calls, 2);
    assert_projected(&result);
}

#[test]
fn solve_after_adding_gdp_and_source_variables_keeps_the_source_set_queryable() {
    let model = Model::new("incremental projected GDP solve");

    variable!(model, -2.0 <= x <= 10.0);
    boolean_variable!(model, left);
    boolean_variable!(model, right);
    disjunct_constraint!(model, left, x >= 1.0);
    disjunct_constraint!(model, right, x <= 3.0);
    disjunction!(model, [left, right], AtLeastOne);

    let options = GdpReformulationOptions::new(
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 }),
    );
    let mut solver = PrimalSolver::default().with_gdp(options);
    solver.solve(&model, &()).unwrap();

    variable!(model, 5.0 <= extra <= 6.0);
    boolean_variable!(model, low);
    boolean_variable!(model, high);
    disjunct_constraint!(model, low, extra >= 5.0);
    disjunct_constraint!(model, high, extra <= 6.0);
    disjunction!(model, [low, high], AtLeastOne);

    let result = solver.solve(&model, &()).unwrap();
    assert_eq!(solver.into_inner().calls, 2);
    assert_eq!(model.gdp_reformulations().len(), 2);

    let reports = model.gdp_reformulations();
    let generated: rustc_hash::FxHashSet<VarId> =
        reports.iter().flat_map(|report| report.variables.iter().map(|id| id.raw_id())).collect();
    let retained: rustc_hash::FxHashSet<VarId> = reports
        .iter()
        .flat_map(|report| {
            report
                .hull_variables
                .iter()
                .map(|copy| copy.activation_kind.variable().raw_id())
                .chain(report.hull_patterns.iter().map(|pattern| pattern.weight.raw_id()))
                .chain(report.hull_branch_sums.iter().map(|sum| sum.weight.raw_id()))
        })
        .collect();

    let primal = result.primal().expect("all source variables remain present");
    for index in 0..model.num_variables() {
        let id = VarId(u32::try_from(index).expect("variable index"));
        if !generated.contains(&id) {
            assert!(primal.contains_key(&id), "source variable {id:?} was dropped");
        }
    }
    let leaked: Vec<_> =
        primal.keys().filter(|id| generated.contains(id) && !retained.contains(id)).collect();
    assert!(
        leaked.is_empty(),
        "generated variables without an activation contract remained: {leaked:?}"
    );

    assert_eq!(result.value_of(extra).unwrap(), Some(1.0));
    assert_eq!(result.boolean_value_of(low).unwrap(), Some(true));
    for report in reports.iter() {
        for handle in report.hull_activations() {
            assert!(result.hull_activation_value_of(handle).unwrap().is_some());
        }
    }
}

#[test]
fn sos_then_gdp_projects_only_original_declarations_and_refreshes_after_more_sos() {
    let model = Model::new("SOS then GDP projection");

    variable!(model, 0.0 <= x <= 4.0);
    variable!(model, 0.0 <= y <= 4.0);
    let original = constraint!(model, original, x + y <= 4.0);
    model.add_sos_constraint("first SOS", SosType::Sos1, [(x, 1.0), (y, 2.0)]);
    let first = model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    assert_ne!(first[0].variables, [] as [GeneratedVarId; 0]);

    variable!(model, 0.0 <= later <= 3.0);
    boolean_variable!(model, active);
    disjunct_constraint!(model, active, later <= 2.0);

    let mut solver = DualSolver.with_gdp(Hull::default());
    for _ in 0..2 {
        let result = solver.solve(&model, &()).unwrap();
        assert_eq!(result.primal().unwrap().len(), 4);
        for variable in [x, y, later, active.binary()] {
            assert!(result.value_of(variable).unwrap().is_some());
        }
        for id in &first[0].variables {
            assert!(!result.primal().unwrap().contains_key(&id.raw_id()));
        }
        assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [original.row_id()]);
        assert_eq!(result.semantic_dual.len(), 1);
        assert_eq!(result.dual_of(original).unwrap(), Some(7.0));
    }

    model.add_sos_constraint("second SOS", SosType::Sos2, [(x, 1.0), (y, 2.0)]);
    model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    let result = solver.solve(&model, &()).unwrap();
    assert_eq!(result.primal().unwrap().len(), 4);
    assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [original.row_id()]);
    assert_eq!(result.semantic_dual.len(), 1);
    assert_eq!(result.dual_of(original).unwrap(), Some(7.0));
}

#[test]
fn indicator_then_gdp_drops_generated_indicator_rows() {
    let model = Model::new("indicator then GDP projection");

    variable!(model, 0.0 <= x <= 4.0);
    variable!(model, trigger, Bin);
    let original = constraint!(model, original, x <= 4.0);
    indicator_constraint!(model, trigger == 1 => x <= 2.0);

    let artifacts = model.reformulate_indicators(IndicatorReformulationOptions::default()).unwrap();
    assert_ne!(artifacts[0].constraints, [] as [GeneratedConstraintId; 0]);

    boolean_variable!(model, active);
    disjunct_constraint!(model, active, x <= 3.0);

    let result = DualSolver.with_gdp(BigM::default()).solve(&model, &()).unwrap();
    assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [original.row_id()]);
    assert_eq!(result.primal().unwrap().len(), 3);
}

#[test]
fn nested_gdp_projection_keeps_declared_booleans_and_explicit_activation_weights() {
    let model = Model::new("nested GDP source projection");

    variable!(model, 0.0 <= x <= 4.0);
    let original = constraint!(model, original, x <= 4.0);
    boolean_variable!(model, parent);
    boolean_variable!(model, sibling);
    boolean_variable!(model, child);
    boolean_variable!(model, other_child);
    disjunct_constraint!(model, parent, x <= 3.0);
    disjunct_constraint!(model, child, x >= 1.0);
    disjunct_constraint!(model, other_child, x <= 2.0);
    disjunction!(model, [parent, sibling], AtLeastOne);
    disjunction!(model, [child, other_child], parent = parent);
    let hull =
        Hull::default().with_inclusive_policy(HullInclusivePolicy::Enumerate { max_patterns: 3 });
    let mut solver = DualSolver.with_gdp(hull);

    for _ in 0..2 {
        let result = solver.solve(&model, &()).unwrap();
        let primal = result.primal().unwrap();
        for variable in [x, parent.binary(), sibling.binary(), child.binary(), other_child.binary()]
        {
            assert!(result.value_of(variable).unwrap().is_some());
        }
        let reports = model.gdp_reformulations();
        let retained: rustc_hash::FxHashSet<_> = reports
            .iter()
            .flat_map(|report| {
                report
                    .hull_variables
                    .iter()
                    .filter_map(|copy| copy.activation_kind.variable().generated())
                    .chain(report.hull_patterns.iter().map(|pattern| pattern.weight))
                    .chain(report.hull_branch_sums.iter().map(|sum| sum.weight))
            })
            .collect();
        assert!(!retained.is_empty());
        assert_eq!(primal.len(), 5 + retained.len());

        for report in reports.iter() {
            for activation in report.hull_activations() {
                assert!(result.hull_activation_value_of(activation).unwrap().is_some());
            }
            for id in &report.variables {
                assert_eq!(primal.contains_key(&id.raw_id()), retained.contains(id));
            }
        }
        assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [original.row_id()]);
    }
}

#[test]
fn sos_only_history_is_projected_even_without_a_gdp_report() {
    let model = Model::new("SOS-only projection");

    variable!(model, 0.0 <= x <= 4.0);
    variable!(model, 0.0 <= y <= 4.0);
    let original = constraint!(model, original, x + y <= 4.0);
    model.add_sos_constraint("SOS", SosType::Sos1, [(x, 1.0), (y, 2.0)]);
    model.reformulate_sos(SosReformulationOptions::default()).unwrap();
    assert!(model.gdp_reformulations().is_empty());
    let result = DualSolver.with_gdp(BigM::default()).solve(&model, &()).unwrap();
    assert_eq!(result.primal().unwrap().len(), 2);
    assert_eq!(result.dual.keys().copied().collect::<Vec<_>>(), [original.row_id()]);
    assert_eq!(result.dual_of(original).unwrap(), Some(7.0));

    for report in model.sos_reformulations().iter() {
        for row in &report.constraints {
            let raw = row.raw_id();
            let handle = model
                .require_scalar_dual_handle(model.constraint_handle_from_id(raw).unwrap())
                .unwrap();
            assert!(!result.dual.contains_key(&raw));
            assert!(!result.semantic_dual.contains_key(&handle.id()));
            assert_eq!(result.dual_of(handle).unwrap(), None);
        }
    }
}
