//! Optimal resource investment and scheduling of tests for new product development.
//!
//! This example implements the ten-test instance in Section 5 of:
//!
//! Maravelias, C. T., & Grossmann, I. E. (2004).
//! Optimal resource investment and scheduling of tests for new product development.
//! Computers & Chemical Engineering, 28(6–7), 1021–1038.
//! <https://doi.org/10.1016/j.compchemeng.2003.09.019>
//!
//! Two products share scientists and experimental equipment.
//! Tests may use an existing unit, a newly acquired unit, or an
//! outsourced (dummy) unit. The model decides the test schedule,
//! resource assignments, and whether and when to acquire the two
//! new units. Its objective is expected discounted income minus
//! expected discounted testing and investment costs.
//!
//! This is the fixed-duration, one-unit-per-category specialization of the
//! paper's strengthened model M*. It uses the FIS precedence closure, the
//! disaggregated discounting formulation (26)-(29), and valid sequencing
//! cuts.
//!
//! The paper does not list the discounting grid used for Section 5. Here the
//! exponent is approximated at 0.1-wide points from 0 to -1.7, which covers the
//! feasible range for the 100-month horizon.
//!
//! Run with HiGHS:
//!
//! ```text
//! cargo run -p oximo --example resource_investment_scheduling
//! ```

use std::time::Duration;

use oximo::prelude::*;
use oximo::solvers::Highs;

const N_TESTS: usize = 10;
const N_RESOURCES: usize = 6;
const N_PRODUCTS: usize = 2;
const HORIZON: f64 = 100.0;
const DISCOUNT_RATE: f64 = 0.0075;
const MAX_INCOME: f64 = 500.0;
const INCOME_BREAKPOINTS: [f64; 2] = [24.0, 48.0];
const MONTHLY_INCOME_LOSS: [f64; 2] = [8.0, 5.0];

// All monetary values are in $10,000.
const RESOURCE_NAMES: [&str; N_RESOURCES] = ["A1", "A2", "A3", "B1", "B2", "B3"];
const SCIENTIST_RESOURCES: [usize; 3] = [0, 1, 2];
const EQUIPMENT_RESOURCES: [usize; 3] = [3, 4, 5];
const NEW_RESOURCES: [usize; 2] = [1, 4];
const IN_HOUSE_RESOURCES: [usize; 4] = [0, 1, 3, 4];
const INSTALLATION_COST: [f64; 2] = [20.0, 30.0];

struct TestData {
    product: usize,
    duration: f64,
    success_probability: f64,
    fixed_cost: f64,
    resource_cost: [f64; N_RESOURCES],
    predecessors: &'static [usize],
}

const TESTS: [TestData; N_TESTS] = [
    TestData {
        product: 0,
        duration: 12.0,
        success_probability: 1.0,
        fixed_cost: 10.0,
        resource_cost: [5.0, 5.0, 20.0, 6.0, 5.0, 20.0],
        predecessors: &[],
    },
    TestData {
        product: 0,
        duration: 13.0,
        success_probability: 0.7,
        fixed_cost: 15.0,
        resource_cost: [4.0, 5.0, 16.0, 5.0, 5.0, 22.0],
        predecessors: &[0],
    },
    TestData {
        product: 0,
        duration: 12.0,
        success_probability: 0.95,
        fixed_cost: 20.0,
        resource_cost: [6.0, 6.0, 18.0, 3.0, 3.0, 15.0],
        predecessors: &[1],
    },
    TestData {
        product: 0,
        duration: 20.0,
        success_probability: 1.0,
        fixed_cost: 40.0,
        resource_cost: [10.0, 8.0, 30.0, 2.0, 2.0, 12.0],
        predecessors: &[],
    },
    TestData {
        product: 0,
        duration: 18.0,
        success_probability: 1.0,
        fixed_cost: 60.0,
        resource_cost: [3.0, 3.0, 15.0, 5.0, 5.0, 30.0],
        predecessors: &[],
    },
    TestData {
        product: 0,
        duration: 15.0,
        success_probability: 0.6,
        fixed_cost: 20.0,
        resource_cost: [8.0, 8.0, 22.0, 10.0, 10.0, 35.0],
        predecessors: &[2, 4],
    },
    TestData {
        product: 1,
        duration: 8.0,
        success_probability: 1.0,
        fixed_cost: 20.0,
        resource_cost: [5.0, 5.0, 22.0, 2.0, 2.0, 16.0],
        predecessors: &[],
    },
    TestData {
        product: 1,
        duration: 15.0,
        success_probability: 0.8,
        fixed_cost: 30.0,
        resource_cost: [6.0, 9.0, 24.0, 6.0, 5.0, 20.0],
        predecessors: &[6],
    },
    TestData {
        product: 1,
        duration: 21.0,
        success_probability: 1.0,
        fixed_cost: 60.0,
        resource_cost: [5.0, 2.0, 18.0, 4.0, 4.0, 18.0],
        predecessors: &[],
    },
    TestData {
        product: 1,
        duration: 17.0,
        success_probability: 0.7,
        fixed_cost: 40.0,
        resource_cost: [4.0, 4.0, 12.0, 6.0, 6.0, 24.0],
        predecessors: &[7, 8],
    },
];

#[derive(Clone, Copy)]
enum AcquisitionCase {
    Disabled,
    Allowed,
}

impl AcquisitionCase {
    const fn name(self) -> &'static str {
        match self {
            Self::Disabled => "Case 1: acquisition disabled",
            Self::Allowed => "Case 2: acquisition allowed",
        }
    }

    const fn allows_acquisition(self) -> bool {
        matches!(self, Self::Allowed)
    }
}

struct ModelSets {
    tests: Set<usize>,
    resources: Set<usize>,
    products: Set<usize>,
    grid_points: Set<usize>,
    income_segments: Set<usize>,
    new_units: Set<usize>,
}

impl ModelSets {
    fn new(grid_size: usize) -> Self {
        Self {
            tests: Set::range(0..N_TESTS),
            resources: Set::range(0..N_RESOURCES),
            products: Set::range(0..N_PRODUCTS),
            grid_points: Set::range(0..grid_size),
            income_segments: Set::range(0..INCOME_BREAKPOINTS.len()),
            new_units: Set::range(0..NEW_RESOURCES.len()),
        }
    }
}

struct DiscountGrid {
    exponent: Vec<f64>,
    factor: Vec<f64>,
}

impl DiscountGrid {
    fn section_5() -> Self {
        let exponent: Vec<f64> = (0..=17).map(|point| -0.1 * f64::from(point)).collect();
        let factor = exponent.iter().map(|value| value.exp()).collect();
        Self { exponent, factor }
    }

    fn len(&self) -> usize {
        self.exponent.len()
    }
}

struct ModelVariables<'a> {
    start_time: IndexedVar<'a, usize>,
    completion_time: IndexedVar<'a, usize>,
    precedes: IndexedVar<'a, (usize, usize)>,
    uses_resource: IndexedVar<'a, (usize, usize)>,
    installed: IndexedVar<'a, usize>,
    installation_time: IndexedVar<'a, usize>,
    income_excess_time: IndexedVar<'a, (usize, usize)>,
    lambda_fixed: IndexedVar<'a, (usize, usize)>,
    lambda_used: IndexedVar<'a, (usize, usize, usize)>,
    lambda_unused: IndexedVar<'a, (usize, usize, usize)>,
    lambda_install: IndexedVar<'a, (usize, usize)>,
}

struct CaseSummary {
    npv: f64,
    completion: [f64; N_PRODUCTS],
    income: [f64; N_PRODUCTS],
    fixed_cost: [f64; N_PRODUCTS],
    resource_cost: [f64; N_PRODUCTS],
    installation_cost: f64,
    installed: [bool; NEW_RESOURCES.len()],
    installation_time: [f64; NEW_RESOURCES.len()],
    start_time: [f64; N_TESTS],
    uses_resource: [[bool; N_RESOURCES]; N_TESTS],
}

// Table 5 values converted from US$1,000 to the model's US$10,000 units.
struct PaperCase {
    completion: [f64; N_PRODUCTS],
    income: [f64; N_PRODUCTS],
    fixed_cost: [f64; N_PRODUCTS],
    resource_cost: [f64; N_PRODUCTS],
    installation_cost: f64,
    npv: f64,
}

const PAPER_CASES: [PaperCase; 2] = [
    PaperCase {
        completion: [52.0, 40.0],
        income: [256.0, 372.0],
        fixed_cost: [119.68, 134.56],
        resource_cost: [135.86, 91.41],
        installation_cost: 0.0,
        npv: 146.49,
    },
    PaperCase {
        completion: [52.0, 40.0],
        income: [256.0, 372.0],
        fixed_cost: [119.68, 134.56],
        resource_cost: [82.68, 66.01],
        installation_cost: 50.0,
        npv: 175.07,
    },
];

fn product_tests(product: usize) -> std::ops::Range<usize> {
    match product {
        0 => 0..6,
        1 => 6..10,
        _ => unreachable!("there are exactly two products"),
    }
}

fn technological_precedence_closure() -> [[bool; N_TESTS]; N_TESTS] {
    let mut precedes = [[false; N_TESTS]; N_TESTS];
    for (test, data) in TESTS.iter().enumerate() {
        for &predecessor in data.predecessors {
            precedes[predecessor][test] = true;
        }
    }
    for middle in 0..N_TESTS {
        for before in 0..N_TESTS {
            for after in 0..N_TESTS {
                precedes[before][after] |= precedes[before][middle] && precedes[middle][after];
            }
        }
    }
    precedes
}

fn declare_model_variables<'a>(model: &'a Model, sets: &ModelSets) -> ModelVariables<'a> {
    let tests = &sets.tests;
    let resources = &sets.resources;
    let products = &sets.products;
    let grid_points = &sets.grid_points;
    let income_segments = &sets.income_segments;
    let new_units = &sets.new_units;

    variable!(model, 0.0 <= start_time[test in tests] <= HORIZON);
    variable!(model, 0.0 <= completion_time[product in products] <= HORIZON);
    variable!(model, precedes[test in tests, other in tests if test != other], Bin);
    variable!(model, uses_resource[test in tests, resource in resources], Bin);
    variable!(model, installed[unit in new_units], Bin);
    variable!(model, 0.0 <= installation_time[unit in new_units] <= HORIZON);
    variable!(model,
        income_excess_time[product in products, segment in income_segments] >= 0.0);

    // Lambda^1, lambda^2, bar-lambda^2, and lambda^3 in model M*.
    variable!(model, lambda_fixed[test in tests, point in grid_points] >= 0.0);
    variable!(model,
        lambda_used[test in tests, resource in resources, point in grid_points] >= 0.0);
    variable!(model,
        lambda_unused[test in tests, resource in resources, point in grid_points] >= 0.0);
    variable!(model, lambda_install[unit in new_units, point in grid_points] >= 0.0);

    ModelVariables {
        start_time,
        completion_time,
        precedes,
        uses_resource,
        installed,
        installation_time,
        income_excess_time,
        lambda_fixed,
        lambda_used,
        lambda_unused,
        lambda_install,
    }
}

fn add_timing_and_sequencing_constraints(
    model: &Model,
    sets: &ModelSets,
    variables: &ModelVariables<'_>,
) {
    let tests = &sets.tests;

    // (1): each product completes after all of its tests.
    constraint!(model, finish[test in tests],
        variables.start_time[test] + TESTS[test].duration
            <= variables.completion_time[TESTS[test].product]);

    // (3), (4), and (4*): precedes[test, other] implies non-overlap. Every
    // test in this instance is eligible for every resource category, so the
    // cross-product form (4*) covers every pair belonging to different products.
    constraint!(model, ordered[test in tests, other in tests if test != other],
        variables.start_time[test] + TESTS[test].duration
            <= variables.start_time[other]
                + HORIZON * (1.0 - variables.precedes[test, other]));

    // (2) plus FIS: fix direct and transitively implied technological precedences.
    let technological_precedence = technological_precedence_closure();
    for (before, row) in technological_precedence.iter().enumerate() {
        for (after, &is_precedence) in row.iter().enumerate() {
            if is_precedence {
                model.fix(variables.precedes[(before, after)], 1.0).unwrap();
                model.fix(variables.precedes[(after, before)], 0.0).unwrap();
            }
        }
    }

    // Valid cycle and transitivity cuts strengthen the sequencing relaxation.
    constraint!(model, no_two_cycle[test in tests, other in tests if test < other],
        variables.precedes[test, other] + variables.precedes[other, test] <= 1.0);
    constraint!(model, transitive[test in tests, middle in tests, other in tests
        if test != middle && middle != other && test != other],
        variables.precedes[test, middle] + variables.precedes[middle, other]
            - variables.precedes[test, other] <= 1.0);
}

fn add_resource_constraints(
    model: &Model,
    sets: &ModelSets,
    variables: &ModelVariables<'_>,
    case: AcquisitionCase,
) {
    let tests = &sets.tests;

    // (7)-(8): N_min = N_max = 1, so every test uses exactly one scientist
    // group and one equipment unit.
    constraint!(model, scientist[test in tests],
        sum!(variables.uses_resource[test, resource] for resource in SCIENTIST_RESOURCES) == 1.0);
    constraint!(model, equipment[test in tests],
        sum!(variables.uses_resource[test, resource] for resource in EQUIPMENT_RESOURCES) == 1.0);

    // (6) and (10): new units must be installed before an assigned test starts.
    for (unit, &resource) in NEW_RESOURCES.iter().enumerate() {
        if !case.allows_acquisition() {
            model.fix(variables.installed[unit], 0.0).unwrap();
        }
        for test in 0..N_TESTS {
            constraint!(model, variables.uses_resource[test, resource] <= variables.installed[unit]);
            constraint!(model,
                variables.start_time[test]
                    >= variables.installation_time[unit]
                        - HORIZON * (1.0 - variables.uses_resource[test, resource]));
        }
    }

    // (9): in-house units have unit capacity.
    // Dummy outsourcing units A3 and B3 deliberately have unlimited capacity.
    for &resource in &IN_HOUSE_RESOURCES {
        for test in 0..N_TESTS {
            for other in (test + 1)..N_TESTS {
                constraint!(model,
                    variables.uses_resource[test, resource]
                        + variables.uses_resource[other, resource]
                        - variables.precedes[test, other]
                        - variables.precedes[other, test]
                        <= 1.0);
            }
        }
    }
}

fn add_discounting_constraints(
    model: &Model,
    sets: &ModelSets,
    variables: &ModelVariables<'_>,
    discount_grid: &DiscountGrid,
) {
    let grid_points = &sets.grid_points;

    // (11), (14), and (26)-(29): a test is conducted only if every test
    // scheduled before it in the same product succeeds. The probability in
    // the exponent therefore belongs to the preceding test, not the current one.
    for (test, test_data) in TESTS.iter().enumerate() {
        let same_product = product_tests(test_data.product);
        constraint!(
            model,
            sum!(variables.lambda_fixed[test, point] for point in grid_points) == 1.0
        );
        constraint!(
            model,
            sum!(discount_grid.exponent[point] * variables.lambda_fixed[test, point]
                for point in grid_points)
                == -DISCOUNT_RATE * variables.start_time[test]
                    + sum!(
                        TESTS[other].success_probability.ln()
                            * variables.precedes[other, test]
                        for other in same_product if other != test
                    )
        );

        for resource in 0..N_RESOURCES {
            constraint!(model,
                sum!(variables.lambda_used[test, resource, point] for point in grid_points)
                    == variables.uses_resource[test, resource]);
            constraint!(model,
                sum!(variables.lambda_unused[test, resource, point] for point in grid_points)
                    == 1.0 - variables.uses_resource[test, resource]);
            for point in 0..discount_grid.len() {
                constraint!(model,
                    variables.lambda_fixed[test, point]
                        == variables.lambda_used[test, resource, point]
                            + variables.lambda_unused[test, resource, point]);
            }
        }
    }

    // (13) and (17): discount each installed unit's acquisition cost.
    for unit in 0..NEW_RESOURCES.len() {
        constraint!(
            model,
            sum!(variables.lambda_install[unit, point] for point in grid_points)
                == variables.installed[unit]
        );
        constraint!(
            model,
            sum!(discount_grid.exponent[point] * variables.lambda_install[unit, point]
                for point in grid_points)
                == -DISCOUNT_RATE * variables.installation_time[unit]
        );
    }
}

fn set_npv_objective(
    model: &Model,
    sets: &ModelSets,
    variables: &ModelVariables<'_>,
    discount_grid: &DiscountGrid,
) {
    let tests = &sets.tests;
    let resources = &sets.resources;
    let products = &sets.products;
    let grid_points = &sets.grid_points;
    let income_segments = &sets.income_segments;
    let new_units = &sets.new_units;

    // (21) and (22): $5M maximum income, reduced after months 24 and 48.
    constraint!(model,
        income_segment[product in products, segment in income_segments],
        variables.income_excess_time[product, segment]
            >= variables.completion_time[product] - INCOME_BREAKPOINTS[segment]);

    let income = 2.0 * MAX_INCOME
        - sum!(MONTHLY_INCOME_LOSS[segment]
            * variables.income_excess_time[product, segment]
            for product in products, segment in income_segments);

    // (18) and (20): expected discounted fixed, utilization, and installation costs.
    let fixed_cost = sum!(
        TESTS[test].fixed_cost * discount_grid.factor[point]
            * variables.lambda_fixed[test, point]
        for test in tests, point in grid_points
    );
    let resource_cost = sum!(
        TESTS[test].resource_cost[resource] * discount_grid.factor[point]
            * variables.lambda_used[test, resource, point]
        for test in tests, resource in resources, point in grid_points
    );
    let installation_cost = sum!(
        INSTALLATION_COST[unit] * discount_grid.factor[point]
            * variables.lambda_install[unit, point]
        for unit in new_units, point in grid_points
    );

    // (23): maximize net present value.
    objective!(model, Max, income - fixed_cost - resource_cost - installation_cost);
}

fn summarize_solution(
    result: &SolverResult,
    variables: &ModelVariables<'_>,
    discount_grid: &DiscountGrid,
) -> CaseSummary {
    let value = |expr| result.value_of(expr).unwrap().unwrap_or(0.0);
    let completion = std::array::from_fn(|product| value(variables.completion_time[product]));
    let mut income_value = [MAX_INCOME; N_PRODUCTS];
    let mut fixed_cost_value = [0.0; N_PRODUCTS];
    let mut resource_cost_value = [0.0; N_PRODUCTS];
    for product in 0..N_PRODUCTS {
        for (segment, &monthly_loss) in MONTHLY_INCOME_LOSS.iter().enumerate() {
            income_value[product] -=
                monthly_loss * value(variables.income_excess_time[(product, segment)]);
        }
        for test in product_tests(product) {
            for point in 0..discount_grid.len() {
                fixed_cost_value[product] += TESTS[test].fixed_cost
                    * discount_grid.factor[point]
                    * value(variables.lambda_fixed[(test, point)]);
                for resource in 0..N_RESOURCES {
                    resource_cost_value[product] += TESTS[test].resource_cost[resource]
                        * discount_grid.factor[point]
                        * value(variables.lambda_used[(test, resource, point)]);
                }
            }
        }
    }
    let installation_cost_value = (0..NEW_RESOURCES.len())
        .flat_map(|unit| (0..discount_grid.len()).map(move |point| (unit, point)))
        .map(|(unit, point)| {
            INSTALLATION_COST[unit]
                * discount_grid.factor[point]
                * value(variables.lambda_install[(unit, point)])
        })
        .sum();
    let installed = std::array::from_fn(|unit| value(variables.installed[unit]) > 0.5);
    let installation_time = std::array::from_fn(|unit| value(variables.installation_time[unit]));
    let start_time = std::array::from_fn(|test| value(variables.start_time[test]));
    let uses_resource = std::array::from_fn(|test| {
        std::array::from_fn(|resource| value(variables.uses_resource[(test, resource)]) > 0.5)
    });

    CaseSummary {
        npv: result.objective().unwrap_or(0.0),
        completion,
        income: income_value,
        fixed_cost: fixed_cost_value,
        resource_cost: resource_cost_value,
        installation_cost: installation_cost_value,
        installed,
        installation_time,
        start_time,
        uses_resource,
    }
}

fn print_case(case: AcquisitionCase, result: &SolverResult, summary: &CaseSummary) {
    let name = case.name();
    println!("\n{name}");
    println!("{}", "=".repeat(name.len()));
    println!("Status: {:?}", result.termination);
    if let Some(gap) = result.gap {
        println!("MIP gap: {:.2}%", 100.0 * gap);
    }
    if result.termination != TerminationStatus::Optimal {
        println!("The schedule below is a feasible incumbent, not a certified optimum.");
    }
    println!("NPV: ${:.4} million", result.objective().unwrap_or(0.0) / 100.0);
    println!(
        "Completion: P1 = {:.1} months, P2 = {:.1} months",
        summary.completion[0], summary.completion[1]
    );
    for (unit, &resource) in NEW_RESOURCES.iter().enumerate() {
        if summary.installed[unit] {
            println!(
                "Acquire {} at month {:.1}",
                RESOURCE_NAMES[resource], summary.installation_time[unit]
            );
        }
    }

    println!(
        "\n{:<6} {:<7} {:>7} {:>7} {:<10} {:<10}",
        "Test", "Product", "Start", "Finish", "Scientists", "Equipment"
    );
    println!("{}", "-".repeat(58));
    let mut schedule: Vec<(usize, f64)> = summary.start_time.iter().copied().enumerate().collect();
    schedule.sort_by(|left, right| left.1.total_cmp(&right.1));
    for (test, start_value) in schedule {
        let scientist = SCIENTIST_RESOURCES
            .into_iter()
            .find(|&resource| summary.uses_resource[test][resource])
            .unwrap();
        let equipment = EQUIPMENT_RESOURCES
            .into_iter()
            .find(|&resource| summary.uses_resource[test][resource])
            .unwrap();
        println!(
            "{:<6} P{:<6} {:>7.1} {:>7.1} {:<10} {:<10}",
            test + 1,
            TESTS[test].product + 1,
            start_value,
            start_value + TESTS[test].duration,
            RESOURCE_NAMES[scientist],
            RESOURCE_NAMES[equipment]
        );
    }
}

fn solve_case(case: AcquisitionCase) -> Result<CaseSummary, Box<dyn std::error::Error>> {
    let discount_grid = DiscountGrid::section_5();
    let sets = ModelSets::new(discount_grid.len());
    let model = Model::new(case.name());
    let variables = declare_model_variables(&model, &sets);

    add_timing_and_sequencing_constraints(&model, &sets, &variables);
    add_resource_constraints(&model, &sets, &variables, case);
    add_discounting_constraints(&model, &sets, &variables, &discount_grid);
    set_npv_objective(&model, &sets, &variables, &discount_grid);

    let options =
        HighsOptions::default().time_limit(Duration::from_secs(320)).mip_gap(0.01).verbose(false);
    let result = Highs.solve(&model, &options)?;
    if !result.has_solution() {
        return Err(format!(
            "{}: HiGHS terminated with {:?} and returned no feasible schedule",
            case.name(),
            result.termination
        )
        .into());
    }

    let summary = summarize_solution(&result, &variables, &discount_grid);
    print_case(case, &result, &summary);
    Ok(summary)
}

fn print_product_row(
    label: &str,
    model_case_1: [f64; N_PRODUCTS],
    paper_case_1: [f64; N_PRODUCTS],
    model_case_2: [f64; N_PRODUCTS],
    paper_case_2: [f64; N_PRODUCTS],
    scale: f64,
) {
    println!(
        "{label:<24}{:>8.1}{:>8.1}{:>8.1}{:>8.1}{:>8.1}{:>8.1}{:>8.1}{:>8.1}",
        model_case_1[0] * scale,
        paper_case_1[0] * scale,
        model_case_1[1] * scale,
        paper_case_1[1] * scale,
        model_case_2[0] * scale,
        paper_case_2[0] * scale,
        model_case_2[1] * scale,
        paper_case_2[1] * scale
    );
}

fn print_comparison(cases: &[CaseSummary; 2]) {
    // Model coefficients are in $10,000.
    // Table 5 reports money in $1,000.
    const TO_PAPER_UNITS: f64 = 10.0;

    println!("\nComparison with Table 5");
    println!("{}", "=".repeat(88));
    println!("{:<24}{:^32}{:^32}", "", "Case 1", "Case 2");
    println!("{:<24}{:^16}{:^16}{:^16}{:^16}", "", "P1", "P2", "P1", "P2");
    println!(
        "{:<24}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
        "", "Model", "Paper", "Model", "Paper", "Model", "Paper", "Model", "Paper"
    );
    println!("{}", "-".repeat(88));

    print_product_row(
        "Completion time",
        cases[0].completion,
        PAPER_CASES[0].completion,
        cases[1].completion,
        PAPER_CASES[1].completion,
        1.0,
    );
    print_product_row(
        "Income (US$ 1,000)",
        cases[0].income,
        PAPER_CASES[0].income,
        cases[1].income,
        PAPER_CASES[1].income,
        TO_PAPER_UNITS,
    );
    print_product_row(
        "CT1 (US$ 1,000)",
        cases[0].fixed_cost,
        PAPER_CASES[0].fixed_cost,
        cases[1].fixed_cost,
        PAPER_CASES[1].fixed_cost,
        TO_PAPER_UNITS,
    );
    print_product_row(
        "CT2 (US$ 1,000)",
        cases[0].resource_cost,
        PAPER_CASES[0].resource_cost,
        cases[1].resource_cost,
        PAPER_CASES[1].resource_cost,
        TO_PAPER_UNITS,
    );
    println!(
        "{:<24}{:>8}{:>8}{:>8}{:>8}{:>8.1}{:>8.1}{:>8}{:>8}",
        "CT3 (US$ 1,000)",
        "-",
        "-",
        "",
        "",
        cases[1].installation_cost * TO_PAPER_UNITS,
        PAPER_CASES[1].installation_cost * TO_PAPER_UNITS,
        "",
        ""
    );
    println!(
        "{:<24}{:>8.1}{:>8.1}{:>8}{:>8}{:>8.1}{:>8.1}{:>8}{:>8}",
        "NPV (US$ 1,000)",
        cases[0].npv * TO_PAPER_UNITS,
        PAPER_CASES[0].npv * TO_PAPER_UNITS,
        "",
        "",
        cases[1].npv * TO_PAPER_UNITS,
        PAPER_CASES[1].npv * TO_PAPER_UNITS,
        "",
        ""
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let no_acquisition = solve_case(AcquisitionCase::Disabled)?;
    let acquisition = solve_case(AcquisitionCase::Allowed)?;
    print_comparison(&[no_acquisition, acquisition]);
    Ok(())
}
