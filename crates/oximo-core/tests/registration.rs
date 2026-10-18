use oximo_core::prelude::*;
use oximo_core::{FunctionRepresentationRef, ModelId, ScalarFunctionData, SemanticFunction};
use oximo_expr::{ExprId, ExprIdRemap};

struct MustNotLower<'a>(Expr<'a, Affine>);

impl Function for MustNotLower<'_> {
    type Owned = ScalarFunctionData<Affine>;
    fn semantic_data(&self) -> Self::Owned {
        ScalarFunctionData::from_expression(self.0)
    }
}
impl FunctionInSet<LessThan> for MustNotLower<'_> {}
impl<'a> LowerConstraint<LessThan> for MustNotLower<'a> {
    type Ir = oximo_core::function_set::AlgebraicConstraintIr<'a>;
    fn lower(self, _: LessThan) -> Self::Ir {
        panic!("implicit alternative lowering")
    }
}

#[test]
fn direct_macro_bulk_and_indexed_registration_never_invoke_alternative_lowering() {
    for threads in [1, 6] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let m = Model::new("one registration contract");
            variable!(m, x);
            let input = || Constraint::new(MustNotLower(x), LessThan(2.0));
            let direct = m.add_constraint("direct", input());
            let scalar = m.__add_constraint("scalar", input());
            m.add_constraints([("bulk".into(), input())]);
            let rows =
                m.__add_constraints_over("indexed", &Set::range(0..1024), |_: usize| input());
            for handle in [
                direct.erase(),
                scalar.erase(),
                m.constraint_handle("bulk").unwrap(),
                rows.get(1023_usize).unwrap().erase(),
            ] {
                let row = m.constraints().algebraic()[handle.index()].clone();
                assert_eq!(row.lhs, x.id());
                assert_eq!(row.upper, 2.0);
                let typed = m
                    .try_typed_handle::<ScalarFunctionData<Affine>, LessThan>(handle)
                    .unwrap()
                    .unwrap();
                m.set_constraint_set(typed, LessThan(3.0)).unwrap();
                assert_eq!(m.constraints().algebraic()[handle.index()].upper, 3.0);
            }
            let mut alternatives = Vec::new();
            m.visit_alternative_lowerings(|id| alternatives.push(id));
            assert_eq!(alternatives, []);
        });
    }
}

#[derive(Clone, Debug)]
struct ScalarView {
    model: ModelId,
    root: ExprId,
}
impl SemanticFunction for ScalarView {
    fn model_id(&self) -> ModelId {
        self.model
    }
    fn rebind_model(&mut self, id: ModelId) {
        self.model = id;
    }
    fn visit_expressions(&self, visitor: &mut dyn FnMut(ExprId)) {
        visitor(self.root);
    }
    fn remap(&mut self, remap: ExprIdRemap) {
        self.root = remap.apply(self.root);
    }
    fn representation(&self) -> FunctionRepresentationRef<'_> {
        FunctionRepresentationRef::Scalar(self.root)
    }
}
impl FunctionInSet<LessThan> for ScalarView {}

#[test]
fn indexed_registration_accepts_owned_functions_without_a_lowering_and_remaps_roots() {
    rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap().install(|| {
        let m = Model::new("owned scalar representations");
        variable!(m, x);
        let rows = m.__add_constraints_over("rows", &Set::range(0..1024), |i: usize| {
            let expression = x + f64::from(u32::try_from(i).unwrap());
            Constraint::new(
                ScalarView { model: x.model_id(), root: expression.id() },
                LessThan(2.0),
            )
        });
        let record = m.semantic_constraint(rows.get(1023_usize).unwrap()).unwrap().unwrap();
        let root = record.pair.function.expressions()[0];
        assert_eq!(root, m.constraints().algebraic()[rows.get(1023_usize).unwrap().index()].lhs);
        assert!(root.index() < m.arena().len());
        m.set_constraint_set(rows.get(1023_usize).unwrap(), LessThan(8.0)).unwrap();
        assert_eq!(m.constraints().algebraic()[rows.get(1023_usize).unwrap().index()].upper, 8.0);
    });
}

#[test]
fn owned_scalar_inputs_reject_foreign_and_missing_roots_before_registration() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let m = Model::new("invalid scalar roots");
    variable!(m, x);
    let foreign = Model::new("foreign");
    for function in [
        ScalarView { model: foreign.id(), root: x.id() },
        ScalarView { model: m.id(), root: ExprId(u32::MAX) },
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(
                || m.__add_constraint("invalid", Constraint::new(function.clone(), LessThan(2.0)))
            ))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(
                || m.add_constraint("invalid", Constraint::new(function, LessThan(2.0)))
            ))
            .is_err()
        );
        assert_eq!(m.num_constraints(), 0);
    }
}

#[cfg(feature = "gdp")]
#[test]
fn conditional_registration_uses_the_declared_representation() {
    let m = Model::new("conditional scalar semantics");
    variable!(m, x);
    let branch = m.add_boolean("branch");
    branch.context().add_constraint("row", Constraint::new(MustNotLower(x), LessThan(2.0)));
    let snapshot = m.gdp_snapshot();
    assert_eq!(snapshot.rows.len(), 1);
    assert_eq!(snapshot.rows[0].lhs, x.id());
    assert_eq!(snapshot.rows[0].upper, 2.0);
}

struct ScalarViewFunction<'a> {
    expression: Expr<'a, Affine>,
    owner: ModelId,
}

impl Function for ScalarViewFunction<'_> {
    type Owned = ScalarView;
    fn semantic_data(&self) -> Self::Owned {
        ScalarView { model: self.owner, root: self.expression.id() }
    }
}
impl FunctionInSet<LessThan> for ScalarViewFunction<'_> {}
impl<'a> IntoAffineFunction<'a> for ScalarViewFunction<'a> {
    fn into_affine_function(self) -> ScalarAffineFunction<'a> {
        self.expression.into_affine_function()
    }
}

#[test]
fn indicators_preserve_the_declared_function_and_check_its_ownership() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let m = Model::new("indicator semantics");
    variable!(m, x);
    variable!(m, enabled, Binary);
    let foreign = Model::new("foreign");
    let input = |owner| Constraint::new(ScalarViewFunction { expression: x, owner }, LessThan(2.0));
    assert!(
        catch_unwind(AssertUnwindSafe(|| m.add_indicator_constraint(
            "row",
            enabled,
            true,
            input(foreign.id())
        )))
        .is_err()
    );
    assert_eq!(m.num_indicator_constraints(), 0);
    let row = m.add_indicator_constraint("row", enabled, true, input(m.id()));
    let record = m.semantic_constraint(row).unwrap().unwrap();
    let function =
        record.pair.function.downcast_ref::<oximo_core::IndicatorFunctionData>().unwrap();
    assert!(function.consequent.downcast_ref::<ScalarView>().is_some());
}
