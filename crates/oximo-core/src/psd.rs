//! Real symmetric affine positive-semidefinite constraints.
//!
//! A PSD constraint requires `F(x)` to have nonnegative eigenvalues, where
//! `F(x)` is a real symmetric matrix whose entries are affine in the decision
//! variables. Symmetry is represented by [`SymmetricMatrix`] itself. PSD
//! membership is registered separately from the scalar variables that form it.
//!
//! # Modeling
//!
//! ```
//! use oximo_core::prelude::*;
//!
//! let model = Model::new("minimum trace");
//! let x = model.add_symmetric_variable("X", 2);
//! let positivity = psd_constraint!(model, positivity, &x);
//! constraint!(model, x[0, 1] == 1.0);
//! objective!(model, Min, x.trace());
//! assert_eq!(model.num_variables(), 3);
//! assert_eq!(x[(0, 1)].var_id(), x[(1, 0)].var_id());
//! assert_eq!(model.kind(), ModelKind::SDP);
//! model.set_psd_active(positivity, false).unwrap();
//! assert_eq!(model.kind(), ModelKind::LP);
//! ```
//!
//! [`crate::symmetric_variable!`] is equivalent to declaring the matrix with
//! [`Model::add_symmetric_variable`]. Its independent entries are free continuous
//! scalar variables. [`crate::psd_constraint!`] accepts anonymous, named,
//! computed-name and indexed-family forms. Indexed registration validates the
//! entire family before merging worker expressions and adding constraints.
//!
//! # Typed functions and validation
//!
//! [`SymmetricAffineFunction`] pairs with [`PositiveSemidefiniteCone`] in the
//! [`crate::Constraint`] API. Numeric entries and constant/affine expressions
//! convert through [`IntoSymmetricAffineFunction`] while dynamic expressions
//! receive a runtime affine check. Statically quadratic or nonlinear entries
//! are excluded. All expression entries must belong to the registering model.
//! The cone dimension must match the matrix side dimension.

use oximo_expr::{Affine, Constant, Dynamic, Expr, ExprArenaCell, ExprId, ModelId};
use smol_str::SmolStr;

use crate::function_set::{
    ConstraintIr, Function, FunctionInSet, IntoAffineFunction, LowerConstraint, Set,
};
use crate::{Model, SymmetricMatrix};

/// The mathematical cone of real symmetric PSD matrices.
#[derive(Clone, Copy, Debug)]
pub struct PositiveSemidefiniteCone {
    pub side_dimension: usize,
}
impl Set for PositiveSemidefiniteCone {}

/// Symbolic affine symmetric matrix retaining rebindable arena expressions.
#[derive(Clone, Debug)]
pub struct SymmetricAffineFunction<'a>(pub(crate) SymmetricMatrix<Expr<'a, Affine>>);
impl Function for SymmetricAffineFunction<'_> {}
impl FunctionInSet<PositiveSemidefiniteCone> for SymmetricAffineFunction<'_> {}
impl<'a> SymmetricAffineFunction<'a> {
    /// Construct and check that all entries have the same arena owner.
    ///
    /// # Panics
    /// Panics when entries belong to different arenas.
    pub fn new(matrix: SymmetricMatrix<Expr<'a, Affine>>) -> Self {
        let first = matrix[(0, 0)];
        for entry in matrix.upper_triangle() {
            assert!(
                std::ptr::eq(first.arena, entry.arena),
                "matrix entries belong to different models"
            );
        }
        Self(matrix)
    }

    pub fn matrix(&self) -> &SymmetricMatrix<Expr<'a, Affine>> {
        &self.0
    }
}

/// Entry conversion for an affine matrix. Static quadratic/nonlinear entries are excluded.
pub trait IntoAffineMatrixEntry<'a>: Copy {
    fn into_affine_matrix_entry(self, arena: &'a ExprArenaCell) -> Expr<'a, Affine>;
}
macro_rules! expression_entries {
    ($($d:ty),*) => { $(impl<'a> IntoAffineMatrixEntry<'a> for Expr<'a, $d> {
        fn into_affine_matrix_entry(self, arena: &'a ExprArenaCell) -> Expr<'a, Affine> {
            assert!(std::ptr::eq(self.arena, arena), "matrix belongs to a different model");
            self.into_affine_function().expression()
        }
    })* };
}
expression_entries!(Constant, Affine, Dynamic);
macro_rules! numeric_entries {
    ($($t:ty),*) => { $(impl<'a> IntoAffineMatrixEntry<'a> for $t {
        fn into_affine_matrix_entry(self, arena: &'a ExprArenaCell) -> Expr<'a, Affine> {
            Expr::constant(arena, f64::from(self)).into()
        }
    })* };
}
numeric_entries!(f64, f32, i32, u32);

/// Convert an explicitly symmetric matrix into a checked symbolic affine function.
pub trait IntoSymmetricAffineFunction<'a> {
    fn into_symmetric_affine_function(self, model: &'a Model) -> SymmetricAffineFunction<'a>
    where
        Self: Sized,
    {
        self.__into_symmetric_affine_function(model.__sum_context())
    }
    #[doc(hidden)]
    fn __into_symmetric_affine_function(
        self,
        arena: &'a ExprArenaCell,
    ) -> SymmetricAffineFunction<'a>;
}
impl<'a, T: IntoAffineMatrixEntry<'a>> IntoSymmetricAffineFunction<'a> for &SymmetricMatrix<T> {
    fn __into_symmetric_affine_function(
        self,
        arena: &'a ExprArenaCell,
    ) -> SymmetricAffineFunction<'a> {
        SymmetricAffineFunction::new(self.map(|&entry| entry.into_affine_matrix_entry(arena)))
    }
}
impl<'a, T: IntoAffineMatrixEntry<'a>> IntoSymmetricAffineFunction<'a> for SymmetricMatrix<T> {
    fn __into_symmetric_affine_function(
        self,
        arena: &'a ExprArenaCell,
    ) -> SymmetricAffineFunction<'a> {
        (&self).__into_symmetric_affine_function(arena)
    }
}
impl<'a> IntoSymmetricAffineFunction<'a> for SymmetricAffineFunction<'a> {
    fn __into_symmetric_affine_function(self, arena: &'a ExprArenaCell) -> Self {
        for entry in self.0.upper_triangle() {
            let _ = entry.into_affine_matrix_entry(arena);
        }
        self
    }
}

/// Checked PSD lowering output retaining expression provenance.
#[derive(Clone, Debug)]
pub struct PsdConstraintIr<'a>(pub(crate) SymmetricAffineFunction<'a>);
impl crate::function_set::sealed::Sealed for PsdConstraintIr<'_> {}
impl ConstraintIr for PsdConstraintIr<'_> {
    type Handle = PsdConstraintHandle;
    fn register(self, model: &Model, name: SmolStr) -> Self::Handle {
        model.register_psd_ir(name, self)
    }
}
impl<'a> LowerConstraint<PositiveSemidefiniteCone> for SymmetricAffineFunction<'a> {
    type Ir = PsdConstraintIr<'a>;
    fn lower(self, set: PositiveSemidefiniteCone) -> Self::Ir {
        assert_eq!(self.0.side_dimension(), set.side_dimension, "PSD cone dimension mismatch");
        PsdConstraintIr(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PsdConstraintId(pub u32);
impl PsdConstraintId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Model-bound handle to a PSD matrix constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PsdConstraintHandle {
    id: PsdConstraintId,
    model_id: ModelId,
}
impl PsdConstraintHandle {
    pub(crate) const fn new(id: PsdConstraintId, model_id: ModelId) -> Self {
        Self { id, model_id }
    }

    pub const fn id(self) -> PsdConstraintId {
        self.id
    }

    pub const fn model_id(self) -> ModelId {
        self.model_id
    }

    pub fn index(self) -> usize {
        self.id.index()
    }
}

/// Registered PSD constraint; entries use ordinary upper-column triangle coordinates.
#[derive(Clone, Debug)]
pub struct PsdConstraint {
    pub name: SmolStr,
    pub matrix: SymmetricMatrix<ExprId>,
    pub active: bool,
}

impl<'a, D: oximo_expr::Degree> crate::IntoFunction for SymmetricMatrix<Expr<'a, D>>
where
    Expr<'a, D>: IntoAffineFunction<'a>,
{
    type Function = SymmetricAffineFunction<'a>;
    fn into_function(self) -> Self::Function {
        SymmetricAffineFunction::new(self.map(|&entry| entry.into_affine_function().expression()))
    }
}
