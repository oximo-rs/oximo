//! Conservative static expression degrees. Dynamic expressions retain runtime classification.
mod sealed {
    pub trait Sealed {
        const MAY_BE_QUADRATIC: bool = true;
        type Publication: PublicationState;
    }

    pub trait PublicationState: Send + Sync {
        fn new(published: bool) -> Self;
        fn is_published(&self) -> bool;
        fn flag(&self) -> Option<&std::sync::atomic::AtomicBool>;
    }

    impl PublicationState for () {
        #[inline]
        fn new(_published: bool) -> Self {}
        #[inline]
        fn is_published(&self) -> bool {
            true
        }
        #[inline]
        fn flag(&self) -> Option<&std::sync::atomic::AtomicBool> {
            None
        }
    }

    #[derive(Debug)]
    pub struct ComputedPublication(std::sync::atomic::AtomicBool);

    impl PublicationState for ComputedPublication {
        #[inline]
        fn new(published: bool) -> Self {
            Self(std::sync::atomic::AtomicBool::new(published))
        }
        #[inline]
        fn is_published(&self) -> bool {
            self.0.load(std::sync::atomic::Ordering::Acquire)
        }
        #[inline]
        fn flag(&self) -> Option<&std::sync::atomic::AtomicBool> {
            Some(&self.0)
        }
    }
}

pub(crate) use sealed::PublicationState;

pub(crate) fn may_be_quadratic<D: Degree>() -> bool {
    D::MAY_BE_QUADRATIC
}

/// A sealed upper bound on polynomial degree.
pub trait Degree: sealed::Sealed + Copy + std::fmt::Debug + Send + Sync + 'static {
    /// Degree of a computed expression, without a reusable leaf identity.
    type Computed: ExpressionDegree<Computed = Self::Computed>;
}

/// Degrees of computed, non-Copy expression handles.
#[doc(hidden)]
pub trait ExpressionDegree: Degree {}

/// Degree markers reserved for immutable variable and parameter identities.
#[doc(hidden)]
pub trait LeafDegree: Degree + sealed::Sealed<Publication = ()> {}

/// Result degree of addition and subtraction.
pub trait AddDegree<R: Degree>: Degree {
    type Output: ExpressionDegree;
}

/// Result degree of multiplication.
pub trait MulDegree<R: Degree>: Degree {
    type Output: ExpressionDegree;
}

/// Result degree of expression division.
pub trait DivDegree<R: Degree>: Degree {
    type Output: ExpressionDegree;
}

/// Constant expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Constant;
impl sealed::Sealed for Constant {
    const MAY_BE_QUADRATIC: bool = false;
    type Publication = sealed::ComputedPublication;
}
impl Degree for Constant {
    type Computed = Self;
}
impl ExpressionDegree for Constant {}

/// Affine expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Affine;
impl sealed::Sealed for Affine {
    const MAY_BE_QUADRATIC: bool = false;
    type Publication = sealed::ComputedPublication;
}
impl Degree for Affine {
    type Computed = Self;
}
impl ExpressionDegree for Affine {}

/// Quadratic expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Quadratic;
impl sealed::Sealed for Quadratic {
    type Publication = sealed::ComputedPublication;
}
impl Degree for Quadratic {
    type Computed = Self;
}
impl ExpressionDegree for Quadratic {}

/// Nonlinear expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Nonlinear;
impl sealed::Sealed for Nonlinear {
    type Publication = sealed::ComputedPublication;
}
impl Degree for Nonlinear {
    type Computed = Self;
}
impl ExpressionDegree for Nonlinear {}

/// Dynamic expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Dynamic;
impl sealed::Sealed for Dynamic {
    type Publication = sealed::ComputedPublication;
}
impl Degree for Dynamic {
    type Computed = Self;
}
impl ExpressionDegree for Dynamic {}

// Explicit tables keep static propagation auditable; no user-supplied degree markers.
macro_rules! pair {
    ($left:ty, $right:ty => $add:ty, $mul:ty, $div:ty) => {
        impl AddDegree<$right> for $left {
            type Output = $add;
        }
        impl MulDegree<$right> for $left {
            type Output = $mul;
        }
        impl DivDegree<$right> for $left {
            type Output = $div;
        }
    };
}

pair!(Constant, Constant => Constant, Constant, Nonlinear);
pair!(Constant, Affine => Affine, Affine, Nonlinear);
pair!(Constant, Quadratic => Quadratic, Quadratic, Nonlinear);
pair!(Constant, Nonlinear => Nonlinear, Nonlinear, Nonlinear);
pair!(Constant, Dynamic => Dynamic, Dynamic, Dynamic);
pair!(Affine, Constant => Affine, Affine, Nonlinear);
pair!(Affine, Affine => Affine, Quadratic, Nonlinear);
pair!(Affine, Quadratic => Quadratic, Nonlinear, Nonlinear);
pair!(Affine, Nonlinear => Nonlinear, Nonlinear, Nonlinear);
pair!(Affine, Dynamic => Dynamic, Dynamic, Dynamic);
pair!(Quadratic, Constant => Quadratic, Quadratic, Nonlinear);
pair!(Quadratic, Affine => Quadratic, Nonlinear, Nonlinear);
pair!(Quadratic, Quadratic => Quadratic, Nonlinear, Nonlinear);
pair!(Quadratic, Nonlinear => Nonlinear, Nonlinear, Nonlinear);
pair!(Quadratic, Dynamic => Dynamic, Dynamic, Dynamic);
pair!(Nonlinear, Constant => Nonlinear, Nonlinear, Nonlinear);
pair!(Nonlinear, Affine => Nonlinear, Nonlinear, Nonlinear);
pair!(Nonlinear, Quadratic => Nonlinear, Nonlinear, Nonlinear);
pair!(Nonlinear, Nonlinear => Nonlinear, Nonlinear, Nonlinear);
pair!(Nonlinear, Dynamic => Dynamic, Dynamic, Dynamic);
pair!(Dynamic, Constant => Dynamic, Dynamic, Dynamic);
pair!(Dynamic, Affine => Dynamic, Dynamic, Dynamic);
pair!(Dynamic, Quadratic => Dynamic, Dynamic, Dynamic);
pair!(Dynamic, Nonlinear => Dynamic, Dynamic, Dynamic);
pair!(Dynamic, Dynamic => Dynamic, Dynamic, Dynamic);

/// Identity of a decision-variable leaf.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct VariableDegree;
impl sealed::Sealed for VariableDegree {
    const MAY_BE_QUADRATIC: bool = false;
    type Publication = ();
}
impl Degree for VariableDegree {
    type Computed = Affine;
}
impl LeafDegree for VariableDegree {}

/// Identity of a parameter leaf; arithmetic produces ordinary degrees.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct ParameterDegree;
impl sealed::Sealed for ParameterDegree {
    const MAY_BE_QUADRATIC: bool = false;
    type Publication = ();
}
impl Degree for ParameterDegree {
    type Computed = Constant;
}
impl LeafDegree for ParameterDegree {}

macro_rules! leaf_pair {
    ($left:ty, $right:ty => $normal_left:ty, $normal_right:ty) => {
        pair!($left, $right =>
            <$normal_left as AddDegree<$normal_right>>::Output,
            <$normal_left as MulDegree<$normal_right>>::Output,
            <$normal_left as DivDegree<$normal_right>>::Output);
    };
}
leaf_pair!(Constant, VariableDegree => Constant, Affine);
leaf_pair!(Constant, ParameterDegree => Constant, Constant);
leaf_pair!(Affine, VariableDegree => Affine, Affine);
leaf_pair!(Affine, ParameterDegree => Affine, Constant);
leaf_pair!(Quadratic, VariableDegree => Quadratic, Affine);
leaf_pair!(Quadratic, ParameterDegree => Quadratic, Constant);
leaf_pair!(Nonlinear, VariableDegree => Nonlinear, Affine);
leaf_pair!(Nonlinear, ParameterDegree => Nonlinear, Constant);
leaf_pair!(Dynamic, VariableDegree => Dynamic, Affine);
leaf_pair!(Dynamic, ParameterDegree => Dynamic, Constant);
leaf_pair!(VariableDegree, Constant => Affine, Constant);
leaf_pair!(VariableDegree, Affine => Affine, Affine);
leaf_pair!(VariableDegree, Quadratic => Affine, Quadratic);
leaf_pair!(VariableDegree, Nonlinear => Affine, Nonlinear);
leaf_pair!(VariableDegree, Dynamic => Affine, Dynamic);
leaf_pair!(VariableDegree, VariableDegree => Affine, Affine);
leaf_pair!(VariableDegree, ParameterDegree => Affine, Constant);
leaf_pair!(ParameterDegree, Constant => Constant, Constant);
leaf_pair!(ParameterDegree, Affine => Constant, Affine);
leaf_pair!(ParameterDegree, Quadratic => Constant, Quadratic);
leaf_pair!(ParameterDegree, Nonlinear => Constant, Nonlinear);
leaf_pair!(ParameterDegree, Dynamic => Constant, Dynamic);
leaf_pair!(ParameterDegree, VariableDegree => Constant, Affine);
leaf_pair!(ParameterDegree, ParameterDegree => Constant, Constant);
