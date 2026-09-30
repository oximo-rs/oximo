//! Conservative static expression degrees. Dynamic expressions retain runtime classification.
mod sealed {
    pub trait Sealed {}
}

/// A sealed upper bound on polynomial degree.
pub trait Degree: sealed::Sealed + Copy + std::fmt::Debug + Send + Sync + 'static {}

/// Result degree of addition and subtraction.
pub trait AddDegree<R: Degree>: Degree {
    type Output: Degree;
}

/// Result degree of multiplication.
pub trait MulDegree<R: Degree>: Degree {
    type Output: Degree;
}

/// Result degree of expression division.
pub trait DivDegree<R: Degree>: Degree {
    type Output: Degree;
}

/// Constant expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Constant;
impl sealed::Sealed for Constant {}
impl Degree for Constant {}

/// Affine expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Affine;
impl sealed::Sealed for Affine {}
impl Degree for Affine {}

/// Quadratic expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Quadratic;
impl sealed::Sealed for Quadratic {}
impl Degree for Quadratic {}

/// Nonlinear expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Nonlinear;
impl sealed::Sealed for Nonlinear {}
impl Degree for Nonlinear {}

/// Dynamic expression degree.
#[derive(Copy, Clone, Debug)]
pub struct Dynamic;
impl sealed::Sealed for Dynamic {}
impl Degree for Dynamic {}

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
