use oximo_core::prelude::*;
use oximo_expr::QuadraticTerms;

fn main() {
    let mut coefficients = QuadraticTerms::default();
    coefficients.quadratic.push((VarId(0), VarId(0), f64::NAN));
}
