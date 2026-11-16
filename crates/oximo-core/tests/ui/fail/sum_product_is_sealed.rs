use oximo_expr::sum_support::{ProductTag, SumProduct};
use oximo_expr::{Affine, Expr};

struct Weight;

impl std::ops::Mul<()> for Weight {
    type Output = ();
    fn mul(self, _: ()) {}
}

// A custom Mul may use the ordinary fallback, but cannot add fused dispatch.
impl SumProduct<Weight, ()> for &&ProductTag<Weight, ()> {
    type Output = ();
    fn sum_product(self, _: (Weight, ())) {}
}

// An existing optimized receiver cannot be reused with different operands.
impl<'a> SumProduct<Weight, ()> for &&ProductTag<f64, Expr<'a, Affine>> {
    type Output = ();
    fn sum_product(self, _: (Weight, ())) {}
}

fn main() {}
