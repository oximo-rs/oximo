use oximo_expr::sum_support::{ProductTag, ProductValue, SumProduct};
use oximo_expr::{Affine, Expr};

struct Weight;

impl std::ops::Mul<()> for Weight {
    type Output = ();
    fn mul(self, _: ()) {}
}

// A custom Mul may use the ordinary fallback, but cannot add fused dispatch.
impl SumProduct<Weight, ()> for &&ProductTag<Weight, ()> {
    type Output = ();
    type PreparedLeft = Weight;
    fn prepare_left(self, left: Weight) -> Weight {
        left
    }
    fn sum_product(self, _: (Weight, ())) {}
}

// An existing optimized receiver cannot be reused with different operands.
impl<'a> SumProduct<Weight, ()>
    for &&ProductTag<ProductValue<f64>, ProductValue<Expr<'a, Affine>>>
{
    type Output = ();
    type PreparedLeft = Weight;
    fn prepare_left(self, left: Weight) -> Weight {
        left
    }
    fn sum_product(self, _: (Weight, ())) {}
}

fn main() {}
