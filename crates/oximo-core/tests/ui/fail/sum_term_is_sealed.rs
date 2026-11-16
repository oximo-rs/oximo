use oximo_expr::sum_support::SumTerm;
use oximo_expr::{Affine, Expr, ExprArenaCell};

struct CustomTerm;

impl<'a> SumTerm<'a> for CustomTerm {
    type Degree = Affine;
    type Buffer = ();
    type Evaluated = Expr<'a, Affine>;

    fn evaluate(self) -> Expr<'a, Affine> {
        unimplemented!()
    }

    fn buffer(_: Option<&'a ExprArenaCell>, _: usize) {}

    fn push(_: &mut (), _: Self) {}

    fn finish(_: ()) -> Option<Expr<'a, Affine>> {
        None
    }
}

fn main() {}
