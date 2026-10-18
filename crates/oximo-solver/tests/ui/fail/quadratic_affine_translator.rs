use oximo_core::prelude::*;
use oximo_solver::{AffineRowTranslator, TranslationContext};
fn main() {
    let m = Model::new("typed translation");
    variable!(m, x);
    let quadratic = Function::semantic_data(&x.square().into_function());
    TranslationContext::new(&m).unwrap().translate(&AffineRowTranslator, &quadratic, &LessThan(1.0)).unwrap();
}
