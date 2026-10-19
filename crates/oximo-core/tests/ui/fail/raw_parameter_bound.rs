use oximo_core::prelude::*;
fn main(){let model=Model::new("raw bound");model.__var("x").ub(Bound::Parameter(ParamId(0))).build();}
