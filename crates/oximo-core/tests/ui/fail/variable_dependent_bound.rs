use oximo_core::prelude::*;
fn main(){let model=Model::new("variable bound");variable!(model,y);variable!(model,x<=y);}
