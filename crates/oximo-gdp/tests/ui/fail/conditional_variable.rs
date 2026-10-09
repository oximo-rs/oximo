use oximo_gdp::prelude::*;
fn main() {
    let m=Model::new("conditional variable");
    disjunct!(m,branch,|d| {variable!(d,x);});
}
