use oximo_gdp::prelude::*;
fn main() {
    let m=Model::new("conditional cone");
    variable!(m,x);variable!(m,t);
    disjunct!(m,branch,|d| {soc_constraint!(d,[x]<=t);});
}
