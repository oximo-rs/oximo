use oximo_gdp::prelude::*;
fn main() {
    let m=Model::new("conditional objective");
    variable!(m,x);
    disjunct!(m,branch,|d| {objective!(d,Min,x);});
}
