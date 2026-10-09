use oximo_gdp::prelude::*;
fn main() {
    let m=Model::new("compile");
    variable!(m, 0.0<=x[i in 0..2, j in 0..2]<=4.0);
    boolean_variable!(m, y[i in 0..2, j in 0..2]);
    let rows=disjunct_constraint!(m,y[i,j],row[i in 0..2,j in 0..2],x[i,j].square()<=2.0);
    assert_eq!(rows.len(),4);
    let nested=disjunct!(m,outer,|d| {
        let a=disjunct!(d,a,|a| {constraint!(a,x[0,0]<=1.0);});
        let b=disjunct!(d,b,|_b| {});
        disjunction!(d,[a,b]);
    });
    disjunction!(m,[nested]);
    logical_constraint!(m,exactly(1,[y[0,0],y[1,1]]));
    m.reformulate_gdp(BigM::default()).unwrap();
}
