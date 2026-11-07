use oximo_core::prelude::*;

struct Owned(f64);
impl std::ops::Add for Owned {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl std::ops::Neg for Owned {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

fn main() {
    let m = Model::new("ordinary Rust arithmetic");
    variable!(m, x);
    objective!(m, Min, {
        let prefix = String::from("a");
        let name = prefix + "b";
        let left = Owned(1.0);
        let right = Owned(2.0);
        let total = left + -right;
        (name.len() as f64 + total.0) * x
    });
    param!(
        m,
        p = {
            let prefix = String::from("a");
            (prefix + "b").len() as f64
        }
    );
    let _ = sum!(m, {
        let prefix = String::from("a");
        (prefix + "b").len() as f64 * x
    } for _i in 0..2);
    constraint!(m, p * x <= 4.0);
}
