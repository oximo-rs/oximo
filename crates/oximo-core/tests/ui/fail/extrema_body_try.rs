use oximo_core::prelude::*;

fn main() -> Result<(), ()> {
    let model = Model::new("extrema_try");
    variable!(model, x);
    let term = Ok::<_, ()>(x);
    let _ = min!(term? for i in 0..3);
    let _ = max!(term? for i in 0..3 if i > 0);
    Ok(())
}
