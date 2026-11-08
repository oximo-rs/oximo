use oximo_core::ExprNode;
use oximo_core::prelude::*;

fn domain() -> Result<std::ops::Range<usize>, &'static str> {
    Ok(0..3)
}

fn allowed(i: usize) -> Result<bool, &'static str> {
    Ok(i != 1)
}

async fn async_extrema() -> Result<(), &'static str> {
    let model = Model::new("async_extrema");
    variable!(model, x[i in 0..3]);
    let _ = min!(x[i] for i in async { domain() }.await? if async { allowed(i) }.await?);
    let _ = max!(x[i] for i in async { domain() }.await? if async { allowed(i) }.await?);
    Ok(())
}

fn main() -> Result<(), &'static str> {
    let model = Model::new("fallible_extrema");
    variable!(model, x[i in 0..3]);
    let lo = min!(x[i] for i in domain()?);
    let hi = max!(x[i] for i in domain()?);
    let filtered_lo = min!(x[i] for i in 0..3 if allowed(i)?);
    let filtered_hi = max!(x[i] for i in 0..3 if allowed(i)?);
    let nested_lo = min!(x[j] for i in domain()?, j in Ok::<_, &str>(0..i)? if allowed(j)?);
    let nested_hi = max!(x[j] for i in domain()?, j in Ok::<_, &str>(0..i)? if allowed(j)?);
    let arena = model.arena();
    assert!(matches!(arena.get(lo.id()), ExprNode::Min(children) if children.len() == 3));
    assert!(matches!(arena.get(hi.id()), ExprNode::Max(children) if children.len() == 3));
    for expr in [filtered_lo, nested_lo] {
        assert!(matches!(arena.get(expr.id()), ExprNode::Min(children) if children.len() == 2));
    }
    for expr in [filtered_hi, nested_hi] {
        assert!(matches!(arena.get(expr.id()), ExprNode::Max(children) if children.len() == 2));
    }
    let _future = async_extrema();
    Ok(())
}
