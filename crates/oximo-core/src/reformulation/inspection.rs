//! Borrow generated registry entries without copying model metadata.

use std::cell::Ref;

use crate::{Model, Variable};

pub(crate) fn variables<'model>(
    model: &'model Model,
    indices: impl ExactSizeIterator<Item = usize>,
) -> impl ExactSizeIterator<Item = Ref<'model, Variable>> {
    let variables = model.variables();
    indices.map(move |index| {
        Ref::map(Ref::clone(&variables), |variables| {
            variables.get(index).expect("unknown generated variable ID")
        })
    })
}
