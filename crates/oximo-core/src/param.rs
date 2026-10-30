use oximo_expr::ParamId;
use smol_str::SmolStr;
use std::ops::Deref;
use std::sync::Arc;

/// Parameter identity (id + name). Parameters are scalars referenced
/// symbolically in expressions and re-bound between solves.
///
/// The current numeric value lives in the model's expression arena
/// (the single source of truth) so re-binding does not have to
/// keep two copies in sync. Read it with [`Model::param_value`] /
/// [`Model::param_value_of`].
///
/// An indexed family of parameters (one `Parameter` per key) is built by the
/// indexed form of the `param!` macro and surfaced as an
/// [`IndexedParam`](crate::IndexedParam).
///
/// [`Model::param_value`]: crate::Model::param_value
/// [`Model::param_value_of`]: crate::Model::param_value_of
/// [`Model::set_param_idx`]: crate::Model::set_param_idx
#[derive(Clone, Debug)]
pub struct Parameter {
    pub id: ParamId,
    pub name: SmolStr,
}

/// Immutable parameter declarations captured from one model.
#[derive(Clone, Debug)]
pub struct ParameterDeclarations {
    pub(crate) model_id: oximo_expr::ModelId,
    pub(crate) parameters: Option<Arc<Vec<Parameter>>>,
}

impl ParameterDeclarations {
    /// Exact model whose declarations were captured.
    #[must_use]
    pub fn model_id(&self) -> oximo_expr::ModelId {
        self.model_id
    }

    /// Compare the first `count` declaration IDs and names.
    #[must_use]
    pub fn matches_prefix(&self, other: &[Parameter], count: usize) -> bool {
        if count > self.len() || count > other.len() {
            return false;
        }
        if count == 0 {
            return true;
        }
        if self.as_ptr() == other.as_ptr() {
            return true;
        }
        self.iter().zip(other.iter()).take(count).all(|(a, b)| a.id == b.id && a.name == b.name)
    }
}

impl Deref for ParameterDeclarations {
    type Target = [Parameter];

    fn deref(&self) -> &Self::Target {
        self.parameters.as_deref().map_or(&[], Vec::as_slice)
    }
}
