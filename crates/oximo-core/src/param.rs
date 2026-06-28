use oximo_expr::ParamId;
use smol_str::SmolStr;

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
