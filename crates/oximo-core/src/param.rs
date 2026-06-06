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
/// For now, we support scalar parameters only.
///
/// TODO: Add support for more parameters
///
/// [`Model::param_value`]: crate::Model::param_value
/// [`Model::param_value_of`]: crate::Model::param_value_of
#[derive(Clone, Debug)]
pub struct Parameter {
    pub id: ParamId,
    pub name: SmolStr,
}
