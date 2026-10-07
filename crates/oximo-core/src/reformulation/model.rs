//! Shared wrapper for independently transformed models.

use std::ops::Deref;

use crate::Model;

/// An independent transformed model plus source-to-generated provenance.
///
/// The underlying model retains each transformation's history.
/// Methods specific to a reformulation live alongside its implementation.
#[derive(Debug)]
pub struct ReformulatedModel {
    model: Model,
}

impl ReformulatedModel {
    /// Wrap a model after an explicit transformation has completed.
    #[doc(hidden)]
    pub fn __from_model(model: Model) -> Self {
        Self { model }
    }

    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[must_use]
    pub fn into_model(self) -> Model {
        self.model
    }
}

impl Deref for ReformulatedModel {
    type Target = Model;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl AsRef<Model> for ReformulatedModel {
    fn as_ref(&self) -> &Model {
        &self.model
    }
}
