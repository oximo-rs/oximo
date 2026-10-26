//! Explicit coordinates emitted by a reformulation.

use crate::{ConstraintId, VarId};

macro_rules! generated_id {
    ($name:ident, $raw:ty) => {
        /// A generated coordinate. Use `raw_id` explicitly for registry-level work.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name($raw);

        impl $name {
            /// Mark a registry coordinate emitted by a reformulation.
            /// This unchecked registry escape hatch trusts the caller's provenance claim.
            pub const fn from_raw(id: $raw) -> Self {
                Self(id)
            }

            /// Access the underlying registry ID, explicitly discarding provenance.
            pub const fn raw_id(self) -> $raw {
                self.0
            }

            pub fn index(self) -> usize {
                self.0.index()
            }
        }
    };
}

generated_id!(GeneratedVarId, VarId);
generated_id!(GeneratedConstraintId, ConstraintId);

/// A Hull mapping may refer to a source coordinate or another generated copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReformulationVarId {
    Source(VarId),
    Generated(GeneratedVarId),
}

impl ReformulationVarId {
    /// Return the original source coordinate, if this is a source reference.
    pub const fn source(self) -> Option<VarId> {
        match self {
            Self::Source(id) => Some(id),
            Self::Generated(_) => None,
        }
    }

    /// Return the generated coordinate, if this is a generated reference.
    pub const fn generated(self) -> Option<GeneratedVarId> {
        match self {
            Self::Generated(id) => Some(id),
            Self::Source(_) => None,
        }
    }

    /// Access the registry ID, explicitly discarding provenance.
    pub const fn raw_id(self) -> VarId {
        match self {
            Self::Source(id) => id,
            Self::Generated(id) => id.raw_id(),
        }
    }

    pub fn index(self) -> usize {
        self.raw_id().index()
    }
}
