use oximo_expr::VarId;
use rustc_hash::{FxBuildHasher, FxHashSet};
use smol_str::SmolStr;

use crate::model::Model;
use crate::reformulation::{
    ReformulatedModel, ReformulationError, SosReformulationArtifacts, SosReformulationOptions,
};

/// The two standard special ordered set types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SosType {
    /// At most one member may be nonzero.
    Sos1,
    /// At most two adjacent (by weight) members may be nonzero.
    Sos2,
}

impl SosType {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sos1 => "SOS1",
            Self::Sos2 => "SOS2",
        }
    }
}

impl std::fmt::Display for SosType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SosMember {
    pub variable: VarId,
    pub weight: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SosConstraintId(pub u32);

impl SosConstraintId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Typed semantic reference to a variable vector in a special ordered set.
pub type SosConstraintHandle =
    crate::ConstraintHandle<crate::VariableVectorData, crate::SpecialOrderedSet>;

impl SosConstraintHandle {
    /// # Errors
    /// Returns an ownership or reformulation error before changing the model.
    pub fn to_reformulated_model(
        self,
        model: &Model,
        options: SosReformulationOptions,
    ) -> Result<ReformulatedModel, ReformulationError> {
        model.ensure_model_id(self.model_id())?;
        model.to_reformulated_sos_constraint_model(self.sos_id(), options)
    }

    /// # Errors
    /// Returns an ownership or reformulation error before changing the model.
    pub fn reformulate(
        self,
        model: &Model,
        options: SosReformulationOptions,
    ) -> Result<Option<SosReformulationArtifacts>, ReformulationError> {
        model.ensure_model_id(self.model_id())?;
        model.reformulate_sos_constraint(self.sos_id(), options)
    }
}

/// An explicit SOS1 or SOS2 constraint.
#[derive(Clone, Debug)]
pub struct SosConstraint {
    pub name: SmolStr,
    pub sos_type: SosType,
    pub members: Vec<SosMember>,
    pub active: bool,
}

/// Validate an SOS member list before it is stored in a model.
pub(crate) fn validate_members(name: &str, members: &[SosMember]) {
    assert!(!members.is_empty(), "SOS constraint {name:?} has no members");
    let mut vars = FxHashSet::with_capacity_and_hasher(members.len(), FxBuildHasher);
    let mut weights = FxHashSet::with_capacity_and_hasher(members.len(), FxBuildHasher);
    for member in members {
        assert!(member.weight.is_finite(), "SOS constraint {name:?} has a non-finite weight");
        assert!(
            vars.insert(member.variable),
            "SOS constraint {name:?} contains a duplicate variable"
        );
        let key = match member.weight.to_bits() {
            0 | 0x8000_0000_0000_0000 => 0,
            bits => bits,
        };
        assert!(weights.insert(key), "SOS constraint {name:?} contains duplicate weights");
    }
}
