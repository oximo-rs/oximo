/// The domain of a variable, which determines the type of values it can take.
///
/// Real: any real number.
/// Integer: any integer.
/// Binary: 0 or 1.
/// SemiContinuous: either 0 or any value >= threshold.
/// SemiInteger: either 0 or any integer >= threshold.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum Domain {
    #[default]
    Real,
    Integer,
    Binary,
    SemiContinuous {
        threshold: f64,
    },
    SemiInteger {
        threshold: f64,
    },
}

impl Domain {
    /// Whether this domain is integer-valued (Integer, Binary, SemiInteger)
    pub fn is_integer(self) -> bool {
        matches!(self, Self::Integer | Self::Binary | Self::SemiInteger { .. })
    }

    /// The semicontinuity gap floor: `Some(threshold)` for `SemiContinuous` /
    /// `SemiInteger`, `None` otherwise. Such a variable takes either 0 or a
    /// value `>= threshold`, so backends emit `threshold` as the lower bound.
    pub fn semi_threshold(self) -> Option<f64> {
        match self {
            Self::SemiContinuous { threshold } | Self::SemiInteger { threshold } => Some(threshold),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Domain;

    #[test]
    fn semi_threshold_only_for_semi_domains() {
        assert_eq!(Domain::Real.semi_threshold(), None);
        assert_eq!(Domain::Integer.semi_threshold(), None);
        assert_eq!(Domain::Binary.semi_threshold(), None);
        assert_eq!(Domain::SemiContinuous { threshold: 2.0 }.semi_threshold(), Some(2.0));
        assert_eq!(Domain::SemiInteger { threshold: 1.0 }.semi_threshold(), Some(1.0));
    }
}
