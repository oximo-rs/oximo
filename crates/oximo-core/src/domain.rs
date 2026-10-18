use std::fmt;

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

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Real => f.write_str("real"),
            Self::Integer => f.write_str("integer"),
            Self::Binary => f.write_str("binary"),
            Self::SemiContinuous { threshold } => write!(f, "semi-continuous({threshold})"),
            Self::SemiInteger { threshold } => write!(f, "semi-integer({threshold})"),
        }
    }
}

impl Domain {
    /// Whether this domain is integer-valued (Integer, Binary, SemiInteger)
    #[must_use]
    pub fn is_integer(self) -> bool {
        matches!(self, Self::Integer | Self::Binary | Self::SemiInteger { .. })
    }

    /// Integrality or a semicontinuity disjunction requires a discrete modeling capability.
    pub fn requires_discrete_choice(self) -> bool {
        !matches!(self, Self::Real)
    }

    /// The semicontinuity gap floor: `Some(threshold)` for `SemiContinuous` /
    /// `SemiInteger`, `None` otherwise. Such a variable takes either 0 or a
    /// value `>= threshold`, so backends emit `threshold` as the lower bound.
    #[must_use]
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

    #[test]
    fn display_uses_user_facing_ascii_labels() {
        let labels = [
            Domain::Real.to_string(),
            Domain::Integer.to_string(),
            Domain::Binary.to_string(),
            Domain::SemiContinuous { threshold: 3.0 }.to_string(),
            Domain::SemiInteger { threshold: 2.5 }.to_string(),
        ];
        assert_eq!(
            labels,
            ["real", "integer", "binary", "semi-continuous(3)", "semi-integer(2.5)"]
        );
        assert!(labels.iter().all(|label| label.is_ascii()));
    }

    #[test]
    fn default_domain_is_real() {
        assert_eq!(Domain::default(), Domain::Real);
    }

    #[test]
    fn integer_domains_are_integer() {
        assert!(Domain::Integer.is_integer());
        assert!(Domain::Binary.is_integer());
        assert!(Domain::SemiInteger { threshold: 2.0 }.is_integer());
    }

    #[test]
    fn non_integer_domains_are_not_integer() {
        assert!(!Domain::Real.is_integer());
        assert!(!Domain::SemiContinuous { threshold: 2.0 }.is_integer());
    }
}
