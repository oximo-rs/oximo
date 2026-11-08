/// Neumaier summation with ordinary IEEE arithmetic for nonfinite inputs.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn new(value: f64) -> Self {
        Self { sum: value, correction: 0.0 }
    }

    pub(crate) fn from_parts(sum: f64, correction: f64) -> Self {
        Self { sum, correction }
    }

    pub(crate) fn parts(self) -> (f64, f64) {
        (self.sum, self.correction)
    }

    pub(crate) fn add(&mut self, value: f64) {
        Self::add_parts(&mut self.sum, &mut self.correction, value);
    }

    #[inline]
    pub(crate) fn add_parts(sum: &mut f64, correction: &mut f64, value: f64) {
        let next = *sum + value;
        if next.is_finite() {
            *correction += if sum.abs() >= value.abs() {
                (*sum - next) + value
            } else {
                (value - next) + *sum
            };
        } else {
            // Avoid introducing NaN from inf-inf in the correction.
            *correction = 0.0;
        }
        *sum = next;
    }

    pub(crate) fn value(self) -> f64 {
        self.sum + self.correction
    }
}

#[cfg(test)]
mod tests {
    use super::CompensatedSum;

    #[test]
    fn finite_cancellation_keeps_small_terms_and_nonfinite_results_propagate() {
        for (values, expected) in [
            ([1e16, 1.0, -1e16], 1.0),
            ([f64::INFINITY, 1.0, 0.0], f64::INFINITY),
            ([f64::NEG_INFINITY, 1.0, 0.0], f64::NEG_INFINITY),
            ([f64::MAX, f64::MAX, 0.0], f64::INFINITY),
            ([f64::INFINITY, f64::NEG_INFINITY, 0.0], f64::NAN),
            ([f64::NAN, 1.0, 0.0], f64::NAN),
        ] {
            let mut sum = CompensatedSum::default();
            for value in values {
                sum.add(value);
            }
            if expected.is_nan() {
                assert!(sum.value().is_nan());
            } else {
                assert_eq!(sum.value(), expected);
            }
        }
    }
}
