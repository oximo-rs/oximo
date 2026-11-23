//! Dense storage for explicit second-order cone dual vectors.

use oximo_core::{SocConstraintId, SocForm, SocHead, SocKind};

/// Dual vectors of explicit second-order cones, keyed by [`SocConstraintId`].
///
/// Each vector is in declared component order:
/// `[t, w...]` for a standard cone and
/// `[u, v, w...]` for a rotated one.
#[derive(Clone, Debug, Default)]
pub struct SocDuals {
    /// `(start, len)` per cone ID.
    spans: Vec<(u32, u32)>,
    values: Vec<f64>,
    /// Number of present entries.
    present: usize,
}

impl PartialEq for SocDuals {
    fn eq(&self, other: &Self) -> bool {
        self.present == other.present && self.iter().eq(other.iter())
    }
}

/// A recorded cone dual viewed with its cone kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SocDualRef<'a> {
    kind: SocKind,
    components: &'a [f64],
}

impl<'a> SocDualRef<'a> {
    /// View `components` as a `kind` cone dual, or `None` when there is no
    /// term component after the head.
    #[must_use]
    pub fn new(kind: SocKind, components: &'a [f64]) -> Option<Self> {
        (components.len() >= kind.min_dimension()).then_some(Self { kind, components })
    }

    /// The cone kind.
    #[must_use]
    pub const fn kind(self) -> SocKind {
        self.kind
    }

    /// The head multipliers.
    ///
    /// # Panics
    /// It never panics since [`Self::new`] checks that the head fits.
    #[must_use]
    pub fn head(self) -> SocHead<f64> {
        SocHead::from_prefix(self.kind, self.components).expect("checked dual dimension")
    }

    /// The multipliers of the norm terms `w`.
    #[must_use]
    pub fn terms(self) -> &'a [f64] {
        &self.components[self.kind.head_len()..]
    }

    /// Every component in declared order: `[zt, zw...]` or `[zu, zv, zw...]`.
    #[must_use]
    pub const fn as_slice(self) -> &'a [f64] {
        self.components
    }

    /// The norm-form bound multiplier `zt` of a standard cone, or `None` for a
    /// rotated one.
    #[must_use]
    pub fn standard_bound(self) -> Option<f64> {
        match self.head() {
            SocHead::Standard(t) => Some(t),
            SocHead::Rotated(..) => None,
        }
    }
}

impl SocDuals {
    /// An empty map.
    #[must_use]
    pub const fn new() -> Self {
        Self { spans: Vec::new(), values: Vec::new(), present: 0 }
    }

    /// An empty map with room for `cones` IDs and `values` total components.
    #[must_use]
    pub fn with_capacity(cones: usize, values: usize) -> Self {
        Self { spans: Vec::with_capacity(cones), values: Vec::with_capacity(values), present: 0 }
    }

    fn range(span: (u32, u32)) -> std::ops::Range<usize> {
        let start = span.0 as usize;
        start..start + span.1 as usize
    }

    /// The dual vector of `id`, if recorded.
    #[must_use]
    pub fn get(&self, id: SocConstraintId) -> Option<&[f64]> {
        let &span = self.spans.get(id.index())?;
        (span.1 > 0).then(|| &self.values[Self::range(span)])
    }

    /// The dual vector of `id` for in-place adjustment, if recorded.
    pub fn get_mut(&mut self, id: SocConstraintId) -> Option<&mut [f64]> {
        let &span = self.spans.get(id.index())?;
        (span.1 > 0).then(|| &mut self.values[Self::range(span)])
    }

    /// Record `dual` for `id`, replacing any previous vector, and return the
    /// stored components.
    ///
    /// # Panics
    /// Panics if the total number of stored components exceeds `u32::MAX`.
    pub fn insert(&mut self, id: SocConstraintId, dual: &[f64]) -> &mut [f64] {
        if let Some(previous) = self.get_mut(id)
            && previous.len() == dual.len()
        {
            previous.copy_from_slice(dual);
            return self.get_mut(id).expect("present");
        }
        self.insert_iter(id, dual.iter().copied())
    }

    /// Record the components yielded by `dual` for `id`.
    ///
    /// # Panics
    /// Panics if the total number of stored components exceeds `u32::MAX`.
    pub fn insert_iter(
        &mut self,
        id: SocConstraintId,
        dual: impl IntoIterator<Item = f64>,
    ) -> &mut [f64] {
        let index = id.index();
        if self.spans.len() <= index {
            self.spans.resize(index + 1, (0, 0));
        }
        let start = self.values.len();
        self.values.extend(dual);
        let len = self.values.len() - start;
        let span = if len == 0 {
            (0, 0)
        } else {
            (
                u32::try_from(start).expect("SOC dual storage overflow"),
                u32::try_from(len).expect("SOC dual storage overflow"),
            )
        };

        let previous = std::mem::replace(&mut self.spans[index], span);
        match (previous.1 > 0, len > 0) {
            (false, true) => self.present += 1,
            (true, false) => self.present -= 1,
            _ => {}
        }

        &mut self.values[start..start + len]
    }

    /// Record the dual `z` of a cone that was solved in standard form
    /// ([`SocForm::into_standard`] for a rotated cone).
    /// For a rotated cone the input `[z0, z1, zw...]` becomes
    /// `[(z0+z1)/sqrt2, (z0-z1)/sqrt2, zw...]`.
    pub fn insert_from_standard(&mut self, id: SocConstraintId, kind: SocKind, z: &[f64]) {
        let stored = self.insert(id, z);
        if kind == SocKind::Rotated
            && let [z0, z1, ..] = stored
        {
            let r = std::f64::consts::FRAC_1_SQRT_2;
            (*z0, *z1) = ((*z0 + *z1) * r, (*z0 - *z1) * r);
        }
    }

    /// Record the cone dual recovered from the multiplier `lambda` of the
    /// squared lowering `||w||^2 - t^2 <= 0` (standard) or
    /// `||w||^2 - 2 u v <= 0` (rotated).
    pub fn insert_from_squared_row(
        &mut self,
        id: SocConstraintId,
        head: SocHead<f64>,
        terms: impl IntoIterator<Item = f64>,
        lambda: f64,
    ) {
        self.insert_from_squared_row_with_signs(id, head, terms, lambda, [0.0; 2]);
    }

    /// Like [`Self::insert_from_squared_row`], also adding the magnitude of
    /// each head's `head >= 0` sign-row multiplier to that head component.
    pub fn insert_from_squared_row_with_signs(
        &mut self,
        id: SocConstraintId,
        head: SocHead<f64>,
        terms: impl IntoIterator<Item = f64>,
        lambda: f64,
        sign: [f64; 2],
    ) {
        let scale = 2.0 * lambda.abs();
        let (head, head_len) = match head {
            SocHead::Standard(t) => ([t * scale + sign[0].abs(), 0.0], 1),
            SocHead::Rotated(u, v) => ([v * scale + sign[0].abs(), u * scale + sign[1].abs()], 2),
        };
        self.insert_iter(
            id,
            head.into_iter().take(head_len).chain(terms.into_iter().map(|w| -w * scale)),
        );
    }

    /// Recover each cone's dual from the multiplier of its squared lowering
    /// (see [`Self::insert_from_squared_row`]).
    #[must_use]
    pub fn from_squared_rows<'f>(
        forms: impl IntoIterator<Item = (SocConstraintId, &'f SocForm)>,
        multiplier: impl Fn(SocConstraintId) -> Option<f64>,
        primal: impl Fn(oximo_expr::VarId) -> Option<f64>,
    ) -> Self {
        Self::from_squared_rows_with_signs(forms, multiplier, |_, _| Some(0.0), primal)
    }

    /// Like [`Self::from_squared_rows`] for lowerings that also emit a
    /// `head >= 0` row per head with variables.
    #[must_use]
    pub fn from_squared_rows_with_signs<'f>(
        forms: impl IntoIterator<Item = (SocConstraintId, &'f SocForm)>,
        multiplier: impl Fn(SocConstraintId) -> Option<f64>,
        sign_multiplier: impl Fn(SocConstraintId, usize) -> Option<f64>,
        primal: impl Fn(oximo_expr::VarId) -> Option<f64>,
    ) -> Self {
        let value_of = |terms: &oximo_expr::LinearTerms<'_>| {
            terms
                .coeffs
                .iter()
                .try_fold(terms.constant, |sum, &(v, c)| primal(v).map(|value| sum + c * value))
        };

        let mut duals = Self::new();
        let mut members = Vec::new();
        for (id, form) in forms {
            let Some(lambda) = multiplier(id) else { continue };
            let Some(head) = form.head.as_ref().try_map(|t| value_of(t).ok_or(())).ok() else {
                continue;
            };
            members.clear();
            if form.terms.iter().any(|w| value_of(w).map(|w| members.push(w)).is_none()) {
                continue;
            }
            let mut sign = [0.0; 2];
            let mut complete = true;
            for (k, h) in form.head.iter().enumerate() {
                if h.coeffs.iter().all(|&(_, c)| c == 0.0) {
                    continue;
                }
                match sign_multiplier(id, k) {
                    Some(m) => sign[k] = m,
                    None => complete = false,
                }
            }
            if !complete {
                continue;
            }
            duals.insert_from_squared_row_with_signs(
                id,
                head,
                members.iter().copied(),
                lambda,
                sign,
            );
        }

        duals
    }

    /// Remove the entry for `id`.
    pub fn remove(&mut self, id: SocConstraintId) {
        if let Some(span) = self.spans.get_mut(id.index())
            && span.1 > 0
        {
            *span = (0, 0);
            self.present -= 1;
        }
    }

    /// Keep only the entries for which `keep` returns true.
    pub fn retain(&mut self, mut keep: impl FnMut(SocConstraintId, &[f64]) -> bool) {
        let mut removed = false;
        for (index, span) in self.spans.iter_mut().enumerate() {
            if span.1 > 0 && !keep(slot_id(index), &self.values[Self::range(*span)]) {
                *span = (0, 0);
                self.present -= 1;
                removed = true;
            }
        }

        if removed || self.has_garbage() {
            self.compact();
        }
    }

    /// Drop every entry with an ID at or above `cones`.
    pub fn truncate(&mut self, cones: usize) {
        if cones < self.spans.len() {
            self.present -= self.spans[cones..].iter().filter(|span| span.1 > 0).count();
            self.spans.truncate(cones);
            self.compact();
        }
    }

    /// Remove every entry.
    pub fn clear(&mut self) {
        self.spans.clear();
        self.values.clear();
        self.present = 0;
    }

    /// Number of recorded cones.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.present
    }

    /// Whether no cone has a recorded dual.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.present == 0
    }

    /// One past the largest ID slot allocated so far.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.spans.len()
    }

    /// Recorded `(id, dual)` pairs in ID order.
    pub fn iter(&self) -> impl Iterator<Item = (SocConstraintId, &[f64])> + '_ {
        self.spans
            .iter()
            .enumerate()
            .filter(|(_, span)| span.1 > 0)
            .map(|(index, &span)| (slot_id(index), &self.values[Self::range(span)]))
    }

    /// Whether replaced or removed components still occupy the value buffer.
    fn has_garbage(&self) -> bool {
        self.spans.iter().map(|span| span.1 as usize).sum::<usize>() != self.values.len()
    }

    /// Drop replaced or removed components and trailing empty slots.
    fn compact(&mut self) {
        while self.spans.last().is_some_and(|span| span.1 == 0) {
            self.spans.pop();
        }
        if !self.has_garbage() {
            return;
        }

        let mut values = Vec::with_capacity(self.spans.iter().map(|span| span.1 as usize).sum());
        for span in &mut self.spans {
            if span.1 > 0 {
                let start = values.len();
                values.extend_from_slice(&self.values[Self::range(*span)]);
                span.0 = u32::try_from(start).expect("SOC dual storage overflow");
            }
        }

        self.values = values;
    }
}

/// The ID of a slot index.
#[expect(clippy::cast_possible_truncation)]
fn slot_id(index: usize) -> SocConstraintId {
    debug_assert!(u32::try_from(index).is_ok());
    SocConstraintId(index as u32)
}

impl<V: AsRef<[f64]>> Extend<(SocConstraintId, V)> for SocDuals {
    fn extend<T: IntoIterator<Item = (SocConstraintId, V)>>(&mut self, iter: T) {
        for (id, dual) in iter {
            self.insert(id, dual.as_ref());
        }
    }
}

impl<V: AsRef<[f64]>> FromIterator<(SocConstraintId, V)> for SocDuals {
    fn from_iter<T: IntoIterator<Item = (SocConstraintId, V)>>(iter: T) -> Self {
        let mut duals = Self::new();
        duals.extend(iter);
        duals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_replace_remove_and_compact() {
        let mut duals = SocDuals::new();
        duals.insert(SocConstraintId(2), &[1.0, 0.5]);
        duals.insert(SocConstraintId(0), &[3.0, 1.0, 1.0]);
        assert_eq!(duals.len(), 2);
        assert_eq!(duals.slot_count(), 3);
        assert_eq!(duals.get(SocConstraintId(1)), None);
        assert_eq!(duals.get(SocConstraintId(2)), Some([1.0, 0.5].as_slice()));

        duals.insert(SocConstraintId(2), &[2.0, 0.0]);
        assert_eq!(duals.values.len(), 5);
        assert_eq!(duals.len(), 2);
        assert_eq!(duals.get(SocConstraintId(2)), Some([2.0, 0.0].as_slice()));

        duals.insert(SocConstraintId(2), &[2.0, 0.0, 1.0]);
        assert_eq!(duals.values.len(), 8);

        duals.retain(|_, _| true);
        assert_eq!(duals.values.len(), 6);

        duals.retain(|_, z| z[0] < 2.5);
        assert_eq!(duals.len(), 1);
        assert_eq!(duals.values.len(), 3);
        assert_eq!(
            duals.iter().collect::<Vec<_>>(),
            vec![(SocConstraintId(2), [2.0, 0.0, 1.0].as_slice())]
        );

        duals.truncate(2);
        assert!(duals.is_empty());
        assert_eq!(duals.slot_count(), 0);

        duals.insert(SocConstraintId(1), &[1.0]);
        duals.insert(SocConstraintId(1), &[]);
        assert!(duals.is_empty());

        duals.remove(SocConstraintId(7));
        assert!(duals.is_empty());
    }

    #[test]
    fn standard_rewrite_dual_maps_back_into_the_rotated_dual_cone() {
        let mut duals = SocDuals::new();
        duals.insert_from_standard(SocConstraintId(0), SocKind::Rotated, &[2.0, 1.0, 0.5]);
        duals.insert_from_standard(SocConstraintId(1), SocKind::Standard, &[2.0, 1.0]);
        let rotated = SocDualRef::new(SocKind::Rotated, duals.get(SocConstraintId(0)).unwrap())
            .expect("rotated dual");
        let SocHead::Rotated(zu, zv) = rotated.head() else { panic!("rotated head") };
        assert!(zu >= 0.0 && zv >= 0.0);
        assert!(2.0 * zu * zv >= rotated.terms().iter().map(|z| z * z).sum::<f64>() - 1e-12);
        assert!(((zu + zv) * std::f64::consts::FRAC_1_SQRT_2 - 2.0).abs() < 1e-12);
        assert_eq!(rotated.standard_bound(), None);
        assert_eq!(duals.get(SocConstraintId(1)), Some([2.0, 1.0].as_slice()));
        assert_eq!(SocDualRef::new(SocKind::Rotated, &[1.0, 1.0]), None);
    }

    #[test]
    fn sign_row_multipliers_add_to_head_duals_at_the_origin() {
        let mut duals = SocDuals::new();
        duals.insert_from_squared_row_with_signs(
            SocConstraintId(0),
            SocHead::Rotated(0.0, 0.0),
            [0.0],
            0.0,
            [0.5, -0.25],
        );
        assert_eq!(duals.get(SocConstraintId(0)), Some([0.5, 0.25, 0.0].as_slice()));
    }

    #[test]
    fn squared_row_dual_lies_in_the_dual_cone() {
        let mut duals = SocDuals::new();
        duals.insert_from_squared_row(SocConstraintId(0), SocHead::Rotated(2.0, 3.0), [3.0], 0.5);
        let z = duals.get(SocConstraintId(0)).unwrap();
        assert_eq!(z, [3.0, 2.0, -3.0].as_slice());
        assert!(2.0 * z[0] * z[1] >= z[2] * z[2]);
    }
}
