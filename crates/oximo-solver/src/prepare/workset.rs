//! Snapshot-local numeric polynomials shared by classification, matching and upload.
//!
//! Locks protect publication and byte accounting only.
//! Each root has its own OnceLock, so equal-root consumers extract once and
//! independent roots can fold concurrently.
//! Metadata and coefficient budgets are both bounded.

use super::{CACHE_SHARDS, Extracted};
use oximo_expr::{ExprId, QuadraticTerms};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const ROOTS_PER_SHARD: usize = 1024;
const BYTES_PER_SHARD: usize = 2 * 1024 * 1024;
type Value = Arc<OnceLock<Option<Arc<QuadraticTerms>>>>;

#[derive(Debug, Default)]
struct Root {
    value: Option<Value>,
    bytes: usize,
}

#[derive(Debug, Default)]
struct Entries {
    roots: FxHashMap<ExprId, Root>,
    order: VecDeque<ExprId>,
    bytes: usize,
}

#[derive(Debug, Default)]
#[repr(align(64))]
struct Shard(Mutex<Entries>);

/// Counters count numeric quadratic extraction, including failures.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QuadraticAnalysisStats {
    pub attempts: usize,
    pub streamed_attempts: usize,
    pub hits: usize,
    pub roots: usize,
    pub re_extractions: usize,
    pub evictions: usize,
    pub bypasses: usize,
    pub retained_bytes: usize,
    pub peak_retained_bytes: usize,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct Counters {
    attempts: AtomicUsize,
    streamed_attempts: AtomicUsize,
    hits: AtomicUsize,
    roots: AtomicUsize,
    re_extractions: AtomicUsize,
    evictions: AtomicUsize,
    bypasses: AtomicUsize,
    bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
}

#[derive(Debug, Default)]
pub(super) struct QuadraticWorkset {
    shards: OnceLock<Box<[Shard; CACHE_SHARDS]>>,
    #[cfg(test)]
    counters: Counters,
}

pub(super) enum Lookup {
    Missing,
    Retry,
    Ready(Option<Extracted<QuadraticTerms>>),
}

// Compile instrumentation out of ordinary builds.
macro_rules! count {
    ($self:ident, $counter:ident) => {
        #[cfg(test)]
        $self.counters.$counter.fetch_add(1, Ordering::Relaxed);
    };
}

impl QuadraticWorkset {
    pub(super) fn lookup(&self, expr: ExprId) -> Lookup {
        let Some(shards) = self.shards.get() else { return Lookup::Missing };
        let shard = &shards[(expr.0.wrapping_mul(0x9e37_79b9) >> 28) as usize];
        let entries = shard.0.lock().expect("polynomial workset poisoned");
        let Some(root) = entries.roots.get(&expr) else { return Lookup::Missing };
        let Some(value) = root.value.as_ref().and_then(|cell| cell.get()) else {
            return Lookup::Retry;
        };
        count!(self, hits);
        Lookup::Ready(value.clone().map(Extracted::Shared))
    }

    #[cfg(test)]
    pub(super) fn record_streamed_attempt(&self) {
        count!(self, streamed_attempts);
    }

    pub(super) fn extract(
        &self,
        expr: ExprId,
        extract: impl FnOnce() -> Option<Arc<QuadraticTerms>>,
    ) -> Option<Extracted<QuadraticTerms>> {
        let shards =
            self.shards.get_or_init(|| Box::new(std::array::from_fn(|_| Shard::default())));
        let shard = &shards[(expr.0.wrapping_mul(0x9e37_79b9) >> 28) as usize];
        let value = {
            let mut entries = shard.0.lock().expect("polynomial workset poisoned");
            if let Some(root) = entries.roots.get_mut(&expr) {
                if let Some(value) = &root.value {
                    count!(self, hits);
                    Arc::clone(value)
                } else {
                    count!(self, re_extractions);
                    let value = Arc::new(OnceLock::new());
                    root.value = Some(Arc::clone(&value));
                    value
                }
            } else if entries.roots.len() < ROOTS_PER_SHARD {
                count!(self, roots);
                let value = Arc::new(OnceLock::new());
                entries.roots.insert(expr, Root { value: Some(Arc::clone(&value)), bytes: 0 });
                value
            } else {
                drop(entries);
                count!(self, bypasses);
                count!(self, attempts);
                return extract().map(Extracted::Shared);
            }
        };

        value
            .get_or_init(|| {
                count!(self, attempts);
                let result = extract();
                let bytes = result.as_ref().map_or(0, |q| {
                    q.linear.capacity() * std::mem::size_of::<(oximo_expr::VarId, f64)>()
                        + q.hessian.capacity()
                            * std::mem::size_of::<(oximo_expr::VarId, oximo_expr::VarId, f64)>()
                });
                let mut entries = shard.0.lock().expect("polynomial workset poisoned");
                if bytes > BYTES_PER_SHARD {
                    entries.roots.get_mut(&expr).expect("admitted root").value = None;
                    count!(self, bypasses);
                    return result;
                }

                // All pending roots have byte reservations but are not removable
                // until their OnceLock publishes.
                let candidates = entries.order.len();
                for _ in 0..candidates {
                    if entries.bytes + bytes <= BYTES_PER_SHARD {
                        break;
                    }
                    let old = entries.order.pop_front().expect("candidate");
                    let root = entries.roots.get_mut(&old).expect("retained root");
                    if root.value.as_ref().is_some_and(|cell| cell.get().is_some()) {
                        root.value = None;
                        let removed = std::mem::take(&mut root.bytes);
                        entries.bytes -= removed;
                        #[cfg(test)]
                        self.counters.bytes.fetch_sub(removed, Ordering::Relaxed);
                        count!(self, evictions);
                    } else {
                        entries.order.push_back(old);
                    }
                }

                if entries.bytes + bytes <= BYTES_PER_SHARD {
                    entries.bytes += bytes;
                    entries.roots.get_mut(&expr).expect("admitted root").bytes = bytes;
                    entries.order.push_back(expr);
                    #[cfg(test)]
                    {
                        let total = self.counters.bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                        self.counters.peak_bytes.fetch_max(total, Ordering::Relaxed);
                    }
                } else {
                    entries.roots.get_mut(&expr).expect("admitted root").value = None;
                    count!(self, bypasses);
                }
                result
            })
            .clone()
            .map(Extracted::Shared)
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> QuadraticAnalysisStats {
        let c = &self.counters;
        QuadraticAnalysisStats {
            attempts: c.attempts.load(Ordering::Relaxed),
            streamed_attempts: c.streamed_attempts.load(Ordering::Relaxed),
            hits: c.hits.load(Ordering::Relaxed),
            roots: c.roots.load(Ordering::Relaxed),
            re_extractions: c.re_extractions.load(Ordering::Relaxed),
            evictions: c.evictions.load(Ordering::Relaxed),
            bypasses: c.bypasses.load(Ordering::Relaxed),
            retained_bytes: c.bytes.load(Ordering::Relaxed),
            peak_retained_bytes: c.peak_bytes.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximo_expr::VarId;

    #[expect(
        clippy::unnecessary_wraps,
        reason = "test extractor has the fallible production signature"
    )]
    fn polynomial(entries: usize) -> Option<Arc<QuadraticTerms>> {
        Some(Arc::new(QuadraticTerms {
            hessian: vec![(VarId(0), VarId(0), 2.0); entries],
            ..QuadraticTerms::default()
        }))
    }

    #[test]
    fn first_results_and_failures_are_reused() {
        let workset = QuadraticWorkset::default();
        let first = workset.extract(ExprId(0), || polynomial(10)).unwrap();
        let second = workset.extract(ExprId(0), || panic!("duplicate extraction")).unwrap();
        let (Extracted::Shared(first), Extracted::Shared(second)) = (first, second) else {
            panic!("shared")
        };
        assert!(Arc::ptr_eq(&first, &second));
        assert!(workset.extract(ExprId(1), || None).is_none());
        assert!(workset.extract(ExprId(1), || panic!("repeated failure")).is_none());

        let stats = workset.stats();
        assert_eq!((stats.attempts, stats.roots, stats.hits, stats.re_extractions), (2, 2, 2, 0));
    }

    #[test]
    fn equal_roots_extract_once_while_independent_roots_fold_concurrently() {
        let workset = QuadraticWorkset::default();
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        workset.extract(ExprId(1), || polynomial(10)).unwrap()
                    })
                })
                .collect();
            for thread in threads {
                assert_eq!(thread.join().unwrap().hessian.len(), 10);
            }
        });
        assert_eq!(workset.stats().attempts, 1);

        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for id in [2, 3] {
                let barrier = &barrier;
                let workset = &workset;
                scope.spawn(move || {
                    workset.extract(ExprId(id), || {
                        barrier.wait();
                        polynomial(10)
                    })
                });
            }
        });
        assert_eq!(workset.stats().attempts, 3);
    }

    #[test]
    fn byte_budget_evicts_without_invalidating_returned_coefficients() {
        let workset = QuadraticWorkset::default();
        let shard = |id: u32| id.wrapping_mul(0x9e37_79b9) >> 28;
        let ids: Vec<_> = (0..1000).filter(|&id| shard(id) == 0).take(3).collect();
        let size = BYTES_PER_SHARD / std::mem::size_of::<(VarId, VarId, f64)>() / 2 + 1;
        let first = workset.extract(ExprId(ids[0]), || polynomial(size)).unwrap();
        workset.extract(ExprId(ids[1]), || polynomial(size));
        assert_eq!(first.hessian.len(), size);
        assert_eq!(workset.stats().evictions, 1);

        workset.extract(ExprId(ids[0]), || polynomial(size));
        assert_eq!(workset.stats().re_extractions, 1);

        let previous_evictions = workset.stats().evictions;
        workset.extract(ExprId(ids[2]), || polynomial(size * 4));
        assert_eq!(workset.stats().evictions, previous_evictions);
        assert!(workset.stats().retained_bytes <= BYTES_PER_SHARD);
        assert!(workset.stats().peak_retained_bytes <= BYTES_PER_SHARD);
    }

    #[test]
    fn root_metadata_is_bounded_including_failed_extractions() {
        let workset = QuadraticWorkset::default();
        for id in (0..100_000_u32)
            .filter(|id| id.wrapping_mul(0x9e37_79b9) >> 28 == 0)
            .take(ROOTS_PER_SHARD + 1)
        {
            assert!(workset.extract(ExprId(id), || None).is_none());
        }
        assert_eq!(workset.stats().roots, ROOTS_PER_SHARD);
        assert_eq!(workset.stats().bypasses, 1);
    }
}
