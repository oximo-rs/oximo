use std::cell::RefCell;
use std::sync::Arc;

use oximo_expr::ModelId;

/// Registry prefixes inherited from an ancestor at a fork boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelPrefix {
    pub variables: usize,
    pub parameters: usize,
    pub algebraic_constraints: usize,
    pub soc_constraints: usize,
    pub psd_constraints: usize,
    pub sos_constraints: usize,
    pub indicator_constraints: usize,
}

/// Immutable ancestor snapshots, shared by repeated forks and deeper branches.
#[derive(Default)]
pub(super) struct ModelLineage {
    parent: Option<Arc<Ancestor>>,
    fork_cache: RefCell<Option<Arc<Ancestor>>>,
}

struct Ancestor {
    model_id: ModelId,
    prefix: ModelPrefix,
    depth: usize,
    parent: Option<Arc<Self>>,
}

impl ModelLineage {
    pub(super) fn forked(&self, model_id: ModelId, prefix: ModelPrefix) -> Self {
        let mut cache = self.fork_cache.borrow_mut();
        let ancestor = match cache.as_ref() {
            Some(cached) if cached.prefix == prefix => Arc::clone(cached),
            _ => {
                let ancestor = Arc::new(Ancestor {
                    model_id,
                    prefix,
                    depth: self.depth(),
                    parent: self.parent.clone(),
                });
                *cache = Some(Arc::clone(&ancestor));
                ancestor
            }
        };
        Self { parent: Some(ancestor), fork_cache: RefCell::default() }
    }

    pub(super) fn depth(&self) -> usize {
        self.parent.as_ref().map_or(0, |parent| parent.depth + 1)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (ModelId, &ModelPrefix)> + '_ {
        std::iter::successors(self.parent.as_deref(), |node| node.parent.as_deref())
            .map(|node| (node.model_id, &node.prefix))
    }
}

impl Drop for Ancestor {
    fn drop(&mut self) {
        // The final descendant can own a very deep chain.
        // Release unique nodes iteratively.
        let mut parent = self.parent.take();
        while let Some(node) = parent {
            match Arc::try_unwrap(node) {
                Ok(mut unique) => parent = unique.parent.take(),
                Err(_) => break,
            }
        }
    }
}
