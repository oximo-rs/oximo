use rustc_hash::FxHashMap;
use smol_str::SmolStr;

use crate::solver::Solver;

/// Runtime registry of solver backends. Used by CLI / server entry points
/// that pick a backend by name at runtime. Library users that statically
/// know the backend should construct it directly.
#[derive(Default)]
pub struct SolverRegistry {
    factories: FxHashMap<SmolStr, Box<dyn Fn() -> Box<dyn Solver>>>,
}

impl std::fmt::Debug for SolverRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverRegistry")
            .field("backends", &self.factories.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SolverRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F>(&mut self, name: impl Into<SmolStr>, factory: F)
    where
        F: Fn() -> Box<dyn Solver> + 'static,
    {
        self.factories.insert(name.into(), Box::new(factory));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn Solver>> {
        self.factories.get(name).map(|f| f())
    }

    pub fn names(&self) -> impl Iterator<Item = &SmolStr> {
        self.factories.keys()
    }
}
