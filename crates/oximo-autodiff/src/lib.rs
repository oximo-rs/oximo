#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "enzyme", feature(autodiff))]
#![forbid(unsafe_code)]

mod error;
#[cfg(any(feature = "enzyme", test))]
mod refresh;
pub mod slot;
pub mod sparsity;
pub mod tape;

#[cfg(feature = "enzyme")]
mod enzyme;
#[cfg(feature = "enzyme")]
mod evaluator;
#[cfg(feature = "enzyme")]
mod linearize;

pub use error::AutodiffError;
pub use slot::{FunctionSlot, SlotKind};
pub use tape::{CompiledBatch, CompiledExpr, Tape, TapeScratch, params_snapshot};

#[cfg(feature = "enzyme")]
pub use evaluator::NlpEvaluator;
#[cfg(all(feature = "enzyme", feature = "benchmark-support"))]
#[doc(hidden)]
pub use evaluator::benchmark_support;
#[cfg(feature = "enzyme")]
pub use linearize::gradient_at;
