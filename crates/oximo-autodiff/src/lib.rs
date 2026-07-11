#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "enzyme", feature(autodiff))]
#![forbid(unsafe_code)]

mod error;
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
pub use tape::Tape;

#[cfg(feature = "enzyme")]
pub use evaluator::NlpEvaluator;
#[cfg(feature = "enzyme")]
pub use linearize::gradient_at;
