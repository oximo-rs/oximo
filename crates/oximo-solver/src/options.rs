use std::collections::BTreeMap;
use std::time::Duration;

use smol_str::SmolStr;

#[derive(Clone, Debug, PartialEq)]
pub enum OptionValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl From<bool> for OptionValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for OptionValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<i32> for OptionValue {
    fn from(v: i32) -> Self {
        Self::Int(v.into())
    }
}
impl From<f64> for OptionValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<&str> for OptionValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
}
impl From<String> for OptionValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

/// Free-form option bag passed to a [`crate::Solver`]. Each backend documents
/// the keys it honors. Common conventions: `time_limit` (seconds, Float),
/// `threads` (Int), `mip_gap` (Float), `presolve` (Bool/Str), `verbose`
/// (Bool).
#[derive(Clone, Debug, Default)]
pub struct SolverOptions {
    pub entries: BTreeMap<SmolStr, OptionValue>,
}

impl SolverOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(mut self, key: impl Into<SmolStr>, value: impl Into<OptionValue>) -> Self {
        self.entries.insert(key.into(), value.into());
        self
    }

    pub fn time_limit(self, d: Duration) -> Self {
        self.set("time_limit", d.as_secs_f64())
    }

    pub fn threads(self, n: i64) -> Self {
        self.set("threads", n)
    }

    pub fn verbose(self, on: bool) -> Self {
        self.set("verbose", on)
    }

    pub fn mip_gap(self, gap: f64) -> Self {
        self.set("mip_gap", gap)
    }

    pub fn get(&self, key: &str) -> Option<&OptionValue> {
        self.entries.get(key)
    }
}
