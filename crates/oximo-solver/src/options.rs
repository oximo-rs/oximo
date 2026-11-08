use std::time::Duration;

/// Options valid for every solver, including NLP backends.
/// Each backend's options struct embeds this via [`HasUniversal`]; the
/// [`UniversalOptionsExt`] blanket impl then provides typed builder setters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UniversalOptions {
    pub time_limit: Option<Duration>,
    pub threads: Option<u32>,
    pub verbose: Option<bool>,
}

/// Implemented by every backend-specific options struct.
/// Gives the [`UniversalOptionsExt`] blanket impl access to the embedded
/// [`UniversalOptions`].
pub trait HasUniversal {
    fn universal(&self) -> &UniversalOptions;
    fn universal_mut(&mut self) -> &mut UniversalOptions;
}

/// Builder setters available on every backend options struct.
pub trait UniversalOptionsExt: HasUniversal + Sized {
    #[must_use]
    fn time_limit(mut self, d: Duration) -> Self {
        self.universal_mut().time_limit = Some(d);
        self
    }

    #[must_use]
    fn threads(mut self, n: u32) -> Self {
        self.universal_mut().threads = Some(n);
        self
    }

    #[must_use]
    fn verbose(mut self, on: bool) -> Self {
        self.universal_mut().verbose = Some(on);
        self
    }
}

impl<T: HasUniversal> UniversalOptionsExt for T {}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[derive(Default)]
    struct TestOpts {
        pub universal: UniversalOptions,
    }
    impl HasUniversal for TestOpts {
        fn universal(&self) -> &UniversalOptions {
            &self.universal
        }
        fn universal_mut(&mut self) -> &mut UniversalOptions {
            &mut self.universal
        }
    }

    #[test]
    fn universal_default_is_all_none() {
        let u = UniversalOptions::default();
        assert!(u.time_limit.is_none());
        assert!(u.threads.is_none());
        assert!(u.verbose.is_none());
    }

    #[test]
    fn universal_builder_chain() {
        let o = TestOpts::default().time_limit(Duration::from_secs(60)).threads(4).verbose(false);
        assert_eq!(o.universal.time_limit, Some(Duration::from_secs(60)));
        assert_eq!(o.universal.threads, Some(4));
        assert_eq!(o.universal.verbose, Some(false));
    }

    #[test]
    #[expect(
        clippy::redundant_clone,
        reason = "the regression verifies options implement Clone correctly"
    )]
    fn universal_options_clone_eq() {
        let a = TestOpts::default().threads(2).verbose(true);
        let b = UniversalOptions { threads: Some(2), verbose: Some(true), ..Default::default() };
        assert_eq!(a.universal, b);
        let c = a.universal.clone();
        assert_eq!(c, b);
    }

    #[test]
    fn last_set_wins_for_same_field() {
        let o = TestOpts::default().threads(1).threads(8);
        assert_eq!(o.universal.threads, Some(8));
    }
}
