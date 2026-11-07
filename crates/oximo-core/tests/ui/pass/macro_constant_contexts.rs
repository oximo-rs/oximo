use oximo_core::prelude::*;

fn first<const N: usize>(values: [f64; N]) -> f64 {
    values[0]
}

fn main() {
    let m = Model::new("constant contexts");
    variable!(m, x);
    // Model-context rewriting leaves constant contexts untouched.
    objective!(m, Min, {
        const N: usize = 2;
        const OFFSET: isize = -1;
        static LENGTH: usize = N + 1;
        const fn length(n: usize) -> usize {
            n + 1
        }
        fn name(prefix: String) -> String {
            prefix + "b"
        }
        struct Nested {
            values: [f64; N + 1],
        }
        let data: [f64; N + 1] = [1.0; N + 1];
        let nested = Nested { values: data };
        let _ = (OFFSET, LENGTH, length(N), name(String::from("a")));
        let _ = const { N + 1 };
        first::<{ N + 1 }>(nested.values) * x
    });
    // Parsing macro values retains ordinary Rust arithmetic.
    param!(
        m,
        p = {
            const N: usize = 2;
            const OFFSET: isize = -1;
            static LENGTH: usize = N + 1;
            const fn length(n: usize) -> usize {
                n + 1
            }
            fn name(prefix: String) -> String {
                prefix + "b"
            }
            struct Nested {
                values: [f64; N + 1],
            }
            let data: [f64; N + 1] = [1.0; N + 1];
            let nested = Nested { values: data };
            let _ = (OFFSET, LENGTH, length(N), name(String::from("a")));
            let _ = const { N + 1 };
            first::<{ N + 1 }>(nested.values)
        }
    );
    // Standalone sums also run the context-aware rewriter.
    let _ = sum!(m, {
        const N: usize = 2;
        let data = [1.0; N + 1];
        first::<{ N + 1 }>(data) * x
    } for _i in 0..2);
    constraint!(m, p * x <= 5.0);
}
