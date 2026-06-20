# oximo-macros

Internal procedural macros backing oximo's modeling surface:
`variable!`, `constraint!`, `objective!`, `sum!`, `set!`, and `param!`.

This crate is an implementation detail, do not depend on it directly.
The macros are re-exported through `oximo-core` and `oximo::prelude`, which is the
supported entry point:

```rust,ignore
use oximo::prelude::*;
```

The macros expand to the typed builder API in `oximo-core` (`Model`, `Set`,
`Expr`, `sum_over`, ...). See the `oximo` crate docs for the macro grammar and examples.
