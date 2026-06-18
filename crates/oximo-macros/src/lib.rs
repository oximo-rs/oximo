#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Delimiter, Spacing, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Ident;

mod bind;
mod constraint;
mod index;
mod objective;
mod param;
mod sum;
mod variable;

use bind::{Binds, IndexBind};

/// Resolve the path prefix used to reach `__macro_support`. Prefers the umbrella
/// `oximo` crate (which re-exports the support module) and falls back to
/// `oximo-core`.
fn oximo_root() -> TokenStream2 {
    fn to_path(found: &FoundCrate, fallback: &str) -> TokenStream2 {
        let name = match found {
            FoundCrate::Itself => fallback,
            FoundCrate::Name(n) => n.as_str(),
        };
        let id = Ident::new(name, proc_macro2::Span::call_site());
        quote!(::#id)
    }

    if let Ok(found) = crate_name("oximo") {
        return to_path(&found, "oximo");
    }
    if let Ok(found) = crate_name("oximo-core") {
        return to_path(&found, "oximo_core");
    }
    quote!(::oximo_core)
}

/// `variable!(model, spec)`, declare a decision variable (or an indexed family)
/// and bind it to a local of the same name. See the crate docs for the grammar.
#[proc_macro]
pub fn variable(input: TokenStream) -> TokenStream {
    variable::expand(input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

/// `constraint!(model, [name|name[idx]], lhs <op> rhs)`, register a constraint,
/// an auto-named anonymous constraint, or an indexed family of constraints.
#[proc_macro]
pub fn constraint(input: TokenStream) -> TokenStream {
    constraint::expand(input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

/// `objective!(model, Min|Max, expr)`, set the model objective and sense.
#[proc_macro]
pub fn objective(input: TokenStream) -> TokenStream {
    objective::expand(input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

/// `sum!(body for pat in domain[, pat in domain ...])`, algebraic summation,
/// lowered to nested `sum_over` folds.
#[proc_macro]
pub fn sum(input: TokenStream) -> TokenStream {
    sum::expand(input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

/// `param!(model, name = value)`, declare a re-bindable scalar parameter and
/// bind it to a local of the same name.
#[proc_macro]
pub fn param(input: TokenStream) -> TokenStream {
    param::expand(input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

// ---------------------------------------------------------------------------
// Shared token-walking helpers. The macros must accept forms that are not valid
// `syn::Expr` (indexed `name[i in set]`, chained `lb <= x <= ub`), so a few
// splits are done at the raw token-tree level.
// ---------------------------------------------------------------------------

/// Relational operator recognized inside `constraint!`/`variable!`.
#[derive(Copy, Clone, PartialEq, Eq)]
enum RelOp {
    Le,
    Ge,
    Eq,
}

impl RelOp {
    /// The `Relate` method this operator maps to.
    fn method(self) -> Ident {
        let name = match self {
            RelOp::Le => "le",
            RelOp::Ge => "ge",
            RelOp::Eq => "eq",
        };
        Ident::new(name, proc_macro2::Span::call_site())
    }
}

/// Split a token stream on top-level commas.
fn split_top_commas(ts: TokenStream2) -> Vec<TokenStream2> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for tt in ts {
        if let TokenTree::Punct(p) = &tt {
            if p.as_char() == ',' {
                out.push(cur.drain(..).collect());
                continue;
            }
        }
        cur.push(tt);
    }
    out.push(cur.into_iter().collect());
    out
}

/// Split a token stream on top-level relational operators (`==`, `<=`, `>=`),
/// returning the intervening segments and the operators between them.
fn split_relops(ts: TokenStream2) -> (Vec<TokenStream2>, Vec<RelOp>) {
    let tts: Vec<TokenTree> = ts.into_iter().collect();
    let mut segs: Vec<TokenStream2> = Vec::new();
    let mut ops: Vec<RelOp> = Vec::new();
    let mut cur: Vec<TokenTree> = Vec::new();

    let mut i = 0;
    while i < tts.len() {
        if let TokenTree::Punct(p1) = &tts[i] {
            if p1.spacing() == Spacing::Joint && i + 1 < tts.len() {
                if let TokenTree::Punct(p2) = &tts[i + 1] {
                    let op = match (p1.as_char(), p2.as_char()) {
                        ('<', '=') => Some(RelOp::Le),
                        ('>', '=') => Some(RelOp::Ge),
                        ('=', '=') => Some(RelOp::Eq),
                        _ => None,
                    };
                    if let Some(op) = op {
                        segs.push(cur.drain(..).collect());
                        ops.push(op);
                        i += 2;
                        continue;
                    }
                }
            }
        }
        cur.push(tts[i].clone());
        i += 1;
    }
    segs.push(cur.into_iter().collect());
    (segs, ops)
}

/// A parsed `name` or `name[binds]` "core" of a `variable!`/`constraint!`
/// declaration. `cond` holds an optional `if` filter on the index family.
struct Named {
    name: Ident,
    binds: Option<Vec<IndexBind>>,
    cond: Option<syn::Expr>,
}

/// Parse a `name`/`name[i in dom, ...]` core out of a token segment.
fn parse_named(seg: TokenStream2) -> syn::Result<Named> {
    let tts: Vec<TokenTree> = seg.into_iter().collect();
    let span = tts.first().map_or_else(proc_macro2::Span::call_site, TokenTree::span);
    let TokenTree::Ident(name) = tts
        .first()
        .cloned()
        .ok_or_else(|| syn::Error::new(span, "expected a variable/constraint name identifier"))?
    else {
        return Err(syn::Error::new(span, "expected a name identifier"));
    };

    let (binds, cond) = match tts.get(1) {
        None => (None, None),
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => {
            let parsed: Binds = syn::parse2(g.stream())?;
            if parsed.binds.is_empty() {
                return Err(syn::Error::new(
                    g.span(),
                    "index family needs at least one binding, e.g. `name[i in domain]`",
                ));
            }
            (Some(parsed.binds), parsed.cond)
        }
        Some(other) => {
            return Err(syn::Error::new(other.span(), "expected `[index in domain, ...]`"));
        }
    };
    Ok(Named { name, binds, cond })
}

/// Build an owned `Set` token expression from one or more index bindings.
fn build_set(binds: &[IndexBind], root: &TokenStream2) -> TokenStream2 {
    let mut iter = binds.iter().map(|b| {
        let dom = &b.domain;
        quote!(#root::__macro_support::as_set(&(#dom)))
    });
    let first = iter.next().expect("at least one index binding");
    iter.fold(first, |acc, s| quote!(#root::__macro_support::product(&(#acc), &(#s))))
}
