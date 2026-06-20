//! `set!(name = ...)`, bind a local to an index [`Set`].
//!
//! Two forms, distinguished by a top-level `in`:
//!
//! - plain expression: `set!(items = 0..5)`, `set!(routes = plants * markets)`:
//!   the right side is normalized to an owned `Set` via `as_set`. A top-level `*`
//!   is treated as a Cartesian product and lowered to `product(&as_set(&a), ..)`,
//!   so the operands are borrowed and need no `&`.
//! - comprehension: `set!(arcs = (p, q) in &plants * &plants if p != q)` or the
//!   multi-bind product `set!(arcs = i in plants, j in plants if i != j)`. Reuses
//!   the same `pat in domain[, ...][ if cond]` grammar as the `constraint!` index
//!   family, lowering the trailing `if` to `filter_keys` over by-value keys.

use proc_macro2::{Spacing, TokenStream as TokenStream2, TokenTree};
use quote::quote;

use crate::bind::{Binds, filtered_set};
use crate::{build_set, oximo_root};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let tts: Vec<TokenTree> = input.into_iter().collect();
    let span = tts.first().map_or_else(proc_macro2::Span::call_site, TokenTree::span);

    let TokenTree::Ident(name) =
        tts.first().cloned().ok_or_else(|| syn::Error::new(span, "expected `set!(name = ...)`"))?
    else {
        return Err(syn::Error::new(span, "expected a set name identifier"));
    };

    match tts.get(1) {
        Some(TokenTree::Punct(p)) if p.as_char() == '=' && p.spacing() == Spacing::Alone => {}
        _ => {
            return Err(syn::Error::new(name.span(), "expected `=` after the set name"));
        }
    }

    let rhs: TokenStream2 = tts[2..].iter().cloned().collect();
    if rhs.is_empty() {
        return Err(syn::Error::new(name.span(), "expected a set expression after `=`"));
    }

    let root = oximo_root();

    let set = if has_top_level_in(&rhs) {
        let binds: Binds = syn::parse2(rhs)?;
        if binds.binds.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "set comprehension needs at least one `pat in domain`",
            ));
        }
        let built = build_set(&binds.binds, &root);
        filtered_set(built, &binds.binds, binds.cond.as_ref(), &root)
    } else {
        // Plain expression: a top-level `*` is a Cartesian product.
        let mut iter = split_top_mul(rhs)
            .into_iter()
            .map(|seg| quote!( #root::__macro_support::as_set(&(#seg)) ));
        let first = iter.next().expect("non-empty rhs yields at least one operand");
        iter.fold(first, |acc, seg| quote!( #root::__macro_support::product(&(#acc), &(#seg)) ))
    };

    Ok(quote!( let #name = #set; ))
}

/// Split a token stream on top-level `*` (the set-product operator).
fn split_top_mul(ts: TokenStream2) -> Vec<TokenStream2> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for tt in ts {
        if matches!(&tt, TokenTree::Punct(p) if p.as_char() == '*') {
            out.push(cur.drain(..).collect());
            continue;
        }
        cur.push(tt);
    }
    out.push(cur.into_iter().collect());
    out
}

/// Whether the right-hand side contains a top-level `in`, marking it as a
/// comprehension (`pat in domain`) rather than a plain set expression.
fn has_top_level_in(ts: &TokenStream2) -> bool {
    ts.clone().into_iter().any(|tt| matches!(&tt, TokenTree::Ident(id) if *id == "in"))
}
