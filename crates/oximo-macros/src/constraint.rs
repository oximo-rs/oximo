//! `constraint!(model, [name | name[i in dom, ...]], lhs <op> rhs)`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Expr;

use crate::bind::family_closure_param;
use crate::{Named, build_set, oximo_root, parse_named, split_relops, split_top_commas};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let parts = split_top_commas(input);
    let mut parts = parts.into_iter();
    let model = parts.next().expect("split always yields at least one segment");
    let model: Expr = syn::parse2(model)?;

    let first = parts.next().ok_or_else(|| {
        syn::Error::new(proc_macro2::Span::call_site(), "constraint! needs a relational expression")
    })?;
    let second = parts.next();
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(extra, "unexpected trailing tokens in constraint!"));
    }

    let root = oximo_root();

    match second {
        None => {
            let rel = build_relation(first, &root)?;
            Ok(quote!( (#model).__add_constraint_auto(#rel) ))
        }
        Some(rel_tokens) => {
            let Named { name, binds } = parse_named(first)?;
            let name_str = name.to_string();
            match binds {
                None => {
                    let rel = build_relation(rel_tokens, &root)?;
                    Ok(quote!( (#model).__add_constraint(#name_str, #rel) ))
                }
                Some(binds) => {
                    let param = family_closure_param(&binds);
                    let set = build_set(&binds, &root);
                    let rel = build_relation(rel_tokens, &root)?;
                    Ok(quote! {
                        (#model).__add_constraints_over(#name_str, &(#set), |#param| #rel);
                    })
                }
            }
        }
    }
}

/// Split `lhs <op> rhs` on its single relational operator and build the
/// `Relate::{le,ge,eq}(lhs, rhs)` call that yields a `ConstraintExpr`.
fn build_relation(tokens: TokenStream2, root: &TokenStream2) -> syn::Result<TokenStream2> {
    let (segs, ops) = split_relops(tokens);
    let [lhs, rhs] = <[TokenStream2; 2]>::try_from(segs).map_err(|_| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "constraint must contain exactly one of `==`, `<=`, or `>=`",
        )
    })?;
    let [op] = ops.as_slice() else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "constraint must contain exactly one of `==`, `<=`, or `>=`",
        ));
    };
    let method = op.method();
    let lhs: Expr = syn::parse2(lhs)?;
    let rhs: Expr = syn::parse2(rhs)?;
    Ok(quote!( #root::__macro_support::Relate::#method(#lhs, #rhs) ))
}
