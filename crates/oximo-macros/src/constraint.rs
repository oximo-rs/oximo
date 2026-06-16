//! `constraint!(model, [name | name = expr | name[i in dom, ...]], lhs <op> rhs)`.

use proc_macro2::{Spacing, TokenStream as TokenStream2, TokenTree};
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
            // Computed name at run-time: `constraint!(m, name = expr, lhs OP rhs)`.
            // Lets per-element constraints built in a `for` loop stay named
            // (`format!("mb_{s}_{n}")`).
            if let Some(name_expr) = computed_name(&first) {
                let rel = build_relation(rel_tokens, &root)?;
                return Ok(quote!( (#model).__add_constraint(#name_expr, #rel) ));
            }

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

/// Detect the computed-name form `name = EXPR` in the name slot, returning the
/// `EXPR` tokens. The literal marker `name` keeps a bare ident a literal name.
fn computed_name(first: &TokenStream2) -> Option<TokenStream2> {
    let mut it = first.clone().into_iter();
    match it.next()? {
        TokenTree::Ident(id) if id == "name" => {}
        _ => return None,
    }
    match it.next()? {
        TokenTree::Punct(p) if p.as_char() == '=' && p.spacing() == Spacing::Alone => {}
        _ => return None,
    }
    let expr: TokenStream2 = it.collect();
    (!expr.is_empty()).then_some(expr)
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
    let lhs: Expr = syn::parse2(crate::index::rewrite_index_subscripts(lhs))?;
    let rhs: Expr = syn::parse2(crate::index::rewrite_index_subscripts(rhs))?;
    Ok(quote!( #root::__macro_support::Relate::#method(#lhs, #rhs) ))
}
