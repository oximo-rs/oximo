//! `param!(model, name = value)` and its indexed form
//! `param!(model, name[i in dom[, ...]][ if cond] = value)`.

use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Expr;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::{Named, build_set, oximo_root, parse_comma_list, parse_named};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let mut parts = parse_comma_list(input, "missing macro argument")?.into_iter();
    let model_ts = parts.next().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "param! needs a model and `name = value`")
    })?;
    let model: Expr = syn::parse2(model_ts)?;

    let spec = parts
        .next()
        .ok_or_else(|| syn::Error::new_spanned(&model, "param! needs `name = value`"))?;
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(extra, "unexpected trailing tokens in param!"));
    }

    let (core, value_ts) = split_assignment(spec)?;
    let Named { name, binds, cond } = parse_named(core)?;
    let name_str = name.to_string();
    let value = crate::index::parse_value_expr(value_ts)?;

    match binds {
        None => Ok(quote!( let #name = (#model).__param(#name_str, f64::from(#value)); )),
        Some(binds) => {
            let root = oximo_root();
            let param = family_closure_param(&binds);
            let used = mark_bindings_used(&binds);
            let set = build_set(&binds, &root)?;
            let set = filtered_set(set, &binds, cond.as_ref(), &root);
            Ok(quote! {
                let #name = {
                    let __set = #set;
                    (#model).__indexed_param(#name_str, &__set, move |#param| { #used f64::from(#value) })
                };
            })
        }
    }
}

/// Split a `param!` spec on its top-level assignment `=`, returning the
/// `name`/`name[...]` core and the value tokens.
fn split_assignment(spec: TokenStream2) -> syn::Result<(TokenStream2, TokenStream2)> {
    let tts: Vec<TokenTree> = spec.clone().into_iter().collect();
    let pos = tts
        .iter()
        .position(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == '='))
        .ok_or_else(|| syn::Error::new_spanned(spec, "param! needs `name = value`"))?;
    let core: TokenStream2 = tts[..pos].iter().cloned().collect();
    let value: TokenStream2 = tts[pos + 1..].iter().cloned().collect();
    if value.is_empty() {
        // point at the `=` token if possible
        let eq_span = tts.get(pos).map_or_else(Span::call_site, proc_macro2::TokenTree::span);
        return Err(syn::Error::new(eq_span, "param! value is missing after `=`"));
    }
    Ok((core, value))
}
