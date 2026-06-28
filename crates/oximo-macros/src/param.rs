//! `param!(model, name = value)` and its indexed form
//! `param!(model, name[i in dom[, ...]][ if cond] = value)`.

use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Expr;

use crate::bind::{filtered_set, masked_closure_param};
use crate::{Named, build_set, oximo_root, parse_named, split_top_commas};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let mut parts = split_top_commas(input).into_iter();
    let model = parts.next().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "param! needs a model and `name = value`")
    })?;
    let model: Expr = syn::parse2(model)?;

    let spec = parts
        .next()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "param! needs `name = value`"))?;
    // Tolerate a single trailing empty segment (trailing comma); reject real extra args.
    for seg in parts {
        if !seg.is_empty() {
            return Err(syn::Error::new(Span::call_site(), "unexpected trailing tokens in param!"));
        }
    }

    let (core, value_ts) = split_assignment(spec)?;
    let Named { name, binds, cond } = parse_named(core)?;
    let name_str = name.to_string();

    match binds {
        None => {
            let value: Expr = syn::parse2(value_ts)?;
            Ok(quote!( let #name = (#model).__param(#name_str, f64::from(#value)); ))
        }
        Some(binds) => {
            let root = oximo_root();
            let value = crate::index::rewrite_index_subscripts(value_ts);
            let param = masked_closure_param(&binds, &value);
            let set = build_set(&binds, &root);
            let set = filtered_set(set, &binds, cond.as_ref(), &root);
            Ok(quote! {
                let #name = {
                    let __set = #set;
                    (#model).__indexed_param(#name_str, &__set, move |#param| f64::from(#value))
                };
            })
        }
    }
}

/// Split a `param!` spec on its top-level assignment `=`, returning the
/// `name`/`name[...]` core and the value tokens.
fn split_assignment(spec: TokenStream2) -> syn::Result<(TokenStream2, TokenStream2)> {
    let tts: Vec<TokenTree> = spec.into_iter().collect();
    let pos = tts
        .iter()
        .position(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == '='))
        .ok_or_else(|| syn::Error::new(Span::call_site(), "param! needs `name = value`"))?;
    let core: TokenStream2 = tts[..pos].iter().cloned().collect();
    let value: TokenStream2 = tts[pos + 1..].iter().cloned().collect();
    if value.is_empty() {
        return Err(syn::Error::new(Span::call_site(), "param! value is missing after `=`"));
    }
    Ok((core, value))
}
