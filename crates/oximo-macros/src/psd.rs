use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use quote::quote;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::{Named, build_set, oximo_root, parse_named, split_top_commas};

pub(crate) fn symmetric_variable(input: TokenStream) -> syn::Result<TokenStream> {
    let parts = split_top_commas(input);
    if parts.len() != 2 {
        return Err(syn::Error::new(
            Span::call_site(),
            "expected symmetric_variable!(model, X[n])",
        ));
    }
    let model: syn::Expr = syn::parse2(parts[0].clone())?;
    let mut tokens = parts[1].clone().into_iter();
    let (Some(TokenTree::Ident(name)), Some(TokenTree::Group(dim)), None) =
        (tokens.next(), tokens.next(), tokens.next())
    else {
        return Err(syn::Error::new_spanned(&parts[1], "expected X[n]"));
    };
    if dim.delimiter() != Delimiter::Bracket {
        return Err(syn::Error::new(dim.span(), "expected X[n]"));
    }
    let n: syn::Expr = syn::parse2(dim.stream())?;
    let text = name.to_string();
    Ok(quote!(let #name = (#model).add_symmetric_variable(#text, #n);))
}

pub(crate) fn constraint(input: TokenStream) -> syn::Result<TokenStream> {
    let parts = split_top_commas(input);
    if !(2..=3).contains(&parts.len()) {
        return Err(syn::Error::new(
            Span::call_site(),
            "expected psd_constraint!(model, [name,] matrix)",
        ));
    }
    let mut sums = crate::sum::ModelSums::new(syn::parse2(parts[0].clone())?);
    let model = sums.receiver();
    let root = oximo_root();
    if parts.len() == 2 {
        let matrix =
            sums.rewrite(syn::parse2(crate::index::rewrite_index_subscripts(parts[1].clone()))?)?;
        return Ok(sums.wrap(quote!((#model).__add_psd_constraint_auto(#matrix))));
    }
    if let Some(name) = crate::constraint::computed_name(&parts[1]) {
        let matrix =
            sums.rewrite(syn::parse2(crate::index::rewrite_index_subscripts(parts[2].clone()))?)?;
        return Ok(sums.wrap(quote!((#model).add_psd_constraint(#name, #matrix))));
    }
    let Named { name, binds, cond } = parse_named(parts[1].clone())?;
    let text = name.to_string();
    if let Some(binds) = &binds {
        sums.indexed(binds);
    }
    let matrix =
        sums.rewrite(syn::parse2(crate::index::rewrite_index_subscripts(parts[2].clone()))?)?;
    let expansion = if let Some(binds) = binds {
        let param = family_closure_param(&binds);
        let used = mark_bindings_used(&binds);
        let set = filtered_set(build_set(&binds, &root)?, &binds, cond.as_ref(), &root);
        quote!((#model).__add_psd_constraints_over(#text, &(#set), |#param| { #used #matrix }))
    } else {
        quote!((#model).add_psd_constraint(#text, #matrix))
    };
    Ok(sums.wrap(expansion))
}
