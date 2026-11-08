use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use quote::quote;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::{Named, build_set, oximo_root, parse_comma_list, parse_named};

pub(crate) fn symmetric_variable(input: TokenStream) -> syn::Result<TokenStream> {
    let parts = parse_comma_list(input, "missing macro argument")?;
    if parts.len() != 2 {
        return Err(syn::Error::new(
            Span::call_site(),
            "expected symmetric_variable!(model, X[n])",
        ));
    }
    let mut parts = parts.into_iter();
    let model: syn::Expr = syn::parse2(parts.next().expect("validated model argument"))?;
    let specification = parts.next().expect("validated matrix specification");
    let mut tokens = specification.clone().into_iter();
    let (Some(TokenTree::Ident(name)), Some(TokenTree::Group(dim)), None) =
        (tokens.next(), tokens.next(), tokens.next())
    else {
        return Err(syn::Error::new_spanned(&specification, "expected X[n]"));
    };
    if dim.delimiter() != Delimiter::Bracket {
        return Err(syn::Error::new(dim.span(), "expected X[n]"));
    }
    let n: syn::Expr = syn::parse2(dim.stream())?;
    let text = name.to_string();
    Ok(quote!(let #name = (#model).add_symmetric_variable(#text, #n);))
}

pub(crate) fn constraint(input: TokenStream) -> syn::Result<TokenStream> {
    let parts = parse_comma_list(input, "missing macro argument")?;
    if !(2..=3).contains(&parts.len()) {
        return Err(syn::Error::new(
            Span::call_site(),
            "expected psd_constraint!(model, [name,] matrix)",
        ));
    }
    let mut parts = parts.into_iter();
    let mut sums =
        crate::sum::ModelSums::new(syn::parse2(parts.next().expect("validated model argument"))?);
    let model = sums.receiver();
    let root = oximo_root();
    let name_or_matrix = parts.next().expect("validated constraint argument");
    let Some(matrix_input) = parts.next() else {
        let matrix = sums.rewrite(crate::index::parse_value_expr(name_or_matrix)?)?;
        return Ok(sums.wrap(quote!((#model).__add_psd_constraint_auto(#matrix))));
    };
    if let Some(name) = crate::constraint::computed_name(&name_or_matrix) {
        let matrix = sums.rewrite(crate::index::parse_value_expr(matrix_input)?)?;
        return Ok(sums.wrap(quote!((#model).add_psd_constraint(#name, #matrix))));
    }
    let Named { name, binds, cond } = parse_named(name_or_matrix)?;
    let text = name.to_string();
    if let Some(binds) = &binds {
        sums.indexed(binds);
    }
    let matrix = sums.rewrite(crate::index::parse_value_expr(matrix_input)?)?;
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
