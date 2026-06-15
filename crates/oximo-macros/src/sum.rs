//! `sum!(body for pat in domain[, pat in domain ...][ if cond])`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Token};

use crate::IndexBind;
use crate::oximo_root;

struct SumInput {
    body: Expr,
    binds: Vec<IndexBind>,
    cond: Option<Expr>,
}

impl Parse for SumInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let body = input.parse::<Expr>()?;
        input.parse::<Token![for]>()?;
        let binds = Punctuated::<IndexBind, Token![,]>::parse_separated_nonempty(input)?;
        let cond = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(input.parse::<Expr>()?)
        } else {
            None
        };
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after `sum!` clauses"));
        }
        Ok(Self { body, binds: binds.into_iter().collect(), cond })
    }
}

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let SumInput { body, binds, cond } = syn::parse2(input)?;
    let root = oximo_root();

    let Some(cond) = cond else {
        let mut expr = quote!(#body);
        for b in binds.iter().rev() {
            let param = b.closure_param();
            let domain = &b.domain;
            expr = quote!( #root::__macro_support::sum_over(&(#domain), |#param| #expr) );
        }
        return Ok(expr);
    };

    // Filtered: iterate the (decoded) keys with nested `for` loops, skipping
    // keys that fail `cond`, and accumulate only the matching terms.
    let mut inner = quote! {
        if #cond {
            let __term = #body;
            __acc = ::core::option::Option::Some(match __acc {
                ::core::option::Option::Some(__a) => __a + __term,
                ::core::option::Option::None => __term,
            });
        }
    };
    for b in binds.iter().rev() {
        let pat = &b.pat;
        let domain = &b.domain;
        let keys = if let Some(ty) = b.keys_of_type() {
            quote!( #root::__macro_support::keys_of::<#ty, _>(&(#domain)) )
        } else {
            quote!( #root::__macro_support::keys_of(&(#domain)) )
        };
        inner = quote! {
            for #pat in #keys {
                #inner
            }
        };
    }
    Ok(quote! {{
        let mut __acc = ::core::option::Option::None;
        #inner
        __acc.expect("sum! with an `if` filter produced no terms")
    }})
}
