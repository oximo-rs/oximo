//! `min!` and `max!` indexed-domain expressions.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Token};

use crate::IndexBind;
use crate::oximo_root;

pub(crate) enum Extremum {
    Min,
    Max,
}

struct ExtremaInput {
    body: Expr,
    binds: Vec<IndexBind>,
    cond: Option<Expr>,
}

impl Parse for ExtremaInput {
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
            return Err(input.error("unexpected tokens after extremum clauses"));
        }
        Ok(Self { body, binds: binds.into_iter().collect(), cond })
    }
}

pub(crate) fn expand(input: TokenStream2, kind: Extremum) -> syn::Result<TokenStream2> {
    let input = crate::index::rewrite_index_subscripts(input);
    let ExtremaInput { body, binds, cond } = syn::parse2(input)?;
    let root = oximo_root();
    let name = match kind {
        Extremum::Min => "min",
        Extremum::Max => "max",
    };
    let constructor = format_ident!("{name}");
    let empty_diagnostic =
        format!("{name}! produced no terms (empty domain or every key filtered out)");
    let emit = syn::Ident::new("__emit", proc_macro2::Span::mixed_site());

    // Keep body control flow local, even though traversal uses generated loops.
    let mut inner = if let Some(cond) = &cond {
        quote! {
            if #cond {
                #emit.push((|| #body)());
            }
        }
    } else {
        quote! {
            #emit.push((|| #body)());
        }
    };
    for b in binds.iter().rev() {
        let pat = &b.pat;
        let used = crate::bind::mark_bindings_used(std::slice::from_ref(b));
        let domain = &b.domain;
        let keys = if let Some(ty) = b.keys_of_type() {
            quote!( #root::__macro_support::keys_of::<#ty, _>(&(#domain)) )
        } else {
            quote!( #root::__macro_support::keys_of(&(#domain)) )
        };
        inner = quote! {
            for #pat in #keys {
                #used
                #inner
            }
        };
    }

    Ok(quote! {{
        let mut #emit = #root::__macro_support::ExtremaBuilder::#constructor();
        #inner
        #emit.finish().expect(#empty_diagnostic)
    }})
}
