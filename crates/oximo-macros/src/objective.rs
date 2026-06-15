//! `objective!(model, Min|Max, expr)`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token};

struct ObjectiveInput {
    model: Expr,
    sense: Ident,
    expr: Expr,
}

impl Parse for ObjectiveInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let model = input.parse::<Expr>()?;
        input.parse::<Token![,]>()?;
        let sense = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;
        let expr = input.parse::<Expr>()?;
        Ok(Self { model, sense, expr })
    }
}

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let ObjectiveInput { model, sense, expr } = syn::parse2(input)?;

    let method = match sense.to_string().as_str() {
        "Min" | "Minimize" | "min" => quote!(__minimize),
        "Max" | "Maximize" | "max" => quote!(__maximize),
        _ => {
            return Err(syn::Error::new(sense.span(), "objective sense must be `Min` or `Max`"));
        }
    };

    Ok(quote!( (#model).#method(#expr) ))
}
