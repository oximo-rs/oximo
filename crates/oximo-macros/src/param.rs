//! `param!(model, name = value)`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token};

struct ParamInput {
    model: Expr,
    name: Ident,
    value: Expr,
}

impl Parse for ParamInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let model = input.parse::<Expr>()?;
        input.parse::<Token![,]>()?;
        let name = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let value = input.parse::<Expr>()?;
        Ok(Self { model, name, value })
    }
}

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let ParamInput { model, name, value } = syn::parse2(input)?;
    let name_str = name.to_string();
    Ok(quote!( let #name = (#model).__param(#name_str, f64::from(#value)); ))
}
