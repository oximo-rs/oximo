//! `objective!(model, Min|Max, expr)`. The sense also accepts the long forms
//! `Minimize`/`Maximize` and lowercase `min`/`max`.

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
    // Rewrite `q[i, j, k]` index sugar in the objective expression.
    let input = crate::index::rewrite_index_subscripts(input);
    let ObjectiveInput { model, sense, expr } = syn::parse2(input)?;

    let method = match sense.to_string().as_str() {
        "Min" | "Minimize" | "min" => quote!(__minimize),
        "Max" | "Maximize" | "max" => quote!(__maximize),
        _ => {
            return Err(syn::Error::new(
                sense.span(),
                "objective sense must be `Min`/`min`/`Minimize` or `Max`/`max`/`Maximize`",
            ));
        }
    };

    Ok(quote!( (#model).#method(#expr) ))
}
