//! `objective!(m, Min|Max, expr)` sets the model objective and sense. The sense
//! also accepts the long forms `Minimize`/`Maximize` and lowercase `min`/`max`.
//!
//! `objective!(m, Feasibility)` (also `feasibility`/`feas`) declares a feasibility
//! problem: no objective to optimize, just find any point satisfying the model.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token};

enum ObjectiveInput {
    Feasibility { model: Expr },
    Optimize { model: Expr, sense: Ident, expr: Expr },
}

fn is_feasibility_kw(ident: &Ident) -> bool {
    matches!(ident.to_string().as_str(), "Feasibility" | "feasibility" | "Feas" | "feas")
}

impl Parse for ObjectiveInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let model = input.parse::<Expr>()?;
        input.parse::<Token![,]>()?;
        let sense = input.parse::<Ident>()?;

        // Two-argument form: `objective!(m, Feasibility)`.
        if input.is_empty() {
            if is_feasibility_kw(&sense) {
                return Ok(Self::Feasibility { model });
            }
            return Err(syn::Error::new(
                sense.span(),
                "expected an objective expression after the sense, or \
                 `Feasibility` for a feasibility problem",
            ));
        }

        // Three-argument form: `objective!(m, Min|Max, expr)`.
        input.parse::<Token![,]>()?;
        let expr = input.parse::<Expr>()?;
        if is_feasibility_kw(&sense) {
            return Err(syn::Error::new(
                sense.span(),
                "`Feasibility` takes no objective expression; write `objective!(m, Feasibility)`",
            ));
        }
        Ok(Self::Optimize { model, sense, expr })
    }
}

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    // Rewrite `q[i, j, k]` index sugar in the objective expression.
    let input = crate::index::rewrite_index_subscripts(input);

    match syn::parse2(input)? {
        ObjectiveInput::Feasibility { model } => Ok(quote!( (#model).__feasibility() )),
        ObjectiveInput::Optimize { model, sense, expr } => {
            let method = match sense.to_string().as_str() {
                "Min" | "Minimize" | "min" => quote!(__minimize),
                "Max" | "Maximize" | "max" => quote!(__maximize),
                _ => {
                    return Err(syn::Error::new(
                        sense.span(),
                        "objective sense must be `Min`/`min`/`Minimize` or \
                         `Max`/`max`/`Maximize`",
                    ));
                }
            };
            Ok(quote!( (#model).#method(#expr) ))
        }
    }
}
