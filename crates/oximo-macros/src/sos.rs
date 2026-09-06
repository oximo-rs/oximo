use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, quote};
use syn::Expr;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::constraint::computed_name;
use crate::{Named, build_set, oximo_root, parse_named, split_top_commas};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let parts = split_top_commas(input);
    let mut parts = parts.into_iter();
    let model_ts = parts.next().ok_or_else(|| err("needs a model expression"))?;
    let model: Expr = syn::parse2(model_ts)?;
    let first = parts.next().ok_or_else(|| {
        syn::Error::new_spanned(&model, "sos_constraint! needs a name or SOS type")
    })?;
    let second = parts.next().ok_or_else(|| {
        syn::Error::new_spanned(&first, "sos_constraint! needs SOS1/SOS2 and a member list")
    })?;
    let third = parts.next();
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(extra, "sos_constraint! unexpected trailing tokens"));
    }
    let root = oximo_root();
    let named = third.is_some();
    let (sos_type, members) = if let Some(third) = third {
        (parse_type(&second, &root)?, parse_members(&third)?)
    } else {
        (parse_type(&first, &root)?, parse_members(&second)?)
    };

    if !named {
        return Ok(match &members {
            Members::Explicit(members) => {
                quote!((#model).__add_sos_constraint_auto(#sos_type, [#(#members),*]))
            }
            Members::Auto(members) => {
                quote!((#model).__add_sos_constraint_auto_weights(#sos_type, [#(#members),*]))
            }
        });
    }
    if let Some(name_expr) = computed_name(&first) {
        return Ok(match &members {
            Members::Explicit(members) => {
                quote!((#model).add_sos_constraint(#name_expr, #sos_type, [#(#members),*]))
            }
            Members::Auto(members) => quote! {
                (#model).add_sos_constraint_auto_weights(#name_expr, #sos_type, [#(#members),*])
            },
        });
    }
    let Named { name, binds, cond } = parse_named(first)?;
    let name_str = name.to_string();
    match binds {
        None => Ok(match &members {
            Members::Explicit(members) => {
                quote!((#model).add_sos_constraint(#name_str, #sos_type, [#(#members),*]))
            }
            Members::Auto(members) => quote! {
                (#model).add_sos_constraint_auto_weights(#name_str, #sos_type, [#(#members),*])
            },
        }),
        Some(binds) => {
            let param = family_closure_param(&binds);
            let used = mark_bindings_used(&binds);
            let set = build_set(&binds, &root)?;
            let set = filtered_set(set, &binds, cond.as_ref(), &root);
            Ok(match &members {
                Members::Explicit(members) => quote! {
                    (#model).__add_sos_constraints_over(
                        #name_str,
                        &(#set),
                        #sos_type,
                        |#param| { #used [#(#members),*] },
                    );
                },
                Members::Auto(members) => quote! {
                    (#model).__add_sos_constraints_over_auto_weights(
                        #name_str,
                        &(#set),
                        #sos_type,
                        |#param| { #used [#(#members),*] },
                    );
                },
            })
        }
    }
}

fn parse_type(tokens: &TokenStream2, root: &TokenStream2) -> syn::Result<TokenStream2> {
    let mut it = tokens.clone().into_iter();
    let Some(TokenTree::Ident(id)) = it.next() else {
        return Err(syn::Error::new_spanned(tokens, "sos_constraint! expected SOS1 or SOS2"));
    };
    if let Some(extra) = it.next() {
        return Err(syn::Error::new_spanned(extra, "sos_constraint! expected SOS1 or SOS2"));
    }
    match id.to_string().as_str() {
        "SOS1" => Ok(quote!(#root::SosType::Sos1)),
        "SOS2" => Ok(quote!(#root::SosType::Sos2)),
        _ => Err(syn::Error::new(id.span(), "expected SOS1 or SOS2")),
    }
}

enum Members {
    Explicit(Vec<TokenStream2>),
    Auto(Vec<TokenStream2>),
}

fn parse_members(tokens: &TokenStream2) -> syn::Result<Members> {
    let mut it = tokens.clone().into_iter();
    let Some(TokenTree::Group(group)) = it.next() else {
        return Err(syn::Error::new_spanned(
            tokens,
            "sos_constraint! members must be written [var, ...] or [(var, weight), ...]",
        ));
    };
    if it.next().is_some() || group.delimiter() != proc_macro2::Delimiter::Bracket {
        return Err(syn::Error::new_spanned(
            tokens,
            "sos_constraint! members must be written [var, ...] or [(var, weight), ...]",
        ));
    }
    let parts: Vec<_> =
        split_top_commas(group.stream()).into_iter().filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        return Err(syn::Error::new(
            group.span(),
            "sos_constraint! the member list cannot be empty",
        ));
    }
    let explicit = parts.iter().all(is_pair);
    let auto = parts.iter().all(|part| !is_pair(part));
    if !explicit && !auto {
        return Err(syn::Error::new_spanned(
            tokens,
            "sos_constraint! do not mix inferred members with explicit `(var, weight)` members",
        ));
    }
    if auto {
        return parts
            .into_iter()
            .map(|part| {
                let expr: Expr = syn::parse2(crate::index::rewrite_index_subscripts(part))?;
                Ok(expr.into_token_stream())
            })
            .collect::<syn::Result<Vec<_>>>()
            .map(Members::Auto);
    }
    parts
        .into_iter()
        .map(|part| {
            let mut outer = part.clone().into_iter();
            let Some(TokenTree::Group(pair)) = outer.next() else {
                return Err(syn::Error::new_spanned(
                    &part,
                    "sos_constraint! SOS member needs exactly `(var, weight)`",
                ));
            };
            if outer.next().is_some() || pair.delimiter() != proc_macro2::Delimiter::Parenthesis {
                return Err(syn::Error::new_spanned(
                    &part,
                    "sos_constraint! SOS member needs exactly `(var, weight)`",
                ));
            }
            let mut p = split_top_commas(pair.stream()).into_iter();
            let expr = p.next().ok_or_else(|| {
                syn::Error::new_spanned(
                    &part,
                    "sos_constraint! SOS member needs a variable and weight",
                )
            })?;
            let weight = p.next().ok_or_else(|| {
                syn::Error::new_spanned(
                    &part,
                    "sos_constraint! SOS member needs a variable and weight",
                )
            })?;
            if p.next().is_some() {
                return Err(syn::Error::new_spanned(
                    &part,
                    "sos_constraint! SOS member needs exactly `(var, weight)`",
                ));
            }
            let expr: Expr = syn::parse2(crate::index::rewrite_index_subscripts(expr))?;
            let weight: Expr = syn::parse2(weight)?;
            Ok(quote!((#expr, #weight)))
        })
        .collect::<syn::Result<Vec<_>>>()
        .map(Members::Explicit)
}

fn is_pair(tokens: &TokenStream2) -> bool {
    let mut it = tokens.clone().into_iter();
    matches!(it.next(), Some(TokenTree::Group(group)) if group.delimiter() == proc_macro2::Delimiter::Parenthesis)
}

fn err(message: &str) -> syn::Error {
    syn::Error::new(Span::call_site(), format!("sos_constraint! {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(source: &str) -> TokenStream2 {
        source.parse().expect("valid token stream")
    }

    #[test]
    fn parses_inferred_and_explicit_members() {
        match parse_members(&tokens("[x, y[i]]")).expect("inferred members") {
            Members::Auto(members) => assert_eq!(members.len(), 2),
            Members::Explicit(_) => panic!("expected inferred members"),
        }
        match parse_members(&tokens("[(x, 1.0), (y, 2.0)]")).expect("explicit members") {
            Members::Explicit(members) => assert_eq!(members.len(), 2),
            Members::Auto(_) => panic!("expected explicit members"),
        }
    }

    #[test]
    fn rejects_malformed_member_lists() {
        for source in ["x", "[]", "[(x, 1.0), y]", "[(x)]", "[(x, 1.0, 2.0)]"] {
            assert!(parse_members(&tokens(source)).is_err(), "expected rejection for {source}");
        }
    }

    #[test]
    fn parses_only_supported_types() {
        assert!(parse_type(&tokens("SOS1"), &tokens("::root")).is_ok());
        assert!(parse_type(&tokens("SOS2"), &tokens("::root")).is_ok());
        assert!(parse_type(&tokens("SOC"), &tokens("::root")).is_err());
        assert!(parse_type(&tokens("SOS1 SOS2"), &tokens("::root")).is_err());
    }
}
