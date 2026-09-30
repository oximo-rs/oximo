//! `soc_constraint!(model, [name | name = expr | name[i in dom, ...]], [terms] <= bound)`,
//! register the second-order cone constraint `||terms||_2 <= bound`. Every
//! term and the bound must be affine (validated at registration). The
//! relation must be a bracketed term list on the left of a single `<=`.

use proc_macro2::{Delimiter, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Expr;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::constraint::computed_name;
use crate::{
    Named, RelOp, build_set, next_seg, oximo_root, parse_named, split_relops, split_top_commas,
};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let parts = split_top_commas(input);
    let mut parts = parts.into_iter();
    let model_ts = parts.next().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "soc_constraint! needs a model expression")
    })?;
    let mut sums = crate::sum::ModelSums::new(syn::parse2(model_ts)?);
    let model = sums.receiver();

    let first = parts.next().ok_or_else(|| {
        syn::Error::new_spanned(&model, "soc_constraint! needs `[terms] <= bound`")
    })?;
    let second = parts.next();
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(
            extra,
            "unexpected trailing tokens in soc_constraint!",
        ));
    }

    let root = oximo_root();

    let expanded = match second {
        None => {
            let (terms, bound) = parse_relation(first, &mut sums)?;
            quote!( (#model).__add_soc_constraint_auto([#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #bound) )
        }
        Some(rel_tokens) => {
            // Computed name at run-time: `soc_constraint!(m, name = expr, ..)`.
            if let Some(name_expr) = computed_name(&first) {
                let (terms, bound) = parse_relation(rel_tokens, &mut sums)?;
                return Ok(sums.wrap(quote!(
                    (#model).add_soc_constraint(#name_expr, [#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #bound)
                )));
            }

            let Named { name, binds, cond } = parse_named(first)?;
            let name_str = name.to_string();
            if let Some(binds) = &binds {
                sums.indexed(binds);
            }
            let (terms, bound) = parse_relation(rel_tokens, &mut sums)?;
            match binds {
                None => quote!(
                    (#model).add_soc_constraint(#name_str, [#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #bound)
                ),
                Some(binds) => {
                    let param = family_closure_param(&binds);
                    let used = mark_bindings_used(&binds);
                    let set = build_set(&binds, &root)?;
                    let set = filtered_set(set, &binds, cond.as_ref(), &root);
                    quote! {
                        (#model).__add_soc_constraints_over(
                            #name_str,
                            &(#set),
                            |#param| { #used ([#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #bound) },
                        );
                    }
                }
            }
        }
    };
    Ok(sums.wrap(expanded))
}

/// Parse the relation `[term, term, ...] <= bound` into the term expressions
/// and the bound expression.
fn parse_relation(
    tokens: TokenStream2,
    sums: &mut crate::sum::ModelSums,
) -> syn::Result<(Vec<Expr>, Expr)> {
    const SHAPE: &str = "a SOC constraint must be written `[term, term, ...] <= bound`";
    let (segs, ops) = split_relops(&tokens);
    let (lhs, rhs) = match (segs.len(), ops.as_slice()) {
        (2, [RelOp::Le]) => {
            let mut segs = segs.into_iter();
            (next_seg(&mut segs)?, next_seg(&mut segs)?)
        }
        _ => return Err(syn::Error::new_spanned(tokens, SHAPE)),
    };

    let mut lhs_tts = lhs.into_iter();
    let group = match (lhs_tts.next(), lhs_tts.next()) {
        (Some(TokenTree::Group(g)), None) if g.delimiter() == Delimiter::Bracket => g,
        (Some(other), _) => return Err(syn::Error::new(other.span(), SHAPE)),
        (None, _) => return Err(syn::Error::new_spanned(tokens, SHAPE)),
    };

    let terms = split_top_commas(group.stream())
        .into_iter()
        .filter(|seg| !seg.is_empty())
        .map(|ts| parse_seg(ts, sums))
        .collect::<syn::Result<Vec<Expr>>>()?;
    if terms.is_empty() {
        return Err(syn::Error::new(group.span(), "a SOC constraint needs at least one term"));
    }

    let bound = parse_seg(rhs, sums)?;
    Ok((terms, bound))
}

fn parse_seg(ts: TokenStream2, sums: &mut crate::sum::ModelSums) -> syn::Result<Expr> {
    sums.rewrite(syn::parse2(crate::index::rewrite_index_subscripts(ts))?)
}
