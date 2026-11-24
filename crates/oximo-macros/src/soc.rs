//! `soc_constraint!(model, [name | name = expr | name[i in dom, ...]], relation)`,
//! register an explicit second-order cone constraint.
//! The relation is:
//! `[terms] <= bound` or `[terms] in soc(bound)`, the standard cone
//! `||terms||_2 <= bound`, or
//! `[terms] in rotated(u, v)`, the rotated cone `||terms||_2^2 <= 2 u v` with `u, v >= 0`.
//! Every term and head must be affine (validated at registration).

use proc_macro2::{Delimiter, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Expr;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::constraint::computed_name;
use crate::{
    Named, RelOp, build_set, next_seg, oximo_root, parse_comma_list, parse_named, split_relops,
};

const SHAPE: &str = "a SOC constraint must be written `[term, ...] <= bound`, `[term, ...] in soc(bound)` or `[term, ...] in rotated(u, v)`";

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let parts = parse_comma_list(input, "missing macro argument")?;
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
            let (terms, head) = parse_relation(first, &mut sums)?;
            let head = head.tokens(&root);
            quote!( (#model).__add_soc_constraint_auto([#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #head) )
        }
        Some(rel_tokens) => {
            // Computed name at run-time: `soc_constraint!(m, name = expr, ..)`.
            if let Some(name_expr) = computed_name(&first) {
                let (terms, head) = parse_relation(rel_tokens, &mut sums)?;
                let head = head.tokens(&root);
                return Ok(sums.wrap(quote!(
                    (#model).__add_soc_constraint_head(#name_expr, [#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #head)
                )));
            }

            let Named { name, binds, cond } = parse_named(first)?;
            let name_str = name.to_string();
            if let Some(binds) = &binds {
                sums.indexed(binds);
            }
            let (terms, head) = parse_relation(rel_tokens, &mut sums)?;
            let head = head.tokens(&root);
            match binds {
                None => quote!(
                    (#model).__add_soc_constraint_head(#name_str, [#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #head)
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
                            |#param| { #used ([#(#root::__macro_support::IntoAffineFunction::into_affine_function(#terms)),*], #head) },
                        );
                    }
                }
            }
        }
    };
    Ok(sums.wrap(expanded))
}

/// The cone head written after the term list.
enum Head {
    /// `<= bound` or `in soc(bound)`.
    Standard(Expr),
    /// `in rotated(u, v)`.
    Rotated(Expr, Expr),
}

impl Head {
    fn tokens(&self, root: &TokenStream2) -> TokenStream2 {
        match self {
            Self::Standard(bound) => quote!(#bound),
            Self::Rotated(u, v) => quote!(#root::__macro_support::RotatedHead(#u, #v)),
        }
    }
}

/// Parse `[term, ...] <= bound`, `[term, ...] in soc(bound)` or
/// `[term, ...] in rotated(u, v)` into the term expressions and the head.
fn parse_relation(
    tokens: TokenStream2,
    sums: &mut crate::sum::ModelSums,
) -> syn::Result<(Vec<Expr>, Head)> {
    let tts: Vec<TokenTree> = tokens.clone().into_iter().collect();
    let in_position = tts.iter().position(|tt| matches!(tt, TokenTree::Ident(i) if i == "in"));
    let (lhs, head) = if let Some(at) = in_position {
        let lhs: TokenStream2 = tts[..at].iter().cloned().collect();
        let head = parse_cone_set(&tts[at + 1..], &tokens, sums)?;
        (lhs, head)
    } else {
        let (segs, ops) = split_relops(&tokens);
        match (segs.len(), ops.as_slice()) {
            (2, [RelOp::Le]) => {
                let mut segs = segs.into_iter();
                let lhs = next_seg(&mut segs)?;
                let rhs = next_seg(&mut segs)?;
                (lhs, Head::Standard(parse_seg(rhs, sums)?))
            }
            _ => return Err(syn::Error::new_spanned(tokens, SHAPE)),
        }
    };

    let mut lhs_tts = lhs.into_iter();
    let group = match (lhs_tts.next(), lhs_tts.next()) {
        (Some(TokenTree::Group(g)), None) if g.delimiter() == Delimiter::Bracket => g,
        (Some(other), _) => return Err(syn::Error::new(other.span(), SHAPE)),
        (None, _) => return Err(syn::Error::new_spanned(tokens, SHAPE)),
    };

    let terms = parse_comma_list(group.stream(), "missing SOC term")?
        .into_iter()
        .map(|ts| parse_seg(ts, sums))
        .collect::<syn::Result<Vec<Expr>>>()?;
    if terms.is_empty() {
        return Err(syn::Error::new(group.span(), "a SOC constraint needs at least one term"));
    }
    Ok((terms, head))
}

/// Parse `soc(t)` or `rotated(u, v)` after the `in` keyword.
fn parse_cone_set(
    tts: &[TokenTree],
    all: &TokenStream2,
    sums: &mut crate::sum::ModelSums,
) -> syn::Result<Head> {
    const CONE: &str = "expected `soc(bound)` or `rotated(u, v)` after `in`";
    let (kind, args) = match tts {
        [TokenTree::Ident(kind), TokenTree::Group(args)]
            if (kind == "soc" || kind == "rotated")
                && args.delimiter() == Delimiter::Parenthesis =>
        {
            (kind, args)
        }
        [first, ..] => return Err(syn::Error::new(first.span(), CONE)),
        [] => return Err(syn::Error::new_spanned(all, CONE)),
    };
    let heads = parse_comma_list(args.stream(), "missing cone head")?;
    if kind == "soc" {
        let [t]: [TokenStream2; 1] = heads
            .try_into()
            .map_err(|_| syn::Error::new(args.span(), "`soc(bound)` takes exactly one bound"))?;
        return Ok(Head::Standard(parse_seg(t, sums)?));
    }
    let [u, v]: [TokenStream2; 2] = heads
        .try_into()
        .map_err(|_| syn::Error::new(args.span(), "`rotated(u, v)` takes exactly two heads"))?;
    Ok(Head::Rotated(parse_seg(u, sums)?, parse_seg(v, sums)?))
}

fn parse_seg(ts: TokenStream2, sums: &mut crate::sum::ModelSums) -> syn::Result<Expr> {
    sums.rewrite(crate::index::parse_value_expr(ts)?)
}
