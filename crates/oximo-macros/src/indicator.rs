//! `indicator_constraint!(model, [name|name[idx]], binary == 0|1 => relation)`.

use proc_macro2::{Spacing, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Expr;

use crate::bind::{family_closure_param, filtered_set, mark_bindings_used};
use crate::constraint::{Relations, build_relations, computed_name};
use crate::{Named, build_set, next_seg, oximo_root, parse_named, split_relops, split_top_commas};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let parts = split_top_commas(input);
    let mut parts = parts.into_iter();
    let model_ts = parts.next().ok_or_else(|| err("needs a model expression"))?;
    let mut sums = crate::sum::ModelSums::new(syn::parse2(model_ts)?);
    let model = sums.receiver();
    let first = parts.next().ok_or_else(|| err("needs an implication"))?;
    let second = parts.next();
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(
            extra,
            "indicator_constraint! unexpected trailing tokens",
        ));
    }
    let (name, implication) = match second {
        None => (None, first),
        Some(v) => (Some(first), v),
    };
    let (trigger_tokens, consequent_tokens) = split_arrow(implication)?;
    let (trigger, active_value) = parse_trigger(trigger_tokens, &mut sums)?;
    let root = oximo_root();

    if let Some(name_tokens) = name {
        if let Some(name_expr) = computed_name(&name_tokens) {
            let rel = build_relations(consequent_tokens, &root, &mut sums)?;
            return Ok(sums.wrap(register_computed(&model, name_expr, trigger, active_value, rel)));
        }
        let Named { name, binds, cond } = parse_named(name_tokens)?;
        let name_str = name.to_string();
        if let Some(binds) = &binds {
            sums.indexed(binds);
        }
        let rel = build_relations(consequent_tokens, &root, &mut sums)?;
        let expanded = match binds {
            None => register_named(&model, &name_str, trigger, active_value, rel),
            Some(binds) => {
                let set = build_set(&binds, &root)?;
                let set = filtered_set(set, &binds, cond.as_ref(), &root);
                register_family(&model, &name_str, set, &binds, trigger, active_value, rel)
            }
        };
        Ok(sums.wrap(expanded))
    } else {
        let rel = build_relations(consequent_tokens, &root, &mut sums)?;
        Ok(sums.wrap(register_anonymous(&model, trigger, active_value, rel)))
    }
}

fn split_arrow(tokens: TokenStream2) -> syn::Result<(TokenStream2, TokenStream2)> {
    let tts: Vec<_> = tokens.into_iter().collect();
    let mut found = None;
    for i in 0..tts.len().saturating_sub(1) {
        if let (TokenTree::Punct(a), TokenTree::Punct(b)) = (&tts[i], &tts[i + 1])
            && a.as_char() == '='
            && a.spacing() == Spacing::Joint
            && b.as_char() == '>'
        {
            if found.is_some() {
                return Err(err("must contain exactly one `=>`"));
            }
            found = Some(i);
        }
    }
    let i = found.ok_or_else(|| err("expected `binary == 0|1 => relation`"))?;
    let left: TokenStream2 = tts[..i].iter().cloned().collect();
    let right: TokenStream2 = tts[i + 2..].iter().cloned().collect();
    if left.is_empty() || right.is_empty() {
        return Err(err("malformed implication"));
    }
    Ok((left, right))
}

fn parse_trigger(
    tokens: TokenStream2,
    sums: &mut crate::sum::ModelSums,
) -> syn::Result<(Expr, bool)> {
    let (segs, ops) = split_relops(&tokens);
    if segs.len() != 2 || ops.as_slice() != [crate::RelOp::Eq] {
        return Err(syn::Error::new_spanned(
            tokens,
            "indicator trigger must be `binary == 0` or `binary == 1`",
        ));
    }
    let mut segs = segs.into_iter();
    let trigger =
        sums.rewrite(syn::parse2(crate::index::rewrite_index_subscripts(next_seg(&mut segs)?))?)?;
    let value_tokens = next_seg(&mut segs)?;
    let value: syn::LitInt = syn::parse2(value_tokens.clone()).map_err(|_| {
        syn::Error::new_spanned(value_tokens, "indicator trigger value must be literal 0 or 1")
    })?;
    let value = match value.base10_parse::<u8>()? {
        0 => false,
        1 => true,
        _ => return Err(syn::Error::new_spanned(value, "indicator trigger value must be 0 or 1")),
    };
    Ok((trigger, value))
}

fn register_anonymous(model: &Expr, trigger: Expr, value: bool, rel: Relations) -> TokenStream2 {
    match rel {
        Relations::Single(r) => {
            quote!((#model).__add_indicator_constraint_auto(#trigger, #value, #r))
        }
        Relations::Range { mid, lo, hi } => {
            quote!((#model).__add_indicator_range_auto(#trigger, #value, #mid, #lo, #hi))
        }
    }
}

fn register_named(
    model: &Expr,
    name: &str,
    trigger: Expr,
    value: bool,
    rel: Relations,
) -> TokenStream2 {
    match rel {
        Relations::Single(r) => {
            quote!((#model).__add_indicator_constraint(#name, #trigger, #value, #r))
        }
        Relations::Range { mid, lo, hi } => {
            quote!((#model).__add_indicator_range(#name, #trigger, #value, #mid, #lo, #hi))
        }
    }
}

fn register_computed(
    model: &Expr,
    name: TokenStream2,
    trigger: Expr,
    value: bool,
    rel: Relations,
) -> TokenStream2 {
    match rel {
        Relations::Single(r) => {
            quote!((#model).__add_indicator_constraint(#name, #trigger, #value, #r))
        }
        Relations::Range { mid, lo, hi } => {
            quote!({ let __name = #name; (#model).__add_indicator_range(&__name, #trigger, #value, #mid, #lo, #hi) })
        }
    }
}

fn register_family(
    model: &Expr,
    name: &str,
    set: TokenStream2,
    binds: &[crate::IndexBind],
    trigger: Expr,
    value: bool,
    rel: Relations,
) -> TokenStream2 {
    let param = family_closure_param(binds);
    let used = mark_bindings_used(binds);
    match rel {
        Relations::Single(r) => quote!((#model).__add_indicator_constraints_over(
            #name, &(#set), |#param| { #used (#trigger, #value, #r) }
        )),
        Relations::Range { mid, lo, hi } => quote!((#model).__add_indicator_ranges_over(
            #name, &(#set), |#param| { #used (#trigger, #value, #mid, #lo, #hi) }
        )),
    }
}

fn err(message: &str) -> syn::Error {
    syn::Error::new(Span::call_site(), format!("indicator_constraint! {message}"))
}
