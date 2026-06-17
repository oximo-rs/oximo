//! `constraint!(model, [name | name = expr | name[i in dom, ...]], <relation>)`,
//! where `<relation>` is `lhs <op> rhs` or a two-sided range `lo <= e <= hi`
//! (`hi >= e >= lo`). A range lowers to two rows, `{name}_lo` and `{name}_hi`.

// TODO: Improve so that ranges don't lower to two rows.

use proc_macro2::{Spacing, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::Expr;

use crate::bind::{family_closure_param, filtered_set};
use crate::{Named, RelOp, build_set, oximo_root, parse_named, split_relops, split_top_commas};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let parts = split_top_commas(input);
    let mut parts = parts.into_iter();
    let model = parts.next().expect("split always yields at least one segment");
    let model: Expr = syn::parse2(model)?;

    let first = parts.next().ok_or_else(|| {
        syn::Error::new(Span::call_site(), "constraint! needs a relational expression")
    })?;
    let second = parts.next();
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(extra, "unexpected trailing tokens in constraint!"));
    }

    let root = oximo_root();

    match second {
        None => Ok(register_anonymous(&model, build_relations(first, &root)?, &root)),
        Some(rel_tokens) => {
            // Computed name at run-time: `constraint!(m, name = expr, <relation>)`.
            if let Some(name_expr) = computed_name(&first) {
                let rel = build_relations(rel_tokens, &root)?;
                return Ok(register_computed(&model, &name_expr, rel, &root));
            }

            let Named { name, binds, cond } = parse_named(first)?;
            let name_str = name.to_string();
            let rel = build_relations(rel_tokens, &root)?;
            match binds {
                None => Ok(register_named(&model, &name_str, rel, &root)),
                Some(binds) => {
                    let param = family_closure_param(&binds);
                    let set = build_set(&binds, &root);
                    let set = filtered_set(set, &binds, cond.as_ref(), &root);
                    Ok(register_family(&model, &name_str, &set, &param, rel, &root))
                }
            }
        }
    }
}

/// Detect the computed-name form `name = EXPR` in the name slot, returning the
/// `EXPR` tokens. The literal marker `name` keeps a bare ident a literal name.
fn computed_name(first: &TokenStream2) -> Option<TokenStream2> {
    let mut it = first.clone().into_iter();
    match it.next()? {
        TokenTree::Ident(id) if id == "name" => {}
        _ => return None,
    }
    match it.next()? {
        TokenTree::Punct(p) if p.as_char() == '=' && p.spacing() == Spacing::Alone => {}
        _ => return None,
    }
    let expr: TokenStream2 = it.collect();
    (!expr.is_empty()).then_some(expr)
}

/// A constraint relation: a single `lhs <op> rhs`, or a two-sided range that
/// lowers to two rows.
#[allow(clippy::large_enum_variant)]
enum Relations {
    Single(TokenStream2),
    Range { mid: Expr, lo: Expr, hi: Expr },
}

fn parse_seg(ts: TokenStream2) -> syn::Result<Expr> {
    syn::parse2(crate::index::rewrite_index_subscripts(ts))
}

/// Split the relation on its relational operators. One operator yields a
/// [`Relations::Single`]. Two like operators (`<= <=` or `>= >=`) a
/// [`Relations::Range`].
fn build_relations(tokens: TokenStream2, root: &TokenStream2) -> syn::Result<Relations> {
    let (segs, ops) = split_relops(tokens);
    match (segs.len(), ops.as_slice()) {
        (2, [op]) => {
            let method = op.method();
            let mut segs = segs.into_iter();
            let lhs = parse_seg(segs.next().unwrap())?;
            let rhs = parse_seg(segs.next().unwrap())?;
            Ok(Relations::Single(quote!( #root::__macro_support::Relate::#method(#lhs, #rhs) )))
        }
        (3, [a, b]) => {
            let mut segs = segs.into_iter();
            let s0 = parse_seg(segs.next().unwrap())?;
            let mid = parse_seg(segs.next().unwrap())?;
            let s2 = parse_seg(segs.next().unwrap())?;
            match (a, b) {
                // lo <= mid <= hi
                (RelOp::Le, RelOp::Le) => Ok(Relations::Range { mid, lo: s0, hi: s2 }),
                // hi >= mid >= lo
                (RelOp::Ge, RelOp::Ge) => Ok(Relations::Range { mid, lo: s2, hi: s0 }),
                _ => Err(syn::Error::new(
                    Span::call_site(),
                    "a two-sided range must use `<=` twice (`lo <= e <= hi`) or `>=` twice (`hi >= e >= lo`)",
                )),
            }
        }
        _ => Err(syn::Error::new(
            Span::call_site(),
            "constraint must contain exactly one `==`/`<=`/`>=`, or be a two-sided range `lo <= e <= hi`",
        )),
    }
}

/// The two relation tokens for a range: `mid >= lo` (the `_lo` row) and
/// `mid <= hi` (the `_hi` row).
fn range_rows(lo: &Expr, hi: &Expr, root: &TokenStream2) -> (TokenStream2, TokenStream2) {
    (
        quote!( #root::__macro_support::Relate::ge(__mid, #lo) ),
        quote!( #root::__macro_support::Relate::le(__mid, #hi) ),
    )
}

fn register_anonymous(model: &Expr, rel: Relations, root: &TokenStream2) -> TokenStream2 {
    match rel {
        Relations::Single(r) => quote!( (#model).__add_constraint_auto(#r) ),
        Relations::Range { mid, lo, hi } => {
            let (lo_rel, hi_rel) = range_rows(&lo, &hi, root);
            quote! {{
                let __mid = #mid;
                (#model).__add_constraint_auto(#lo_rel);
                (#model).__add_constraint_auto(#hi_rel);
            }}
        }
    }
}

fn register_named(
    model: &Expr,
    name_str: &str,
    rel: Relations,
    root: &TokenStream2,
) -> TokenStream2 {
    match rel {
        Relations::Single(r) => quote!( (#model).__add_constraint(#name_str, #r) ),
        Relations::Range { mid, lo, hi } => {
            let (lo_rel, hi_rel) = range_rows(&lo, &hi, root);
            quote! {{
                let __mid = #mid;
                (#model).__add_constraint(::core::concat!(#name_str, "_lo"), #lo_rel);
                (#model).__add_constraint(::core::concat!(#name_str, "_hi"), #hi_rel);
            }}
        }
    }
}

fn register_computed(
    model: &Expr,
    name_expr: &TokenStream2,
    rel: Relations,
    root: &TokenStream2,
) -> TokenStream2 {
    match rel {
        Relations::Single(r) => quote!( (#model).__add_constraint(#name_expr, #r) ),
        Relations::Range { mid, lo, hi } => {
            let (lo_rel, hi_rel) = range_rows(&lo, &hi, root);
            quote! {{
                let __mid = #mid;
                let __name = #name_expr;
                (#model).__add_constraint(::std::format!("{__name}_lo"), #lo_rel);
                (#model).__add_constraint(::std::format!("{__name}_hi"), #hi_rel);
            }}
        }
    }
}

fn register_family(
    model: &Expr,
    name_str: &str,
    set: &TokenStream2,
    param: &TokenStream2,
    rel: Relations,
    root: &TokenStream2,
) -> TokenStream2 {
    match rel {
        Relations::Single(r) => quote! {
            (#model).__add_constraints_over(#name_str, &(#set), |#param| #r);
        },
        Relations::Range { mid, lo, hi } => {
            let (lo_rel, hi_rel) = range_rows(&lo, &hi, root);
            quote! {{
                let __set = #set;
                (#model).__add_constraints_over(::core::concat!(#name_str, "_lo"), &__set, |#param| {
                    let __mid = #mid;
                    #lo_rel
                });
                (#model).__add_constraints_over(::core::concat!(#name_str, "_hi"), &__set, |#param| {
                    let __mid = #mid;
                    #hi_rel
                });
            }}
        }
    }
}
