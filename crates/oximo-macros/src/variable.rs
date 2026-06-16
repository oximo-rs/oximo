//! `variable!(model, spec[, Bin|Int])`.
//!
//! Accepted `spec` shapes (where `name` may be `name` or `name[i in dom, ...]`):
//! `name`, `name >= lb`, `name <= ub`, `lb <= name <= ub`. Bounds may reference
//! the index variables (e.g. `b[i in I] <= b_max[i]`), in which case they are
//! lowered to the builder's per-key `.lb_by` / `.ub_by`.

use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, quote};
use syn::{Ident, Pat};

use crate::bind::filtered_set;
use crate::{
    IndexBind, Named, RelOp, build_set, oximo_root, parse_named, split_relops, split_top_commas,
};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let mut parts = split_top_commas(input).into_iter();
    let model = parts.next().expect("split always yields at least one segment");
    let model: syn::Expr = syn::parse2(model)?;

    let spec = parts.next().ok_or_else(|| {
        syn::Error::new(proc_macro2::Span::call_site(), "variable! needs a `name`/bounds spec")
    })?;
    let domain = parts.next();
    if let Some(extra) = parts.next() {
        return Err(syn::Error::new_spanned(extra, "unexpected trailing tokens in variable!"));
    }

    let domain_method = match domain {
        None => quote!(),
        Some(ts) => {
            let id: Ident = syn::parse2(ts)?;
            match id.to_string().as_str() {
                "Bin" | "Binary" => quote!(.binary()),
                "Int" | "Integer" => quote!(.integer()),
                "Real" | "Cont" | "Continuous" => quote!(),
                _ => {
                    return Err(syn::Error::new(
                        id.span(),
                        "domain must be `Bin`, `Int`, or `Real`",
                    ));
                }
            }
        }
    };

    // Split the bound spec on relational operators and identify the core name.
    let (segs, ops) = split_relops(spec);
    let (core, lb, ub) = match (segs.len(), ops.as_slice()) {
        (1, []) => (segs[0].clone(), None, None),
        (2, [RelOp::Le]) => (segs[0].clone(), None, Some(segs[1].clone())),
        (2, [RelOp::Ge]) => (segs[0].clone(), Some(segs[1].clone()), None),
        (3, [RelOp::Le, RelOp::Le]) => {
            (segs[1].clone(), Some(segs[0].clone()), Some(segs[2].clone()))
        }
        _ => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "variable! bounds must be `name`, `name >= lb`, `name <= ub`, or `lb <= name <= ub`",
            ));
        }
    };

    // Bound expressions are value expressions, so `q[i, j]` index sugar applies.
    let lb = lb.map(crate::index::rewrite_index_subscripts);
    let ub = ub.map(crate::index::rewrite_index_subscripts);

    let Named { name, binds, cond } = parse_named(core)?;
    let name_str = name.to_string();
    let root = oximo_root();

    let mut idents = Vec::new();
    if let Some(binds) = &binds {
        for b in binds {
            collect_idents(&b.pat.to_token_stream(), &mut idents);
        }
    }
    let binds_slice = binds.as_deref();

    let bound_method = |val: &TokenStream2, kind: BoundKind| -> TokenStream2 {
        match binds_slice {
            Some(bs) if references_any(val, &idents) => {
                let param = bound_closure_param(bs, val);
                match kind {
                    BoundKind::Lb => quote!(.lb_by(move |#param| f64::from(#val))),
                    BoundKind::Ub => quote!(.ub_by(move |#param| f64::from(#val))),
                }
            }
            _ => match kind {
                BoundKind::Lb => quote!(.lb(f64::from(#val))),
                BoundKind::Ub => quote!(.ub(f64::from(#val))),
            },
        }
    };

    let mut bounds = TokenStream2::new();
    if let Some(lb) = &lb {
        bounds.extend(bound_method(lb, BoundKind::Lb));
    }
    if let Some(ub) = &ub {
        bounds.extend(bound_method(ub, BoundKind::Ub));
    }

    let expanded = match binds {
        None => quote! {
            let #name = (#model).__var(#name_str) #domain_method #bounds .build();
        },
        Some(binds) => {
            let set = build_set(&binds, &root);
            let set = filtered_set(set, &binds, cond.as_ref(), &root);
            quote! {
                let #name = {
                    let __set = #set;
                    (#model).__indexed_var(#name_str, &__set) #domain_method #bounds .build()
                };
            }
        }
    };
    Ok(expanded)
}

#[derive(Copy, Clone)]
enum BoundKind {
    Lb,
    Ub,
}

/// Closure parameter for a per-key (`.lb_by`/`.ub_by`) bound: each index the
/// bound does not reference is replaced with `_`, so the generated closure never
/// has an unused parameter. The builder's key type pins the parameter, so it is
/// left bare for inference unless the user annotated every binding.
fn bound_closure_param(binds: &[IndexBind], bound: &TokenStream2) -> TokenStream2 {
    let masked = binds.iter().map(|b| mask_pat(&b.pat, bound));
    let pattern = if binds.len() == 1 {
        let p = mask_pat(&binds[0].pat, bound);
        quote!(#p)
    } else {
        quote!( (#(#masked),*) )
    };

    let tys: Option<Vec<&syn::Type>> = binds.iter().map(|b| b.ty.as_ref()).collect();
    match tys {
        Some(tys) if binds.len() == 1 => {
            let ty = tys[0];
            quote!(#pattern: #ty)
        }
        Some(tys) => quote!( #pattern: (#(#tys),*) ),
        None => pattern,
    }
}

/// Replace each bare-ident sub-pattern the bound does not reference with `_`,
/// recursing into tuple patterns.
fn mask_pat(pat: &Pat, bound: &TokenStream2) -> TokenStream2 {
    match pat {
        Pat::Tuple(t) => {
            let elems = t.elems.iter().map(|e| mask_pat(e, bound));
            quote!( (#(#elems),*) )
        }
        Pat::Ident(pi) if pi.subpat.is_none() && pi.by_ref.is_none() => {
            if references_any(bound, &[pi.ident.to_string()]) { quote!(#pat) } else { quote!(_) }
        }
        _ => quote!(#pat),
    }
}

/// Collect every identifier appearing in a token stream (recursing into groups).
fn collect_idents(ts: &TokenStream2, out: &mut Vec<String>) {
    for tt in ts.clone() {
        match tt {
            TokenTree::Ident(id) => out.push(id.to_string()),
            TokenTree::Group(g) => collect_idents(&g.stream(), out),
            _ => {}
        }
    }
}

/// Whether a token stream references any of the given identifiers.
fn references_any(ts: &TokenStream2, idents: &[String]) -> bool {
    ts.clone().into_iter().any(|tt| match tt {
        TokenTree::Ident(id) => idents.contains(&id.to_string()),
        TokenTree::Group(g) => references_any(&g.stream(), idents),
        _ => false,
    })
}
