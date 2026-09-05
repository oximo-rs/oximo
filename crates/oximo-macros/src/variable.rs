//! `variable!(model, spec[, domain][, kw = val ...])`.
//!
//! Accepted `spec` shapes (where `name` may be `name` or `name[i in dom, ...]`):
//! `name`, `name >= lb`, `name <= ub`, `lb <= name <= ub`. Bounds may reference
//! the index variables (e.g. `b[i in I] <= b_max[i]`), in which case they are
//! lowered to the builder's per-key `.lb_by` / `.ub_by`.
//!
//! After the spec you can pass, in any order, an optional positional `domain`
//! token (`Bin`/`Int`/`Real` and aliases, or a call `SemiCont(thr)` /
//! `SemiContinuous(thr)` / `SemiInt(thr)` / `SemiInteger(thr)`) and any of the
//! keyword args `lb = expr`, `ub = expr`, `domain = <domain>`, `initial = expr`,
//! `fix = expr`. Keyword `lb`/`ub` behave exactly like the relational form (and
//! may reference the index). `initial`/`fix` are scalar-only.

use proc_macro2::{Spacing, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, quote};

use crate::bind::{filtered_set, masked_closure_param, references_any};
use crate::{Named, RelOp, build_set, oximo_root, parse_named, split_relops, split_top_commas};

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let mut parts = split_top_commas(input).into_iter();
    let model_ts = parts.next().ok_or_else(|| {
        syn::Error::new(proc_macro2::Span::call_site(), "variable! needs a model expression")
    })?;
    let model: syn::Expr = syn::parse2(model_ts)?;

    let spec = parts
        .next()
        .ok_or_else(|| syn::Error::new_spanned(&model, "variable! needs a `name`/bounds spec"))?;

    let root = oximo_root();

    let Trailing { domain: domain_ts, lb: kw_lb, ub: kw_ub, initial: kw_initial, fix: kw_fix } =
        parse_trailing(parts)?;
    let domain_method = match domain_ts {
        None => quote!(),
        Some(ts) => domain_method(ts, &root)?,
    };

    // Split the bound spec on relational operators and identify the core name.
    let (segs, ops) = split_relops(&spec);
    let (core, rel_lb, rel_ub) = match (segs.len(), ops.as_slice()) {
        (1, []) => (segs[0].clone(), None, None),
        (2, [RelOp::Le]) => (segs[0].clone(), None, Some(segs[1].clone())),
        (2, [RelOp::Ge]) => (segs[0].clone(), Some(segs[1].clone()), None),
        (3, [RelOp::Le, RelOp::Le]) => {
            (segs[1].clone(), Some(segs[0].clone()), Some(segs[2].clone()))
        }
        _ => {
            return Err(syn::Error::new_spanned(
                spec,
                "variable! bounds must be `name`, `name >= lb`, `name <= ub`, or `lb <= name <= ub`",
            ));
        }
    };

    // Merge relational and keyword bounds, where the same bound from both is an error.
    let lb = merge_bound(rel_lb, kw_lb, "lb")?;
    let ub = merge_bound(rel_ub, kw_ub, "ub")?;

    // `fix` pins both bounds, so explicit lb/ub alongside it is contradictory.
    if let Some(fix_kw) = kw_fix.as_ref()
        && (lb.is_some() || ub.is_some())
    {
        return Err(syn::Error::new(
            fix_kw.ident.span(),
            "`fix` sets both bounds. Do not combine it with `lb`/`ub`",
        ));
    }

    // Bound expressions are value expressions, so `q[i, j]` index sugar applies.
    let lb = lb.map(crate::index::rewrite_index_subscripts);
    let ub = ub.map(crate::index::rewrite_index_subscripts);

    let Named { name, binds, cond } = parse_named(core)?;
    let name_str = name.to_string();

    // `initial`/`fix` lower to scalar `VarBuilder` methods the indexed builder
    // lacks, so reject them on a family with a clear message.
    if binds.is_some()
        && let Some(kw) = kw_initial.as_ref().or(kw_fix.as_ref())
    {
        return Err(syn::Error::new(
            kw.ident.span(),
            "`initial`/`fix` is not supported on an indexed family. Use `m.set_initial` / \
             `m.fix` per element",
        ));
    }

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
                let param = masked_closure_param(bs, val);
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

    let extras = scalar_extras(kw_initial, kw_fix);

    let expanded = match binds {
        None => quote! {
            let #name = (#model).__var(#name_str) #domain_method #bounds #extras .build();
        },
        Some(binds) => {
            let set = build_set(&binds, &root)?;
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

/// Build the scalar-only `.initial(..)`/`.fix(..)` chain from the keyword args.
/// Both are empty for an indexed family (rejected in `expand`).
fn scalar_extras(initial: Option<Kw>, fix: Option<Kw>) -> TokenStream2 {
    let mut extras = TokenStream2::new();
    if let Some(kw) = initial {
        let init = crate::index::rewrite_index_subscripts(kw.value);
        extras.extend(quote!(.initial(f64::from(#init))));
    }
    if let Some(kw) = fix {
        let fix_val = crate::index::rewrite_index_subscripts(kw.value);
        extras.extend(quote!(.fix(f64::from(#fix_val))));
    }
    extras
}

/// Keyword attribute `kw = value` with the keyword's span preserved for
/// diagnostics. Stored for `lb`/`ub`/`domain`/`initial`/`fix`.
struct Kw {
    ident: proc_macro2::Ident,
    value: TokenStream2,
}

/// The trailing modifiers of a `variable!` declaration: an optional domain
/// (positional token or `domain =` keyword) plus the keyword bound/initial/
/// fix expressions.
struct Trailing {
    domain: Option<TokenStream2>,
    lb: Option<Kw>,
    ub: Option<Kw>,
    initial: Option<Kw>,
    fix: Option<Kw>,
}

/// Parse the segments after the spec into [`Trailing`], collecting one positional
/// domain token and the `lb`/`ub`/`domain`/`initial`/`fix` keyword args (in any
/// order). Errors on a repeated keyword, a second positional token, or a domain
/// given both ways.
fn parse_trailing(parts: impl Iterator<Item = TokenStream2>) -> syn::Result<Trailing> {
    let mut positional_domain: Option<TokenStream2> = None;
    let mut kw_domain: Option<Kw> = None;
    let mut lb: Option<Kw> = None;
    let mut ub: Option<Kw> = None;
    let mut initial: Option<Kw> = None;
    let mut fix: Option<Kw> = None;

    for seg in parts {
        if seg.is_empty() {
            continue; // tolerate a trailing comma
        }
        if let Some((kw, val)) = parse_keyword(&seg) {
            let slot = match kw.to_string().as_str() {
                "lb" => &mut lb,
                "ub" => &mut ub,
                "domain" => &mut kw_domain,
                "initial" | "init" => &mut initial,
                "fix" => &mut fix,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &kw,
                        format!("`{kw}` is not a valid variable! keyword"),
                    ));
                }
            };
            if slot.is_some() {
                return Err(syn::Error::new_spanned(&kw, format!("`{kw}` specified twice")));
            }
            *slot = Some(Kw { ident: kw, value: val });
        } else if positional_domain.is_some() {
            return Err(syn::Error::new_spanned(
                &seg,
                "unexpected trailing tokens in variable! (only one positional domain token is \
                 allowed; use `lb =`/`ub =`/`domain =`/`initial =`/`fix =` for the rest)",
            ));
        } else {
            positional_domain = Some(seg);
        }
    }

    let domain = match (positional_domain, kw_domain) {
        (Some(_), Some(d)) => {
            return Err(syn::Error::new(d.ident.span(), "domain specified twice"));
        }
        (Some(d), None) => Some(d),
        (None, Some(d)) => Some(d.value),
        (None, None) => None,
    };
    Ok(Trailing { domain, lb, ub, initial, fix })
}

/// Recognize a trailing `kw = value` keyword segment, returning the keyword ident
/// and the value tokens. A segment is a keyword iff it starts with an identifier
/// followed by a lone `=` (so `==`/`<=`/`>=` are not mistaken for it). Whether
/// `kw` is a known keyword is decided by the caller's slot match, whose fallback
/// reports invalid keywords. Anything that is not an `ident = value` pair (a bare
/// domain token, `SemiCont(thr)`) returns `None`.
fn parse_keyword(seg: &TokenStream2) -> Option<(proc_macro2::Ident, TokenStream2)> {
    let tts: Vec<TokenTree> = seg.clone().into_iter().collect();
    let TokenTree::Ident(kw) = tts.first()? else {
        return None;
    };
    match tts.get(1)? {
        TokenTree::Punct(p) if p.as_char() == '=' && p.spacing() == Spacing::Alone => {}
        _ => return None,
    }
    Some((kw.clone(), tts[2..].iter().cloned().collect()))
}

/// Merge a bound coming from the relational spec with one given as a keyword.
/// Specifying the same bound twice is an error.
fn merge_bound(
    rel: Option<TokenStream2>,
    kw: Option<Kw>,
    which: &str,
) -> syn::Result<Option<TokenStream2>> {
    match (rel, kw) {
        (Some(_), Some(kw)) => {
            Err(syn::Error::new(kw.ident.span(), format!("`{which}` specified twice")))
        }
        (Some(b), None) => Ok(Some(b)),
        (None, Some(kw)) => Ok(Some(kw.value)),
        (None, None) => Ok(None),
    }
}

/// Map the trailing domain token to the builder method. Accepts the bare-ident
/// forms (`Bin`/`Int`/`Real` and aliases) and the call forms `SemiCont(thr)`/
/// `SemiContinuous(thr)`/`SemiInt(thr)`/`SemiInteger(thr)`, where `thr` is the
/// semicontinuous threshold (`f64`).
fn domain_method(ts: TokenStream2, root: &TokenStream2) -> syn::Result<TokenStream2> {
    const HELP: &str = "domain must be `Bin`, `Int`, `Real`, `SemiCont(thr)`, or `SemiInt(thr)`";
    match syn::parse2::<syn::Expr>(ts)? {
        syn::Expr::Path(p) => {
            let id = p.path.get_ident().ok_or_else(|| syn::Error::new_spanned(&p, HELP))?;
            match id.to_string().as_str() {
                "Bin" | "Binary" => Ok(quote!(.binary())),
                "Int" | "Integer" => Ok(quote!(.integer())),
                "Real" | "Cont" | "Continuous" => Ok(quote!()),
                "SemiCont" | "SemiContinuous" | "SemiInt" | "SemiInteger" => {
                    Err(syn::Error::new_spanned(
                        id,
                        format!("`{id}` needs a threshold, e.g. `{id}(1.0)`"),
                    ))
                }
                _ => Err(syn::Error::new_spanned(id, HELP)),
            }
        }
        syn::Expr::Call(call) => {
            let syn::Expr::Path(fp) = &*call.func else {
                return Err(syn::Error::new_spanned(&call.func, HELP));
            };
            let func = fp.path.get_ident().ok_or_else(|| syn::Error::new_spanned(fp, HELP))?;
            let variant = match func.to_string().as_str() {
                "SemiCont" | "SemiContinuous" => quote!(SemiContinuous),
                "SemiInt" | "SemiInteger" => quote!(SemiInteger),
                _ => return Err(syn::Error::new_spanned(func, HELP)),
            };
            let [thr] = call.args.iter().collect::<Vec<_>>()[..] else {
                return Err(syn::Error::new_spanned(
                    &call,
                    format!("`{func}` takes exactly one threshold argument, e.g. `{func}(1.0)`"),
                ));
            };
            Ok(quote! {
                .domain(#root::__macro_support::Domain::#variant { threshold: f64::from(#thr) })
            })
        }
        other => Err(syn::Error::new_spanned(other, HELP)),
    }
}

#[derive(Copy, Clone)]
enum BoundKind {
    Lb,
    Ub,
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
