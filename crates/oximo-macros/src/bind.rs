//! Parsing for index bindings (`pat in domain`, optionally `pat: ty in domain`)
//! shared by `variable!`, `constraint!`, and `sum!`.

use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Pat, Token, Type};

/// One `pat in domain` clause, with an optional explicit key type.
///
/// The type annotation is only needed when the domain is a `Set` whose keys are
/// not plain integers (strings, tuples).
/// The macros decode keys through `FromIndexKey`,
/// and a bare identifier defaults to `usize`.
pub(crate) struct IndexBind {
    pub(crate) pat: Pat,
    pub(crate) ty: Option<Type>,
    pub(crate) domain: Expr,
}

impl Parse for IndexBind {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let pat = Pat::parse_single(input)?;
        let ty = if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
            Some(input.parse::<Type>()?)
        } else {
            None
        };
        input.parse::<Token![in]>()?;
        let domain = input.parse::<Expr>()?;
        Ok(Self { pat, ty, domain })
    }
}

/// A comma-separated, optionally trailing list of [`IndexBind`]s, with an
/// optional trailing `if cond` filter (`name[i in S if cond]`).
pub(crate) struct Binds {
    pub(crate) binds: Vec<IndexBind>,
    pub(crate) cond: Option<Expr>,
}

impl Parse for Binds {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut binds = Vec::new();
        while !input.is_empty() && !input.peek(Token![if]) {
            binds.push(input.parse::<IndexBind>()?);
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        let cond = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(input.parse::<Expr>()?)
        } else {
            None
        };
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after index bindings"));
        }
        Ok(Self { binds, cond })
    }
}

impl IndexBind {
    /// Whether the domain is written as a range expression.
    pub(crate) fn is_range_literal(&self) -> bool {
        matches!(self.domain, Expr::Range(_))
    }

    /// Closure parameter for a single-index `sum!` term. The domain is consumed
    /// directly via `SumDomain`.
    pub(crate) fn closure_param(&self) -> TokenStream2 {
        let pat = &self.pat;
        if let Some(ty) = &self.ty {
            quote!(#pat: #ty)
        } else if self.is_range_literal() {
            quote!(#pat: usize)
        } else {
            quote!(#pat)
        }
    }

    /// Key type to pin when iterating this binding via `keys_of::<K, _>`.
    pub(crate) fn keys_of_type(&self) -> Option<TokenStream2> {
        if let Some(ty) = &self.ty {
            Some(quote!(#ty))
        } else if self.is_range_literal() {
            Some(quote!(usize))
        } else {
            None
        }
    }
}

/// Wrap a built `Set` token expression in `filter_keys(.., |pat| cond)` when the
/// family carries an `if` filter (`name[i in dom if cond]`), otherwise return the
/// set unchanged.
pub(crate) fn filtered_set(
    set: TokenStream2,
    binds: &[IndexBind],
    cond: Option<&Expr>,
    root: &TokenStream2,
) -> TokenStream2 {
    match cond {
        None => set,
        Some(cond) => {
            let param = family_closure_param(binds);
            let used = mark_bindings_used(binds);
            quote!( #root::__macro_support::filter_keys_owned((#set), |#param| { #used #cond }) )
        }
    }
}

/// Closure parameter for an index family decoded as a whole key.
/// The `Set<K>` passed to `__add_constraints_over` pins `K`, so the
/// pattern is left bare unless the user annotated every binding.
pub(crate) fn family_closure_param(binds: &[IndexBind]) -> TokenStream2 {
    let pats = binds.iter().map(|b| &b.pat);
    let pattern = if let [single] = binds {
        let pat = &single.pat;
        quote!(#pat)
    } else {
        quote!( (#(#pats),*) )
    };

    let tys: Option<Vec<&Type>> = binds.iter().map(|b| b.ty.as_ref()).collect();
    match tys {
        Some(tys) if binds.len() == 1 => {
            let ty = tys[0];
            quote!(#pattern: #ty)
        }
        Some(tys) => quote!( #pattern: (#(#tys),*) ),
        None => pattern,
    }
}

/// Statements (`let _ = &ident;` per bound name) that mark every index binding
/// as used inside a generated closure or loop. A family expands to two closures over
/// the same names (filter + rule), each of which may ignore a subset
/// (guard-only keys).
pub(crate) fn mark_bindings_used(binds: &[IndexBind]) -> TokenStream2 {
    let mut idents = Vec::new();
    for b in binds {
        collect_pat_idents(&b.pat, &mut idents);
    }
    let stmts = idents.iter().map(|id| quote!(let _ = &#id;));
    quote!(#(#stmts)*)
}

/// Collect the identifiers actually bound by a pattern.
fn collect_pat_idents(pat: &Pat, out: &mut Vec<syn::Ident>) {
    match pat {
        Pat::Ident(pi) => {
            let name = pi.ident.to_string();
            if name != "_" && !name.starts_with('_') {
                out.push(pi.ident.clone());
            }
            if let Some((_, sub)) = &pi.subpat {
                collect_pat_idents(sub, out);
            }
        }
        Pat::Tuple(t) => t.elems.iter().for_each(|e| collect_pat_idents(e, out)),
        Pat::TupleStruct(ts) => ts.elems.iter().for_each(|e| collect_pat_idents(e, out)),
        Pat::Struct(s) => s.fields.iter().for_each(|f| collect_pat_idents(&f.pat, out)),
        Pat::Slice(s) => s.elems.iter().for_each(|e| collect_pat_idents(e, out)),
        Pat::Reference(r) => collect_pat_idents(&r.pat, out),
        Pat::Paren(p) => collect_pat_idents(&p.pat, out),
        Pat::Type(t) => collect_pat_idents(&t.pat, out),
        Pat::Or(o) => o.cases.iter().for_each(|c| collect_pat_idents(c, out)),
        _ => {}
    }
}

/// Whether a token stream references any of the given identifiers.
pub(crate) fn references_any(ts: &TokenStream2, idents: &[String]) -> bool {
    ts.clone().into_iter().any(|tt| match tt {
        TokenTree::Ident(id) => idents.contains(&id.to_string()),
        TokenTree::Group(g) => references_any(&g.stream(), idents),
        _ => false,
    })
}
