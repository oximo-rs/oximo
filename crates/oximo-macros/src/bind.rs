//! Parsing for index bindings (`pat in domain`, optionally `pat: ty in domain`)
//! shared by `variable!`, `constraint!`, and `sum!`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
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

/// A comma-separated, optionally trailing list of [`IndexBind`]s
pub(crate) struct Binds(pub(crate) Vec<IndexBind>);

impl Parse for Binds {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let parsed = Punctuated::<IndexBind, Token![,]>::parse_terminated(input)?;
        Ok(Self(parsed.into_iter().collect()))
    }
}

impl IndexBind {
    /// The key type for this binding: the explicit annotation, else `usize`.
    pub(crate) fn key_type(&self) -> TokenStream2 {
        if let Some(ty) = &self.ty { quote!(#ty) } else { quote!(usize) }
    }

    /// A typed closure parameter for a single-index closure (`|pat: ty|`).
    pub(crate) fn closure_param(&self) -> TokenStream2 {
        let pat = &self.pat;
        let ty = self.key_type();
        quote!(#pat: #ty)
    }
}

/// Build the closure parameter for an index family decoded as a whole key:
/// single binding -> `pat: ty`; multiple -> `(p0, p1, ...): (t0, t1, ...)`.
pub(crate) fn family_closure_param(binds: &[IndexBind]) -> TokenStream2 {
    if let [single] = binds {
        return single.closure_param();
    }
    let pats = binds.iter().map(|b| &b.pat);
    let tys = binds.iter().map(IndexBind::key_type);
    quote!( (#(#pats),*): (#(#tys),*) )
}
