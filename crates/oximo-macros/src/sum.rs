//! Indexed sums with an explicit or enclosing model's expression context.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, Ident, Token};

use crate::{IndexBind, oximo_root};

struct SumInput {
    model: Option<Expr>,
    body: Expr,
    binds: Vec<IndexBind>,
    cond: Option<Expr>,
}

impl Parse for SumInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let first = input.parse::<Expr>()?;
        let (model, body) = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            (Some(first), input.parse::<Expr>()?)
        } else {
            (None, first)
        };
        input.parse::<Token![for]>()?;
        let binds = Punctuated::<IndexBind, Token![,]>::parse_separated_nonempty(input)?;
        let cond = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(input.parse::<Expr>()?)
        } else {
            None
        };
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after `sum!` clauses"));
        }
        Ok(Self { model, body, binds: binds.into_iter().collect(), cond })
    }
}

#[derive(Clone)]
struct Context {
    model: Expr,
    arena: Ident,
}

/// Bind a model once and project its thread-safe expression context before
/// constructing an indexed callback. The registries never enter the callback.
pub(crate) struct ModelSums {
    context: Context,
    receiver: Ident,
    captures: Captures,
}

impl ModelSums {
    pub(crate) fn new(model: Expr) -> Self {
        Self {
            context: Context {
                model,
                arena: Ident::new("__oximo_model_arena", Span::mixed_site()),
            },
            receiver: Ident::new("__oximo_model_receiver", Span::mixed_site()),
            captures: Captures::default(),
        }
    }

    pub(crate) fn receiver(&self) -> Expr {
        let receiver = &self.receiver;
        syn::parse_quote!(#receiver)
    }

    pub(crate) fn indexed(&mut self, binds: &[IndexBind]) {
        self.captures.hoist = true;
        for bind in binds {
            self.captures.block_pattern(&bind.pat);
        }
    }

    pub(crate) fn rewrite(&mut self, expr: Expr) -> syn::Result<Expr> {
        rewrite(expr, Some(&self.context), &mut self.captures)
    }

    pub(crate) fn wrap(&self, body: TokenStream2) -> TokenStream2 {
        let model = &self.context.model;
        let arena = &self.context.arena;
        let receiver = &self.receiver;
        let captures = self
            .captures
            .bindings
            .iter()
            .map(|(name, model)| quote!(let #name = (#model).__sum_context();));
        quote! {{
            let #receiver = &(#model);
            let #arena = #receiver.__sum_context();
            #(#captures)*
            #body
        }}
    }
}

#[derive(Default)]
struct Captures {
    hoist: bool,
    bindings: Vec<(Ident, Expr)>,
    blocked: Vec<String>,
}

impl Captures {
    fn block_pattern(&mut self, pat: &syn::Pat) {
        struct Names<'a>(&'a mut Vec<String>);
        impl VisitMut for Names<'_> {
            fn visit_pat_ident_mut(&mut self, pat: &mut syn::PatIdent) {
                self.0.push(pat.ident.to_string());
                visit_mut::visit_pat_ident_mut(self, pat);
            }
        }
        Names(&mut self.blocked).visit_pat_mut(&mut pat.clone());
    }

    fn stable_key(&self, model: &Expr) -> Option<String> {
        let key = stable_model(model)?;
        let root = key.split(['.', ':', ' ']).next()?;
        (!self.blocked.iter().any(|name| name == root)).then_some(key)
    }
}

// Parentheses and shared borrows do not change the model named by an anchor.
fn stable_model(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Paren(e) => stable_model(&e.expr),
        Expr::Group(e) => stable_model(&e.expr),
        Expr::Reference(e) if e.mutability.is_none() => stable_model(&e.expr),
        Expr::Path(_) => Some(expr.to_token_stream().to_string()),
        Expr::Field(e) => {
            Some(format!("{}.{}", stable_model(&e.base)?, e.member.to_token_stream()))
        }
        _ => None,
    }
}

/// Visit Rust expressions, expanding nested oximo sums with the same context.
struct Rewriter<'a> {
    context: Option<&'a Context>,
    error: Option<syn::Error>,
    captures: &'a mut Captures,
}

impl VisitMut for Rewriter<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if self.error.is_some() {
            return;
        }
        let scope = self.captures.blocked.len();
        if let Expr::Macro(call) = expr
            && is_sum_path(&call.mac.path)
        {
            match expand_in(call.mac.tokens.clone(), self.context, self.captures)
                .and_then(syn::parse2)
            {
                Ok(expanded) => *expr = expanded,
                Err(error) => self.error = Some(error),
            }
        } else {
            visit_mut::visit_expr_mut(self, expr);
        }
        self.captures.blocked.truncate(scope);
    }

    fn visit_expr_match_mut(&mut self, expr: &mut syn::ExprMatch) {
        self.visit_attributes_mut(&mut expr.attrs);
        self.visit_expr_mut(&mut expr.expr);
        for arm in &mut expr.arms {
            let arm_scope = self.captures.blocked.len();
            self.visit_attributes_mut(&mut arm.attrs);
            self.visit_pat_mut(&mut arm.pat);
            self.visit_expr_mut(&mut arm.body);
            self.captures.blocked.truncate(arm_scope);
        }
    }

    fn visit_expr_let_mut(&mut self, expr: &mut syn::ExprLet) {
        self.visit_attributes_mut(&mut expr.attrs);
        self.visit_expr_mut(&mut expr.expr);
        self.visit_pat_mut(&mut expr.pat);
    }

    fn visit_expr_for_loop_mut(&mut self, expr: &mut syn::ExprForLoop) {
        self.visit_attributes_mut(&mut expr.attrs);
        if let Some(label) = &mut expr.label {
            self.visit_label_mut(label);
        }
        self.visit_expr_mut(&mut expr.expr);
        self.visit_pat_mut(&mut expr.pat);
        self.visit_block_mut(&mut expr.body);
    }

    fn visit_pat_mut(&mut self, pat: &mut syn::Pat) {
        self.captures.block_pattern(pat);
    }

    fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
        if let syn::Stmt::Macro(call) = stmt
            && is_sum_path(&call.mac.path)
        {
            let semi = call.semi_token;
            let mut expr =
                Expr::Macro(syn::ExprMacro { attrs: call.attrs.clone(), mac: call.mac.clone() });
            self.visit_expr_mut(&mut expr);
            *stmt = syn::Stmt::Expr(expr, semi);
        } else {
            visit_mut::visit_stmt_mut(self, stmt);
        }
    }

    fn visit_local_mut(&mut self, local: &mut syn::Local) {
        self.visit_attributes_mut(&mut local.attrs);
        if let Some(init) = &mut local.init {
            self.visit_expr_mut(&mut init.expr);
            if let Some((_, diverge)) = &mut init.diverge {
                self.visit_expr_mut(diverge);
            }
        }
        self.visit_pat_mut(&mut local.pat);
    }

    fn visit_item_mut(&mut self, _item: &mut syn::Item) {
        // A nested function/item cannot capture the surrounding arena.
    }
}

fn rewrite(
    mut expr: Expr,
    context: Option<&Context>,
    captures: &mut Captures,
) -> syn::Result<Expr> {
    let mut rewriter = Rewriter { context, error: None, captures };
    rewriter.visit_expr_mut(&mut expr);
    match rewriter.error {
        Some(error) => Err(error),
        None => Ok(expr),
    }
}

fn is_sum_path(path: &syn::Path) -> bool {
    if path.is_ident("sum") {
        return true;
    }
    let root: syn::Path = syn::parse2(oximo_root()).expect("Oximo root is a path");
    path.segments.len() == 2
        && path.segments.last().is_some_and(|part| part.ident == "sum")
        && path.segments.first().is_some_and(|part| {
            part.ident == "oximo"
                || part.ident == "oximo_core"
                || part.ident == root.segments[0].ident
        })
}

pub(crate) fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    expand_in(input, None, &mut Captures::default())
}

fn expand_in(
    input: TokenStream2,
    inherited: Option<&Context>,
    captures: &mut Captures,
) -> syn::Result<TokenStream2> {
    let input = crate::index::rewrite_index_subscripts(input)?;
    let SumInput { model, body, binds, cond } = syn::parse2(input)?;
    let root = oximo_root();
    let mut anchor = TokenStream2::new();
    let context = match model {
        Some(model) => {
            if let Some(parent) = inherited
                && let Some(key) = captures.stable_key(&model)
                && stable_model(&parent.model).as_ref() == Some(&key)
            {
                Some(parent.clone())
            } else if captures.hoist && captures.stable_key(&model).is_some() {
                let arena = Ident::new(
                    &format!("__oximo_sum_capture_{}", captures.bindings.len()),
                    Span::mixed_site(),
                );
                captures.bindings.push((arena.clone(), model.clone()));
                Some(Context { model, arena })
            } else {
                let arena = Ident::new("__oximo_sum_model", Span::mixed_site());
                anchor = quote!(let #arena = (#model).__sum_context(););
                Some(Context { model, arena })
            }
        }
        None => inherited.cloned(),
    };
    let scope = captures.blocked.len();
    for bind in &binds {
        captures.block_pattern(&bind.pat);
    }
    let body = rewrite(body, context.as_ref(), captures)?;
    let cond = cond.map(|cond| rewrite(cond, context.as_ref(), captures)).transpose()?;
    captures.blocked.truncate(scope);

    let Some(cond) = cond else {
        let mut expr = quote!(#body);
        for b in binds.iter().rev() {
            let param = b.closure_param();
            let used = crate::bind::mark_bindings_used(std::slice::from_ref(b));
            let domain = &b.domain;
            expr = if let Some(context) = &context {
                let arena = &context.arena;
                quote!(#root::__macro_support::sum_over_in(
                    #arena, &(#domain), |#param| { #used #expr },
                ))
            } else {
                quote!(#root::__macro_support::sum_over(&(#domain), |#param| { #used #expr }))
            };
        }
        return Ok(quote! {{ #anchor #expr }});
    };

    let terms = Ident::new("__terms", Span::mixed_site());
    let mut inner = quote! {
        if #cond {
            #terms.push(#body);
        }
    };
    for b in binds.iter().rev() {
        let pat = &b.pat;
        let used = crate::bind::mark_bindings_used(std::slice::from_ref(b));
        let domain = &b.domain;
        let keys = if let Some(ty) = b.keys_of_type() {
            quote!(#root::__macro_support::keys_of::<#ty, _>(&(#domain)))
        } else {
            quote!(#root::__macro_support::keys_of(&(#domain)))
        };
        inner = quote! {
            for #pat in #keys {
                #used
                #inner
            }
        };
    }
    let result = if let Some(context) = &context {
        let arena = &context.arena;
        quote!(#root::__macro_support::sum_terms_in(#arena, #terms))
    } else {
        quote! {
            ::core::assert!(
                !#terms.is_empty(),
                "sum! with an `if` filter produced no terms; use sum!(model, ...)"
            );
            #root::__macro_support::sum_terms(#terms)
        }
    };
    Ok(quote! {{
        #anchor
        let mut #terms = ::std::vec::Vec::new();
        #inner
        #result
    }})
}
