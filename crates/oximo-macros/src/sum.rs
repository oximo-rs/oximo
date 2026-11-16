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
    unfused: bool,
    body: Expr,
    binds: Vec<IndexBind>,
    cond: Option<Expr>,
}

impl Parse for SumInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (mut unfused, first) = parse_body(input)?;
        let (model, body) = if !unfused && input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let parsed = parse_body(input)?;
            unfused = parsed.0;
            (Some(first), parsed.1)
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
        Ok(Self { model, unfused, body, binds: binds.into_iter().collect(), cond })
    }
}

fn parse_body(input: ParseStream) -> syn::Result<(bool, Expr)> {
    let fork = input.fork();
    let unfused = fork.parse::<Ident>().is_ok_and(|ident| ident == "unfused")
        && fork.peek(Token![:])
        && !fork.peek(Token![::]);
    if unfused {
        input.parse::<Ident>()?;
        input.parse::<Token![:]>()?;
    }
    Ok((unfused, input.parse()?))
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
    unfused: bool,
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
    // Runtime model context must not be projected into constant expressions.
    fn visit_expr_repeat_mut(&mut self, expression: &mut syn::ExprRepeat) {
        self.visit_attributes_mut(&mut expression.attrs);
        self.visit_expr_mut(&mut expression.expr);
    }

    fn visit_expr_const_mut(&mut self, _expression: &mut syn::ExprConst) {}

    fn visit_type_mut(&mut self, _ty: &mut syn::Type) {}

    fn visit_generic_argument_mut(&mut self, argument: &mut syn::GenericArgument) {
        match argument {
            syn::GenericArgument::Const(_) | syn::GenericArgument::AssocConst(_) => {}
            _ => visit_mut::visit_generic_argument_mut(self, argument),
        }
    }

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
    let SumInput { model, unfused, body, mut binds, cond } = syn::parse2(input)?;
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
    let previous_unfused = captures.unfused;
    captures.unfused |= unfused;
    let unfused = captures.unfused;
    for bind in &mut binds {
        if captures.unfused {
            bind.domain = rewrite(bind.domain.clone(), None, captures)?;
        }
        captures.block_pattern(&bind.pat);
    }
    let body = rewrite(body, context.as_ref(), captures)?;
    let body = if captures.unfused { quote!(#body) } else { fuse_product(&body, &root) };
    let cond = cond.map(|cond| rewrite(cond, context.as_ref(), captures)).transpose()?;
    captures.unfused = previous_unfused;
    captures.blocked.truncate(scope);

    if cond.is_none()
        && (unfused
            || binds.len() == 1
            || binds.iter().skip(1).any(|bind| has_control_flow(&bind.domain)))
    {
        let mut expr = body;
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
    }

    Ok(expand_loops(anchor, body, &binds, cond, context.as_ref(), &root))
}

fn expand_loops(
    anchor: TokenStream2,
    body: TokenStream2,
    binds: &[IndexBind],
    cond: Option<Expr>,
    context: Option<&Context>,
    root: &TokenStream2,
) -> TokenStream2 {
    let terms = Ident::new("__terms", Span::mixed_site());
    let filtered = cond.is_some();
    // Only literal inner ranges have an iteration count independent of outer
    // bindings and body side effects.
    let inner_count = if filtered {
        None
    } else {
        binds
            .iter()
            .skip(1)
            .try_fold(1_usize, |count, bind| count.checked_mul(literal_range_len(&bind.domain)?))
    };
    let mut inner = if let Some(cond) = cond {
        quote! { if #cond { #terms.push(#body); } }
    } else {
        TokenStream2::new()
    };
    for (index, b) in binds.iter().rev().enumerate() {
        let pat = &b.pat;
        let used = crate::bind::mark_bindings_used(std::slice::from_ref(b));
        let domain = &b.domain;
        let keys = if let Some(ty) = b.keys_of_type() {
            quote!(#root::__macro_support::keys_of::<#ty, _>(&(#domain)))
        } else {
            quote!(#root::__macro_support::keys_of(&(#domain)))
        };
        let count = (index + 1 == binds.len()).then_some(inner_count).flatten();
        let iter = Ident::new("__oximo_sum_keys", Span::mixed_site());
        let loop_keys = if count.is_some() { quote!(#iter) } else { keys.clone() };
        let loop_body = if !filtered && index == 0 {
            let key = Ident::new("__oximo_sum_key", Span::mixed_site());
            let param = b.closure_param();

            // Match the iterator callback's ownership.
            quote! {
                for #key in #loop_keys {
                    #terms.push(#root::__macro_support::evaluate_sum_callback(
                        #key, |#param| { #used #body }
                    ));
                }
            }
        } else {
            quote! {
                for #pat in #loop_keys {
                    #used
                    #inner
                }
            }
        };
        inner = if let Some(count) = count {
            let count = count as u128;
            let length = Ident::new("__oximo_sum_count", Span::mixed_site());
            quote! {
                match #keys {
                    #iter => {
                        #terms.reserve(
                            <::core::primitive::usize as ::core::convert::TryFrom<
                                ::core::primitive::u128
                            >>::try_from(#count).ok().and_then(|#length| {
                                ::core::iter::Iterator::size_hint(&#iter).0.checked_mul(#length)
                            }).unwrap_or(0)
                        );
                        #loop_body
                    }
                }
            }
        } else {
            loop_body
        };
    }
    let arena = if let Some(context) = &context {
        let arena = &context.arena;
        quote!(::core::option::Option::Some(#arena))
    } else {
        quote!(::core::option::Option::None)
    };
    let empty_message = if filtered {
        "sum! with an `if` filter produced no terms; use sum!(model, ...)"
    } else {
        "sum_over on empty domain"
    };
    quote! {{
        #anchor
        let mut #terms = #root::__macro_support::SumCollector::new(#arena);
        #inner
        #terms.finish().expect(#empty_message)
    }}
}

fn literal_range_len(domain: &Expr) -> Option<usize> {
    let range = match domain {
        Expr::Range(range) if range.attrs.is_empty() => range,
        Expr::Paren(paren) if paren.attrs.is_empty() => return literal_range_len(&paren.expr),
        _ => return None,
    };
    let integer = |bound: &Option<Box<Expr>>| {
        let Expr::Lit(literal) = bound.as_deref()? else { return None };
        let syn::Lit::Int(integer) = &literal.lit else { return None };
        if !literal.attrs.is_empty() {
            return None;
        }
        integer.base10_parse::<usize>().ok()
    };
    let start = integer(&range.start)?;
    let end = integer(&range.end)?;
    match range.limits {
        syn::RangeLimits::HalfOpen(_) => Some(end.saturating_sub(start)),
        syn::RangeLimits::Closed(_) if start <= end => end.checked_sub(start)?.checked_add(1),
        syn::RangeLimits::Closed(_) => Some(0),
    }
}

fn has_control_flow(body: &Expr) -> bool {
    struct ControlFlow(bool);
    impl VisitMut for ControlFlow {
        fn visit_expr_mut(&mut self, expression: &mut Expr) {
            if matches!(
                expression,
                Expr::Return(_) | Expr::Try(_) | Expr::Yield(_) | Expr::Macro(_)
            ) {
                self.0 = true;
            } else {
                visit_mut::visit_expr_mut(self, expression);
            }
        }
    }
    let mut control_flow = ControlFlow(false);
    control_flow.visit_expr_mut(&mut body.clone());
    control_flow.0
}

fn fuse_product(body: &Expr, root: &TokenStream2) -> TokenStream2 {
    if has_control_flow(body) {
        return quote!(#body);
    }
    let product = match body {
        Expr::Paren(expr) if expr.attrs.is_empty() => return fuse_product(&expr.expr, root),
        Expr::Group(expr) if expr.attrs.is_empty() => return fuse_product(&expr.expr, root),
        Expr::Binary(expr) if matches!(expr.op, syn::BinOp::Mul(_)) && expr.attrs.is_empty() => {
            expr
        }
        _ => return quote!(#body),
    };
    let left = &product.left;
    let right = &product.right;
    let operands = Ident::new("__oximo_product_operands", Span::mixed_site());
    let tag = Ident::new("__oximo_product_tag", Span::mixed_site());
    quote! {{
        use #root::__macro_support::SumProduct as _;
        match (#left, #right) {
            #operands => {
                let #tag = #root::__macro_support::product_tag(&#operands);
                (&&#tag).sum_product(#operands)
            }
        }
    }}
}
