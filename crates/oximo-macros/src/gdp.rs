//! GDP declarations macros.

use crate::bind::{IndexBind, family_closure_param, filtered_set, mark_bindings_used};
use crate::{Named, build_set, oximo_root, parse_named, split_top_commas};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Expr;

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Boolean,
    Disjunct,
    Disjunction,
    Logic,
}

enum DeclarationName {
    Anonymous,
    Named(Box<Named>),
    Computed(Box<Expr>),
}

impl DeclarationName {
    fn parse(tokens: TokenStream) -> syn::Result<Self> {
        if let Some(expression) = crate::constraint::computed_name(&tokens) {
            Ok(Self::Computed(Box::new(syn::parse2(expression)?)))
        } else {
            parse_named(tokens).map(|named| Self::Named(Box::new(named)))
        }
    }

    fn bindings(&self) -> Option<&[IndexBind]> {
        match self {
            Self::Named(named) => named.binds.as_deref(),
            _ => None,
        }
    }

    fn tokens(&self, receiver: &Expr) -> TokenStream {
        match self {
            Self::Anonymous => quote!(#receiver.__gdp_auto_name("gdp")),
            Self::Named(named) => {
                let name = &named.name;
                quote!(stringify!(#name))
            }
            Self::Computed(expression) => quote!(#expression),
        }
    }

    /// The same registration path handles scalar declarations and family members.
    fn expand(
        &self,
        receiver: &Expr,
        root: &TokenStream,
        operation: impl Fn(Option<TokenStream>) -> TokenStream,
    ) -> syn::Result<TokenStream> {
        if let Self::Named(named) = self
            && let Some(binds) = &named.binds
        {
            let name = self.tokens(receiver);
            let set = build_set(binds, root)?;
            let set = filtered_set(set, binds, named.cond.as_ref(), root);
            let param = family_closure_param(binds);
            let used = mark_bindings_used(binds);
            let op = operation(Some(quote!(__oximo_gdp_name)));

            Ok(
                quote!(#receiver.__gdp_family_over(#name, &(#set), |#param, __oximo_gdp_name| { #used #op })),
            )
        } else {
            let name = (!matches!(self, Self::Anonymous)).then(|| self.tokens(receiver));

            Ok(operation(name))
        }
    }
}

fn parse_expression(tokens: TokenStream) -> syn::Result<Expr> {
    syn::parse2(crate::index::rewrite_index_subscripts(tokens))
}

struct ConditionalDeclaration {
    receiver: Expr,
    indicator: Expr,
    name: DeclarationName,
    relation: TokenStream,
}

impl ConditionalDeclaration {
    fn parse(input: TokenStream) -> syn::Result<Self> {
        let mut parts = split_top_commas(input).into_iter();
        let receiver = parse_expression(
            parts.next().ok_or_else(|| syn::Error::new(Span::call_site(), "expected a model"))?,
        )?;
        let indicator =
            parse_expression(parts.next().ok_or_else(|| {
                syn::Error::new(Span::call_site(), "expected a Boolean indicator")
            })?)?;
        let first = parts
            .next()
            .ok_or_else(|| syn::Error::new(Span::call_site(), "expected a conditional relation"))?;
        let (name, relation) = match parts.next() {
            Some(relation) => (DeclarationName::parse(first)?, relation),
            None => (DeclarationName::Anonymous, first),
        };

        if let Some(extra) = parts.next() {
            return Err(syn::Error::new_spanned(
                extra,
                "unexpected trailing GDP declaration tokens",
            ));
        }

        Ok(Self { receiver, indicator, name, relation })
    }
}

pub(crate) fn conditional(input: TokenStream) -> syn::Result<TokenStream> {
    let ConditionalDeclaration { receiver, indicator, name, relation } =
        ConditionalDeclaration::parse(input)?;
    let root = oximo_root();
    let mut sums = crate::sum::ModelSums::new(receiver);

    if let Some(binds) = name.bindings() {
        sums.indexed(binds);
    }

    let receiver = sums.receiver();
    let indicator = sums.rewrite(indicator)?;
    let relation = crate::constraint::build_relations(relation, &root, &mut sums)?;
    let expansion = name.expand(&receiver, &root, |name| {
        let row = match (&relation, name) {
            (crate::constraint::Relations::Single(relation), Some(name)) => {
                quote!(__oximo_context.add_constraint(#name, #relation))
            }
            (crate::constraint::Relations::Single(relation), None) => {
                quote!(__oximo_context.__add_constraint_auto(#relation))
            }
            (crate::constraint::Relations::Range { mid, lo, hi }, Some(name)) => {
                quote!(__oximo_context.__add_range(&(#name), #mid, #lo, #hi))
            }
            (crate::constraint::Relations::Range { mid, lo, hi }, None) => {
                quote!(__oximo_context.__add_range_auto(#mid, #lo, #hi))
            }
        };
        quote!({ let __oximo_context = #receiver.__gdp_context(#indicator); #row })
    })?;

    Ok(sums.wrap(expansion))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Selector {
    ExactlyOne,
    AtLeastOne,
}

impl Selector {
    fn tokens(self, root: &TokenStream) -> TokenStream {
        match self {
            Self::ExactlyOne => quote!(#root::__macro_support::DisjunctionKind::ExactlyOne),
            Self::AtLeastOne => quote!(#root::__macro_support::DisjunctionKind::AtLeastOne),
        }
    }
}

enum DisjunctionOption {
    Selector(Selector),
    Parent(Box<Expr>),
}

impl DisjunctionOption {
    fn from_expression(expression: Expr) -> syn::Result<Self> {
        match expression {
            Expr::Path(path) if path.path.is_ident("ExactlyOne") => {
                Ok(Self::Selector(Selector::ExactlyOne))
            }
            Expr::Path(path) if path.path.is_ident("AtLeastOne") => {
                Ok(Self::Selector(Selector::AtLeastOne))
            }
            Expr::Assign(assign) if matches!(&*assign.left, Expr::Path(path) if path.path.is_ident("parent")) => {
                Ok(Self::Parent(assign.right))
            }
            other => Err(syn::Error::new_spanned(
                other,
                "expected ExactlyOne, AtLeastOne or parent = indicator",
            )),
        }
    }

    fn starts_option(tokens: &TokenStream) -> bool {
        parse_expression(tokens.clone())
            .is_ok_and(|expression| Self::from_expression(expression).is_ok())
    }
}

struct DeclarationOptions {
    selector: Selector,
    parent: Option<Expr>,
}

impl DeclarationOptions {
    fn parse(options: Vec<TokenStream>, kind: Kind) -> syn::Result<Self> {
        let mut selector = None;
        let mut parent = None;

        for tokens in options {
            if !matches!(kind, Kind::Disjunction) {
                return Err(syn::Error::new_spanned(tokens, "unexpected GDP option"));
            }

            match DisjunctionOption::from_expression(parse_expression(tokens.clone())?)? {
                DisjunctionOption::Selector(value) => {
                    if let Some(previous) = selector {
                        let message = if previous == value {
                            "duplicate disjunction selector"
                        } else {
                            "conflicting disjunction selectors"
                        };

                        return Err(syn::Error::new_spanned(tokens, message));
                    }
                    selector = Some(value);
                }
                DisjunctionOption::Parent(expression) => {
                    if parent.is_some() {
                        return Err(syn::Error::new_spanned(tokens, "duplicate parent option"));
                    }
                    parent = Some(*expression);
                }
            }
        }

        Ok(Self { selector: selector.unwrap_or(Selector::ExactlyOne), parent })
    }
}

struct Declaration {
    receiver: Expr,
    name: DeclarationName,
    body: Option<Expr>,
    options: DeclarationOptions,
}

impl Declaration {
    fn parse(input: TokenStream, kind: Kind) -> syn::Result<Self> {
        let mut parts = split_top_commas(input).into_iter();
        let receiver = parse_expression(parts.next().ok_or_else(|| {
            syn::Error::new(Span::call_site(), "expected a model or disjunct context")
        })?)?;
        let first = parts
            .next()
            .ok_or_else(|| syn::Error::new(Span::call_site(), "expected a GDP declaration"))?;
        let mut second = parts.next();
        let mut options: Vec<_> = parts.collect();

        if matches!(kind, Kind::Disjunction)
            && (second.as_ref().is_some_and(DisjunctionOption::starts_option)
                || DeclarationName::parse(first.clone()).is_err())
            && let Some(option) = second.take()
        {
            options.insert(0, option);
        }

        let options = DeclarationOptions::parse(options, kind)?;
        let (name, body) = if matches!(kind, Kind::Boolean) {
            if second.is_some() {
                return Err(syn::Error::new_spanned(
                    first,
                    "boolean_variable! expects only a name or indexed name",
                ));
            }
            (DeclarationName::Named(Box::new(parse_named(first)?)), None)
        } else if let Some(body) = second {
            (DeclarationName::parse(first)?, Some(parse_expression(body)?))
        } else {
            (DeclarationName::Anonymous, Some(parse_expression(first)?))
        };

        Ok(Self { receiver, name, body, options })
    }
}

pub(crate) fn expand(input: TokenStream, kind: Kind) -> syn::Result<TokenStream> {
    let Declaration { receiver, name, body, options } = Declaration::parse(input, kind)?;
    let root = oximo_root();
    let mut sums = crate::sum::ModelSums::new(receiver);

    if let Some(binds) = name.bindings() {
        sums.indexed(binds);
    }

    let receiver = sums.receiver();
    let parent = options.parent.map(|parent| sums.rewrite(parent)).transpose()?;
    let body = body.map(|body| sums.rewrite(body)).transpose()?;
    let selector = options.selector.tokens(&root);
    let expansion = name.expand(&receiver, &root, |explicit_name| {
        let declaration_name = explicit_name.unwrap_or_else(|| name.tokens(&receiver));

        match kind {
            Kind::Boolean => quote!(#receiver.add_boolean(#declaration_name)),
            Kind::Disjunct => quote!(#receiver.__gdp_disjunct(#declaration_name, #body)),
            Kind::Logic => quote!(#receiver.__gdp_logic(#declaration_name, #body)),
            Kind::Disjunction => {
                if let Some(parent) = &parent {
                    let declaration_name = if matches!(name, DeclarationName::Anonymous) {
                        quote!(__oximo_context.__gdp_auto_name("gdp"))
                    } else {
                        declaration_name
                    };
                    quote!({
                        let __oximo_context = #receiver.__gdp_context(#parent);
                        __oximo_context.__gdp_disjunction(#declaration_name, #body, #selector)
                    })
                } else {
                    quote!(#receiver.__gdp_disjunction(#declaration_name, #body, #selector))
                }
            }
        }
    })?;
    let expansion = sums.wrap(expansion);

    Ok(if let (Kind::Boolean, DeclarationName::Named(named)) = (kind, name) {
        let name = named.name;
        quote!(let #name = #expansion;)
    } else {
        expansion
    })
}
