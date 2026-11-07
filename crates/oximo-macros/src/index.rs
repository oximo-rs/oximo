//! Token rewriter for multi-index access `q[i, j, k]`.
//!
//! Inside the modeling macros' value expressions, an index-position bracket with
//! top-level commas (`q[i, j, k]`) is rewritten to a parenthesized, auto-ref'd
//! tuple (`q[(&(i), &(j), &(k))]`). A comma subscript is a syntax error for every
//! std type (`Vec`/array/slice/`HashMap` all use single-arg or chained `x[i][j]`
//! indexing), so a multi-arg bracket in index position is an
//! [`IndexedVar`](../../oximo_core/struct.IndexedVar.html) access.

use proc_macro2::{Delimiter, Group, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, quote};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Expr, Token};

use crate::parse_comma_list;

/// Parse a Rust value expression, retaining only multi-index syntax sugar.
pub(crate) fn parse_value_expr(ts: TokenStream2) -> syn::Result<syn::Expr> {
    syn::parse2(rewrite_index_subscripts(ts)?)
}

/// Rust keywords that may precede a `[` without it being a subscript.
/// After these, a bracket starts a pattern or expression, not an index.
fn is_keyword(id: &str) -> bool {
    matches!(
        id,
        "for"
            | "in"
            | "if"
            | "else"
            | "while"
            | "loop"
            | "match"
            | "let"
            | "return"
            | "move"
            | "as"
            | "where"
            | "mut"
            | "ref"
            | "yield"
            | "await"
            | "break"
            | "continue"
            | "const"
    )
}

/// Whether a freshly emitted token leaves us in "index position".
fn subscriptable(tt: &TokenTree) -> bool {
    match tt {
        TokenTree::Ident(id) => !is_keyword(&id.to_string()),
        TokenTree::Group(g) => {
            matches!(g.delimiter(), Delimiter::Parenthesis | Delimiter::Bracket)
        }
        TokenTree::Literal(_) => true,
        TokenTree::Punct(_) => false,
    }
}

/// Rewrite every multi-arg index subscript in a value-expression token stream.
pub(crate) fn rewrite_index_subscripts(ts: TokenStream2) -> syn::Result<TokenStream2> {
    rewrite_subscripts(ts, false)
}

fn rewrite_subscripts(ts: TokenStream2, defer_errors: bool) -> syn::Result<TokenStream2> {
    let mut out: Vec<TokenTree> = Vec::new();
    let mut index_pos = false;

    for tt in ts {
        if let TokenTree::Group(g) = &tt {
            let delim = g.delimiter();
            let macro_input = matches!(
                out.as_slice(),
                [.., TokenTree::Ident(id), TokenTree::Punct(p)]
                    if p.as_char() == '!' && !is_keyword(&id.to_string())
            );
            let inner = rewrite_subscripts(g.stream(), defer_errors || macro_input)?;

            if delim == Delimiter::Bracket && index_pos {
                match parse_index_components(inner.clone()) {
                    Ok(elems) if elems.len() >= 2 => {
                        let tuple = build_ref_tuple(&elems);
                        let mut newg = Group::new(Delimiter::Bracket, tuple);
                        newg.set_span(g.span());
                        out.push(TokenTree::Group(newg));
                        // the resulting `]` can be subscripted again
                        index_pos = true;
                        continue;
                    }
                    Err(error) if !defer_errors => return Err(error),

                    // The enclosing macro owns its input grammar.
                    _ => {}
                }
            }

            let mut newg = Group::new(delim, inner);
            newg.set_span(g.span());
            let g_tt = TokenTree::Group(newg);
            index_pos = subscriptable(&g_tt);
            out.push(g_tt);
        } else {
            index_pos = subscriptable(&tt);
            out.push(tt);
        }
    }

    Ok(out.into_iter().collect())
}

fn parse_index_components(ts: TokenStream2) -> syn::Result<Vec<TokenStream2>> {
    // Rust expressions can contain commas in generic arguments or closure
    // parameters without a surrounding token group.
    if let Ok(elems) = Punctuated::<Expr, Token![,]>::parse_terminated.parse2(ts.clone()) {
        return Ok(elems.into_iter().map(ToTokens::into_token_stream).collect());
    }

    // This token walk also reaches nested macro input such as `x[i in domain]`.
    parse_comma_list(ts, "missing index component")
}

/// `[e0, e1, ..]` -> `(&(e0), &(e1), ..)`, leaving an element that already begins
/// with `&` as-is.
fn build_ref_tuple(elems: &[TokenStream2]) -> TokenStream2 {
    let refd = elems.iter().map(ref_elem);
    quote!( ( #(#refd),* ) )
}

fn ref_elem(e: &TokenStream2) -> TokenStream2 {
    if let Some(TokenTree::Punct(p)) = e.clone().into_iter().next()
        && p.as_char() == '&'
    {
        return e.clone();
    }
    quote!( &(#e) )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(ts: TokenStream2) -> String {
        ts.to_string().chars().filter(|c| !c.is_whitespace()).collect()
    }

    fn rewrite(src: &str) -> String {
        norm(rewrite_index_subscripts(src.parse().unwrap()).unwrap())
    }

    fn unchanged(src: &str) {
        let parsed: TokenStream2 = src.parse().unwrap();
        assert_eq!(norm(rewrite_index_subscripts(parsed.clone()).unwrap()), norm(parsed), "{src}");
    }

    #[test]
    fn rewrites_multi_index() {
        assert_eq!(rewrite("q[i, j, k]"), "q[(&(i),&(j),&(k))]");
        assert_eq!(rewrite("s[p, n]"), "s[(&(p),&(n))]");
        assert_eq!(rewrite("s[p, n + 1]"), "s[(&(p),&(n+1))]");
    }

    #[test]
    fn rewrites_multi_index_with_one_trailing_comma() {
        assert_eq!(rewrite("q[i, j,]"), "q[(&(i),&(j))]");
        assert_eq!(rewrite("q[i, j, k,]"), "q[(&(i),&(j),&(k))]");
        assert_eq!(rewrite("q[&i, j,]"), "q[(&i,&(j))]");
    }

    #[test]
    fn rejects_missing_index_components() {
        for src in [
            "q[,i,j]",
            "q[i,,j]",
            "q[i,j,,]",
            "q[i,j,,,]",
            "q[,i,j,]",
            "q[i,,j,]",
            "q[,i]",
            "q[i,,]",
            "q[,]",
            "foo(q[i,,j])",
            "q[r[i,,j], k]",
            "opaque!(tokens[i,,j]) + q[i,,j]",
            "!(q[i,,j] == other)",
            "return !(q[i,,j] == other)",
            "if !(q[i,,j] == other) {}",
        ] {
            let error = rewrite_index_subscripts(src.parse().unwrap()).unwrap_err();
            assert_eq!(error.to_string(), "missing index component", "{src}");
        }
        assert!(parse_value_expr("q[]".parse().unwrap()).is_err());
    }

    #[test]
    fn keeps_explicit_ref_components() {
        assert_eq!(rewrite("q[&i, j, k]"), "q[(&i,&(j),&(k))]");
    }

    #[test]
    fn leaves_single_index_and_chains() {
        unchanged("cost[i]");
        unchanged("sigma[i][j]");
        unchanged("a[i + 1]");
    }

    #[test]
    fn leaves_commas_within_single_index_expressions() {
        unchanged("weights[index::<u8, u16>()]");
        unchanged("weights[index::<u8, u16,>()]");
        unchanged("map[|i, j| i + j]");
    }

    #[test]
    fn rewrites_multi_index_with_generic_arguments() {
        assert_eq!(rewrite("q[index::<u8, u16>(), j,]"), "q[(&(index::<u8,u16>()),&(j))]");
    }

    #[test]
    fn leaves_nested_macro_binding_syntax() {
        unchanged("variable!(model, x[i in 0..2]);");
    }

    #[test]
    fn defers_missing_components_inside_macro_input() {
        unchanged("opaque!(tokens[,i,j])");
        unchanged("opaque![tokens[i,,j]]");
        unchanged("path::opaque! { tokens[i,j,,] }");
        unchanged("opaque!({ foo(tokens[i,,j]) })");
    }

    #[test]
    fn rewrites_valid_subscripts_alongside_opaque_macro_input() {
        assert_eq!(
            rewrite("opaque!(tokens[i,,j], q[i,j], sum!(r[a,b] for a in domain))"),
            "opaque!(tokens[i,,j],q[(&(i),&(j))],sum!(r[(&(a),&(b))]foraindomain))",
        );
    }

    #[test]
    fn leaves_already_tupled() {
        unchanged("q[(i, j, k)]");
        unchanged("q[(&i, &j, k)]");
    }

    #[test]
    fn leaves_array_and_macro_literals() {
        unchanged("[1, 2, 3]");
        unchanged("vec![1, 2, 3]");
        unchanged("let a = [1, 2, 3];");
        unchanged("foo([1, 2, 3])");
    }

    #[test]
    fn leaves_slice_patterns_after_keyword() {
        unchanged("for [a, b] in pairs {}");
    }

    #[test]
    fn rewrites_inside_nested_groups() {
        assert_eq!(rewrite("(q[i, j] + r[a, b])"), "(q[(&(i),&(j))]+r[(&(a),&(b))])");
        assert_eq!(rewrite("foo(q[i, j])"), "foo(q[(&(i),&(j))])");
    }
}
