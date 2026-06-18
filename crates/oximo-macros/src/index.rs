//! Token rewriter for multi-index access `q[i, j, k]`.
//!
//! Inside the modeling macros' value expressions, an index-position bracket with
//! top-level commas (`q[i, j, k]`) is rewritten to a parenthesized, auto-ref'd
//! tuple (`q[(&(i), &(j), &(k))]`). A comma subscript is a syntax error for every
//! std type (`Vec`/array/slice/`HashMap` all use single-arg or chained `x[i][j]`
//! indexing), so a multi-arg bracket in index position is an
//! [`IndexedVar`](../../oximo_core/struct.IndexedVar.html) access.

use proc_macro2::{Delimiter, Group, TokenStream as TokenStream2, TokenTree};
use quote::quote;

use crate::split_top_commas;

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
pub(crate) fn rewrite_index_subscripts(ts: TokenStream2) -> TokenStream2 {
    let mut out: Vec<TokenTree> = Vec::new();
    let mut index_pos = false;

    for tt in ts {
        if let TokenTree::Group(g) = &tt {
            let delim = g.delimiter();
            let inner = rewrite_index_subscripts(g.stream());

            if delim == Delimiter::Bracket && index_pos {
                let mut elems = split_top_commas(inner.clone());
                // trailing comma
                elems.retain(|e| !e.is_empty());
                if elems.len() >= 2 {
                    let tuple = build_ref_tuple(&elems);
                    let mut newg = Group::new(Delimiter::Bracket, tuple);
                    newg.set_span(g.span());
                    out.push(TokenTree::Group(newg));
                    // the resulting `]` can be subscripted again
                    index_pos = true;
                    continue;
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

    out.into_iter().collect()
}

/// `[e0, e1, ..]` -> `(&(e0), &(e1), ..)`, leaving an element that already begins
/// with `&` as-is.
fn build_ref_tuple(elems: &[TokenStream2]) -> TokenStream2 {
    let refd = elems.iter().map(ref_elem);
    quote!( ( #(#refd),* ) )
}

fn ref_elem(e: &TokenStream2) -> TokenStream2 {
    if let Some(TokenTree::Punct(p)) = e.clone().into_iter().next() {
        if p.as_char() == '&' {
            return e.clone();
        }
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
        norm(rewrite_index_subscripts(src.parse().unwrap()))
    }

    fn unchanged(src: &str) {
        let parsed: TokenStream2 = src.parse().unwrap();
        assert_eq!(norm(rewrite_index_subscripts(parsed.clone())), norm(parsed), "{src}");
    }

    #[test]
    fn rewrites_multi_index() {
        assert_eq!(rewrite("q[i, j, k]"), "q[(&(i),&(j),&(k))]");
        assert_eq!(rewrite("s[p, n]"), "s[(&(p),&(n))]");
        assert_eq!(rewrite("s[p, n + 1]"), "s[(&(p),&(n+1))]");
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
