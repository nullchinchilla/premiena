use std::iter::once;

use regex_syntax::{
    hir::{Hir, HirKind, Literal},
    Parser,
};

use crate::automaton::{Automaton, Nfa, Nfst};

/// Transforms a regexp to an NFA.
pub fn regex_to_nfa(s: &str) -> anyhow::Result<Nfa> {
    let hir: Hir = Parser::new().parse(s.trim())?;
    hir_to_nfa(&hir)
}

/// Transforms a regexp hir to an NFA
fn hir_to_nfa(hir: &Hir) -> anyhow::Result<Nfa> {
    Ok(match hir.kind() {
        HirKind::Empty => Nfa::empty(),
        HirKind::Literal(Literal::Unicode(c)) => Nfa::from(String::from_iter(once(*c)).as_str()),
        HirKind::Class(_) => anyhow::bail!("classes not supported"),
        HirKind::Anchor(_) => anyhow::bail!("anchors not supported"),
        HirKind::WordBoundary(_) => anyhow::bail!("word boundaries not supported"),
        HirKind::Repetition(r) => {
            let inner = hir_to_nfa(&r.hir)?;
            inner.star()
        }
        HirKind::Group(g) => hir_to_nfa(&g.hir)?,
        HirKind::Concat(cc) => cc.iter().try_fold(Nfa::empty(), |a, b| {
            Ok::<_, anyhow::Error>(a.concat(&hir_to_nfa(b)?))
        })?,
        HirKind::Alternation(cc) => {
            let (a, b) = cc.split_first().unwrap();
            b.iter().try_fold(hir_to_nfa(a)?, |a, b| {
                Ok::<_, anyhow::Error>(a.union(&hir_to_nfa(b)?))
            })?
        }
        _ => anyhow::bail!("unsupported syntax"),
    })
}

/// NFA extensions
pub trait NfaExt {
    fn intro(self) -> Nfst;
    fn ignore(self, other: &Nfa) -> Nfa;
    fn int_bytes(self) -> Nfa;
}

impl NfaExt for Nfa {
    fn ignore(self, other: &Nfa) -> Nfa {
        Nfst::id_nfa(self)
            .compose(&other.clone().intro())
            .image_nfa()
    }

    fn intro(self) -> Nfst {
        let s = Nfst::id_nfa(self);
        Nfst::id_nfa(Nfa::sigma().concat(&Nfa::sigma()))
            .union(&Nfst::id_nfa(Nfa::empty()).image_cross(&s).star())
            .star()
    }

    fn int_bytes(self) -> Nfa {
        self.intersect(&Nfa::sigma().concat(&Nfa::sigma()).star())
    }
}

/// Obtain a regular language where, for every partition of every element of the language, if the prefix is in l1 then the suffix is in l2.
pub fn if_pre_then_suf(l1: &Nfa, l2: &Nfa) -> Nfa {
    l1.clone().concat(&l2.clone().complement()).complement()
}

/// Obtain a regular language where, for every partition of every element of the language, the prefix is in l1 if the suffix is in l2.
pub fn if_suf_then_pre(l1: &Nfa, l2: &Nfa) -> Nfa {
    l1.clone().complement().concat(l2).complement()
}

/// Obtains a regular language where, for every partition of every element of the language, the prefix is in l1 iff the suffix is in l2.
pub fn pre_iff_suf(l1: &Nfa, l2: &Nfa) -> Nfa {
    if_pre_then_suf(l1, l2).intersect(&if_suf_then_pre(l1, l2))
}

pub fn left_context(sigma: &Nfa, lambda: &Nfa, left: &Nfa, right: &Nfa) -> Nfa {
    // let sigma_star_ignore_left = sigma
    //     .clone()
    //     .star()
    //     .ignore(left)
    //     .determinize_min()
    //     .ignore(&"0".into())
    //     .determinize_min();
    // pre_iff_suf(
    //     &sigma_star_ignore_left
    //         .clone()
    //         .concat(&lambda.clone().ignore(left).ignore(&"0".into()))
    //         .intersect(&sigma_star_ignore_left.clone().concat(left).complement()),
    //     &left.clone().concat(&sigma_star_ignore_left),
    // )
    // .ignore(right)
    if let Some(true) = lambda.lang_iter().next().map(|l| l.is_empty()) {
        // cannot have two elements of the original sigma without a left between them
        sigma.clone().ignore(right).concat(left).star()
    } else {
        pre_iff_suf(
            &Nfa::all().concat(lambda),
            &left.clone().concat(&Nfa::all()),
        )
    }
}

pub fn right_context(sigma: &Nfa, rho: &Nfa, left: &Nfa, right: &Nfa) -> Nfa {
    // let sigma_star_ignore_right = sigma
    //     .clone()
    //     .star()
    //     .ignore(right)
    //     .determinize_min()
    //     .ignore(&"0".into())
    //     .determinize_min();
    // pre_iff_suf(
    //     &sigma_star_ignore_right.clone().concat(right),
    //     &rho.clone()
    //         .ignore(right)
    //         .ignore(&"0".into())
    //         .concat(&sigma_star_ignore_right)
    //         .intersect(&right.clone().concat(&sigma_star_ignore_right).complement()),
    // )
    // .ignore(left)
    //     if rho.lang_iter().next() == Some(vec![]) {
    //         return Nfa::all();
    //     }
    if let Some(true) = rho.lang_iter().next().map(|l| l.is_empty()) {
        // cannot have two elements of the original sigma without a right between them
        right.clone().concat(&sigma.clone().ignore(left)).star()
    } else {
        pre_iff_suf(&Nfa::all().concat(right), &rho.clone().concat(&Nfa::all()))
    }
}
