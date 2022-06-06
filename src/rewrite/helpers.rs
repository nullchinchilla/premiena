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
}

impl NfaExt for Nfa {
    fn ignore(self, other: &Nfa) -> Nfa {
        Nfst::id_nfa(self)
            .compose(&other.clone().intro())
            .image_nfa()
    }

    fn intro(self) -> Nfst {
        let s = Nfst::id_nfa(self);
        Nfst::id_nfa(Nfa::sigma())
            .union(&Nfst::id_nfa(Nfa::empty()).image_cross(&s).star())
            .star()
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

pub fn left_context(lambda: &Nfa, left: &Nfa, right: &Nfa) -> Nfa {
    // alphabet.pre_iff_suf(
    //     &alphabet.id_sigma().star().image_dfa().concat(lambda),
    //     &left.concat(&alphabet.id_sigma().star().image_dfa()),
    // )
    let sigma_star_ignore_left = Nfa::sigma().star().ignore(left);
    pre_iff_suf(
        &sigma_star_ignore_left
            .clone()
            .concat(&lambda.clone().ignore(left))
            .intersect(&sigma_star_ignore_left.clone().concat(left).complement()),
        &left.clone().concat(&sigma_star_ignore_left),
    )
    .ignore(right)
}

pub fn right_context(rho: &Nfa, left: &Nfa, right: &Nfa) -> Nfa {
    // alphabet.pre_iff_suf(
    //     &alphabet.id_sigma().star().image_dfa().concat(right),
    //     &rho.concat(&alphabet.id_sigma().star().image_dfa()),
    // )
    let sigma_star_ignore_right = Nfa::sigma().star().ignore(right);
    pre_iff_suf(
        &sigma_star_ignore_right.clone().concat(right),
        &rho.clone()
            .concat(&sigma_star_ignore_right)
            .intersect(&right.clone().concat(&sigma_star_ignore_right).complement()),
    )
    .ignore(left)
}
