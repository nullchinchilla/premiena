use regex_syntax::hir::{Hir, HirKind, Literal};

use crate::{
    alphabet::Alphabet,
    automaton::{Dfa, Nfst, Symbol},
};

pub fn hir_to_dfa(hir: &Hir) -> Dfa {
    hir_to_nfst(hir).image_dfa()
}

fn hir_to_nfst(hir: &Hir) -> Nfst {
    match hir.kind() {
        HirKind::Empty => Nfst::id_single(None),
        HirKind::Literal(Literal::Unicode(c)) => Nfst::id_single(Some(Symbol::from(*c))),
        HirKind::Class(_) => todo!(),
        HirKind::Anchor(_) => todo!(),
        HirKind::WordBoundary(_) => todo!(),
        HirKind::Repetition(r) => {
            let inner = hir_to_nfst(&r.hir);
            inner.star()
        }
        HirKind::Group(_) => todo!(),
        HirKind::Concat(cc) => cc
            .iter()
            .fold(Nfst::id_single(None), |a, b| a.concat(&hir_to_nfst(b))),
        HirKind::Alternation(cc) => {
            let (a, b) = cc.split_first().unwrap();
            b.iter()
                .fold(hir_to_nfst(a), |a, b| a.union(&hir_to_nfst(b)))
        }
        _ => todo!(),
    }
}

pub fn left_context(alphabet: &Alphabet, lambda: &Dfa, left: &Dfa, right: &Dfa) -> Dfa {
    // alphabet.pre_iff_suf(
    //     &alphabet.id_sigma().star().image_dfa().concat(lambda),
    //     &left.concat(&alphabet.id_sigma().star().image_dfa()),
    // )
    let sigma_star_ignore_left = alphabet.ignore(&alphabet.id_sigma().star().image_dfa(), left);
    alphabet.ignore(
        &alphabet.pre_iff_suf(
            &sigma_star_ignore_left
                .concat(&alphabet.ignore(lambda, left))
                .intersect(
                    &sigma_star_ignore_left
                        .concat(left)
                        .complement(alphabet.sigma()),
                ),
            &left.concat(&sigma_star_ignore_left),
        ),
        right,
    )
}

pub fn right_context(alphabet: &Alphabet, rho: &Dfa, left: &Dfa, right: &Dfa) -> Dfa {
    // alphabet.pre_iff_suf(
    //     &alphabet.id_sigma().star().image_dfa().concat(right),
    //     &rho.concat(&alphabet.id_sigma().star().image_dfa()),
    // )
    let sigma_star_ignore_right = alphabet.ignore(&alphabet.id_sigma().star().image_dfa(), right);
    alphabet.ignore(
        &alphabet.pre_iff_suf(
            &sigma_star_ignore_right.concat(right),
            &rho.concat(&sigma_star_ignore_right).intersect(
                &right
                    .concat(&sigma_star_ignore_right)
                    .complement(alphabet.sigma()),
            ),
        ),
        left,
    )
}
