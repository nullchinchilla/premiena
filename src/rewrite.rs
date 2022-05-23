mod helpers;

use std::sync::Arc;

use regex_syntax::{hir::Hir, Parser};
use tap::Tap;

use crate::{
    alphabet::Alphabet,
    automaton::{Nfst, Symbol},
};

use self::helpers::{hir_to_dfa, left_context, right_context};

#[derive(Clone, Debug)]
/// A single rewrite rule. This can be cheaply cloned!
pub struct RewriteRule {
    from: Arc<Hir>,
    to: Arc<Hir>,
    left_ctx: Arc<Hir>,
    right_ctx: Arc<Hir>,
    alphabet: Alphabet,
}

impl RewriteRule {
    /// Create a new rewrite rule, from string-based regex syntax
    pub fn from_regexes(
        from: &str,
        to: &str,
        left_ctx: &str,
        right_ctx: &str,
        alphabet: Alphabet,
    ) -> regex_syntax::Result<Self> {
        let from = Parser::new().parse(from)?.into();
        let to = Parser::new().parse(to)?.into();
        let left_ctx = Parser::new().parse(left_ctx)?.into();
        let right_ctx = Parser::new().parse(right_ctx)?.into();
        Ok(Self {
            from,
            to,
            left_ctx,
            right_ctx,
            alphabet,
        })
    }

    /// Generate the transducer corresponding to thie rewrite rule
    pub fn transducer(&self) -> Nfst {
        let from = hir_to_dfa(&self.from);
        eprintln!("{}", from.graphviz());
        let to = hir_to_dfa(&self.to);
        let left_ctx = hir_to_dfa(&self.left_ctx);
        let right_ctx = hir_to_dfa(&self.right_ctx);
        let alphabet_with_wings = self.alphabet.clone().tap_mut(|a| {
            a.insert(Symbol::from('<'));
            a.insert(Symbol::from('>'));
        });

        let prologue = self.alphabet.intro(
            &Alphabet::new(['<', '>'].into_iter().map(Symbol::from))
                .id_sigma()
                .image_dfa(),
        );
        let left_context = left_context(
            &alphabet_with_wings,
            &left_ctx,
            &Nfst::id_single(Some(Symbol::from('<'))).image_dfa(),
            &Nfst::id_single(Some(Symbol::from('>'))).image_dfa(),
        );
        let right_context = right_context(
            &alphabet_with_wings,
            &right_ctx,
            &Nfst::id_single(Some(Symbol::from('<'))).image_dfa(),
            &Nfst::id_single(Some(Symbol::from('>'))).image_dfa(),
        );
        let emm = Alphabet::new(['<', '>'].into_iter().map(Symbol::from)).id_sigma();

        let from_emm = self.alphabet.ignore(&from, &emm.image_dfa());
        let to_emm = self.alphabet.ignore(&to, &emm.image_dfa());

        let replace = Nfst::id_dfa(
            &self
                .alphabet
                .ignore(&self.alphabet.id_sigma().image_dfa(), &emm.image_dfa()),
        )
        .concat(
            &Nfst::id_single(Some(Symbol::from('<')))
                .concat(&Nfst::id_dfa(&from_emm).image_cross(&Nfst::id_dfa(&to_emm)))
                .concat(&Nfst::id_single(Some(Symbol::from('>'))))
                .optional(),
        )
        .star();

        prologue
            .compose(&Nfst::id_dfa(&right_context).samelen_intersect(&Nfst::id_dfa(&left_context)))
            .compose(&replace)
            .compose(&prologue.inverse())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rr_construct() {
        let input = hir_to_dfa(&Parser::new().parse("atekta").unwrap());
        let rr = RewriteRule::from_regexes("e", "ei", "", "k", Alphabet::new_alphanum()).unwrap();
        let transducer = rr.transducer();
        eprintln!("TD built");
        for word in Nfst::id_dfa(&input).compose(&transducer).image_dfa().iter() {
            for s in word {
                eprint!("{}", s);
            }
            eprintln!();
        }
    }
}
