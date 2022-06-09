use anyhow::Context;
use once_cell::sync::Lazy;

use crate::automaton::{Automaton, Nfa, Nfst};

use self::helpers::{left_context, regex_to_nfa, right_context, NfaExt};

mod helpers;

/// A rewrite rule.
pub struct RewriteRule {
    pre: Nfa,
    post: Nfa,
    left_ctx: Nfa,
    right_ctx: Nfa,
}

impl RewriteRule {
    /// Create a new rewrite rule from a string
    pub fn from_line(line: &str) -> anyhow::Result<Self> {
        let (rule, context) = line.split_once('/').context("rule must have /")?;
        let (pre, post) = rule.split_once('>').context("rule must have >")?;
        let (left, right) = context.split_once('_').context("rule must have _")?;
        Ok(Self {
            pre: regex_to_nfa(pre)?.determinize_min(),
            post: regex_to_nfa(post)?.determinize_min(),
            left_ctx: regex_to_nfa(left)?.determinize_min(),
            right_ctx: regex_to_nfa(right)?.determinize_min(),
        })
    }

    /// Generate the transducer corresponding to thie rewrite rule
    pub fn transduce(&self, reverse: bool) -> impl Fn(Nfa) -> Nfa {
        static LI: Lazy<Nfa> = Lazy::new(|| Nfa::from("["));
        static LA: Lazy<Nfa> = Lazy::new(|| Nfa::from("<"));
        static LC: Lazy<Nfa> = Lazy::new(|| Nfa::from("("));
        static RI: Lazy<Nfa> = Lazy::new(|| Nfa::from("]"));
        static RA: Lazy<Nfa> = Lazy::new(|| Nfa::from(">"));
        static RC: Lazy<Nfa> = Lazy::new(|| Nfa::from(")"));

        static LRC: Lazy<Nfa> = Lazy::new(|| LA.clone().union(&LC).determinize_min());

        static LEFT: Lazy<Nfa> = Lazy::new(|| LI.clone().union(&LA).union(&LC).determinize_min());
        static RIGHT: Lazy<Nfa> = Lazy::new(|| RI.clone().union(&RA).union(&RC).determinize_min());

        static EMM: Lazy<Nfa> = Lazy::new(|| LEFT.clone().union(&RIGHT.clone()));
        static EMM_0: Lazy<Nfa> = Lazy::new(|| EMM.clone().union(&Nfa::from("0")));
        static NO_WINGS: Lazy<Nfa> =
            Lazy::new(|| Nfa::all().concat(&EMM_0).concat(&Nfa::all()).complement());
        static PROLOGUE: Lazy<Nfst> = Lazy::new(|| {
            Nfst::id_nfa(NO_WINGS.clone())
                .compose(&EMM_0.clone().intro())
                .compose(&Nfst::id_nfa(Nfa::all().int_bytes().determinize()))
                .deepsilon()
        });

        let obligatory = |phi: &Nfa, left: &Nfa, right: &Nfa| {
            Nfa::all()
                .concat(left)
                .concat(&phi.clone().ignore(&EMM_0).determinize())
                .concat(right)
                .concat(&Nfa::all())
                .complement()
                .determinize_min()
        };

        let left_context = left_context(&self.left_ctx, &LEFT, &RIGHT).determinize();
        let right_context = right_context(&self.right_ctx, &LEFT, &RIGHT).determinize();

        // let pre_emm0 = self.pre.clone().ignore(&EMM_0).determinize();
        let post_emm = self.post.clone().ignore(&EMM).determinize();

        let context = right_context.intersect(&left_context).determinize();
        let oblig = obligatory(&self.pre, &LI, &RIGHT)
            .intersect(&obligatory(&self.pre, &LEFT, &RI))
            .determinize_min();

        let sigma_i_0_star = Nfa::all()
            .concat(&LA.clone().union(&LC).union(&RA).union(&RC))
            .concat(&Nfa::all())
            .complement()
            .int_bytes()
            .determinize_min();

        let inner_replace = Nfst::id_nfa(sigma_i_0_star)
            .concat(
                &Nfst::id_nfa(LA.clone())
                    .concat(
                        &Nfst::id_nfa(
                            self.pre
                                .clone()
                                // .ignore(&"0".into())
                                .ignore(&LRC)
                                .determinize(),
                        )
                        .image_cross(&Nfst::id_nfa(
                            self.post
                                .clone()
                                .ignore(&LRC)
                                // .ignore(&"0".into())
                                .determinize(),
                        )),
                    )
                    .concat(&Nfst::id_nfa(RA.clone()))
                    .optional(),
            )
            .star();
        eprintln!("gonna replace");
        let replace =
            Nfst::id_nfa(context.intersect(&oblig).determinize_min()).compose(&inner_replace);
        let replace = if reverse { replace.inverse() } else { replace };
        eprintln!("made replace");
        move |input| {
            let pre_replace = Nfst::id_nfa(input.determinize_min()).compose(&PROLOGUE);
            Nfst::id_nfa(pre_replace.compose(&replace).image_nfa().determinize_min())
                .compose(&PROLOGUE.clone().inverse())
                .image_nfa()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_lenition() {
        let rr = RewriteRule::from_line("b > f / a_o").unwrap();
        let rule = rr.transduce(false);
        // eprintln!("{}", rule.image_nfa().graphviz());
        for s in rule("abo".into())
            .int_bytes()
            .determinize_min()
            // .intersect(
            //     &Nfa::all()
            //         .int_bytes()
            //         .concat(&"b".into())
            //         .concat(&Nfa::all()),
            // )
            .lang_iter_utf8()
        // .filter(|c| c.chars().all(|c| c.is_ascii_graphic()))
        // .take(1000)
        {
            // if s.find('b').is_some() {
            eprintln!("{:?}", s)
            // } else {
            // eprint!(".")
            // }
        }
    }
}
