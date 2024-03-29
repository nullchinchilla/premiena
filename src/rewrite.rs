use std::time::Instant;

use anyhow::Context;
use once_cell::sync::Lazy;

use crate::automaton::{Automaton, Nfa, Nfst};

use self::helpers::{left_context, right_context, NfaExt};
pub use helpers::regex_to_nfa;
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
        let start = Instant::now();
        static LI: Lazy<Nfa> = Lazy::new(|| Nfa::from("["));
        static LA: Lazy<Nfa> = Lazy::new(|| Nfa::from("<"));
        static LC: Lazy<Nfa> = Lazy::new(|| Nfa::from("("));
        static RI: Lazy<Nfa> = Lazy::new(|| Nfa::from("]"));
        static RA: Lazy<Nfa> = Lazy::new(|| Nfa::from(">"));
        static RC: Lazy<Nfa> = Lazy::new(|| Nfa::from(")"));

        static LRC: Lazy<Nfa> = Lazy::new(|| RC.clone().union(&LC).determinize_min());

        static LEFT: Lazy<Nfa> = Lazy::new(|| LI.clone().union(&LA).union(&LC).determinize_min());
        static RIGHT: Lazy<Nfa> = Lazy::new(|| RI.clone().union(&RA).union(&RC).determinize_min());

        static EMM: Lazy<Nfa> = Lazy::new(|| LEFT.clone().union(&RIGHT.clone()));
        static EMM_0: Lazy<Nfa> = Lazy::new(|| EMM.clone().union(&Nfa::from("0")));

        static SIGMA: Lazy<Nfa> = Lazy::new(|| Nfa::sigma().subtract(&EMM_0).determinize_min());

        // static NO_WINGS: Lazy<Nfa> =
        //     Lazy::new(|| Nfa::all().concat(&EMM_0).concat(&Nfa::all()).complement());
        static PROLOGUE: Lazy<Nfst> = Lazy::new(|| {
            // EMM_0.clone().determinize_min().intro()
            let add_hash =
                Nfst::image_cross(&Nfst::id_nfa(Nfa::empty()), &Nfst::id_nfa("%".into()));

            add_hash
                .clone()
                .concat(&Nfst::id_nfa(SIGMA.clone().star().determinize_min()))
                .concat(&add_hash)
                .compose(&EMM_0.clone().determinize_min().intro())
                .deepsilon()
        });

        let obligatory = |phi: &Nfa, left: &Nfa, right: &Nfa| {
            Nfa::all()
                .concat(left)
                .concat(&phi.clone().ignore(&EMM_0))
                .concat(right)
                .concat(&Nfa::all())
                .complement()
        };
        log::trace!("by obligatory: {:?}", start.elapsed());

        let left_context = left_context(&SIGMA, &self.left_ctx, &LEFT, &RIGHT).determinize_min();
        let right_context = right_context(&SIGMA, &self.right_ctx, &LEFT, &RIGHT).determinize_min();
        log::trace!("by LR contexts: {:?}", start.elapsed());

        let context = left_context.clone().intersect(&right_context);
        log::trace!("by unified context: {:?}", start.elapsed());

        let oblig = obligatory(&self.pre, &LI, &RIGHT)
            .intersect(&obligatory(&self.pre, &LEFT, &RI))
            .determinize_min();
        log::trace!("by obligatory + leftright: {:?}", start.elapsed());

        let sigma_i_0_star = Nfa::all()
            .concat(&LA.clone().union(&LC).union(&RA).union(&RC))
            .concat(&Nfa::all())
            .complement()
            .determinize_min();
        log::trace!("by sigmastar: {:?}", start.elapsed());

        let inner_replace = Nfst::id_nfa(sigma_i_0_star)
            .concat(
                &Nfst::id_nfa(LA.clone())
                    .concat(
                        &Nfst::id_nfa(
                            self.pre
                                .clone()
                                // .ignore(&"0".into())
                                .ignore(&LRC)
                                .determinize_min(),
                        )
                        .image_cross(&Nfst::id_nfa(
                            self.post
                                .clone()
                                .ignore(&LRC)
                                // .ignore(&"0".into())
                                .determinize_min(),
                        )),
                    )
                    .concat(&Nfst::id_nfa(RA.clone()))
                    .optional(),
            )
            .star()
            .deepsilon();
        log::trace!("by inner_replace: {:?}", start.elapsed());

        let replace = Nfst::id_nfa(context.clone().intersect(&oblig).determinize_min())
            .compose(&inner_replace);
        log::trace!("by replace: {:?}", start.elapsed());

        let replace = if reverse { replace.inverse() } else { replace };

        move |input| {
            // Nfst::id_nfa(input.determinize_min())
            // .compose(&PROLOGUE)
            // .compose(&Nfst::id_nfa(right_context.clone()))
            // .image_nfa()
            // .determinize_min()
            let pre_replace = Nfst::id_nfa(
                Nfst::id_nfa(input.determinize_min())
                    .compose(&PROLOGUE)
                    .image_nfa()
                    .determinize_min(),
            );
            let post_replace = pre_replace.compose(&replace).image_nfa().determinize_min();

            Nfst::id_nfa(post_replace)
                .compose(&PROLOGUE.clone().inverse())
                .image_nfa()
                .determinize_min()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_rewrite() {
        let _ = env_logger::try_init();
        let rr = RewriteRule::from_line("X > W / _t(t|)(a|o)").unwrap();
        let rule = rr.transduce(false);

        for s in rule(Nfa::from("tXtotXtXtta"))
            .determinize_min()
            .lang_iter_utf8()
            .take(100)
        {
            eprintln!("{:?}", s)
        }
    }
}
