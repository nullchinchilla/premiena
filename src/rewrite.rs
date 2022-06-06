use anyhow::Context;

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
    pub fn transduce(&self, input: Nfa, reverse: bool) -> Nfa {
        let emm = regex_to_nfa("<")
            .unwrap()
            .union(&regex_to_nfa(">").unwrap())
            .determinize_min();
        let no_wings = Nfa::all()
            .concat(&emm)
            .concat(&Nfa::all())
            .determinize_min()
            .complement();

        let prologue = Nfst::id_nfa(no_wings)
            .compose(&emm.clone().intro())
            .deepsilon();

        let left_context = left_context(&self.left_ctx, &Nfa::from("<"), &Nfa::from(">"));
        let right_context = right_context(&self.right_ctx, &"<".into(), &">".into());

        let pre_emm = self.pre.clone().ignore(&emm).determinize_min();
        let post_emm = self.post.clone().ignore(&emm).determinize_min();

        let context = right_context.intersect(&left_context).determinize_min();
        let replace = Nfst::id_nfa(context).compose(
            &Nfst::id_nfa(Nfa::sigma())
                .concat(
                    &Nfst::id_nfa("<".into())
                        .concat(&Nfst::id_nfa(pre_emm).image_cross(&Nfst::id_nfa(post_emm)))
                        .concat(&Nfst::id_nfa(">".into()))
                        .optional(),
                )
                .star(),
        );

        let replace = if reverse { replace.inverse() } else { replace };
        let pre_replace = Nfst::id_nfa(input.determinize_min()).compose(&prologue);
        Nfst::id_nfa(pre_replace.compose(&replace).image_nfa().determinize_min())
            .compose(&prologue.inverse().deepsilon())
            .image_nfa()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_lenition() {
        let rr = RewriteRule::from_line("b > f / a_a").unwrap();
        let rule = rr.transduce("afa".into(), false).deepsilon();
        // eprintln!("{}", rule.image_nfa().graphviz());
        for s in rule.determinize_min().lang_iter_utf8().take(10) {
            eprintln!("{:?}", s)
        }
    }
}
