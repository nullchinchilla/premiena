use crate::automaton::{Dfa, Nfst, Symbol};

/// Alphabet encapsulates an alphabet within a particular segment type.
#[derive(Clone, Debug)]
pub struct Alphabet {
    sigma: im::HashSet<Symbol>,
}

impl Alphabet {
    /// Create a new alphabet struct.
    pub fn new(segments: impl IntoIterator<Item = Symbol>) -> Self {
        Self {
            sigma: segments.into_iter().collect(),
        }
    }

    /// The alphabet itself.
    pub fn sigma(&self) -> &im::HashSet<Symbol> {
        &self.sigma
    }

    /// One character of the alphabet, as a relation.
    pub fn id_sigma(&self) -> Nfst {
        Nfst::id_dfa(
            &self
                .sigma
                .iter()
                .cloned()
                .fold(Nfst::nothing(), |a, b| a.union(&Nfst::id_single(Some(b))))
                .image_dfa(),
        )
    }

    /// Freely introduce elements of the given set
    pub fn intro(&self, s: &Dfa) -> Nfst {
        let s = Nfst::id_dfa(s);
        self.id_sigma()
            .union(&Nfst::id_single(None).image_cross(&s).star())
            .star()
    }

    /// Obtain a regular language that's L but "ignoring" elements of S (i.e. allowing elements of S to appear randomly)
    pub fn ignore(&self, lang: &Dfa, s: &Dfa) -> Dfa {
        Nfst::id_dfa(lang).compose(&self.intro(s)).image_dfa()
    }

    /// Obtain a regular language where, for every partition of every element of the language, if the prefix is in l1 then the suffix is in l2.
    pub fn if_pre_then_suf(&self, l1: &Dfa, l2: &Dfa) -> Dfa {
        l1.concat(&l2.complement(&self.sigma))
            .complement(&self.sigma)
    }

    /// Obtain a regular language where, for every partition of every element of the language, the prefix is in l1 if the suffix is in l2.
    pub fn if_suf_then_pre(&self, l1: &Dfa, l2: &Dfa) -> Dfa {
        l1.complement(&self.sigma)
            .concat(l2)
            .complement(&self.sigma)
    }

    /// Obtains a regular language where, for every partition of every element of the language, the prefix is in l1 iff the suffix is in l2.
    pub fn pre_iff_suf(&self, l1: &Dfa, l2: &Dfa) -> Dfa {
        self.if_pre_then_suf(l1, l2)
            .intersect(&self.if_suf_then_pre(l1, l2))
    }
}

fn left_context(alphabet: &Alphabet, lambda: &Dfa, left: &Dfa, right: &Dfa) -> Dfa {
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

fn right_context(alphabet: &Alphabet, rho: &Dfa, left: &Dfa, right: &Dfa) -> Dfa {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_sigma() {
        let alphabet = Alphabet::new("abcdefghijklmnopqrstuvwxyz".chars().map(Symbol::from));
        eprintln!("{}", alphabet.id_sigma().graphviz());
    }

    #[test]
    fn complement() {
        let alphabet = Alphabet::new("gtac".chars().map(Symbol::from));
        let test = Nfst::id_single(Some(Symbol::from('g')))
            .star()
            .image_dfa()
            .complement(alphabet.sigma());
        eprintln!("{}", test.graphviz());
    }

    #[test]
    fn fullhog() {
        let input = "tapaktapka".chars().fold(Nfst::id_single(None), |a, b| {
            a.concat(&Nfst::id_single(Some(Symbol::from(b))))
        });
        let alphabet = Alphabet::new("abcdefghijklmnopqrstuvwxyz".chars().map(Symbol::from));
        let alphabet_with_brackets =
            Alphabet::new("abcdefghijklmnopqrstuvwxyz<>".chars().map(Symbol::from));

        let prologue = alphabet.intro(
            &Alphabet::new(['<', '>'].into_iter().map(Symbol::from))
                .id_sigma()
                .image_dfa(),
        );
        let left_pattern = Nfst::id_single(Some(Symbol::from('a')));
        let right_pattern = Nfst::id_single(Some(Symbol::from('t')));
        let pre_change = Nfst::id_single(Some(Symbol::from('k')));
        let post_change = Nfst::id_single(Some(Symbol::from('h')));

        let left_context = left_context(
            &alphabet_with_brackets,
            &left_pattern.image_dfa(),
            &Nfst::id_single(Some(Symbol::from('<'))).image_dfa(),
            &Nfst::id_single(Some(Symbol::from('>'))).image_dfa(),
        );
        let right_context = right_context(
            &alphabet_with_brackets,
            &right_pattern.image_dfa(),
            &Nfst::id_single(Some(Symbol::from('<'))).image_dfa(),
            &Nfst::id_single(Some(Symbol::from('>'))).image_dfa(),
        );
        let emm = Alphabet::new(['<', '>'].into_iter().map(Symbol::from)).id_sigma();
        let replace =
            Nfst::id_dfa(&alphabet.ignore(&alphabet.id_sigma().image_dfa(), &emm.image_dfa()))
                .concat(
                    &Nfst::id_single(Some(Symbol::from('<')))
                        .concat(&pre_change.image_cross(&post_change))
                        .concat(&Nfst::id_single(Some(Symbol::from('>'))))
                        .optional(),
                )
                .star();

        let transducer = prologue
            .compose(&Nfst::id_dfa(&right_context))
            .compose(&replace)
            .compose(&Nfst::id_dfa(&left_context))
            .compose(&prologue.inverse());

        eprintln!("made transducer");
        let result = input.compose(&transducer).image_dfa();
        eprintln!("made result");
        // eprintln!("{}", replace.graphviz());

        eprintln!("{}", result.graphviz());
        for out in result.iter().take(20) {
            eprintln!(
                "{}",
                out.into_iter()
                    .fold(String::new(), |a, b| format!("{}{}", a, b))
            );
        }
    }
}
