use crate::automaton::{Dfa, Nfst, Symbol};

/// Alphabet encapsulates an alphabet within a particular segment type.
#[derive(Clone, Debug)]
pub struct Alphabet {
    sigma: im::HashSet<Symbol>,
}

impl Alphabet {
    /// The alphanumeric alphabet, for convenience
    pub fn new_alphanum() -> Self {
        Self::new(
            "abcdefghijklmnopqrstuvwxyz1234567890"
                .chars()
                .map(Symbol::from),
        )
    }

    /// Create a new alphabet struct.
    pub fn new(segments: impl IntoIterator<Item = Symbol>) -> Self {
        Self {
            sigma: segments.into_iter().collect(),
        }
    }

    /// Add more
    pub fn insert(&mut self, c: Symbol) {
        self.sigma.insert(c);
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
}
