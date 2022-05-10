use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Display,
    thread::current,
};

/// A trait that generalizes over a character / phoneme / whatever. Don't implement it; anything that *could* implement it autoimplements it already.
pub trait Segment: Display + Eq + Ord + Clone + 'static {}

impl<T: Display + Eq + Ord + Clone + 'static> Segment for T {}

/// An automaton that transforms one regular language to another.
pub struct Automaton<S: Segment> {
    start_idx: usize,
    transitions: Vec<BTreeMap<Option<S>, BTreeMap<usize, Option<S>>>>,
}

impl<S: Segment> Automaton<S> {
    /// Creates a new automaton that does nothing but do a single-character replacements.
    pub fn single_replace(from: S, to: Option<S>, alphabet: &[S]) -> Self {
        let mut mapping = BTreeMap::new();
        for alph in alphabet {
            let mut inner = BTreeMap::new();
            inner.insert(0, Some(alph.clone()));
            mapping.insert(Some(alph.clone()), inner);
        }
        let mut inner = BTreeMap::new();
        inner.insert(0, to);
        mapping.insert(Some(from), inner);
        Self {
            start_idx: 0,
            transitions: vec![mapping],
        }
    }

    /// Transduces the given string.
    pub fn transduce(&self, i: &[S]) -> Vec<Vec<S>> {
        let mut current_states: BTreeMap<usize, im::Vector<S>> =
            std::iter::once((self.start_idx, im::Vector::new())).collect();
        for ch in i.iter() {
            // expand to the epsilon closure
            let epsilon_closure = current_states
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .filter_map(|(state, word)| {
                    let nexts = self.transitions[state].get(&None)?;
                    Some(nexts.iter().map(move |(next, letter)| {
                        let mut word = word.clone();
                        if let Some(letter) = letter {
                            word.push_back(letter.clone());
                        }
                        (*next, word)
                    }))
                })
                .flatten()
                .collect::<BTreeMap<_, _>>();
            if !epsilon_closure.is_empty() {
                todo!("cannot deal with epsilon transductions yet")
            }
            // go through the states
            let new_states = current_states
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .filter_map(|(state, word)| {
                    let next_states = self.transitions[state].get(&Some(ch.clone()))?;
                    Some(next_states.iter().map(move |(next_state, next_char)| {
                        let mut word = word.clone();
                        if let Some(char) = next_char {
                            word.push_back(char.clone())
                        }
                        (*next_state, word)
                    }))
                })
                .flatten();
            current_states = new_states.collect();
        }
        current_states
            .into_iter()
            .map(|(_, word)| word.into_iter().collect())
            .collect()
    }

    /// Produces the Graphviz representation of the NFST
    pub fn graphviz(&self) -> String {
        let mut lel = String::new();
        lel.push_str("digraph G {\n");
        for (state, table) in self.transitions.iter().enumerate() {
            lel.push_str(&format!("{}\n", state));
            for (k, v) in table.iter() {
                for (next_state, rewrite_to) in v.iter() {
                    let k = k
                        .as_ref()
                        .map(|k| k.to_string())
                        .unwrap_or_else(|| "ε".to_string());
                    let rewrite_to = rewrite_to
                        .as_ref()
                        .map(|k| k.to_string())
                        .unwrap_or_else(|| "ε".to_string());
                    lel.push_str(&format!(
                        "{} -> {} [label=\"{}:{}\"]\n",
                        state, next_state, k, rewrite_to
                    ));
                }
            }
        }
        lel.push_str("}\n");
        lel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playground() {
        let alphabet = ['G', 'T', 'A', 'C'];
        let auto = Automaton::single_replace('T', Some('A'), &alphabet);
        eprintln!("{}", auto.graphviz());
        eprintln!("{:?}", auto.transduce(&['T', 'A', 'A', 'T', 'G', 'C']));
    }
}
