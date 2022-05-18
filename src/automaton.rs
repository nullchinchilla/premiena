use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt::Debug,
    fmt::Display,
    hash::Hash,
};

use genawaiter::sync::Gen;
use tap::Tap;

/// A trait that generalizes over a character / phoneme / whatever. Don't implement it; anything that *could* implement it autoimplements it already.
pub trait Segment: Debug + Eq + Ord + Clone + Hash + 'static {}

impl<T: Eq + Ord + Clone + Hash + Debug + 'static> Segment for T {}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Transition<S: Segment> {
    from_state: usize,
    to_state: usize,
    from_char: Option<S>,
    to_char: Option<S>,
}

/// A nondeterministic finite-state transducer.
pub struct Nfst<S: Segment> {
    start_idx: usize,
    transitions: BTreeSet<Transition<S>>,
    accepting: BTreeSet<usize>,
}

impl<S: Segment> Nfst<S> {
    /// Creates a new automaton that id-matches either epsilon or a single symbol.
    pub fn id_single(symb: Option<S>) -> Self {
        Self {
            start_idx: 0,
            transitions: std::iter::once(Transition {
                from_state: 0,
                to_state: 1,
                from_char: symb.clone(),
                to_char: symb,
            })
            .collect(),
            accepting: std::iter::once(1usize).collect(),
        }
    }

    /// Concatenates this automaton with another one.
    pub fn concat(&self, other: &Self) -> Self {
        let other_offset = self.greatest_stateid() + 1;
        // append their transition set into ours
        let mut new_transitions = self.transitions.clone();
        for mut trans in other.transitions.iter().cloned() {
            trans.from_state += other_offset;
            trans.to_state += other_offset;
            new_transitions.insert(trans);
        }
        for end in self.accepting.iter().copied() {
            new_transitions.insert(Transition {
                from_state: end,
                to_state: other.start_idx + other_offset,
                from_char: None,
                to_char: None,
            });
        }
        Self {
            start_idx: self.start_idx,
            accepting: other
                .accepting
                .iter()
                .cloned()
                .map(|i| i + other_offset)
                .collect(),
            transitions: new_transitions,
        }
        .simplify()
    }

    /// Unions this automaton with another one.
    pub fn union(&self, other: &Self) -> Self {
        let other_offset = self.greatest_stateid() + 1;
        // append their transition set into ours
        let mut new_transitions = self.transitions.clone();
        for mut trans in other.transitions.iter().cloned() {
            trans.from_state += other_offset;
            trans.to_state += other_offset;
            new_transitions.insert(trans);
        }
        // make two other states q and f
        let q = self.greatest_stateid() + other_offset + 1;
        let f = q + 1;
        new_transitions.insert(Transition {
            from_state: q,
            to_state: self.start_idx,
            from_char: None,
            to_char: None,
        });
        new_transitions.insert(Transition {
            from_state: q,
            to_state: other.start_idx + other_offset,
            from_char: None,
            to_char: None,
        });
        for i in self
            .accepting
            .iter()
            .cloned()
            .chain(other.accepting.iter().cloned().map(|d| d + other_offset))
        {
            new_transitions.insert(Transition {
                from_state: i,
                to_state: f,
                from_char: None,
                to_char: None,
            });
        }
        Self {
            start_idx: q,
            transitions: new_transitions,
            accepting: std::iter::once(f).collect(),
        }
        .simplify()
    }

    /// Kleene star for the automaton
    pub fn star(&self) -> Self {
        let mut new_transitions = self.transitions.clone();
        let q = self.greatest_stateid() + 1;
        let f = q + 1;
        new_transitions.insert(Transition {
            from_state: q,
            to_state: f,
            from_char: None,
            to_char: None,
        });
        new_transitions.insert(Transition {
            from_state: q,
            to_state: self.start_idx,
            from_char: None,
            to_char: None,
        });
        for end in self.accepting.iter().copied() {
            new_transitions.insert(Transition {
                from_state: end,
                to_state: f,
                from_char: None,
                to_char: None,
            });
            new_transitions.insert(Transition {
                from_state: end,
                to_state: self.start_idx,
                from_char: None,
                to_char: None,
            });
        }
        Self {
            transitions: new_transitions,
            start_idx: q,
            accepting: std::iter::once(f).collect(),
        }
        .simplify()
    }

    /// Create an NFST corresponding to the Cartesian product of the *images* of the this and that.
    pub fn image_cross(&self, other: &Self) -> Self {
        let mut ab2c = new_ab2c();
        let mut transitions = BTreeSet::new();
        let alphabet: BTreeSet<Option<S>> = self
            .alphabet()
            .into_iter()
            .chain(other.alphabet().into_iter())
            .map(|s| Some(s))
            .chain(std::iter::once(None))
            .collect();
        for x in self.states() {
            for y in other.states() {
                for a in alphabet.iter() {
                    for b in alphabet.iter() {
                        let x_tsns = self.tfun(x, a.clone());
                        let y_tsns = other.tfun(y, b.clone());
                        for x_tsn in x_tsns {
                            for y_tsn in y_tsns.clone() {
                                transitions.insert(Transition {
                                    from_state: ab2c(x, y),
                                    to_state: ab2c(x_tsn.to_state, y_tsn.to_state),
                                    from_char: a.clone(),
                                    to_char: b.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        let mut accepting = BTreeSet::new();
        for x in self.accepting.iter() {
            for y in other.accepting.iter() {
                accepting.insert(ab2c(*x, *y));
            }
        }
        Self {
            start_idx: ab2c(self.start_idx, other.start_idx),
            transitions,
            accepting,
        }
        .simplify()
    }

    fn greatest_stateid(&self) -> usize {
        self.transitions
            .iter()
            .map(|t| t.from_state.max(t.to_state))
            .max()
            .unwrap_or_default()
    }

    fn states(&self) -> BTreeSet<usize> {
        self.transitions
            .iter()
            .flat_map(|t| [t.from_state, t.to_state].into_iter())
            .collect()
    }

    fn alphabet(&self) -> BTreeSet<S> {
        self.transitions
            .iter()
            .flat_map(|d| [d.from_char.clone(), d.to_char.clone()].into_iter())
            .flatten()
            .collect()
    }

    /// Extracts a DFA that represents the regular language containing the paths of the transducer.
    pub fn paths(&self) -> Dfa<(Option<S>, Option<S>)> {
        // first, we create a NFST, so that we can use the existing method
        let temp_nfst: Nfst<(Option<S>, Option<S>)> = Nfst {
            start_idx: self.start_idx,
            transitions: self
                .transitions
                .iter()
                .cloned()
                .map(|t| {
                    let n = (t.from_char, t.to_char);
                    Transition {
                        from_state: t.from_state,
                        to_state: t.to_state,
                        from_char: Some(n.clone()),
                        to_char: Some(n.clone()),
                    }
                })
                .collect(),
            accepting: self.accepting.clone(),
        };
        temp_nfst.image_dfa()
    }

    /// Creates a new automaton that does nothing but do a single-character replacement.
    pub fn single_replace(from: S, to: Option<S>) -> Self {
        Self {
            start_idx: 0,
            transitions: std::iter::once(Transition {
                from_state: 0,
                to_state: 1,
                from_char: Some(from),
                to_char: to,
            })
            .collect(),
            accepting: BTreeSet::new().tap_mut(|s| {
                s.insert(1);
            }),
        }
    }

    /// Produces the reversed version of this automaton.
    pub fn reversed(&self) -> Self {
        let reversed_transitions = self
            .transitions
            .iter()
            .map(|t| {
                let mut t = t.clone();
                std::mem::swap(&mut t.from_char, &mut t.to_char);
                t
            })
            .collect();
        Self {
            start_idx: self.start_idx,
            transitions: reversed_transitions,
            accepting: self.accepting.clone(),
        }
    }

    /// Produces the Graphviz representation of the NFST
    pub fn graphviz(&self) -> String {
        let mut lel = String::new();
        lel.push_str("digraph G {\n");
        lel.push_str("rankdir=LR\n");
        lel.push_str("node [shape=\"circle\"]\n");
        for tsn in self.transitions.iter() {
            let from_char = tsn
                .from_char
                .as_ref()
                .map(|k| format!("{:?}", k))
                .unwrap_or_else(|| "ε".to_string());
            let to_char = tsn
                .to_char
                .as_ref()
                .map(|k| format!("{:?}", k))
                .unwrap_or_else(|| "ε".to_string());
            lel.push_str(&format!(
                "{} -> {} [label=\" {}:{} \"]\n",
                tsn.from_state, tsn.to_state, from_char, to_char
            ));
        }
        for state in self.accepting.iter() {
            lel.push_str(&format!("{} [shape=doublecircle ]\n", state));
        }
        lel.push_str(&format!("S -> {}\n", self.start_idx));
        lel.push_str("S[style=invis label=\"\"]\n");
        lel.push_str("}\n");
        lel
    }

    /// Composes this FST with another one
    pub fn compose(&self, other: &Self) -> Self {
        // function that maps a pair of state ids to the state id in the new fst
        let mut ab2c = {
            let mut tab: BTreeMap<(usize, usize), usize> = BTreeMap::new();
            let mut counter = 0;
            move |a, b| {
                *tab.entry((a, b)).or_insert_with(|| {
                    counter += 1;
                    counter - 1
                })
            }
        };
        // we go through the cartesian product of state ids
        let mut new_transitions = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut dfs_stack = vec![(self.start_idx, other.start_idx)];
        while let Some((i, j)) = dfs_stack.pop() {
            if !seen.insert((i, j)) {
                continue;
            }
            let chars: BTreeSet<Option<S>> = self
                .transitions
                .iter()
                .filter(|t| t.from_state == i)
                .map(|t| t.from_char.clone())
                .collect();
            // first handle epsilon
            for second_stage in other
                .transitions
                .iter()
                .filter(|t| t.from_char == None && t.from_state == j)
            {
                new_transitions.insert(Transition {
                    from_state: ab2c(i, j),
                    to_state: ab2c(i, second_stage.to_state),
                    from_char: None,
                    to_char: second_stage.to_char.clone(),
                });
                dfs_stack.push((i, second_stage.to_state));
            }
            for ch in chars {
                for first_stage in self
                    .transitions
                    .iter()
                    .filter(|t| t.from_state == i && t.from_char == ch)
                {
                    for second_stage in other
                        .transitions
                        .iter()
                        .filter(|t| t.from_state == j && t.from_char == first_stage.to_char)
                    {
                        new_transitions.insert(Transition {
                            from_state: ab2c(i, j),
                            to_state: ab2c(first_stage.to_state, second_stage.to_state),
                            from_char: first_stage.from_char.clone(),
                            to_char: second_stage.to_char.clone(),
                        });
                        dfs_stack.push((first_stage.to_state, second_stage.to_state));
                    }
                    if first_stage.to_char.is_none() {
                        // always implicit epsilon
                        new_transitions.insert(Transition {
                            from_state: ab2c(i, j),
                            to_state: ab2c(first_stage.to_state, j),
                            from_char: first_stage.from_char.clone(),
                            to_char: None,
                        });
                        dfs_stack.push((first_stage.to_state, j));
                    }
                }
            }
        }

        let mut accepting_states = BTreeSet::new();
        for i in self.accepting.iter() {
            for j in other.accepting.iter() {
                accepting_states.insert(ab2c(*i, *j));
            }
        }
        Nfst {
            start_idx: ab2c(self.start_idx, other.start_idx),
            transitions: new_transitions,
            accepting: accepting_states,
        }
    }

    /// Obtains the DFA corresponding to the regular language that is *produced* by this NFST.
    pub fn image_dfa(&self) -> Dfa<S> {
        // create lol
        let mut set_to_num = {
            let mut record: HashMap<BTreeSet<usize>, usize> = HashMap::new();
            let mut counter = 0;
            move |s: BTreeSet<usize>| {
                *record.entry(s).or_insert_with(|| {
                    counter += 1;
                    counter - 1
                })
            }
        };
        // subset construction
        let mut search_queue: Vec<BTreeSet<usize>> =
            vec![std::iter::once(self.start_idx).collect()];
        let mut dfa_table = BTreeMap::new();
        let mut dfa_states = BTreeSet::new();
        let mut dfa_accepting = BTreeSet::new();
        while let Some(top) = search_queue.pop() {
            let top = self.image_epsilon_closure(top);
            let dfa_num = set_to_num(top.clone());
            if dfa_states.contains(&dfa_num) {
                continue;
            }
            if top.iter().any(|t| self.accepting.contains(t)) {
                dfa_accepting.insert(dfa_num);
            }
            dfa_states.insert(dfa_num);
            // find all outgoing chars
            let outgoing_chars: BTreeSet<S> = self
                .transitions
                .iter()
                .filter(|t| top.contains(&t.from_state))
                .filter_map(|t| t.to_char.clone())
                .collect();
            for ch in outgoing_chars {
                let resulting_state: BTreeSet<usize> = self.image_epsilon_closure(
                    top.iter()
                        .flat_map(|s| {
                            self.transitions
                                .iter()
                                .filter(|t| t.from_state == *s && t.to_char == Some(ch.clone()))
                                .map(|t| t.to_state)
                        })
                        .collect(),
                );
                dfa_table.insert((dfa_num, ch.clone()), set_to_num(resulting_state.clone()));
                search_queue.push(resulting_state);
            }
        }
        Dfa {
            start: set_to_num(
                self.image_epsilon_closure(std::iter::once(self.start_idx).collect()),
            ),
            accepting: dfa_accepting,
            table: dfa_table,
        }
    }

    /// Simplifies the NFST.
    pub fn simplify(&self) -> Self {
        let mut set_to_num = {
            let mut record: HashMap<BTreeSet<usize>, usize> = HashMap::new();
            let mut counter = 0;
            move |s: BTreeSet<usize>| {
                *record.entry(s).or_insert_with(|| {
                    counter += 1;
                    counter - 1
                })
            }
        };
        let mut new_table = BTreeSet::new();
        let mut search_queue: Vec<BTreeSet<usize>> =
            vec![std::iter::once(self.start_idx).collect()];
        let mut seen = BTreeSet::new();
        let mut accepting = BTreeSet::new();
        while let Some(top) = search_queue.pop() {
            let top = self.double_epsilon_closure(top);
            let top_no = set_to_num(top.clone());
            if !seen.insert(top_no) {
                continue;
            }
            if top.iter().any(|t| self.accepting.contains(t)) {
                accepting.insert(top_no);
            }
            for state in top {
                for edge in self.transitions.iter().filter(|t| {
                    t.from_state == state && !(t.from_char == None && t.to_char == None)
                }) {
                    let to = self.double_epsilon_closure(BTreeSet::new().tap_mut(|s| {
                        s.insert(edge.to_state);
                    }));
                    let mut edge = edge.clone();
                    edge.from_state = top_no;
                    edge.to_state = set_to_num(to.clone());
                    new_table.insert(edge);
                    search_queue.push(to);
                }
            }
        }
        Self {
            transitions: new_table,
            start_idx: set_to_num(self.double_epsilon_closure(BTreeSet::new().tap_mut(|s| {
                s.insert(self.start_idx);
            }))),
            accepting,
        }
    }

    /// Finds the epsilon-closure of the given set of states.
    fn double_epsilon_closure(&self, mut states: BTreeSet<usize>) -> BTreeSet<usize> {
        let mut ss = states.iter().copied().collect::<Vec<_>>();
        while let Some(top) = ss.pop() {
            let epsilon_neighs: BTreeSet<usize> = self
                .tfun(top, None)
                .into_iter()
                .filter(|t| t.to_char == None && t.from_char == None)
                .map(|t| t.to_state)
                .collect();
            for n in epsilon_neighs {
                if states.insert(n) {
                    ss.push(n);
                }
            }
        }
        states
    }

    /// Finds the *image* epsilon-closure of the given set of states.
    fn image_epsilon_closure(&self, mut states: BTreeSet<usize>) -> BTreeSet<usize> {
        let mut ss = states.iter().copied().collect::<Vec<_>>();
        while let Some(top) = ss.pop() {
            let epsilon_neighs: BTreeSet<usize> = self
                .tfun(top, None)
                .into_iter()
                .map(|t| t.to_state)
                .collect();
            for n in epsilon_neighs {
                if states.insert(n) {
                    ss.push(n);
                }
            }
        }
        states
    }

    /// Applies the transition function.
    fn tfun(&self, s: usize, c: Option<S>) -> BTreeSet<Transition<S>> {
        let mut v: BTreeSet<Transition<S>> = self
            .transitions
            .iter()
            .filter(|t| t.from_state == s && t.to_char == c)
            .cloned()
            .collect();
        if c.is_none() {
            v.insert(Transition {
                from_state: s,
                to_state: s,
                from_char: None,
                to_char: None,
            });
        }
        v
    }

    /// Applies the transition function partially
    fn tfun_partial(&self, s: usize) -> BTreeSet<Transition<S>> {
        self.transitions
            .iter()
            .filter(|t| t.from_state == s)
            .cloned()
            .collect()
    }
}

/// A deterministic finite-state automaton
pub struct Dfa<S: Segment> {
    start: usize,
    accepting: BTreeSet<usize>,
    table: BTreeMap<(usize, S), usize>,
}

impl<S: Segment> Dfa<S> {
    /// Creates the graphviz representation of the DFA.
    pub fn graphviz(&self) -> String {
        let mut lel = String::new();
        lel.push_str("digraph G {\n");
        lel.push_str("rankdir=LR\n");
        lel.push_str("node [shape=\"circle\"]\n");
        for ((from, c), to) in self.table.iter() {
            lel.push_str(&format!("{} -> {} [label=\"{:?}\"]\n", from, to, c));
        }
        for state in self.accepting.iter() {
            lel.push_str(&format!("{} [shape=doublecircle ]\n", state));
        }
        lel.push_str(&format!("S -> {}\n", self.start));
        lel.push_str("S[style=invis label=\"\"]\n");
        lel.push_str("}\n");
        lel
    }

    /// Iterates through the regular language described by this DFA.
    pub fn iter(&self) -> impl Iterator<Item = Vec<S>> + '_ {
        let mut bfs_queue = VecDeque::new();
        bfs_queue.push_back((self.start, vec![]));
        let generator = Gen::new(|co| async move {
            if self.accepting.is_empty() {
                return;
            }
            while let Some((state, word_so_far)) = bfs_queue.pop_front() {
                if self.accepting.contains(&state) {
                    co.yield_(word_so_far.clone()).await;
                }
                // explore further
                for (next_ch, next_state) in self
                    .table
                    .iter()
                    .filter(|(t, u)| t.0 == state)
                    .map(|(t, u)| (t.1.clone(), *u))
                {
                    bfs_queue
                        .push_back((next_state, word_so_far.clone().tap_mut(|w| w.push(next_ch))));
                }
            }
        });
        generator.into_iter()
    }
}

fn new_ab2c() -> impl FnMut(usize, usize) -> usize {
    let mut tab: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut counter = 0;
    move |a, b| {
        *tab.entry((a, b)).or_insert_with(|| {
            counter += 1;
            counter - 1
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thompson() {
        let bstar = Nfst::id_single(Some('b'));
        let nfst = Nfst::id_single(Some('a')).image_cross(&bstar);
        eprintln!("{}", nfst.graphviz());
    }
}
