mod table;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt::Display,
    num::NonZeroU64,
};

use genawaiter::sync::Gen;
use smallvec::SmallVec;
use tap::Tap;

use self::table::{Table, Transition};

/// An opaque symbol, normally representing a Unicode codepoint, but perhaps other things too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(NonZeroU64);

impl From<char> for Symbol {
    fn from(c: char) -> Self {
        let u: u64 = c.into();
        Symbol(NonZeroU64::new(u + 1).unwrap())
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ch = {
            let u: u64 = self.0.into();
            char::from_u32((u - 1) as u32)
        };
        if let Some(ch) = ch {
            ch.fmt(f)
        } else {
            let u: u64 = self.0.into();
            format!("{:#x}", u - 1).fmt(f)
        }
    }
}

impl Symbol {
    fn invalid() -> Self {
        Self(NonZeroU64::new(123456789).unwrap())
    }

    /// Create a symbol from two other symbols
    pub fn from_pair(i: Self, j: Self) -> Self {
        // bit-interleaving
        let mut ret = 0u64;
        let mut i: u64 = i.0.into();
        let mut j: u64 = j.0.into();
        for ctr in 0..64 {
            ret <<= 1;
            if ctr % 2 == 0 {
                ret |= i & 1;
                i >>= 1;
            } else {
                ret |= j & 1;
                j >>= 1;
            }
        }
        Self(ret.reverse_bits().try_into().unwrap())
    }

    /// Decodes as pair
    pub fn decode_as_pair(self) -> (Self, Self) {
        let mut this: u64 = self.0.into();
        let mut i = 0u64;
        let mut j = 0u64;
        for ctr in 0..126 {
            if ctr % 2 == 0 {
                i |= this & 1;
                i <<= 1;
            } else {
                j |= this & 1;
                j <<= 1;
            }
            this >>= 1;
        }
        (
            Self(i.reverse_bits().try_into().unwrap()),
            Self(j.reverse_bits().try_into().unwrap()),
        )
    }
}

/// A nondeterministic finite-state transducer.
#[derive(Clone)]
pub struct Nfst {
    start_idx: usize,
    transitions: Table,
    accepting: im::HashSet<usize>,
}

impl Nfst {
    /// Creates an autommaton that literally matches nothing.
    pub fn nothing() -> Self {
        Self {
            start_idx: 0,
            transitions: Table::new(),
            accepting: Default::default(),
        }
    }

    /// Creates a new automaton that id-matches either epsilon or a single symbol.
    pub fn id_single(symb: Option<Symbol>) -> Self {
        Self {
            start_idx: 0,
            transitions: std::iter::once(Transition {
                from_state: 0,
                to_state: 1,
                from_char: symb,
                to_char: symb,
            })
            .collect(),
            accepting: std::iter::once(1usize).collect(),
        }
    }

    /// Create from paths
    pub fn id_dfa(dfa: &Dfa) -> Self {
        Self {
            start_idx: dfa.start,
            transitions: dfa
                .table
                .iter()
                .map(|trans| Transition {
                    from_char: trans.from_char,
                    to_char: trans.from_char,
                    from_state: trans.from_state,
                    to_state: trans.to_state,
                })
                .collect(),
            accepting: dfa.accepting.clone(),
        }
    }

    /// Concatenates this automaton with another one.
    pub fn concat(&self, other: &Self) -> Self {
        let other_offset = self.transitions.next_free_stateid();
        // append their transition set into ours
        let mut new_transitions = self.transitions.clone();
        for mut trans in other.transitions.iter() {
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
        .eliminate_double_epsilon()
    }

    /// Unions this automaton with another one.
    pub fn union(&self, other: &Self) -> Self {
        if self.accepting.is_empty() {
            return other.clone();
        }
        if other.accepting.is_empty() {
            return self.clone();
        }

        let other_offset = self.transitions.next_free_stateid();
        // append their transition set into ours
        let mut new_transitions = self.transitions.clone();
        for mut trans in other.transitions.iter() {
            trans.from_state += other_offset;
            trans.to_state += other_offset;
            new_transitions.insert(trans);
        }
        // make two other states q and f
        let q = self.transitions.next_free_stateid() + other_offset + 1;
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
        .eliminate_double_epsilon()
    }

    /// Convenience function for optionality
    pub fn optional(&self) -> Self {
        self.union(&Self::id_single(None))
    }

    /// Kleene star for the automaton
    pub fn star(&self) -> Self {
        let mut new_transitions = self.transitions.clone();
        let q = self.transitions.next_free_stateid();
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
        .eliminate_double_epsilon()
    }

    /// Intersection of the two transducers. Only works if both are same-length transducers.
    pub fn samelen_intersect(&self, other: &Self) -> Self {
        let a = self.eliminate_double_epsilon().samelen_eliminate_epsilon();
        let b = other.eliminate_double_epsilon().samelen_eliminate_epsilon();
        let dfa = a.paths().intersect(&b.paths());
        Self::from_paths(&dfa)
    }

    /// Create an NFST corresponding to the Cartesian product of the *images* of the this and that.
    pub fn image_cross(&self, other: &Self) -> Self {
        let mut ab2c = new_ab2c();
        let mut transitions = Table::new();
        let alphabet: BTreeSet<Option<Symbol>> = self
            .alphabet()
            .into_iter()
            .chain(other.alphabet().into_iter())
            .map(Some)
            .chain(std::iter::once(None))
            .collect();
        for x in self.transitions.states() {
            for y in other.transitions.states() {
                for a in alphabet.iter() {
                    for b in alphabet.iter() {
                        let x_tsns = self.transitions.transition(x, *a);
                        let y_tsns = other.transitions.transition(y, *b);
                        for x_tsn in x_tsns {
                            for y_tsn in y_tsns.clone() {
                                transitions.insert(Transition {
                                    from_state: ab2c(x, y),
                                    to_state: ab2c(x_tsn.1, y_tsn.1),
                                    from_char: *a,
                                    to_char: *b,
                                });
                            }
                        }
                    }
                }
            }
        }

        let mut accepting = im::HashSet::new();
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
    }

    /// Extracts a DFA that represents the regular language containing the paths of the transducer.
    pub fn paths(&self) -> Dfa {
        // first, we create a NFST, so that we can use the existing method
        let temp_nfst: Nfst = Nfst {
            start_idx: self.start_idx,
            transitions: self
                .transitions
                .iter()
                .map(|t| {
                    let n: Symbol = Symbol::from_pair(
                        t.from_char.unwrap_or_else(Symbol::invalid),
                        t.to_char.unwrap_or_else(Symbol::invalid),
                    );
                    Transition {
                        from_state: t.from_state,
                        to_state: t.to_state,
                        from_char: Some(n),
                        to_char: Some(n),
                    }
                })
                .collect(),
            accepting: self.accepting.clone(),
        };
        temp_nfst.image_dfa()
    }

    /// Create from a path-language DFA.
    pub fn from_paths(paths: &Dfa) -> Self {
        Self {
            start_idx: paths.start,
            transitions: paths
                .table
                .iter()
                .map(|trans| {
                    let (from_char, to_char): (Symbol, Symbol) =
                        trans.from_char.unwrap().decode_as_pair();
                    Transition {
                        from_char: if from_char == Symbol::invalid() {
                            None
                        } else {
                            Some(from_char)
                        },
                        to_char: if to_char == Symbol::invalid() {
                            None
                        } else {
                            Some(to_char)
                        },
                        from_state: trans.from_state,
                        to_state: trans.to_state,
                    }
                })
                .collect(),
            accepting: paths.accepting.clone(),
        }
    }

    /// Simplifies this NFST.
    pub fn simplify(&self) -> Self {
        Self::from_paths(&self.paths())
    }

    /// Produces the reversed version of this automaton.
    pub fn inverse(&self) -> Self {
        let reversed_transitions = self
            .transitions
            .iter()
            .map(|mut t| {
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
                .map(|k| format!("{}", k))
                .unwrap_or_else(|| "ε".to_string());
            let to_char = tsn
                .to_char
                .as_ref()
                .map(|k| format!("{}", k))
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

        // // label with imbalance
        // for (k, v) in self.epsilon_imbalances() {
        //     lel.push_str(&format!("{} [label=\"{}\"]\n", k, v));
        // }

        lel.push_str("}\n");
        lel
    }

    /// Composes this FST with another one
    pub fn compose(&self, other: &Self) -> Self {
        // function that maps a pair of state ids to the state id in the new fst
        let mut ab2c = new_ab2c();
        // we go through the cartesian product of state ids
        let mut new_transitions = Table::new();
        let mut seen = BTreeSet::new();
        let mut dfs_stack = vec![(self.start_idx, other.start_idx)];
        while let Some((i, j)) = dfs_stack.pop() {
            // dbg!(dfs_stack.len());
            if !seen.insert((i, j)) {
                continue;
            }
            // eprintln!("going through ({},{})", i, j);
            let chars: BTreeSet<Option<Symbol>> = self
                .transitions
                .outgoing_edges(i)
                .into_iter()
                .map(|t| t.from_char)
                .collect();
            // first handle epsilon
            for second_stage in other.transitions.transition(j, None) {
                new_transitions.insert(Transition {
                    from_state: ab2c(i, j),
                    to_state: ab2c(i, second_stage.1),
                    from_char: None,
                    to_char: second_stage.0,
                });
                dfs_stack.push((i, second_stage.1));
            }
            for ch in chars {
                for first_stage in self.transitions.transition(i, ch) {
                    for second_stage in other.transitions.transition(j, first_stage.0) {
                        new_transitions.insert(Transition {
                            from_state: ab2c(i, j),
                            to_state: ab2c(first_stage.1, second_stage.1),
                            from_char: ch,
                            to_char: second_stage.0,
                        });
                        dfs_stack.push((first_stage.1, second_stage.1));
                    }
                    if first_stage.0.is_none() {
                        // always implicit epsilon
                        new_transitions.insert(Transition {
                            from_state: ab2c(i, j),
                            to_state: ab2c(first_stage.1, j),
                            from_char: ch,
                            to_char: None,
                        });
                        dfs_stack.push((first_stage.1, j));
                    }
                }
            }
        }

        let mut accepting_states = im::HashSet::new();
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
    pub fn image_dfa(&self) -> Dfa {
        self.image_dfa_unminimized().minimize()
    }

    fn image_dfa_unminimized(&self) -> Dfa {
        // create lol
        let mut set_to_num = {
            let mut record = HashMap::new();
            let mut counter = 0;
            move |mut s: SmallVec<[usize; 4]>| {
                s.sort_unstable();
                *record.entry(s).or_insert_with(|| {
                    counter += 1;
                    counter - 1
                })
            }
        };
        // subset construction
        let mut search_queue: Vec<im::HashSet<usize>> =
            vec![std::iter::once(self.start_idx).collect()];
        let mut dfa_table = Table::new();
        let mut dfa_states = BTreeSet::new();
        let mut dfa_accepting = im::HashSet::new();

        let mut itercount = 0;

        while let Some(top) = search_queue.pop() {
            itercount += 1;
            let top = top.into_iter().fold(im::HashSet::new(), |a, top| {
                a.union(self.transitions.edge_closure(top, |t| t.to_char.is_none()))
            });
            let dfa_num = set_to_num(top.iter().copied().collect());
            if dfa_states.contains(&dfa_num) {
                continue;
            }
            if top.iter().any(|t| self.accepting.contains(t)) {
                dfa_accepting.insert(dfa_num);
            }
            dfa_states.insert(dfa_num);
            // find all outgoing chars
            let outgoing_chars = top.iter().copied().flat_map(|c| {
                self.transitions
                    .outgoing_edges(c)
                    .into_iter()
                    .filter_map(|e| e.to_char)
            });
            for ch in outgoing_chars {
                let le_next: BTreeSet<usize> = top
                    .iter()
                    .flat_map(|s| {
                        self.transitions
                            .outgoing_edges(*s)
                            .into_iter()
                            .filter(|t| t.to_char == Some(ch))
                            .map(|t| t.to_state)
                        // self.transitions
                        //     .iter()
                        //     .filter(|t| t.from_state == *s && t.to_char == Some(ch.clone()))
                        //     .map(|t| t.to_state)
                    })
                    .collect();
                let resulting_state: im::HashSet<usize> =
                    le_next.into_iter().fold(im::HashSet::new(), |a, b| {
                        a.union(self.transitions.edge_closure(b, |e| e.to_char.is_none()))
                    });
                let resulting_state_num = set_to_num(resulting_state.iter().copied().collect());
                if !dfa_states.contains(&resulting_state_num) {
                    dfa_table.insert(Transition {
                        from_state: dfa_num,
                        to_state: resulting_state_num,
                        from_char: Some(ch),
                        to_char: None,
                    });
                    search_queue.push(resulting_state);
                }
            }
        }

        log::trace!(
            "powerset {} => {} in {} iterations",
            self.transitions.states().len(),
            dfa_table.states().len(),
            itercount
        );
        Dfa {
            start: set_to_num(
                self.transitions
                    .edge_closure(self.start_idx, |e| e.to_char.is_none())
                    .into_iter()
                    .collect(),
            ),
            accepting: dfa_accepting,
            table: dfa_table,
        }
        // .minimize()
        .eliminate_unreachable()
    }

    /// Simplifies the NFST by eliminating all epsilon:epsilon transitions
    fn eliminate_double_epsilon(&self) -> Self {
        let dubeps = |t: &Transition| t.from_char == None && t.to_char == None;
        let mut new_table = Table::new();
        let mut search_queue: Vec<usize> = vec![self.start_idx];
        let mut seen = BTreeSet::new();
        let mut accepting = im::HashSet::new();
        let mut set2num = new_set2num();
        while let Some(top) = search_queue.pop() {
            let top = self.transitions.edge_closure(top, dubeps);
            let top_no = set2num(top.clone());
            if !seen.insert(top_no) {
                continue;
            }

            if top.iter().any(|t| self.accepting.contains(t)) {
                accepting.insert(top_no);
            }
            for state in top.iter().copied() {
                for edge in self.transitions.outgoing_edges(state) {
                    if !dubeps(&edge) {
                        search_queue.push(edge.to_state);
                        let to = set2num(self.transitions.edge_closure(edge.to_state, dubeps));
                        let mut edge = edge.clone();
                        edge.from_state = top_no;
                        edge.to_state = to;
                        new_table.insert(edge);
                    }
                }
            }
        }

        Self {
            transitions: new_table,
            start_idx: set2num(self.transitions.edge_closure(self.start_idx, dubeps)),
            accepting,
        }
    }

    /// Simplifies the NFST by eliminating epsilon-something-else transitions. Does NOT work on NFSTs that don't represent same-length mappings!
    fn samelen_eliminate_epsilon(&self) -> Self {
        self.eliminate_epsilon_half()
            .inverse()
            .eliminate_epsilon_half()
            .inverse()
    }

    fn eliminate_epsilon_half(&self) -> Self {
        let mut ptr = self.clone();
        loop {
            let imbalances = ptr.epsilon_imbalances();
            // find the most imbalanced one ever
            let max_imbalance = imbalances.values().max().copied().unwrap_or_default();
            if max_imbalance == 0 {
                return ptr;
            }
            // first, remove every single p:q transition between two nodes of the max imbalance,
            // replacing it with a p:epsilon and epsilon:q transition
            let mut next_stateno = ptr.transitions.next_free_stateid();
            let mut next_transitions = Table::new();
            for transition in ptr.transitions.iter() {
                if imbalances[&transition.from_state] == max_imbalance
                    && imbalances[&transition.to_state] == max_imbalance
                    && transition.to_state != transition.from_state
                {
                    let new_stateno = next_stateno;
                    next_stateno += 1;
                    next_transitions.insert(Transition {
                        from_state: transition.from_state,
                        to_state: new_stateno,
                        from_char: transition.from_char,
                        to_char: None,
                    });
                    next_transitions.insert(Transition {
                        from_state: new_stateno,
                        to_state: transition.to_state,
                        from_char: None,
                        to_char: transition.to_char,
                    });
                } else {
                    next_transitions.insert(transition.clone());
                }
            }
            for max_imbalance_state in imbalances
                .iter()
                .filter(|p| *p.1 == max_imbalance)
                .map(|p| p.0)
                .copied()
            {
                // for all incoming-outgoing pairs, just connect them directly. then delete this state
                let incoming = next_transitions.incoming_edges(max_imbalance_state);
                let outgoing = next_transitions.outgoing_edges(max_imbalance_state);
                for incoming in incoming {
                    for outgoing in outgoing.iter().cloned() {
                        next_transitions.insert(Transition {
                            from_state: incoming.from_state,
                            to_state: outgoing.to_state,
                            to_char: incoming.to_char,
                            from_char: outgoing.from_char,
                        });
                    }
                }
                next_transitions.retain(|tx| {
                    tx.from_state != max_imbalance_state && tx.to_state != max_imbalance_state
                });
            }
            ptr = Self {
                transitions: next_transitions,
                start_idx: self.start_idx,
                accepting: self.accepting.clone(),
            }
        }
    }

    /// Finds a mapping between the state and the "imbalance"
    fn epsilon_imbalances(&self) -> BTreeMap<usize, i32> {
        let mut toret = BTreeMap::new();
        let mut dfs_stack = vec![(self.start_idx, 0, im::HashSet::new())];
        while let Some((top, top_imbalance, history)) = dfs_stack.pop() {
            toret.insert(top, top_imbalance);
            let children = self.transitions.outgoing_edges(top);
            for child in children {
                let mut history = history.clone();
                if history.insert(child.to_state).is_some() {
                    log::warn!("skipping cyclic link {} -> {}", top, child.to_state);
                } else {
                    let child_imbalance = match (&child.from_char, &child.to_char) {
                        (Some(_), None) => top_imbalance - 1,
                        (None, Some(_)) => top_imbalance + 1,
                        _ => top_imbalance,
                    };
                    dfs_stack.push((child.to_state, child_imbalance, history));
                }
            }
        }
        toret
    }

    fn alphabet(&self) -> BTreeSet<Symbol> {
        self.transitions
            .iter()
            .flat_map(|d| [d.from_char, d.to_char].into_iter())
            .flatten()
            .collect()
    }
}

/// A deterministic finite-state automaton
#[derive(Clone)]
pub struct Dfa {
    start: usize,
    accepting: im::HashSet<usize>,
    table: Table,
}

impl Dfa {
    /// Creates the graphviz representation of the DFA.
    pub fn graphviz(&self) -> String {
        let mut lel = String::new();
        lel.push_str("digraph G {\n");
        lel.push_str("rankdir=LR\n");
        lel.push_str("node [shape=\"circle\"]\n");
        for trans in self.table.iter() {
            lel.push_str(&format!(
                "{} -> {} [label=\"{}\"]\n",
                trans.from_state,
                trans.to_state,
                trans.from_char.unwrap()
            ));
        }
        for state in self.accepting.iter() {
            lel.push_str(&format!("{} [shape=doublecircle ]\n", state));
        }
        lel.push_str(&format!("S -> {}\n", self.start));
        lel.push_str("S[style=invis label=\"\"]\n");
        lel.push_str("}\n");
        lel
    }

    /// Intersection with another DFA.
    pub fn intersect(&self, other: &Self) -> Self {
        let mut new_table = Table::new();
        let mut ab2c = new_ab2c();
        for atrans in self.table.iter() {
            for btrans in other.table.iter() {
                if atrans.from_char == btrans.from_char {
                    new_table.insert(Transition {
                        from_state: ab2c(atrans.from_state, btrans.from_state),
                        to_state: ab2c(atrans.to_state, btrans.to_state),
                        from_char: atrans.from_char,
                        to_char: None,
                    })
                }
            }
        }
        let mut new_accepting = im::HashSet::new();
        for a_accept in self.accepting.iter() {
            for b_accept in other.accepting.iter() {
                new_accepting.insert(ab2c(*a_accept, *b_accept));
            }
        }
        Self {
            start: ab2c(self.start, other.start),
            table: new_table,
            accepting: new_accepting,
        }
    }

    /// Concatenates this DFA with another (by using an NFA/NFST repr as an intermediary)
    pub fn concat(&self, other: &Self) -> Self {
        Nfst::id_dfa(self).concat(&Nfst::id_dfa(other)).image_dfa()
        // todo!()
    }

    /// Complement of this DFA.
    pub fn complement(&self, alphabet: &im::HashSet<Symbol>) -> Self {
        let mut new = self.clone();
        // first, we make sure that every state has every possible output from it in the alphabet
        let junk_state = new.table.next_free_stateid();
        for state in new
            .table
            .states()
            .into_iter()
            .chain(std::iter::once(junk_state))
        {
            for ch in alphabet.iter() {
                if new.table.transition(state, Some(*ch)).is_empty() {
                    new.table.insert(Transition {
                        from_state: state,
                        to_state: junk_state,
                        from_char: (*ch).into(),
                        to_char: None,
                    });
                }
            }
        }
        // invert the accepting states
        let new_accepting = new
            .table
            .states()
            .into_iter()
            .filter(|k| !self.accepting.contains(k))
            .collect();
        new.accepting = new_accepting;
        new.minimize()
    }

    /// Eliminates everything that is unreachable
    fn eliminate_unreachable(mut self) -> Self {
        let reachable = self.table.edge_closure(self.start, |_| true);
        self.table.retain(|k| reachable.contains(&k.to_state));
        self.accepting.retain(|k| reachable.contains(k));
        self
    }

    /// Iterates through the regular language described by this DFA.
    pub fn iter(&self) -> impl Iterator<Item = Vec<Symbol>> + '_ {
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
                for trans in self.table.outgoing_edges(state) {
                    bfs_queue.push_back((
                        trans.to_state,
                        word_so_far
                            .clone()
                            .tap_mut(|w| w.push(trans.from_char.unwrap())),
                    ));
                }
            }
        });
        generator.into_iter()
    }

    /// Reverses the DFA.
    pub fn reverse(&self) -> Self {
        let mut nfa_table: Table = self
            .table
            .iter()
            .map(|trans| Transition {
                from_state: trans.to_state,
                to_state: trans.from_state,
                from_char: trans.from_char,
                to_char: trans.from_char,
            })
            .collect();
        let new_start = self.table.next_free_stateid();
        for accept in self.accepting.iter() {
            nfa_table.insert(Transition {
                from_state: new_start,
                to_state: *accept,
                from_char: None,
                to_char: None,
            })
        }
        let reverse_id_nfst = Nfst {
            transitions: nfa_table,
            start_idx: new_start,
            accepting: std::iter::once(self.start).collect(),
        };
        reverse_id_nfst.image_dfa_unminimized()
    }

    /// Minimizes the DFA.
    pub fn minimize(&self) -> Self {
        self.reverse().reverse()
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

fn new_set2num() -> impl FnMut(im::HashSet<usize>) -> usize {
    let mut tab = BTreeMap::new();
    let mut counter = 0;
    move |a| {
        *tab.entry(a.into_iter().collect::<BTreeSet<_>>())
            .or_insert_with(|| {
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
        let bee = Nfst::id_single(Some('b'.into()));
        let ayy = Nfst::id_single(Some('a'.into()));
        let ayybee = ayy.union(&bee);
        let aa_to_starstar = ayy.concat(&ayy).image_cross(&ayybee.concat(&ayybee));
        let starstar_to_bb = ayybee.concat(&ayybee).image_cross(&bee.concat(&bee));

        let lel = aa_to_starstar.star().union(&starstar_to_bb.star());

        eprintln!("{}", lel.graphviz());
    }

    #[test]
    fn symbol() {
        let i = Symbol::from('b');
        let j = Symbol::from('中');
        let k = Symbol::from_pair(i, j);
        eprintln!("{} {} {}", i, j, k);
        eprintln!("i = {:#b}, j = {:#b}, k = {:#b}", i.0, j.0, k.0);
        let (r, v) = k.decode_as_pair();
        eprintln!("r = {:#b}, v = {:#b}", r.0, v.0);
        assert_eq!(i, r);
        assert_eq!(j, v);
    }
}
