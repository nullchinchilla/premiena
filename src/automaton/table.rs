use tap::Tap;

use super::Symbol;

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transition {
    pub from_state: usize,
    pub to_state: usize,
    pub from_char: Option<Symbol>,
    pub to_char: Option<Symbol>,
}

/// A persistent/copy-on-write NFST transition table with good asymptotic performance, supporting multiple kinds of queries.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct Table {
    // state -> (char -> setof outchar, state)
    forwards: im::HashMap<usize, im::HashMap<Option<Symbol>, im::HashSet<(Option<Symbol>, usize)>>>,
    backwards:
        im::HashMap<usize, im::HashMap<Option<Symbol>, im::HashSet<(Option<Symbol>, usize)>>>,
}

impl Table {
    /// Create an empty new table.
    pub fn new() -> Self {
        Self {
            forwards: im::HashMap::new(),
            backwards: im::HashMap::new(),
        }
    }

    /// All the states.
    pub fn states(&self) -> im::HashSet<usize> {
        self.forwards
            .keys()
            .chain(self.backwards.keys())
            .copied()
            .collect()
    }

    /// Next free stateid
    pub fn next_free_stateid(&self) -> usize {
        self.states().into_iter().max().unwrap_or_default() + 1
    }

    /// Closure, given a particular predicate
    pub fn edge_closure(
        &self,
        start: usize,
        pred: impl Fn(&Transition) -> bool,
    ) -> im::HashSet<usize> {
        let mut group = im::HashSet::new();
        let mut dfa_stack = vec![start];
        while let Some(top) = dfa_stack.pop() {
            group.insert(top);
            let neighs = self.outgoing_edges(top);
            for neigh in neighs.into_iter().filter(&pred) {
                if group.insert(neigh.to_state).is_none() {
                    dfa_stack.push(neigh.to_state);
                }
            }
        }
        group
    }

    /// Inserts one transition.
    pub fn insert(&mut self, transition: Transition) {
        self.forwards
            .entry(transition.from_state)
            .or_default()
            .entry(transition.from_char)
            .or_default()
            .insert((transition.to_char, transition.to_state));
        self.backwards
            .entry(transition.to_state)
            .or_default()
            .entry(transition.to_char)
            .or_default()
            .insert((transition.from_char, transition.from_state));
    }

    /// The basic transition function.
    pub fn transition(
        &self,
        state: usize,
        input: impl Into<Option<Symbol>>,
    ) -> im::HashSet<(Option<Symbol>, usize)> {
        let input: Option<Symbol> = input.into();
        let mut res = if let Some(val) = self.forwards.get(&state) {
            if let Some(val) = val.get(&input) {
                val.clone()
            } else {
                Default::default()
            }
        } else {
            Default::default()
        };
        if input.is_none() {
            res.insert((None, state));
        }
        res
    }

    /// Gets all the edges going out of one node
    pub fn outgoing_edges(&self, state: usize) -> Vec<Transition> {
        if let Some(val) = self.forwards.get(&state) {
            val.iter()
                .flat_map(|p| {
                    p.1.iter().map(|q| Transition {
                        from_state: state,
                        to_state: q.1,
                        from_char: *p.0,
                        to_char: q.0,
                    })
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Gets all the edges going into one node
    pub fn incoming_edges(&self, state: usize) -> Vec<Transition> {
        if let Some(val) = self.backwards.get(&state) {
            val.iter()
                .flat_map(|p| {
                    p.1.iter().map(|q| Transition {
                        to_state: state,
                        from_state: q.1,
                        to_char: *p.0,
                        from_char: q.0,
                    })
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Iterates over all the transitions.
    pub fn iter(&self) -> impl Iterator<Item = Transition> + '_ {
        self.forwards
            .iter()
            .flat_map(|(state, tab)| {
                tab.iter().map(|(ch, nstate)| {
                    nstate.iter().map(|(nch, nstate)| Transition {
                        from_state: *state,
                        to_state: *nstate,
                        from_char: *ch,
                        to_char: *nch,
                    })
                })
            })
            .flatten()
    }

    /// Retains only  the things fitting the predicate.
    pub fn retain(&mut self, f: impl Fn(&Transition) -> bool) {
        *self = self.iter().filter(f).collect()
    }
}

impl FromIterator<Transition> for Table {
    fn from_iter<T: IntoIterator<Item = Transition>>(iter: T) -> Self {
        iter.into_iter().fold(Table::new(), |tab, trans| {
            tab.tap_mut(|tab| tab.insert(trans))
        })
    }
}
