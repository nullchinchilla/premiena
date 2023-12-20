use std::fmt::Display;

use ahash::{AHashMap, AHashSet};
use tap::Tap;

/// A single transition.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transition {
    pub from_state: u32,
    pub to_state: u32,
    pub from_char: Option<u8>,
    pub to_char: Option<u8>,
}

/// A transition table with good asymptotic performance, supporting multiple kinds of queries.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct Table {
    // state -> (char -> setof outchar, state)
    forwards: AHashMap<u32, AHashMap<Option<u8>, AHashSet<(Option<u8>, u32)>>>,
    backwards: AHashMap<u32, AHashMap<Option<u8>, AHashSet<(Option<u8>, u32)>>>,
}

impl Table {
    /// Create an empty new table.
    pub fn new() -> Self {
        Self {
            forwards: AHashMap::default(),
            backwards: AHashMap::default(),
        }
    }

    /// All the states.
    pub fn states(&self) -> AHashSet<u32> {
        self.forwards
            .keys()
            .chain(self.backwards.keys())
            .copied()
            .collect()
    }

    /// Next free stateid
    pub fn next_free_stateid(&self) -> u32 {
        self.states().into_iter().max().unwrap_or_default() + 1
    }

    /// Double-epsilon closure
    pub fn dubeps_closure(&self, start: u32) -> AHashSet<u32> {
        let mut group = AHashSet::default();
        let mut dfs_stack = vec![start];
        while let Some(top) = dfs_stack.pop() {
            group.insert(top);
            if let Some(single_epsilon_neighs) =
                self.forwards.get(&top).and_then(|top| top.get(&None))
            {
                for (other, neigh) in single_epsilon_neighs {
                    if other.is_none() && group.insert(*neigh) {
                        dfs_stack.push(*neigh)
                    }
                }
            }
        }
        group
    }

    /// Closure, given a particular predicate
    fn edge_closure(&self, start: u32, pred: impl Fn(&Transition) -> bool) -> AHashSet<u32> {
        let mut group = AHashSet::default();
        let mut dfs_stack = vec![start];
        while let Some(top) = dfs_stack.pop() {
            group.insert(top);
            let neighs = self.outgoing_edges(top);
            for neigh in neighs.into_iter().filter(&pred) {
                if group.insert(neigh.to_state) {
                    dfs_stack.push(neigh.to_state);
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
        state: u32,
        input: impl Into<Option<u8>>,
    ) -> AHashSet<(Option<u8>, u32)> {
        let input: Option<u8> = input.into();
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
    pub fn outgoing_edges(&self, state: u32) -> impl Iterator<Item = Transition> + '_ {
        self.forwards.get(&state).into_iter().flat_map(move |d| {
            d.iter().flat_map(move |p| {
                p.1.iter().map(move |q| Transition {
                    from_state: state,
                    to_state: q.1,
                    from_char: *p.0,
                    to_char: q.0,
                })
            })
        })
    }

    /// Gets all the edges going into one node
    pub fn incoming_edges(&self, state: u32) -> Vec<Transition> {
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

    /// Flips direction of the table.
    pub fn flip(&mut self) {
        std::mem::swap(&mut self.backwards, &mut self.forwards);
    }
}

impl FromIterator<Transition> for Table {
    fn from_iter<T: IntoIterator<Item = Transition>>(iter: T) -> Self {
        iter.into_iter().fold(Table::new(), |tab, trans| {
            tab.tap_mut(|tab| tab.insert(trans))
        })
    }
}
