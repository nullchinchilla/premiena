use std::collections::BTreeSet;

use ahash::{AHashMap, AHashSet};
use either::Either;

use tap::Tap;

/// A single transition.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
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

    states: BTreeSet<u32>,

    transitions: Vec<Transition>,
}

impl Table {
    /// Create an empty new table.
    pub fn new() -> Self {
        Self {
            forwards: AHashMap::default(),
            states: BTreeSet::new(),

            transitions: Vec::new(),
        }
    }

    /// All the states.
    pub fn states(&self) -> impl Iterator<Item = u32> + '_ {
        self.states.iter().copied()
    }

    /// State count.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Next free stateid
    pub fn next_free_stateid(&self) -> u32 {
        self.states.last().copied().unwrap_or_default() + 1
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

    /// Inserts one transition.
    pub fn insert(&mut self, transition: Transition) {
        self.states.insert(transition.from_state);
        self.states.insert(transition.to_state);
        let new_value = self
            .forwards
            .entry(transition.from_state)
            .or_default()
            .entry(transition.from_char)
            .or_default()
            .insert((transition.to_char, transition.to_state));
        if new_value {
            self.transitions.push(transition)
        }
    }

    /// The basic transition function.
    pub fn transition(
        &self,
        state: u32,
        input: impl Into<Option<u8>>,
    ) -> impl Iterator<Item = (Option<u8>, u32)> + '_ {
        let input = input.into();

        let iter = self
            .forwards
            .get(&state)
            .and_then(|map| map.get(&input).map(|res| res.iter()))
            .into_iter()
            .flatten()
            .copied();
        if input.is_none() {
            Either::Left(iter.chain(std::iter::once((None, state))))
        } else {
            Either::Right(iter)
        }
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

    /// Iterates over all the transitions.
    pub fn iter(&self) -> impl Iterator<Item = Transition> + '_ {
        self.transitions.iter().copied()
    }

    /// Flips direction of the table.
    pub fn flip(&self) -> Self {
        Table::from_iter(self.iter().map(|trans| Transition {
            from_state: trans.to_state,
            to_state: trans.from_state,
            from_char: trans.to_char,
            to_char: trans.from_char,
        }))
    }
}

impl FromIterator<Transition> for Table {
    fn from_iter<T: IntoIterator<Item = Transition>>(iter: T) -> Self {
        iter.into_iter().fold(Table::new(), |tab, trans| {
            tab.tap_mut(|tab| tab.insert(trans))
        })
    }
}
