use std::fmt::Display;

use ahash::{AHashMap, AHashSet};
use tap::Tap;

/// A single, hex symbol
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum Symbol {
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    Sa,
    Sb,
    Sc,
    Sd,
    Se,
    Sf,
}

impl Symbol {
    /// Alphabet of all possible symbols.
    pub const SIGMA: [Symbol; 16] = [
        Symbol::S0,
        Symbol::S1,
        Symbol::S2,
        Symbol::S3,
        Symbol::S4,
        Symbol::S5,
        Symbol::S6,
        Symbol::S7,
        Symbol::S8,
        Symbol::S9,
        Symbol::Sa,
        Symbol::Sb,
        Symbol::Sc,
        Symbol::Sd,
        Symbol::Se,
        Symbol::Sf,
    ];

    fn from_hexdigit(b: u8) -> Self {
        match b {
            0x0 => Self::S0,
            0x1 => Self::S1,
            0x2 => Self::S2,
            0x3 => Self::S3,
            0x4 => Self::S4,
            0x5 => Self::S5,
            0x6 => Self::S6,
            0x7 => Self::S7,
            0x8 => Self::S8,
            0x9 => Self::S9,
            0xa => Self::Sa,
            0xb => Self::Sb,
            0xc => Self::Sc,
            0xd => Self::Sd,
            0xe => Self::Se,
            0xf => Self::Sf,
            _ => panic!("not a hex digit"),
        }
    }

    /// Convert a single byte.
    pub fn from_byte(bayt: u8) -> [Self; 2] {
        let a = (bayt & 0b11110000) >> 4;
        let b = bayt & 0b00001111;
        [Self::from_hexdigit(a), Self::from_hexdigit(b)]
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::S0 => "0".fmt(f),
            Symbol::S1 => "1".fmt(f),
            Symbol::S2 => "2".fmt(f),
            Symbol::S3 => "3".fmt(f),
            Symbol::S4 => "4".fmt(f),
            Symbol::S5 => "5".fmt(f),
            Symbol::S6 => "6".fmt(f),
            Symbol::S7 => "7".fmt(f),
            Symbol::S8 => "8".fmt(f),
            Symbol::S9 => "9".fmt(f),
            Symbol::Sa => "a".fmt(f),
            Symbol::Sb => "b".fmt(f),
            Symbol::Sc => "c".fmt(f),
            Symbol::Sd => "d".fmt(f),
            Symbol::Se => "e".fmt(f),
            Symbol::Sf => "f".fmt(f),
        }
    }
}

/// A single transition.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transition {
    pub from_state: u32,
    pub to_state: u32,
    pub from_char: Option<Symbol>,
    pub to_char: Option<Symbol>,
}

/// A transition table with good asymptotic performance, supporting multiple kinds of queries.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct Table {
    // state -> (char -> setof outchar, state)
    forwards: AHashMap<u32, AHashMap<Option<Symbol>, AHashSet<(Option<Symbol>, u32)>>>,
    backwards: AHashMap<u32, AHashMap<Option<Symbol>, AHashSet<(Option<Symbol>, u32)>>>,
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

    /// Closure, given a particular predicate
    pub fn edge_closure(&self, start: u32, pred: impl Fn(&Transition) -> bool) -> AHashSet<u32> {
        let mut group = AHashSet::default();
        let mut dfa_stack = vec![start];
        while let Some(top) = dfa_stack.pop() {
            group.insert(top);
            let neighs = self.outgoing_edges(top);
            for neigh in neighs.into_iter().filter(&pred) {
                if group.insert(neigh.to_state) {
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
        state: u32,
        input: impl Into<Option<Symbol>>,
    ) -> AHashSet<(Option<Symbol>, u32)> {
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
