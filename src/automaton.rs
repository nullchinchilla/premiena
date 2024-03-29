use std::{
    collections::VecDeque,
    iter::once,
    time::{Duration, Instant},
};

use ahash::{AHashMap, AHashSet};
use genawaiter::sync::Gen;
use itertools::Itertools;

use once_cell::sync::Lazy;
use smallvec::SmallVec;
use tap::Tap;

use crate::table::{Table, Transition};

/// Trait for methods common to NFAs and NFSTs.
pub trait Automaton: Sized {
    fn start_mut(&mut self) -> &mut u32;
    fn start(&self) -> &u32;
    fn table_mut(&mut self) -> &mut Table;
    fn table(&self) -> &Table;
    fn accepting_mut(&mut self) -> &mut AHashSet<u32>;
    fn accepting(&self) -> &AHashSet<u32>;

    /// Concatenates with another automaton
    fn concat(mut self, other: &Self) -> Self {
        // copy the other NFA into this one
        let offset = self.table().next_free_stateid();
        for mut trans in other.table().iter() {
            trans.from_state += offset;
            trans.to_state += offset;
            self.table_mut().insert(trans);
        }
        // epsilon transitions from each of our accepting states to their starting state
        for our_accept in self.accepting().clone() {
            self.table_mut().insert(Transition {
                from_state: our_accept,
                to_state: *other.start() + offset,
                from_char: None,
                to_char: None,
            });
        }
        // accept is their accept
        *self.accepting_mut() = other.accepting().iter().map(|i| *i + offset).collect();
        self
    }

    /// Kleene star of this machine
    fn star(mut self) -> Self {
        let new_start = self.table().next_free_stateid();
        let new_end = new_start + 1;
        let tsn = Transition {
            from_state: new_start,
            to_state: *self.start(),
            from_char: None,
            to_char: None,
        };
        self.table_mut().insert(tsn);
        for end in self.accepting().clone() {
            self.table_mut().insert(Transition {
                from_state: end,
                to_state: new_end,
                from_char: None,
                to_char: None,
            });
        }
        self.table_mut().insert(Transition {
            from_state: new_end,
            to_state: new_start,
            from_char: None,
            to_char: None,
        });
        *self.accepting_mut() = once(new_start).collect();
        *self.start_mut() = new_start;
        self
    }

    /// Union of two automata.
    fn union(mut self, other: &Self) -> Self {
        // copy the other NFA into this one
        let offset = self.table().next_free_stateid();
        for mut trans in other.table().iter() {
            trans.from_state += offset;
            trans.to_state += offset;
            self.table_mut().insert(trans);
        }
        // make start and end
        let new_start = self.table().next_free_stateid();
        let new_end = new_start + 1;
        // connect new start to all the starts and new end to all the ends
        let t = Transition {
            from_state: new_start,
            to_state: *self.start(),
            from_char: None,
            to_char: None,
        };
        self.table_mut().insert(t);
        let t = Transition {
            from_state: new_start,
            to_state: *other.start() + offset,
            from_char: None,
            to_char: None,
        };
        self.table_mut().insert(t);
        for end in self.accepting().clone() {
            self.table_mut().insert(Transition {
                from_state: end,
                to_state: new_end,
                from_char: None,
                to_char: None,
            });
        }
        for end in other.accepting().clone() {
            self.table_mut().insert(Transition {
                from_state: end + offset,
                to_state: new_end,
                from_char: None,
                to_char: None,
            });
        }
        // replace start and end
        *self.start_mut() = new_start;
        *self.accepting_mut() = once(new_end).collect();
        self
    }

    /// Remove all double-epsilon transitions.
    fn deepsilon(mut self) -> Self {
        let mut set_to_num = new_set2num();
        let new_start = set_to_num(
            self.table()
                .dubeps_closure(*self.start())
                .into_iter()
                .collect(),
        );
        let mut dfs_stack = vec![*self.start()];
        let mut seen = AHashSet::default();
        let mut new_table = Table::new();
        let mut new_accepting = AHashSet::default();
        while let Some(top) = dfs_stack.pop() {
            let top = self.table().dubeps_closure(top);
            let top_no = set_to_num(top.iter().copied().collect());
            if top.iter().any(|t| self.accepting().contains(t)) {
                new_accepting.insert(top_no);
            }
            for etop in top {
                for mut edge in self.table().outgoing_edges(etop) {
                    if edge.from_char.is_some() || edge.to_char.is_some() {
                        if seen.insert(edge.to_state) {
                            dfs_stack.push(edge.to_state);
                        }
                        let to = set_to_num(
                            self.table()
                                .dubeps_closure(edge.to_state)
                                .into_iter()
                                .collect(),
                        );
                        edge.from_state = top_no;
                        edge.to_state = to;
                        new_table.insert(edge);
                    }
                }
            }
        }
        *self.table_mut() = new_table;
        *self.start_mut() = new_start;
        *self.accepting_mut() = new_accepting;
        self
    }
}

/// A nondeterministic finite automaton, representing a regular language.
#[derive(Clone)]
pub struct Nfa {
    start: u32,
    // transitions always "produce" epsilon
    table: Table,
    accepting: AHashSet<u32>,
}

impl Automaton for Nfa {
    fn start_mut(&mut self) -> &mut u32 {
        &mut self.start
    }

    fn start(&self) -> &u32 {
        &self.start
    }

    fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    fn table(&self) -> &Table {
        &self.table
    }

    fn accepting_mut(&mut self) -> &mut AHashSet<u32> {
        &mut self.accepting
    }

    fn accepting(&self) -> &AHashSet<u32> {
        &self.accepting
    }
}

impl Nfa {
    /// A NFA that does not contain any strings.
    pub fn null() -> Self {
        Self {
            start: 0,
            table: Table::new(),
            accepting: Default::default(),
        }
    }

    /// A NFA that contains all strings
    pub fn all() -> Self {
        Self::null().complement()
    }

    /// A NFA that contains all integer-byte strings.
    pub fn all_int_bytes() -> Self {
        Self::sigma().concat(&Self::sigma()).star()
    }

    /// A NFA that contains a single symbol
    pub fn sigma() -> Self {
        static SIGMA: Lazy<Nfa> = Lazy::new(|| {
            (0..=u8::MAX)
                .fold(Nfa::null(), |a, b| a.union(&Nfa::single(b)))
                .determinize_min()
        });
        SIGMA.clone()
    }

    /// An NFA that contains only the empty string.
    pub fn empty() -> Self {
        let mut table = Table::new();
        table.insert(Transition {
            from_state: 0,
            to_state: 1,
            from_char: None,
            to_char: None,
        });
        Nfa {
            start: 0,
            table,
            accepting: once(1).collect(),
        }
    }

    /// A single symbol.
    pub fn single(s: u8) -> Self {
        let mut table = Table::new();
        table.insert(Transition {
            from_state: 0,
            to_state: 1,
            from_char: Some(s),
            to_char: Some(s),
        });
        Nfa {
            start: 0,
            table,
            accepting: once(1).collect(),
        }
    }

    /// Compute the epsilon closure of a given state.
    fn epsilon_closure(&self, state: u32) -> AHashSet<u32> {
        let mut closure = AHashSet::new();
        let mut stack = vec![state];

        while let Some(s) = stack.pop() {
            if closure.insert(s) {
                // If the state is not already in the closure, add its epsilon transitions to the stack.
                for (_, t) in self.table().transition(s, None) {
                    stack.push(t);
                }
            }
        }

        closure
    }

    pub fn intersect(mut self, other: &Self) -> Self {
        self = self.deepsilon();
        let other = other.clone().deepsilon();
        let mut new_table = Table::new();
        let mut ab2c = new_ab2c();
        for atrans in self.table.iter() {
            for btrans in other.table.iter() {
                if atrans.from_char == btrans.from_char {
                    new_table.insert(Transition {
                        from_state: ab2c(atrans.from_state, btrans.from_state),
                        to_state: ab2c(atrans.to_state, btrans.to_state),
                        from_char: atrans.from_char,
                        to_char: atrans.from_char,
                    })
                }
            }
        }
        let mut new_accepting = AHashSet::default();
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

    /// Complement of this NFA
    pub fn complement(mut self) -> Self {
        self = self.determinize_min().complete();
        self.accepting = self
            .table
            .states()
            .into_iter()
            .filter(|s| !self.accepting.contains(s))
            .collect();
        self
    }

    /// Subtract another NFA from this NFA.
    pub fn subtract(self, other: &Nfa) -> Self {
        self.intersect(&other.clone().complement())
    }

    /// Helper function to create an NFA based on a bunch of bytes
    pub fn from_bytes(b: &[u8]) -> Self {
        b.iter()
            .map(|b| Self::single(*b))
            .fold(Self::empty(), |a, b| a.concat(&b))
    }

    /// Creates the graphviz representation of the NFA.
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
                trans
                    .from_char
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| String::from("ε"))
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

    /// Reverses the strings in this NFA.
    pub fn reverse(mut self) -> Self {
        self.table = self.table.flip();
        let new_start = self.table.next_free_stateid();
        for old_accept in self.accepting.iter().copied() {
            self.table.insert(Transition {
                from_state: new_start,
                to_state: old_accept,
                from_char: None,
                to_char: None,
            });
        }
        self.accepting = once(self.start).collect();
        self.start = new_start;
        self
    }

    /// Make complete.
    pub fn complete(mut self) -> Self {
        let garbage_state = self.table.next_free_stateid();
        let states = self.table.states().collect_vec();
        for state in states
            .into_iter()
            .chain(once(garbage_state))
            .chain(once(self.start))
        {
            for ch in 0..=u8::MAX {
                if self.table.transition(state, Some(ch)).count() == 0 {
                    self.table.insert(Transition {
                        from_state: state,
                        to_state: garbage_state,
                        from_char: Some(ch),
                        to_char: Some(ch),
                    });
                }
            }
        }
        self
    }

    /// Convert into a DFA.
    pub fn determinize(mut self) -> Self {
        let start = Instant::now();
        let pre_det = self.table.state_count();
        // first remove epsilons. this makes our life easier
        self = self.deepsilon();
        // then we use the powerset construction
        let mut set_to_num = new_set2num();
        let new_start = set_to_num(once(self.start).collect());
        // DFA loop
        let mut set_stack: Vec<AHashSet<u32>> = vec![once(self.start).collect()];
        let mut seen = AHashSet::default();
        let mut new_table = Table::new();
        let mut new_accepting = AHashSet::default();
        let mut step_ctr = 0;
        while let Some(top_set) = set_stack.pop() {
            step_ctr += 1;

            let top_num = set_to_num(top_set.iter().copied().collect());
            if !seen.insert(top_num) {
                continue;
            }
            if top_set.iter().any(|s| self.accepting.contains(s)) {
                new_accepting.insert(top_num);
            }
            for ch in top_set
                .iter()
                .copied()
                .flat_map(|elem| {
                    self.table
                        .outgoing_edges(elem)
                        .map(|e| e.from_char.unwrap())
                })
                .unique()
            {
                let next_set: AHashSet<u32> = top_set
                    .iter()
                    .copied()
                    .flat_map(|elem| {
                        self.table
                            .transition(elem, Some(ch))
                            .into_iter()
                            .map(|v| v.1)
                    })
                    .collect();
                if !next_set.is_empty() {
                    let next_set_num = set_to_num(next_set.iter().copied().collect());
                    new_table.insert(Transition {
                        from_state: top_num,
                        to_state: next_set_num,
                        from_char: Some(ch),
                        to_char: Some(ch),
                    });
                    set_stack.push(next_set);
                }
            }
        }

        if start.elapsed() > Duration::from_millis(50) {
            log::warn!(
                "determinize {} => {} took {} steps ({:?})",
                pre_det,
                new_table.state_count(),
                step_ctr,
                start.elapsed()
            );
        }

        Self {
            start: new_start,
            table: new_table,
            accepting: new_accepting,
        }
    }

    /// Create a minimized, determinized version.
    pub fn determinize_min(self) -> Self {
        self.reverse().determinize().reverse().determinize()
    }

    /// Iterates through the regular language described by this NFA.
    pub fn lang_iter(&self) -> impl Iterator<Item = Vec<u8>> + '_ {
        let this = self.clone().deepsilon();
        let mut bfs_queue = VecDeque::new();
        bfs_queue.push_back((this.start, vec![]));
        let generator = Gen::new(|co| async move {
            if this.accepting.is_empty() {
                return;
            }
            while let Some((state, word_so_far)) = bfs_queue.pop_front() {
                // dbg!(bfs_queue.len());
                // eprintln!("state = {state}, word_so_far = {:?}", word_so_far);
                if this.accepting.contains(&state) {
                    co.yield_(word_so_far.clone()).await;
                }
                // explore further
                for trans in this.table.outgoing_edges(state) {
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

    /// Helper function that iterates through the UTF-8 strings described by this NFA.
    pub fn lang_iter_utf8(&self) -> impl Iterator<Item = String> + '_ {
        self.lang_iter().filter_map(|s| String::from_utf8(s).ok())
    }
}

impl From<&[u8]> for Nfa {
    fn from(b: &[u8]) -> Self {
        Self::from_bytes(b)
    }
}

impl From<&str> for Nfa {
    fn from(b: &str) -> Self {
        Self::from_bytes(b.as_bytes())
    }
}

/// A nondeterministic finite-state transducer.
#[derive(Clone)]
pub struct Nfst {
    start: u32,
    table: Table,
    accepting: AHashSet<u32>,
}

impl Automaton for Nfst {
    fn start_mut(&mut self) -> &mut u32 {
        &mut self.start
    }

    fn start(&self) -> &u32 {
        &self.start
    }

    fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    fn table(&self) -> &Table {
        &self.table
    }

    fn accepting_mut(&mut self) -> &mut AHashSet<u32> {
        &mut self.accepting
    }

    fn accepting(&self) -> &AHashSet<u32> {
        &self.accepting
    }
}

impl Nfst {
    /// Creates a transducer that maps a regular language to itself.
    pub fn id_nfa(nfa: Nfa) -> Self {
        Self {
            start: nfa.start,
            table: nfa.table,
            accepting: nfa.accepting,
        }
    }

    /// Create an NFST corresponding to the Cartesian product of the *images* of the this and that.
    pub fn image_cross(&self, other: &Self) -> Self {
        let mut ab2c = new_ab2c();
        let mut table = Table::new();
        let alphabet = || (0..=u8::MAX).map(Some).chain(std::iter::once(None));
        for x in self.table.states() {
            for y in other.table.states() {
                for a in alphabet() {
                    for b in alphabet() {
                        let x_tsns = self.table.transition(x, a);
                        for x_tsn in x_tsns {
                            let y_tsns = other.table.transition(y, b);
                            for y_tsn in y_tsns {
                                table.insert(Transition {
                                    from_state: ab2c(x, y),
                                    to_state: ab2c(x_tsn.1, y_tsn.1),
                                    from_char: a,
                                    to_char: b,
                                });
                            }
                        }
                    }
                }
            }
        }

        let mut accepting = AHashSet::default();
        for x in self.accepting.iter() {
            for y in other.accepting.iter() {
                accepting.insert(ab2c(*x, *y));
            }
        }
        Self {
            start: ab2c(self.start, other.start),
            table,
            accepting,
        }
    }

    /// Composes this FST with another one
    pub fn compose(&self, other: &Self) -> Self {
        let start = Instant::now();
        // function that maps a pair of state ids to the state id in the new fst
        let mut ab2c = new_ab2c();
        // we go through the cartesian product of state ids
        let mut new_table = Table::new();
        let mut seen = AHashSet::default();
        let mut dfs_stack = vec![(self.start, other.start)];
        let mut counter = 0;
        while let Some((i, j)) = dfs_stack.pop() {
            counter += 1;
            if !seen.insert((i, j)) {
                continue;
            }
            // first handle epsilon
            for second_stage in other.table.transition(j, None) {
                new_table.insert(Transition {
                    from_state: ab2c(i, j),
                    to_state: ab2c(i, second_stage.1),
                    from_char: None,
                    to_char: second_stage.0,
                });
                dfs_stack.push((i, second_stage.1));
            }
            for ch in (0..=u8::MAX).map(Some).chain(once(None)) {
                for first_stage in self.table.transition(i, ch) {
                    for second_stage in other.table.transition(j, first_stage.0) {
                        new_table.insert(Transition {
                            from_state: ab2c(i, j),
                            to_state: ab2c(first_stage.1, second_stage.1),
                            from_char: ch,
                            to_char: second_stage.0,
                        });
                        dfs_stack.push((first_stage.1, second_stage.1));
                    }
                    if first_stage.0.is_none() {
                        // always implicit epsilon
                        new_table.insert(Transition {
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

        let mut accepting_states = AHashSet::default();
        for i in self.accepting.iter() {
            for j in other.accepting.iter() {
                accepting_states.insert(ab2c(*i, *j));
            }
        }
        if start.elapsed().as_secs_f64() > 1.0 {
            log::warn!(
                "compose of {}x{} => {} took {} steps ({:?})",
                self.table.state_count(),
                other.table.state_count(),
                new_table.state_count(),
                counter,
                start.elapsed()
            );
        }
        Nfst {
            start: ab2c(self.start, other.start),
            table: new_table,
            accepting: accepting_states,
        }
    }

    /// Optional.
    pub fn optional(self) -> Self {
        let zero = Self::id_nfa(Nfa::empty());
        self.union(&zero)
    }

    /// NFA representing the image of this NFST.
    pub fn image_nfa(&self) -> Nfa {
        Nfa {
            start: self.start,
            table: self
                .table
                .iter()
                .map(|mut e| {
                    e.from_char = e.to_char;
                    e
                })
                .collect(),
            accepting: self.accepting.clone(),
        }
    }

    /// Inverse of this NFST.
    pub fn inverse(mut self) -> Self {
        self.table = self
            .table
            .iter()
            .map(|mut e| {
                std::mem::swap(&mut e.from_char, &mut e.to_char);
                e
            })
            .collect();
        self
    }

    /// Produces the Graphviz representation of the NFST
    pub fn graphviz(&self) -> String {
        let mut lel = String::new();
        lel.push_str("digraph G {\n");
        lel.push_str("rankdir=LR\n");
        lel.push_str("node [shape=\"circle\"]\n");
        for tsn in self.table.iter() {
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
        lel.push_str(&format!("S -> {}\n", self.start));
        lel.push_str("S[style=invis label=\"\"]\n");

        lel.push_str("}\n");
        lel
    }
}

fn new_set2num() -> impl FnMut(SmallVec<[u32; 4]>) -> u32 {
    let mut tab = AHashMap::default();
    let mut counter = 0;
    move |mut a| {
        a.sort_unstable();
        *tab.entry(a).or_insert_with(|| {
            counter += 1;
            counter - 1
        })
    }
}
fn new_ab2c() -> impl FnMut(u32, u32) -> u32 {
    let mut tab: AHashMap<(u32, u32), u32> = AHashMap::default();
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
    fn simple_nfa() {
        let baa_containing = Nfa::all().concat(&"ba".into()).concat(&Nfa::all());
        let nn = Nfa::from("a")
            .union(&"b".into())
            .star()
            .subtract(&baa_containing)
            .determinize_min();
        eprintln!("{}", nn.graphviz());
        for s in nn.lang_iter_utf8().take(10) {
            if s.is_ascii() {
                eprintln!("{:?}", s);
            }
        }
    }

    #[test]
    fn simple_nfst() {
        let baa_containing = Nfa::all().concat(&"ba".into()).concat(&Nfa::all());
        let nn = Nfa::from("a")
            .union(&"b".into())
            .star()
            .subtract(&baa_containing)
            .determinize_min();
        let nn = Nfst::id_nfa(nn)
            .compose(&Nfst::id_nfa(Nfa::from("aab")))
            .image_nfa();
        eprintln!("{}", nn.graphviz());
        for s in nn.lang_iter_utf8().take(10) {
            if s.is_ascii() {
                eprintln!("{:?}", s);
            }
        }
    }
}
