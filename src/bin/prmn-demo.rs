use std::io::{stdin, BufRead, BufReader};

use anyhow::Context;
use premiena::{
    alphabet::Alphabet,
    automaton::{Nfst, Symbol},
    rewrite::RewriteRule,
};
use rayon::prelude::*;
use std::fmt::Write;

fn main() -> anyhow::Result<()> {
    let input = String::from_utf8_lossy(&std::fs::read(
        &std::env::args().nth(1).context("must pass in a filename")?,
    )?)
    .into_owned();
    let mut rules = vec![];
    for line in input.lines() {
        let (rule, ctx) = line.split_once('/').context("could not find / in rule")?;
        let (from, to) = rule.split_once("->").context("could not find -> in rule")?;
        let (lctx, rctx) = ctx.split_once('_').context("could not find _ in rule")?;
        let rule = RewriteRule::from_regexes(
            from.trim(),
            to.trim(),
            lctx.trim(),
            rctx.trim(),
            Alphabet::new_alphanum(),
        )?;
        dbg!(&(from, to, lctx, rctx));
        rules.push(rule);
    }
    eprintln!("rules parsed");
    let transducers: Vec<Nfst> = rules.par_iter().map(|r| r.transducer()).collect();
    eprintln!("individual transducers produced");
    let big_transduce = transducers
        .into_iter()
        .reduce(|a, b| a.compose(&b))
        .context("must have at least one rule")?;
    eprintln!("big transducer produced");
    eprintln!("{}", big_transduce.graphviz());
    for line in BufReader::new(stdin()).lines() {
        let line = line?;
        let input = line
            .chars()
            .map(Symbol::from)
            .fold(Nfst::id_single(None), |a, b| {
                a.concat(&Nfst::id_single(Some(b)))
            });
        eprintln!("{}", input.graphviz());
        eprintln!("{}", input.compose(&big_transduce).graphviz());
        for res in input.compose(&big_transduce).image_dfa().iter() {
            let res: String = res.into_iter().fold(String::new(), |mut a, b| {
                write!(&mut a, "{}", b).unwrap();
                a
            });
            println!("{}", res)
        }
    }
    Ok(())
}
