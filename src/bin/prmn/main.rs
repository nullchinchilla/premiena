use anyhow::Context;
use argh::FromArgs;
use premiena::{Nfa, RewriteRule};
use rayon::prelude::*;
use scfile::ScFile;
use std::{
    borrow::Cow,
    io::{stdin, BufRead, BufReader},
    path::PathBuf,
    time::Instant,
};
use thiserror::Error;
mod scfile;

#[derive(FromArgs)]
/// Applies a premiena-style sound change file.
struct Args {
    /// path to the sound change YAML file
    #[argh(positional)]
    input: PathBuf,

    /// whether to apply in reverse
    #[argh(switch, short = 'r')]
    reverse: bool,
}

#[derive(Error, Debug)]
enum ExpandError {
    #[error("cannot find variable")]
    UndefinedVariable,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Args = argh::from_env();
    let infile = std::fs::read_to_string(&args.input)?;
    let mut infile: ScFile = serde_yaml::from_str(&infile)?;
    if args.reverse {
        infile.rules.reverse();
    }
    let rules = infile
        .rules
        .par_iter()
        .map(|line| {
            let s = shellexpand::env_with_context(&line, |s| {
                Ok::<_, ExpandError>(Some(
                    infile
                        .variables
                        .get(s)
                        .map(|c| Cow::Owned(c.to_string()))
                        .ok_or(ExpandError::UndefinedVariable)?,
                ))
            })?;
            let start = Instant::now();
            let res = RewriteRule::from_line(s.trim())?.transduce(args.reverse);
            eprintln!("{:?} took {:?}", s, start.elapsed());
            Ok(res)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    eprintln!("rules compiled!");
    for line in BufReader::new(stdin()).lines() {
        let line = line?;
        let mut line = Nfa::from(line.as_str());
        for rule in rules.iter() {
            line = rule(line).determinize_min();
        }
        for res in line
            .lang_iter_utf8()
            .filter(|l| l.chars().all(|c| c.is_ascii_alphabetic()))
            .take(20)
        {
            print!("{} ", res)
        }
        println!();
    }
    Ok(())
}
