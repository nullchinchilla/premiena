use argh::FromArgs;
use premiena::{regex_to_nfa, Nfa, RewriteRule};
use rayon::prelude::*;
use scfile::ScFile;
use std::{
    io::{stdin, BufRead, BufReader},
    path::PathBuf,
    time::Instant,
};

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

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args: Args = argh::from_env();
    let infile = std::fs::read_to_string(&args.input)?;
    let mut infile: ScFile = serde_yaml::from_str(&infile)?;
    if args.reverse {
        infile.rules.reverse();
    }
    let (source, rules) = infile.expand()?;
    let rules = rules
        .iter()
        .map(|line| {
            let start = Instant::now();
            let res = RewriteRule::from_line(line.trim())?.transduce(args.reverse);
            log::debug!("{:?} took {:?}", line, start.elapsed());
            Ok(res)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    log::debug!("rules compiled!");
    for line in BufReader::new(stdin()).lines() {
        let line = format!("#{}#", line?);
        let mut line = Nfa::from(line.as_str());
        for rule in rules.iter() {
            line = rule(line).determinize_min();
        }

        if args.reverse {
            line = line.intersect(&regex_to_nfa(&format!("#{}#", source))?)
        }

        for res in line
            .lang_iter_utf8()
            // .filter(|l| l.chars().all(|c| c.is_ascii_alphabetic()))
            .take(50)
        {
            if res.starts_with('#') && res.ends_with('#') {
                print!("{} ", res.trim_matches('#'))
            }
        }
        println!();
    }
    Ok(())
}
