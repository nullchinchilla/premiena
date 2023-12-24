use argh::FromArgs;
use premiena::{regex_to_nfa, Nfa, RewriteRule};
use rayon::prelude::*;
use scfile::ScFile;
use std::{
    io::{stdin, stdout, BufRead, BufReader, Write},
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
    let start = Instant::now();
    let rules = rules
        .par_iter()
        .map(|line| {
            let start = Instant::now();
            let res = RewriteRule::from_line(line.trim())?.transduce(args.reverse);
            log::debug!("{:?} took {:?}", line, start.elapsed());
            Ok(res)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    log::debug!("rules compiled in {:?}!", start.elapsed());
    for line in BufReader::new(stdin()).lines() {
        let line = line?;
        let mut line = Nfa::from(line.as_str());
        for rule in rules.iter() {
            line = rule(line);
        }

        if args.reverse {
            if let Some(source) = source.as_ref() {
                line = line.intersect(&regex_to_nfa(&source)?)
            }
        }

        for res in line
            .lang_iter_utf8()
            // .filter(|l| l.chars().all(|c| c.is_ascii_alphabetic()))
            .take(50)
        {
            print!("{} ", res);
        }
        println!();
    }
    Ok(())
}
