use std::collections::BTreeMap;

use anyhow::Context;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// YAML-encoded sound change file
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScFile {
    #[serde(default)]
    pub categories: BTreeMap<String, Vec<String>>,
    pub source: String,
    pub rules: Vec<String>,
}

impl ScFile {
    /// Expand all the categories.
    pub fn expand(&self) -> anyhow::Result<(String, Vec<String>)> {
        let mut toret = vec![];
        for rule in self.rules.iter() {
            let (rule, context) = rule
                .split_once('/')
                .context("could not split into rule and context")?;
            // first replace all the categories with their regexes in the context
            let context = self.categories.iter().fold(
                context.to_string(),
                |context, (cat_name, cat_contents)| {
                    context.replace(cat_name, &cat_to_regex(cat_contents))
                },
            );
            // then, we process the categories mentioned in the rule
            let rule_categories = self
                .categories
                .keys()
                .filter(|cat| rule.contains(cat.as_str()))
                .collect_vec();
            if rule_categories.is_empty() {
                // okay this is easy
                toret.push(format!("{}/{}", rule, context));
            } else {
                let cat_length = self.categories[rule_categories[0]].len();
                if rule_categories
                    .iter()
                    .any(|cat| self.categories[*cat].len() != cat_length)
                {
                    anyhow::bail!("categories mentioned in rule not of the same length")
                }
                for i in 0..cat_length {
                    let rule = rule_categories.iter().fold(rule.to_string(), |rule, cat| {
                        rule.replace(*cat, &self.categories[*cat][i])
                    });
                    toret.push(format!("{}/{}", rule, context));
                }
            }
        }
        Ok((
            self.categories.iter().fold(
                self.source.to_string(),
                |source, (cat_name, cat_contents)| {
                    source.replace(cat_name, &cat_to_regex(cat_contents))
                },
            ),
            toret,
        ))
    }
}

fn cat_to_regex(cat: &[String]) -> String {
    format!("({})", cat.join("|"))
}
