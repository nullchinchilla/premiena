use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// YAML-encoded sound change file
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScFile {
    pub variables: BTreeMap<String, String>,
    pub rules: Vec<String>,
}
