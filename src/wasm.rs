```
use crate::ConstraintSet;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct SolveResult {
    pub score: usize,
    pub flag: bool,
    pub positions: Vec<Option<String>>,
    pub error: Option<String>,
}

#[wasm_bindgen]
pub fn solve(constraints_str: &str) -> JsValue {
    let constraints = match ConstraintSet::from_str(constraints_str) {
        Ok(c) => c,
        Err(e) => {
            return serde_wasm_bindgen::to_value(&SolveResult {
                score: 0,
                flag: false,
                positions: vec![],
                error: Some(format!("Invalid constraints: {}", e)),
            })
            .unwrap()
        }
    };

    let (board, score) = constraints.solve();

    let positions: Vec<Option<String>> = board
        .positions()
        .iter()
        .map(|opt_color| opt_color.map(|c| char::from(c).to_string()))
        .collect();

    let result = SolveResult {
        score: score.score,
        flag: score.flag,
        positions,
        error: None,
    };

    serde_wasm_bindgen::to_value(&result).unwrap()
}
