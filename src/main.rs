use dimension::*;
use itertools::Itertools;
use structopt::StructOpt;

fn get_input_colors() -> Vec<Color> {
    vec![
        Color::Black,
        Color::Black,
        Color::Black,
        Color::White,
        Color::White,
        Color::White,
        Color::Blue,
        Color::Blue,
        Color::Blue,
        Color::Green,
        Color::Green,
        Color::Green,
        Color::Orange,
        Color::Orange,
        Color::Orange,
    ]
}

// TODO(trevorm): make this a class method on ColorMix.
fn get_all_color_mixes(k: usize) -> impl Iterator<Item = ColorMix> {
    // Use unique() to filter combinations. In the worst case (15 choose 7|8) this generates 6435
    // combinations with duplicates as some fields are repeated. ColorMix is only 64 bits long so
    // this is unlikely to use more than 100KB memory.
    get_input_colors()
        .into_iter()
        .combinations(k)
        .as_color_mix()
        .unique()
}

// Gets the upper bound score for each board size.
fn get_upper_bound_scores(constraints: &ConstraintSet) -> Vec<(usize, BoardScore)> {
    let mut scores: Vec<(usize, BoardScore)> = Vec::with_capacity(NUM_POSITIONS);
    for k in 1..=NUM_POSITIONS {
        let max_score = constraints.max_score(k);
        let mut high_score = BoardScore::default();

        for m in get_all_color_mixes(k) {
            let score = m.approximate_score(&constraints);
            if score > high_score {
                high_score = score
            }
            if high_score == max_score {
                break;
            }
        }
        scores.push((k, high_score))
    }
    // Return in descending order by score.
    scores.sort_by(|(_, a), (_, b)| a.cmp(&b).reverse());
    scores
}

// TODO(trevorm): tests for this method.
fn solve(constraints: &ConstraintSet) -> (BoardState, BoardScore) {
    for c in constraints.dropped_constraints() {
        println!(
            "Dropping conflicting constraint '{}'; taking 2 point/no flag penalty.",
            c
        );
    }
    let mut best_board = BoardState::default();
    let mut best_score = BoardScore::default();

    for (k, max_score) in get_upper_bound_scores(&constraints) {
        if max_score < best_score {
            break;
        }

        // Iterate over all ColorMixes whose approximate_score() matches max_score, then iterate
        // over those permutations to compute final scores.
        for mix in
            get_all_color_mixes(k).filter(|m| m.approximate_score(&constraints) == max_score)
        {
            for board in BoardState::permutations_from_color_mix(&mix) {
                let score = board.score(&constraints);
                if score > best_score {
                    best_board = board;
                    best_score = score;
                    if best_score == max_score {
                        break;
                    }
                }
            }

            if best_score == max_score {
                break;
            }
        }
    }
    (best_board, best_score)
}

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long = "--board")]
    board: Option<BoardState>,
    #[structopt(short, long = "--constraints", required(true))]
    constraints: ConstraintSet,
}

fn main() {
    let opt = Opt::from_args();
    match opt.board {
        Some(board) => {
            println!("{}", board.score(&opt.constraints))
        }
        None => {
            let (board, score) = solve(&opt.constraints);
            println!("{} {}", board, score)
        }
    }
}
