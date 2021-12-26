use dimension::*;
use itertools::Itertools;
use structopt::StructOpt;

// TODO(trevorm): consider color count constraints when generating this.
// This may not be necessary if blindly scoring all combinations is fast enough.
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

fn get_all_color_mixes(k: usize) -> impl Iterator<Item = ColorMix> {
    get_input_colors()
        .into_iter()
        .combinations(k)
        .as_color_mix()
}

// Gets the upper bound score for each board size.
fn get_upper_bound_scores(constraints: &[Constraint]) -> Vec<(usize, BoardScore)> {
    let mut scores: Vec<(usize, BoardScore)> = Vec::with_capacity(NUM_POSITIONS);
    for k in 1..=NUM_POSITIONS {
        let max_score = BoardScore::new(k, true);
        let mut high_score = BoardScore::default();

        for m in get_all_color_mixes(k) {
            let matching = m.approximate_matching_constraints(constraints);
            let score = BoardScore::with_state(k, constraints.len() - matching, m.has_all_colors());
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
fn solve(constraints: &[Constraint]) -> (BoardState, BoardScore) {
    let mut best_board = BoardState::new();
    let mut best_score = BoardScore::default();

    for (k, max_score) in get_upper_bound_scores(constraints) {
        if max_score < best_score {
            break;
        }

        // Iterate over all ColorMixes whose approximate_score() matches max_score, then iterate
        // over those permutations to compute final scores.
        for m in get_all_color_mixes(k).filter(|m| m.approximate_score(&constraints) == max_score) {
            let color_board_state = BoardState::from(&m);

            // TODO(trevorm): eliminate the bad touch.
            for p in color_board_state.positions.iter().permutations(k) {
                let board = BoardState::with_positions(
                    &p.into_iter().copied().collect::<Vec<Option<Color>>>(),
                );
                let score = board.score(&constraints);
                if score > best_score {
                    best_board = board;
                    best_score = score;
                    if best_score == max_score {
                        break
                    }
                }
            }

            if best_score == max_score {
                break
            }
        }
    }
    (best_board, best_score)
}

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long = "--board")]
    board: Option<BoardState>,
    #[structopt(short, long = "--constraints", use_delimiter(true), required(true))]
    constraints: Vec<Constraint>,
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
