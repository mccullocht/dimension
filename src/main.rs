use dimension::*;
use itertools::Itertools;
use structopt::StructOpt;


// TODO(trevorm): tests for this method.
fn solve(constraints: &[Constraint]) -> (BoardState, BoardScore) {
    // TODO(trevorm): consider color count constraints when generating this.
    let input_colors = vec![
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
    ];
    let mut best_board = BoardState::new();
    let mut best_score = BoardScore::default();
    // TODO(trevorm): structure this as two passes. The first gets the top possible score
    // over the list of colors with just count constraints and emits (k, score).
    // The second iterates over all boards of length k, with cut-outs for those that don't
    // yield at least score from count constraints. This is greedy -- there are corner cases
    // where some non-count constraints have count-like behaviors (X|X) but should be pretty
    // good otherwise.
    // The second pass can be implemented as an iterator that yields all permutations that
    // meet the thresholds from the first.
    for k in (1..=NUM_POSITIONS).rev() {
        let high_score = BoardScore::new(k, true);
        if high_score < best_score {
            break;
        }
        println!("Trying mix of {} spheres", k);

        let mut min_count_score = BoardScore::default();
        for m in input_colors.iter().combinations(k) {
            let color_mix: Vec<Color> = m.into_iter().copied().collect();
            let score = BoardState::with_colors(&color_mix).count_score(&constraints);
            if score > min_count_score {
                min_count_score = score
            }
        }

        for m in input_colors.iter().combinations(k) {
            let color_mix: Vec<Color> = m.into_iter().copied().collect();
            let color_board_state = BoardState::with_colors(&color_mix);
            // Compute the upper bound score based on count constraints. If it's less than the best
            // score then there's no point pursuing this color mix.
            if color_board_state.count_score(&constraints) < min_count_score {
                continue;
            }

            for p in color_board_state.positions.iter().permutations(k) {
                let board = BoardState::with_positions(
                    &p.into_iter().copied().collect::<Vec<Option<Color>>>(),
                );
                let score = board.score(&constraints);
                if score > best_score {
                    best_board = board;
                    best_score = score;
                    if best_score == high_score {
                        break;
                    }
                }
            }

            if best_score == high_score {
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
