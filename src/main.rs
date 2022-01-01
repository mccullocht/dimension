use dimension::*;
use structopt::StructOpt;

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
            let (board, score) = opt.constraints.solve();
            println!("{} {}", board, score)
        }
    }
}
