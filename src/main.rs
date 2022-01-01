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
    for c in opt.constraints.dropped_constraints() {
        println!(
            "Dropping conflicting constraint '{}'; taking 2 point/no flag penalty.",
            c
        );
    }
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
