use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dimension::{BoardState, Color, ColorMix, Constraint, ConstraintSet};
use pprof::criterion::{Output, PProfProfiler};
use std::convert::TryFrom;
use std::str::FromStr;

fn parse_board(s: &str) -> Vec<Option<Color>> {
    let mut out: Vec<Option<Color>> = Vec::with_capacity(s.len());
    for b in s.bytes() {
        match Color::try_from(b) {
            Ok(c) => out.push(Some(c)),
            Err(_) => {
                if b != b'.' {
                    panic!("Unrecognized character {} in board", b)
                }
                out.push(None)
            }
        }
    }
    out
}

fn parse_constraints(s: &str) -> Vec<Constraint> {
    let mut out: Vec<Constraint> = Vec::new();
    for c in s.split(",") {
        out.push(Constraint::from_str(c).unwrap())
    }
    out
}

struct ScoreBenchmarkData {
    name: &'static str,
    board: &'static str,
    constraints: &'static str,
}
const SCORE_DATA: [ScoreBenchmarkData; 2] = [
    ScoreBenchmarkData {
        name: "2",
        board: "KKKWWBBGGOO",
        constraints: "BB,WO",
    },
    ScoreBenchmarkData {
        name: "6",
        board: "BGGGOOOWWWK",
        constraints: "KxB,K/*,*/B,WK,B,B|G",
    },
];

fn bm_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("score");
    for d in SCORE_DATA.iter() {
        let board = BoardState::from_str(d.board).unwrap();
        let constraints = ConstraintSet::with_constraints(
            &Constraint::parse_list(d.constraints).expect(d.constraints),
        );
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| assert!(board.score(&constraints).score > 0));
        });
    }
}

fn bm_create_board(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_board");
    for d in SCORE_DATA.iter() {
        let positions = parse_board(d.board);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| assert!(BoardState::with_positions(&positions).unwrap().is_valid()));
        });
    }
}

fn bm_approximate_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("approximate_score");
    for d in SCORE_DATA.iter() {
        let mix = ColorMix::from_str(d.board).unwrap();
        let constraints = parse_constraints(d.constraints);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| {
                // Computing ColorMix is a fairer comparison to bm_count_score
                assert!(mix.approximate_matching_constraints(&constraints) > 0)
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bm_score, bm_create_board, bm_approximate_score
}
criterion_main!(benches);
