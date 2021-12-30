use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dimension::{BoardState, Color, ColorMix, ConstraintSet};
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

fn parse_mix(s: &str) -> Vec<Color> {
    parse_board(s).into_iter().filter_map(|c| c).collect()
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

fn bm_full_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_score");
    for d in SCORE_DATA.iter() {
        let board = BoardState::from_str(d.board).unwrap();
        let constraints = ConstraintSet::from_str(d.constraints).expect(d.constraints);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| assert!(board.score(&constraints).score > 0));
        });
    }
}

fn bm_create_board(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_board");
    for d in SCORE_DATA.iter() {
        let positions = parse_board(d.board);
        let mix = ColorMix::with_colors(&parse_mix(d.board)).expect(d.board);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| {
                assert!(
                    BoardState::with_positions_and_mix(&positions, &mix)
                        .unwrap()
                        .num_spheres()
                        > 0
                )
            });
        });
    }
}

fn bm_approximate_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("approximate_score");
    for d in SCORE_DATA.iter() {
        let mix = ColorMix::from_str(d.board).unwrap();
        let constraints = ConstraintSet::from_str(d.constraints).expect(d.constraints);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| assert!(mix.approximate_score(&constraints).score > 0));
        });
    }
}

fn bm_create_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_mix");
    for d in SCORE_DATA.iter() {
        let colors = parse_mix(d.board);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| assert!(ColorMix::with_colors(&colors).unwrap().num_spheres() > 0));
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bm_full_score, bm_create_board, bm_approximate_score, bm_create_mix
}
criterion_main!(benches);
