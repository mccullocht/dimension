use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dimension::{BoardState, ColorMix, Constraint};
use pprof::criterion::{Output, PProfProfiler};
use std::str::FromStr;

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
        constraints: "K3K,W4O",
    },
    ScoreBenchmarkData {
        name: "6",
        board: "BGGGOOOWWWK",
        constraints: "KxB,K/*,*/B,W4K,B1B,B|G",
    },
];

fn bm_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("score");
    for d in SCORE_DATA.iter() {
        let board = BoardState::from_str(d.board).unwrap();
        let constraints = parse_constraints(d.constraints);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &d, |b, _d| {
            b.iter(|| assert!(board.score(&constraints).score > 0));
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
    targets = bm_score, bm_approximate_score
}
criterion_main!(benches);
