use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dimension::{BoardState, Constraint};
use pprof::criterion::{PProfProfiler, Output};
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
fn bm_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("score");
    let data = [ScoreBenchmarkData {
        name: "2",
        board: "KKKWWBBGGOO",
        constraints: "K3K,W4O",
    }, ScoreBenchmarkData {
        name: "6",
        board: "BGGGOOOWWWK",
        constraints: "KxB,K/*,*/B,W4K,B1B,B|G"
    }];
    for d in data.iter() {
        let board = BoardState::from_str(d.board).unwrap();
        let constraints = parse_constraints(d.constraints);
        group.bench_with_input(BenchmarkId::from_parameter(d.name), &data[0], |b, _d| {
            b.iter(|| assert!(board.score(&constraints).score > 0));
        });
    }
}

criterion_group!{
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bm_score
}
criterion_main!(benches);
