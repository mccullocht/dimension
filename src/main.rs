use std::convert::TryFrom;
use std::str::FromStr;
use structopt::StructOpt;

// Metadata about outbound edges for a node in the 11 node board graph.
// Each member is a bitmask representing edges of different types.
struct PositionNode {
    below: u16,
    adjacent: u16,
}

// Describe the relationship of each position on the board to all other positions.
// L0      L1                L2
//   1                1
//  /|\              /
// 6 | 2   6--9-----7--2       9    7
// |\|/|     / \   /            \  /
// | 0 |    5   \ /              10
// |/|\|         8--3             |
// 5 | 3          \               8
//  \|/            4
//   4
const BOARD_GRAPH: [PositionNode; 11] = [
    PositionNode {
        below: 0,
        adjacent: 0b01111111110,
    },
    PositionNode {
        below: 0,
        adjacent: 0b00011000101,
    },
    PositionNode {
        below: 0,
        adjacent: 0b00010001011,
    },
    PositionNode {
        below: 0,
        adjacent: 0b00100010101,
    },
    PositionNode {
        below: 0,
        adjacent: 0b00100101001,
    },
    PositionNode {
        below: 0,
        adjacent: 0b01001010001,
    },
    PositionNode {
        below: 0,
        adjacent: 0b01000100011,
    },
    PositionNode {
        below: 0b00000000111,
        adjacent: 0b11100000111,
    },
    PositionNode {
        below: 0b00000011001,
        adjacent: 0b11010011001,
    },
    PositionNode {
        below: 0b00001100001,
        adjacent: 0b10111100001,
    },
    PositionNode {
        below: 0b01110000000,
        adjacent: 0b01110000000,
    },
];

const NUM_SPHERE_COLORS: usize = 5;
#[derive(Copy, Clone, Debug, PartialEq)]
enum Color {
    Black,
    White,
    Blue,
    Green,
    Orange,
}

impl TryFrom<u8> for Color {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'K' => Ok(Color::Black),
            b'W' => Ok(Color::White),
            b'B' => Ok(Color::Blue),
            b'O' => Ok(Color::Orange),
            b'G' => Ok(Color::Green),
            _ => Err("Not a valid Color. Must be in [KWBOG]"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Constraint {
    // All spheres of these two colors must be adjacent to one another. Fmt: C1|C2
    Adjacent(Color, Color),
    // All spheres of these two colors may not be adjacent to one another. Fmt: C1XC2
    NotAdjacent(Color, Color),
    // Exactly N of the specified color. Fmt: CNC
    Count(usize, Color),
    // Exactly N of the sum of these two colors. Fmt: C1NC2
    SumCount(usize, Color, Color),
    // Spheres of this color must be below other spheres (must not be on top!) Fmt: */C
    Below(Color),
    // Spheres of this color must be above other spheres (must not be underneath!) Fmt: C/*
    Above(Color),
    // There must be more spheres of the first color than the second. Fmt: C1>C2
    GreaterThan(Color, Color),
}

fn parse_two_colors(one: u8, two: u8) -> Result<(Color, Color), &'static str> {
    let c1 = Color::try_from(one);
    if c1.is_err() {
        return Err(c1.err().unwrap());
    }
    let c2 = Color::try_from(two);
    if c2.is_err() {
        return Err(c2.err().unwrap());
    }
    return Ok((c1.unwrap(), c2.unwrap()));
}

impl FromStr for Constraint {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 3 {
            return Err("constraints are 3 characters long");
        }

        let v: Vec<u8> = s.bytes().collect();
        match v[1] {
            b'|' => match parse_two_colors(v[0], v[2]) {
                Ok((c1, c2)) => Ok(Constraint::Adjacent(c1, c2)),
                Err(e) => Err(e),
            },
            b'x' => match parse_two_colors(v[0], v[2]) {
                Ok((c1, c2)) => Ok(Constraint::NotAdjacent(c1, c2)),
                Err(e) => Err(e),
            },
            b'1'..=b'6' => {
                let n = v[1] - b'0';
                match parse_two_colors(v[0], v[2]) {
                    Ok((c1, c2)) => {
                        if c1 != c2 {
                            Ok(Constraint::SumCount(n as usize, c1, c2))
                        } else if n <= 3 {
                            Ok(Constraint::Count(n as usize, c1))
                        } else {
                            Err("unknown constraint operator")
                        }
                    }
                    Err(e) => Err(e),
                }
            }
            b'/' => {
                if v[0] == b'*' {
                    match Color::try_from(v[2]) {
                        Ok(c) => Ok(Constraint::Below(c)),
                        Err(e) => Err(e),
                    }
                } else if v[2] == b'*' {
                    match Color::try_from(v[0]) {
                        Ok(c) => Ok(Constraint::Above(c)),
                        Err(e) => Err(e),
                    }
                } else {
                    Err("below/above constraints must have exactly one wildcard [*]")
                }
            }
            b'>' => match parse_two_colors(v[0], v[2]) {
                Ok((c1, c2)) => Ok(Constraint::GreaterThan(c1, c2)),
                Err(e) => Err(e),
            },
            _ => Err("unknown constraint operator"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct ColorInfo {
    // Bitmap of matching positions.
    positions: u16,
    // Bitmap of positions adjacent to positions.
    adjacent: u16,
}

impl ColorInfo {
    fn default() -> ColorInfo {
        ColorInfo {
            positions: 0,
            adjacent: 0,
        }
    }

    fn count(self) -> usize {
        self.positions.count_ones() as usize
    }
}

#[derive(Debug, PartialEq)]
struct BoardStateAnalysis {
    // Information broken down per-color.
    colors: [ColorInfo; NUM_SPHERE_COLORS],
    // Bitmap of set positions.
    positions: u16,
    // Bitmap of all positions below set positions.
    below: u16,
}

impl BoardStateAnalysis {
    fn new(state: &[Option<Color>; BOARD_GRAPH.len()]) -> BoardStateAnalysis {
        let mut r = BoardStateAnalysis {
            colors: [ColorInfo::default(); NUM_SPHERE_COLORS],
            positions: 0,
            below: 0,
        };
        for (i, p) in state.iter().enumerate() {
            if p.is_none() {
                continue;
            }
            r.positions |= 1 << i;
            r.below |= BOARD_GRAPH[i].below;
            let mut c: &mut ColorInfo = &mut r.colors[p.unwrap() as usize];
            c.positions |= 1 << i;
            c.adjacent |= BOARD_GRAPH[i].adjacent;
        }
        r
    }

    fn count(&self) -> usize {
        self.positions.count_ones() as usize
    }

    fn color(&self, c: Color) -> &ColorInfo {
        return &self.colors[c as usize];
    }
}

#[derive(Debug, PartialEq)]
struct BoardState {
    positions: [Option<Color>; BOARD_GRAPH.len()],
    analysis: Option<BoardStateAnalysis>,
}

#[derive(Debug, PartialEq)]
struct BoardScore {
    // Score in [0,11]
    score: usize,
    // Set iff all color spheres are used and no constraints have been violated.
    flag: bool,
}

impl BoardState {
    // Returns false is this configuration is not valid (physically possible).
    // TODO(trevorm): return Result<(), &'static str> to get better error messages.
    pub fn is_valid(&mut self) -> bool {
        if self.analysis.is_none() {
            self.analysis = Some(BoardStateAnalysis::new(&self.positions));
        }
        // Every position that is below another position *must* be set.
        let analysis: &BoardStateAnalysis = self.analysis.as_ref().unwrap();
        if analysis.positions & analysis.below != analysis.below {
            false
        } else {
            analysis.colors.iter().all(|&c| c.count() <= 3)
        }
    }

    pub fn score(&mut self, constraints: &[Constraint]) -> BoardScore {
        if !self.is_valid() {
            return BoardScore {
                score: 0,
                flag: false,
            };
        }

        let analysis = self.analysis.as_ref().unwrap();
        let invalid = constraints.iter().fold(0, |acc, &c| {
            if self.matches_constraint(&c, analysis) {
                acc
            } else {
                acc + 1
            }
        });
        BoardScore {
            score: if invalid * 2 < analysis.count() {
                analysis.count() - (invalid * 2)
            } else {
                0
            },
            flag: invalid == 0 && analysis.colors.iter().all(|&c| c.positions > 0),
        }
    }

    fn matches_constraint(&self, constraint: &Constraint, analysis: &BoardStateAnalysis) -> bool {
        match constraint {
            Constraint::Adjacent(color1, color2) => {
                let c1 = analysis.color(*color1);
                let c2 = analysis.color(*color2);
                c1.positions & c2.adjacent == c1.positions
                    && c2.positions & c1.adjacent == c2.positions
            }
            Constraint::NotAdjacent(color1, color2) => {
                let c1 = analysis.color(*color1);
                let c2 = analysis.color(*color2);
                c1.positions & !c2.adjacent == c1.positions
                    && c2.positions & !c1.adjacent == c2.positions
            }
            Constraint::Count(count, color) => analysis.color(*color).count() == *count,
            Constraint::SumCount(count, color1, color2) => {
                analysis.color(*color1).count() + analysis.color(*color2).count() == *count
            }
            Constraint::Below(color) => {
                let color_positions = analysis.color(*color).positions;
                color_positions & analysis.below == color_positions
            }
            Constraint::Above(color) => analysis.color(*color).positions & analysis.below == 0,
            Constraint::GreaterThan(color1, color2) => {
                analysis.color(*color1).count() > analysis.color(*color2).count()
            }
        }
    }
}

impl FromStr for BoardState {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() > BOARD_GRAPH.len() {
            return Err("cannot fill more than 11 places");
        }

        // NB: if there are less than BOARD_GRAPH.len() characters, the rest are empty.
        let mut values: [Option<Color>; BOARD_GRAPH.len()] = [None; BOARD_GRAPH.len()];
        for (i, c) in s.bytes().enumerate() {
            let r = Color::try_from(c);
            if r.is_ok() {
                values[i] = r.ok()
            } else if c != b'.' {
                return Err("invalid place must be in [KWBOG.]");
            }
        }

        let mut state = BoardState {
            positions: values,
            analysis: None,
        };
        if state.is_valid() {
            Ok(state)
        } else {
            Err("parsed board state is not valid")
        }
    }
}

mod test {
    use super::*;

    #[test]
    fn validate_board_graph() {
        for (i, pos) in BOARD_GRAPH.iter().enumerate() {
            // This node cannot be in any of the adjacent, below, above sets.
            let node_mask: u16 = 1 << i;
            assert_eq!(pos.adjacent & node_mask, 0);
            assert_eq!(pos.below & node_mask, 0);
            let mut below = pos.below;
            while below > 0 {
                let b = below.trailing_zeros();
                assert_eq!(pos.adjacent & (1 << b), 1 << b);
                below &= !(1 << b);
            }
        }
    }

    mod constraint_from_str {
        #[allow(unused_imports)]
        use super::*;

        macro_rules! test {
            ($name: ident, $s: expr, $c: expr) => {
                #[test]
                fn $name() {
                    assert_eq!(Constraint::from_str($s), $c);
                }
            };
        }

        test!(
            adjacent1,
            "K|W",
            Ok(Constraint::Adjacent(Color::Black, Color::White))
        );
        test!(
            adjacent2,
            "W|B",
            Ok(Constraint::Adjacent(Color::White, Color::Blue))
        );
        test!(
            adjacent3,
            "O|G",
            Ok(Constraint::Adjacent(Color::Orange, Color::Green))
        );
        test!(
            not_adjacent1,
            "KxW",
            Ok(Constraint::NotAdjacent(Color::Black, Color::White))
        );
        test!(
            not_adjacent2,
            "WxB",
            Ok(Constraint::NotAdjacent(Color::White, Color::Blue))
        );
        test!(
            not_adjacent3,
            "OxG",
            Ok(Constraint::NotAdjacent(Color::Orange, Color::Green))
        );
        test!(count1, "K1K", Ok(Constraint::Count(1, Color::Black)));
        test!(count2, "W2W", Ok(Constraint::Count(2, Color::White)));
        test!(count3, "O3O", Ok(Constraint::Count(3, Color::Orange)));
        test!(count_0, "G0G", Err("unknown constraint operator"));
        test!(count_4, "G4G", Err("unknown constraint operator"));
        test!(
            sum_count1,
            "K1W",
            Ok(Constraint::SumCount(1, Color::Black, Color::White))
        );
        test!(
            sum_count4,
            "B4O",
            Ok(Constraint::SumCount(4, Color::Blue, Color::Orange))
        );
        test!(
            sum_count5,
            "G5B",
            Ok(Constraint::SumCount(5, Color::Green, Color::Blue))
        );
        test!(
            sum_count6,
            "G6O",
            Ok(Constraint::SumCount(6, Color::Green, Color::Orange))
        );
        test!(sum_count0, "K0W", Err("unknown constraint operator"));
        test!(sum_count7, "K7W", Err("unknown constraint operator"));
        test!(below1, "*/K", Ok(Constraint::Below(Color::Black)));
        test!(below2, "*/O", Ok(Constraint::Below(Color::Orange)));
        test!(above1, "W/*", Ok(Constraint::Above(Color::White)));
        test!(above2, "G/*", Ok(Constraint::Above(Color::Green)));
        test!(
            above_below_invalid,
            "B/W",
            Err("below/above constraints must have exactly one wildcard [*]")
        );
        test!(not_an_operator, "K-W", Err("unknown constraint operator"));
    }

    mod board_state_from_str {
        #[allow(unused_imports)]
        use super::*;
        macro_rules! test {
            ($name: ident, $s: expr, $($c:expr),*) => {
                #[test]
                fn $name() {
                    let mut positions: [Option<Color>; BOARD_GRAPH.len()] = [None; BOARD_GRAPH.len()];
                    let mut p = 0;
                    $(
                        #[allow(unused_assignments)]
                        {
                            assert!(p < BOARD_GRAPH.len());
                            positions[p] = $c;
                            p = p + 1;
                        }
                    )*
                    assert_eq!(BoardState::from_str($s).unwrap().positions, positions, "input={} actual={:?}", $s, positions);
                }
            }
        }
        test!(black, "K", Some(Color::Black));
        test!(white, "W", Some(Color::White));
        test!(blue, "B", Some(Color::Blue));
        test!(green, "G", Some(Color::Green));
        test!(orange, "O", Some(Color::Orange));
        test!(empty, ".O", None, Some(Color::Orange));
        test!(
            eleven,
            "KWWWBBBGGGO",
            Some(Color::Black),
            Some(Color::White),
            Some(Color::White),
            Some(Color::White),
            Some(Color::Blue),
            Some(Color::Blue),
            Some(Color::Blue),
            Some(Color::Green),
            Some(Color::Green),
            Some(Color::Green),
            Some(Color::Orange)
        );

        #[test]
        fn invalid_color() {
            assert_eq!(
                BoardState::from_str("X"),
                Err("invalid place must be in [KWBOG.]")
            )
        }

        #[test]
        fn too_long() {
            assert_eq!(
                BoardState::from_str("KKKWWWOOOGGG"),
                Err("cannot fill more than 11 places")
            )
        }

        #[test]
        fn missing_l0() {
            assert_eq!(
                BoardState::from_str(".KKKWWWOOOG"),
                Err("parsed board state is not valid")
            )
        }

        #[test]
        fn missing_l1() {
            assert_eq!(
                BoardState::from_str("OKKKWWW...G"),
                Err("parsed board state is not valid")
            )
        }

        #[test]
        fn color_limit3() {
            assert_eq!(
                BoardState::from_str("OKKKWWWBBBK"),
                Err("parsed board state is not valid")
            )
        }
    }

    mod board_state_valid {
        #[allow(unused_imports)]
        use super::*;
        macro_rules! test {
            ($name: ident, $s: expr) => {
                #[test]
                fn $name() {
                    assert_eq!(BoardState::from_str($s).unwrap().is_valid(), true);
                }
            };
        }

        test!(short_stack, "WWWB");
        test!(ten_stack, "KWWWBBBGGG");
        test!(eleven_stack, "KWWWBBBGGGO");
    }

    mod board_state_score {
        use super::*;
        #[allow(dead_code)]
        fn parse_constraints(s: &str) -> Result<Vec<Constraint>, String> {
            let mut out = Vec::<Constraint>::new();
            if s.is_empty() {
                return Ok(out);
            }
            for x in s.split(',') {
                match Constraint::from_str(x) {
                    Ok(c) => out.push(c),
                    Err(e) => {
                        let f: String = format!("Could not parse constraint {}: {}", x, e);
                        return Err(f);
                    }
                }
            }
            Ok(out)
        }

        macro_rules! test {
            ($name:ident, $s:expr, $c:expr, $score:expr) => {
                #[test]
                fn $name() {
                    let state = BoardState::from_str($s);
                    let constraints = parse_constraints($c);
                    assert_eq!(state.unwrap().score(&constraints.unwrap()).score, $score);
                }
            };
            ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                #[test]
                fn $name() {
                    let state = BoardState::from_str($s);
                    let constraints = parse_constraints($c);
                    assert_eq!(
                        state.unwrap().score(&constraints.unwrap()),
                        BoardScore {
                            score: $score,
                            flag: $flag
                        }
                    );
                }
            };
        }

        test!(adjacent1, ".KWKWKW", "K|W", 6);
        test!(adjacent2, ".KWKWKW", "W|K", 6);
        test!(adjacent3, "GKWKWKGOOOW", "K|W", 9);
        test!(adjacent4, "GKWKWKGOOOW", "W|K", 9);
        test!(not_adjacent1, ".KWKWKW", "KxW", 4);
        test!(not_adjacent2, ".KWKWKW", "WxK", 4);
        test!(not_adjacent3, "GKK.WW", "KxW", 5);
        test!(not_adjacent4, "GKK.WW", "WxK", 5);
        test!(count1, "B", "B2B", 0);
        test!(count2, "BK", "B2B", 0);
        test!(count3, "BKK", "B2B", 1);
        test!(count4, "BKKB", "B2B", 4);
        test!(count5, "BKKBB", "B2B", 3);
        test!(sum_count1, "B", "B2K", 0);
        test!(sum_count2, "BK", "B2K", 2);
        test!(sum_count3, "BKW", "B2K", 3);
        test!(sum_count4, "BKWB", "B2K", 2);
        test!(sum_count5, "BKWK", "B2K", 2);
        test!(sum_count6, "BKW", "K2B", 3);
        test!(below1, "WWW", "*/W", 1);
        test!(below2, "WWW....B", "*/W", 4);
        test!(below3, "KWWW...B", "*/W", 3);
        test!(below4, "WBBBKKKOOOW", "*/W", 9);
        test!(below5, "GBBBKKKWWW", "*/W", 8);
        test!(above1, "WWW", "W/*", 3);
        test!(above2, "WWW....B", "W/*", 2);
        test!(above3, "KWWW...B", "W/*", 3);
        test!(above4, "WBBBKKKOOOW", "W/*", 9);
        test!(above5, "GBBBKKKOOOW", "W/*", 11);
        test!(above6, "GBBBKKKWWW", "W/*", 10);
        test!(greater_than1, "G", "G>O", 1);
        test!(greater_than2, "GW", "G>O", 2);
        test!(greater_than3, "GWO", "G>O", 1);
        test!(greater_than4, "GWOO", "G>O", 2);
        test!(greater_than5, "GWOOG", "G>O", 3);
        test!(greater_than6, "GWOOGG", "G>O", 6);
        test!(flag_trivial, "KWBOG", "", 5, true);
        test!(flag_constraint, "KWBOG", "K1K", 5, true);
    }
}

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long = "--board", required(true))]
    board: BoardState,
    #[structopt(short, long = "--constraints", use_delimiter(true), required(true))]
    constraints: Vec<Constraint>
}

fn main() {
    let mut opt = Opt::from_args();
    // TODO(trevorm): implement fmt::Display for BoardScore
    println!("{:?}", opt.board.score(&opt.constraints));
}
