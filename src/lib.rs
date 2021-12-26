use itertools::Itertools;
use std::cmp;
use std::convert::TryFrom;
use std::fmt;
use std::str::FromStr;

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
pub const NUM_POSITIONS: usize = 11;
const BOARD_GRAPH: [PositionNode; NUM_POSITIONS] = [
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
pub enum Color {
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
pub enum Constraint {
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

#[derive(Debug)]
pub struct ColorMix {
    counts: [usize; NUM_SPHERE_COLORS],
}

impl ColorMix {
    pub fn with_colors(colors: &[Color]) -> Result<ColorMix, &'static str> {
        if colors.len() > NUM_POSITIONS {
            return Err("ColorMix may not have more than 11 colors");
        }
        let mut mix = ColorMix {
            counts: [0; NUM_SPHERE_COLORS],
        };
        // Most of the cost of this method is branching in the loop. 11 is the common case
        // so we unroll it here.
        if colors.len() == 11 {
            mix.counts[colors[0] as usize] += 1;
            mix.counts[colors[1] as usize] += 1;
            mix.counts[colors[2] as usize] += 1;
            mix.counts[colors[3] as usize] += 1;
            mix.counts[colors[4] as usize] += 1;
            mix.counts[colors[5] as usize] += 1;
            mix.counts[colors[6] as usize] += 1;
            mix.counts[colors[7] as usize] += 1;
            mix.counts[colors[8] as usize] += 1;
            mix.counts[colors[9] as usize] += 1;
            mix.counts[colors[10] as usize] += 1;
        } else {
            for c in colors {
                mix.counts[*c as usize] += 1;
            }
        }
        if mix.counts.iter().all(|&c| c <= 3) {
            Ok(mix)
        } else {
            Err("ColorMix may not have more than 3 of any color")
        }
    }

    pub fn num_spheres(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn has_all_colors(&self) -> bool {
        self.counts.iter().all(|&c| c > 0)
    }

    // Returns an upper bound of the number of matching constraints based entirely on the
    // mix of colors used.
    pub fn approximate_matching_constraints(&self, constraints: &[Constraint]) -> usize {
        return constraints
            .iter()
            .filter(|&c| self.matches_constraint(&c))
            .count();
    }

    fn matches_constraint(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Adjacent(color1, color2) => {
                if *color1 == *color2 {
                    let c = self.counts[*color1 as usize];
                    // There must be zero or 2+ to fulfill this constraint.
                    c == 0 || c > 1
                } else {
                    let c1 = self.counts[*color1 as usize];
                    let c2 = self.counts[*color2 as usize];
                    // To be adjacent there must be 0 of both or more than 0 of both.
                    (c1 == 0 && c2 == 0) || (c1 > 0 && c2 > 0)
                }
            }
            Constraint::Count(count, color) => self.counts[*color as usize] == *count,
            Constraint::SumCount(count, color1, color2) => {
                self.counts[*color1 as usize] + self.counts[*color2 as usize] == *count
            }
            Constraint::GreaterThan(color1, color2) => {
                self.counts[*color1 as usize] > self.counts[*color2 as usize]
            }
            // Approximations of these are too pessimistic given that we are trying
            // to compute an upper bound score. The best approximations only return true
            // when a targeted color is 0, where the flag cannot be set.
            Constraint::NotAdjacent(_, _) => true,
            Constraint::Below(_) => true,
            Constraint::Above(_) => true,
        }
    }

    pub fn approximate_score(&self, constraints: &[Constraint]) -> BoardScore {
        let matching = self.approximate_matching_constraints(constraints);
        BoardScore::with_state(
            self.num_spheres(),
            constraints.len() - matching,
            self.has_all_colors(),
        )
    }
}

impl FromStr for ColorMix {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut colors: Vec<Color> = Vec::with_capacity(s.len());
        for b in s.bytes() {
            match Color::try_from(b) {
                Ok(c) => colors.push(c),
                Err(e) => return Err(e),
            }
        }
        ColorMix::with_colors(&colors)
    }
}

pub struct ColorMixAdaptor<I: Iterator> {
    iter: I,
}

impl<I> Iterator for ColorMixAdaptor<I>
where
    I: Iterator<Item = Vec<Color>>,
{
    type Item = ColorMix;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(c) => {
                    let mix = ColorMix::with_colors(&c);
                    debug_assert!(mix.is_ok());
                    if mix.is_ok() {
                        return mix.ok();
                    }
                }
            }
        }
    }
}

pub trait ColorMixIterator: Iterator {
    fn as_color_mix(self) -> ColorMixAdaptor<Self>
    where
        Self: Sized + Iterator<Item = Vec<Color>>,
    {
        ColorMixAdaptor { iter: self }
    }
}

impl<I: Iterator<Item = Vec<Color>>> ColorMixIterator for I {}

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
    fn new(state: &[Option<Color>; NUM_POSITIONS]) -> BoardStateAnalysis {
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

    pub fn is_valid(&self) -> bool {
        self.positions & self.below == self.below && self.colors.iter().all(|&c| c.count() <= 3)
    }

    fn count(&self) -> usize {
        self.positions.count_ones() as usize
    }

    fn color(&self, c: Color) -> &ColorInfo {
        return &self.colors[c as usize];
    }
}

#[derive(Debug, PartialEq)]
pub struct BoardScore {
    // Score in [0,11]
    pub score: usize,
    // Set iff all color spheres are used and no constraints have been violated.
    pub flag: bool,
}

impl BoardScore {
    pub fn new(score: usize, flag: bool) -> BoardScore {
        BoardScore {
            score: score,
            flag: flag,
        }
    }

    // XXX there's got to be a better way.
    // total      -- the number of spheres in the board state.
    // invalid    -- number of non-matching constraints.
    // all_colors -- true if all colors have at least one sphere.
    pub fn with_state(total: usize, invalid: usize, all_colors: bool) -> BoardScore {
        BoardScore::new(
            if invalid * 2 < total {
                total - (invalid * 2)
            } else {
                0
            },
            invalid == 0 && all_colors,
        )
    }
}

impl Default for BoardScore {
    fn default() -> BoardScore {
        BoardScore {
            score: 0,
            flag: false,
        }
    }
}

impl Ord for BoardScore {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.score
            .cmp(&other.score)
            .then(self.flag.cmp(&other.flag))
    }
}

impl PartialOrd for BoardScore {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl Eq for BoardScore {}

impl fmt::Display for BoardScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={} flag={}", self.score, self.flag)
    }
}

type Positions = [Option<Color>; NUM_POSITIONS];
#[derive(Debug, PartialEq)]
pub struct BoardState {
    positions: Positions,
    analysis: BoardStateAnalysis,
}

impl BoardState {
    // Create a BoardState with the input positions. Returns None if the board is not valid.
    pub fn with_positions(mut positions: &[Option<Color>]) -> Option<BoardState> {
        if positions.len() > NUM_POSITIONS {
            positions = &positions[0..NUM_POSITIONS];
        }
        let mut bp: Positions = [None; NUM_POSITIONS];
        // Unroll the loop in the common case.
        if positions.len() == NUM_POSITIONS {
            bp[0] = positions[0];
            bp[1] = positions[1];
            bp[2] = positions[2];
            bp[3] = positions[3];
            bp[4] = positions[4];
            bp[5] = positions[5];
            bp[6] = positions[6];
            bp[7] = positions[7];
            bp[8] = positions[8];
            bp[9] = positions[9];
            bp[10] = positions[10];
        } else {
            for (i, p) in positions.iter().enumerate() {
                bp[i] = *p;
            }
        }
        let analysis = BoardStateAnalysis::new(&bp);
        if analysis.is_valid() {
            Some(BoardState {
                positions: bp,
                analysis: analysis,
            })
        } else {
            None
        }
    }

    // Produces all BoardState permutation from a single ColorMix.
    pub fn permutations_from_color_mix(mix: &ColorMix) -> impl Iterator<Item = BoardState> {
        let positions: Vec<Option<Color>> =
            BoardState::from(mix).positions.iter().copied().collect();
        positions.into_iter().permutations(11).as_board_state()
    }

    pub fn num_spheres(&self) -> usize {
        self.analysis.count()
    }

    // Returns false is this configuration is not valid (physically possible).
    // TODO(trevorm): return Result<(), &'static str> to get better error messages.
    pub fn is_valid(&self) -> bool {
        self.analysis.is_valid()
    }

    pub fn score(&self, constraints: &[Constraint]) -> BoardScore {
        if !self.is_valid() {
            return BoardScore::default();
        }

        let invalid = constraints.iter().fold(0, |acc, &c| {
            if self.matches_constraint(&c) {
                acc
            } else {
                acc + 1
            }
        });
        BoardScore::with_state(
            self.analysis.count(),
            invalid,
            self.analysis.colors.iter().all(|&c| c.positions > 0),
        )
    }

    fn matches_constraint(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Adjacent(color1, color2) => {
                let c1 = self.analysis.color(*color1);
                let c2 = self.analysis.color(*color2);
                c1.positions & c2.adjacent == c1.positions
                    && c2.positions & c1.adjacent == c2.positions
            }
            Constraint::NotAdjacent(color1, color2) => {
                let c1 = self.analysis.color(*color1);
                let c2 = self.analysis.color(*color2);
                c1.positions & !c2.adjacent == c1.positions
                    && c2.positions & !c1.adjacent == c2.positions
            }
            Constraint::Count(count, color) => self.analysis.color(*color).count() == *count,
            Constraint::SumCount(count, color1, color2) => {
                self.analysis.color(*color1).count() + self.analysis.color(*color2).count()
                    == *count
            }
            Constraint::Below(color) => {
                let color_positions = self.analysis.color(*color).positions;
                color_positions & self.analysis.below == color_positions
            }
            Constraint::Above(color) => {
                self.analysis.color(*color).positions & self.analysis.below == 0
            }
            Constraint::GreaterThan(color1, color2) => {
                self.analysis.color(*color1).count() > self.analysis.color(*color2).count()
            }
        }
    }
}

impl Default for BoardState {
    fn default() -> BoardState {
        BoardState {
            positions: [None; NUM_POSITIONS],
            analysis: BoardStateAnalysis::new(&[None; NUM_POSITIONS]),
        }
    }
}

impl FromStr for BoardState {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() > NUM_POSITIONS {
            return Err("cannot fill more than 11 places");
        }

        // NB: if there are less than NUM_POSITIONS characters, the rest are empty.
        let mut values: [Option<Color>; NUM_POSITIONS] = [None; NUM_POSITIONS];
        for (i, c) in s.bytes().enumerate() {
            let r = Color::try_from(c);
            if r.is_ok() {
                values[i] = r.ok()
            } else if c != b'.' {
                return Err("invalid place must be in [KWBOG.]");
            }
        }

        BoardState::with_positions(&values).ok_or("Parsed board state is not valid")
    }
}

impl fmt::Display for BoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s: [char; NUM_POSITIONS] = ['.'; NUM_POSITIONS];
        for (i, p) in self.positions.iter().enumerate() {
            s[i] = match p {
                Some(c) => match c {
                    Color::Black => 'K',
                    Color::White => 'W',
                    Color::Blue => 'B',
                    Color::Green => 'G',
                    Color::Orange => 'O',
                },
                None => '.',
            }
        }
        let r: String = s.iter().collect();
        write!(f, "{}", r)
    }
}

impl From<&ColorMix> for BoardState {
    fn from(mix: &ColorMix) -> BoardState {
        let mut positions: Positions = [None; NUM_POSITIONS];
        let mut sum = 0;
        &positions[sum..(sum + mix.counts[0])].fill(Some(Color::Black));
        sum += mix.counts[0];
        &positions[sum..(sum + mix.counts[1])].fill(Some(Color::White));
        sum += mix.counts[1];
        &positions[sum..(sum + mix.counts[2])].fill(Some(Color::Blue));
        sum += mix.counts[2];
        &positions[sum..(sum + mix.counts[3])].fill(Some(Color::Green));
        sum += mix.counts[3];
        &positions[sum..(sum + mix.counts[4])].fill(Some(Color::Orange));
        BoardState::with_positions(&positions).unwrap()
    }
}

pub struct BoardStateAdaptor<I: Iterator> {
    iter: I,
}

impl<I> Iterator for BoardStateAdaptor<I>
where
    I: Iterator<Item = Vec<Option<Color>>>,
{
    type Item = BoardState;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(p) => {
                    let board = BoardState::with_positions(&p);
                    if board.is_some() {
                        return board;
                    }
                }
            }
        }
    }
}

pub trait BoardStateIterator: Iterator {
    fn as_board_state(self) -> BoardStateAdaptor<Self>
    where
        Self: Sized + Iterator<Item = Vec<Option<Color>>>,
    {
        BoardStateAdaptor { iter: self }
    }
}

impl<I: Iterator<Item = Vec<Option<Color>>>> BoardStateIterator for I {}

mod test {
    use super::*;

    #[allow(dead_code)]
    fn parse_constraints(s: &str) -> Vec<Constraint> {
        let mut out = Vec::<Constraint>::new();
        if s.is_empty() {
            return out;
        }
        for x in s.split(',') {
            match Constraint::from_str(x) {
                Ok(c) => out.push(c),
                Err(e) => assert!(false, "Could not parse constraint {}: {}", x, e),
            }
        }
        out
    }

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

    mod board_score {
        #[allow(unused_imports)]
        use super::*;

        #[test]
        fn order_score() {
            assert!(BoardScore::new(3, false) < BoardScore::new(5, false));
            assert!(BoardScore::new(7, false) > BoardScore::new(5, false));
            assert_eq!(BoardScore::new(5, false), BoardScore::new(5, false));
        }

        #[test]
        fn order_flag() {
            assert!(BoardScore::new(5, false) < BoardScore::new(5, true));
            assert!(BoardScore::new(5, true) > BoardScore::new(5, false));
            assert_eq!(BoardScore::new(5, false), BoardScore::new(5, false));
            assert_eq!(BoardScore::new(5, true), BoardScore::new(5, true));
        }

        #[test]
        fn format() {
            assert_eq!(format!("{}", BoardScore::default()), "score=0 flag=false");
            assert_eq!(
                format!("{}", BoardScore::new(5, false)),
                "score=5 flag=false"
            );
            assert_eq!(
                format!("{}", BoardScore::new(11, true)),
                "score=11 flag=true"
            );
        }
    }

    mod color_mix_approximate_score {
        #[allow(unused_imports)]
        use super::*;
        macro_rules! test {
            ($name:ident, $s:expr, $c:expr, $score:expr) => {
                #[test]
                fn $name() {
                    let mix = ColorMix::from_str($s);
                    let constraints = parse_constraints($c);
                    assert_eq!(mix.unwrap().approximate_score(&constraints).score, $score);
                }
            };
            ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                #[test]
                fn $name() {
                    let mix = ColorMix::from_str($s);
                    let constraints = parse_constraints($c);
                    assert_eq!(
                        mix.unwrap().approximate_score(&constraints),
                        BoardScore {
                            score: $score,
                            flag: $flag
                        }
                    );
                }
            };
        }

        test!(adjacent1, "KOBG", "K|W", 2);
        test!(adjacent2, "WOBG", "K|W", 2);
        test!(adjacent3, "KWOBG", "K|W", 5);
        // This one isn't adjacent but if you're only looking at counts it's fine.
        test!(adjacent4, "GWWOKB", "K|W", 6);
        // Vacuously true (neither color appears).
        test!(adjacent5, "OBG", "K|W", 3);
        // Vacuously true (black doesn't appear).
        test!(adjacent6, "WOBG", "K|K", 4);
        // Can't match because there's no second black sphere to be next to.
        test!(adjacent7, "KWOBG", "K|K", 3);
        // 2+ spheres of same color is ok though.
        test!(adjacent8, "WKOBGK", "K|K", 6);
        test!(not_adjacent1, "KWKWKW", "KxW", 6);
        test!(not_adjacent2, "GKKWW", "KxW", 5);
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
        test!(below1, "WWW", "*/W", 3);
        test!(below2, "WWWB", "*/W", 4);
        test!(above1, "WWW", "W/*", 3);
        test!(above2, "WWWB", "W/*", 4);
        test!(greater_than1, "G", "G>O", 1);
        test!(greater_than2, "GW", "G>O", 2);
        test!(greater_than3, "GWO", "G>O", 1);
        test!(greater_than4, "GWOO", "G>O", 2);
        test!(greater_than5, "GWOOG", "G>O", 3);
        test!(greater_than6, "GWOOGG", "G>O", 6);
        test!(flag_trivial, "KWBOG", "", 5, true);
        test!(flag_constraint, "KWBOG", "K1K", 5, true);
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
                Err("Parsed board state is not valid")
            )
        }

        #[test]
        fn missing_l1() {
            assert_eq!(
                BoardState::from_str("OKKKWWW...G"),
                Err("Parsed board state is not valid")
            )
        }

        #[test]
        fn color_limit3() {
            assert_eq!(
                BoardState::from_str("OKKKWWWBBBK"),
                Err("Parsed board state is not valid")
            )
        }
    }

    mod board_state_format {
        #[allow(unused_imports)]
        use super::*;
        macro_rules! test {
            ($name: ident, $c: expr, $expected:expr) => {
                #[test]
                fn $name() {
                    let actual = format!("{}", BoardState::with_positions(&$c).unwrap());
                    assert_eq!(
                        $expected, actual,
                        "expected={} actual={}",
                        $expected, actual
                    );
                }
            };
        }

        test!(black, [Some(Color::Black)], "K..........");
        test!(white, [Some(Color::White)], "W..........");
        test!(blue, [Some(Color::Blue)], "B..........");
        test!(green, [Some(Color::Green)], "G..........");
        test!(orange, [Some(Color::Orange)], "O..........");
        test!(empty, [None, Some(Color::Orange)], ".O.........");
        test!(
            eleven,
            [
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
            ],
            "KWWWBBBGGGO"
        );
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
        #[allow(unused_imports)]
        use super::*;
        macro_rules! test {
            ($name:ident, $s:expr, $c:expr, $score:expr) => {
                #[test]
                fn $name() {
                    let state = BoardState::from_str($s);
                    let constraints = parse_constraints($c);
                    assert_eq!(state.unwrap().score(&constraints).score, $score);
                }
            };
            ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                #[test]
                fn $name() {
                    let state = BoardState::from_str($s);
                    let constraints = parse_constraints($c);
                    assert_eq!(
                        state.unwrap().score(&constraints),
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
