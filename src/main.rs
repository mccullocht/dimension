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
enum SphereColor {
    Black,
    White,
    Blue,
    Green,
    Orange,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Constraint {
    // All spheres of these two colors must be adjacent to one another.
    Adjacent(SphereColor, SphereColor),
    // All spheres of these two colors may not be adjacent to one another.
    NonAdjacent(SphereColor, SphereColor),
    // Exactly N of the specified color.
    Count(usize, SphereColor),
    // Exactly N of the sum of these two colors.
    SumCount(usize, SphereColor, SphereColor),
    // Spheres of this color must be below other spheres (must not be on top!)
    Below(SphereColor),
    // Spheres of this color must be above other spheres (must not be underneath!)
    Above(SphereColor),
    // There must be more spheres of the first color than the second.
    GreaterThan(SphereColor, SphereColor),
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

#[derive(Debug)]
struct BoardStateAnalysis {
    // Information broken down per-color.
    colors: [ColorInfo; NUM_SPHERE_COLORS],
    // Bitmap of set positions.
    positions: u16,
    // Bitmap of all positions below any set positions.
    below: u16,
}

impl BoardStateAnalysis {
    fn new(state: &[Option<SphereColor>; BOARD_GRAPH.len()]) -> BoardStateAnalysis {
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

    fn color(&self, c: SphereColor) -> &ColorInfo {
        return &self.colors[c as usize];
    }
}

#[derive(Debug)]
struct BoardState {
    positions: [Option<SphereColor>; BOARD_GRAPH.len()],
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
    pub fn is_valid(&mut self) -> bool {
        if self.analysis.is_none() {
            self.analysis = Some(BoardStateAnalysis::new(&self.positions));
        }
        // Every position that is below another position *must* be set.
        let analysis: &BoardStateAnalysis = self.analysis.as_ref().unwrap();
        return analysis.positions & analysis.below == analysis.below;
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
            Constraint::NonAdjacent(color1, color2) => {
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

#[derive(Debug)]
enum ParseBoardStateError {
    // Use a single character to represent each possible value of Option<SphereColor>
    // K = Some(Black)
    // W = Some(White)
    // B = Some(Blue)
    // G = Some(Green)
    // O = Some(Orange)
    // E = None
    // Any other value is invalid.
    InvalidSphereColor,
    // May have no more than BOARD_GRAPH.len() values.
    TooLong,
}

// TODO(trevorm): test for BoardState::FromStr()
impl FromStr for BoardState {
    type Err = ParseBoardStateError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() > BOARD_GRAPH.len() {
            return Err(ParseBoardStateError::TooLong);
        }

        // NB: if there are less than BOARD_GRAPH.len() characters, the rest are empty.
        let mut values: [Option<SphereColor>; BOARD_GRAPH.len()] = [None; BOARD_GRAPH.len()];
        for (i, c) in s.chars().enumerate() {
            match c {
                'K' => values[i] = Some(SphereColor::Black),
                'W' => values[i] = Some(SphereColor::White),
                'B' => values[i] = Some(SphereColor::Blue),
                'G' => values[i] = Some(SphereColor::Green),
                'O' => values[i] = Some(SphereColor::Orange),
                'E' => values[i] = None,
                _ => return Err(ParseBoardStateError::InvalidSphereColor),
            }
        }

        Ok(BoardState {
            positions: values,
            analysis: None,
        })
    }
}

// TODO(trevorm): ~ddt using macros.
// TODO(trevorm): tests for BoardScore.flag
// TODO(trevorm): more complex constraint tests.
mod test {
    use super::*;

    struct BitVector {
        v: u64,
    }

    impl BitVector {
        fn new(v: u64) -> BitVector {
            BitVector { v: v }
        }
    }

    impl Iterator for BitVector {
        type Item = usize;
        fn next(&mut self) -> Option<Self::Item> {
            let r = self.v.trailing_zeros();
            if r == 64 {
                None
            } else {
                self.v = self.v & !(1 << r);
                Some(r as usize)
            }
        }
    }

    #[test]
    fn validate_board_graph() {
        for (i, pos) in BOARD_GRAPH.iter().enumerate() {
            // This node cannot be in any of the adjacent, below, above sets.
            let node_mask: u16 = 1 << i;
            assert_eq!(pos.adjacent & node_mask, 0);
            assert_eq!(pos.below & node_mask, 0);
            for b in BitVector::new(pos.below as u64) {
                assert_eq!(pos.adjacent & (1 << b), 1 << b);
            }
        }
    }

    #[test]
    fn is_valid() {
        fn valid(s: &str) -> bool {
            BoardState::from_str(s).expect("f").is_valid()
        }

        assert!(valid("WWWB"));
        assert!(!valid("EWWWBBBGGG"));
        assert!(valid("KWWWBBBGGG"));
        assert!(valid("KWWWBBBGGGO"));
        assert!(!valid("KWWWBBBEEEO"));
    }

    fn score(s: &str, constraints: &[Constraint]) -> BoardScore {
        BoardState::from_str(s).expect("f").score(&constraints)
    }

    #[test]
    fn adjacent_constraint() {
        let constraints = [Constraint::Adjacent(SphereColor::Black, SphereColor::White)];

        assert_eq!(score("EKWKWKW", &constraints).score, 6);
        assert_eq!(score("GKWKWKGOOOW", &constraints).score, 9);
    }

    #[test]
    fn non_adjacent_constraint() {
        let constraints = [Constraint::NonAdjacent(
            SphereColor::Black,
            SphereColor::White,
        )];

        assert_eq!(score("EKWKWKW", &constraints).score, 4);
        assert_eq!(score("GKKEWW", &constraints).score, 5);
    }

    #[test]
    fn count_constraint() {
        let constraints = [Constraint::Count(2, SphereColor::Blue)];

        assert_eq!(score("B", &constraints).score, 0);
        assert_eq!(score("BK", &constraints).score, 0);
        assert_eq!(score("BKK", &constraints).score, 1);
        assert_eq!(score("BKKB", &constraints).score, 4);
        assert_eq!(score("BKKBB", &constraints).score, 3);
    }

    #[test]
    fn sum_count_constraint() {
        let constraints = [Constraint::SumCount(
            2,
            SphereColor::Blue,
            SphereColor::Black,
        )];

        assert_eq!(score("B", &constraints).score, 0);
        assert_eq!(score("BK", &constraints).score, 2);
        assert_eq!(score("BKW", &constraints).score, 3);
        assert_eq!(score("BKWB", &constraints).score, 2);
        assert_eq!(score("BKWK", &constraints).score, 2);
    }

    #[test]
    fn below_constraint() {
        let constraints = [Constraint::Below(SphereColor::White)];

        // All on the bottom, nothing below.
        assert_eq!(score("WWW", &constraints).score, 1);
        // Blue spheres on top of them.
        assert_eq!(score("WWWEEEEB", &constraints).score, 4);
        // Blue spheres on top of most of the white spheres.
        assert_eq!(score("KWWWEEEB", &constraints).score, 3);
        // White spheres on the very top!
        assert_eq!(score("WBBBKKKOOOW", &constraints).score, 9);
        // White spheres on the level 1
        assert_eq!(score("GBBBKKKWWW", &constraints).score, 8);
    }

    #[test]
    fn above_constraint() {
        let constraints = [Constraint::Above(SphereColor::White)];

        // All on the bottom but nothing above.
        assert_eq!(score("WWW", &constraints).score, 3);
        // Blue sphere on top of them.
        assert_eq!(score("WWWEEEEB", &constraints).score, 2);
        // Blue sphere on top of most of the white spheres.
        assert_eq!(score("KWWWEEEB", &constraints).score, 3);
        // White sphere on the very top! But some still on the bottom.
        assert_eq!(score("WBBBKKKOOOW", &constraints).score, 9);
        // White sphere on the top.
        assert_eq!(score("GBBBKKKOOOW", &constraints).score, 11);
        // White spheres on level 1.
        assert_eq!(score("GBBBKKKWWW", &constraints).score, 10);
    }

    #[test]
    fn greater_than_constraint() {
        let constraints = [Constraint::GreaterThan(
            SphereColor::Green,
            SphereColor::Orange,
        )];

        assert_eq!(score("G", &constraints).score, 1);
        assert_eq!(score("GW", &constraints).score, 2);
        assert_eq!(score("GWO", &constraints).score, 1);
        assert_eq!(score("GWOO", &constraints).score, 2);
        assert_eq!(score("GWOOG", &constraints).score, 3);
        assert_eq!(score("GWOOGG", &constraints).score, 6);
    }
}

fn main() {
    println!(
        "Option<SphereColor>={} BoardState={} positions={} analysis={}",
        std::mem::size_of::<Option<SphereColor>>(),
        std::mem::size_of::<BoardState>(),
        std::mem::size_of::<[Option<SphereColor>; BOARD_GRAPH.len()]>(),
        std::mem::size_of::<Option<BoardStateAnalysis>>(),
    );
}
