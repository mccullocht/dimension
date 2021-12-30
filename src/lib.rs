#[macro_use]
extern crate lazy_static;

use itertools::Itertools;
use std::cmp;
use std::collections::{HashMap, HashSet};
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
// XXX do the below optimization for valid board state where we memoize the below value based on
// the top 4 bits (L1/L2) sphere configuration.

const NUM_SPHERE_COLORS: usize = 5;
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Color {
    Black,
    White,
    Blue,
    Green,
    Orange,
}
const ALL_COLORS: [Color; NUM_SPHERE_COLORS] = [
    Color::Black,
    Color::White,
    Color::Blue,
    Color::Green,
    Color::Orange,
];

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

impl From<Color> for u8 {
    fn from(value: Color) -> u8 {
        match value {
            Color::Black => b'K',
            Color::White => b'W',
            Color::Blue => b'B',
            Color::Green => b'G',
            Color::Orange => b'O',
        }
    }
}

impl From<Color> for char {
    fn from(value: Color) -> char {
        u8::from(value) as char
    }
}

// TODO(trevorm): move Constraint and ConstraintSet to another file.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Constraint {
    // All spheres of these two colors must be adjacent to one another. Fmt: C1|C2
    Adjacent(Color, Color),
    // All spheres of these two colors may not be adjacent to one another. Fmt: C1XC2
    NotAdjacent(Color, Color),
    // Exactly N of the specified color. Fmt: CNC
    Count(usize, Color),
    // Exactly 4 of the sum of these two colors. Fmt: C1NC2
    SumCount(Color, Color),
    // Spheres of this color must be below other spheres (must not be on top!) Fmt: */C
    Below(Color),
    // Spheres of this color must be above other spheres (must not be underneath!) Fmt: C/*
    Above(Color),
    // There must be more spheres of the first color than the second. Fmt: C1>C2
    GreaterThan(Color, Color),
}

fn get_valid_constraints() -> impl Iterator<Item = Constraint> {
    let adjacent = ALL_COLORS
        .iter()
        .combinations_with_replacement(2)
        .map(|p| Constraint::Adjacent(*p[0], *p[1]));
    let not_adjacent = ALL_COLORS
        .iter()
        .combinations_with_replacement(2)
        .map(|p| Constraint::NotAdjacent(*p[0], *p[1]));
    let count = (1..=2)
        .cartesian_product(ALL_COLORS.iter())
        .map(|(n, c)| Constraint::Count(n, *c));
    let sum_count = [
        (Color::Black, Color::White),
        (Color::White, Color::Orange),
        (Color::Orange, Color::Blue),
        (Color::Blue, Color::Green),
        (Color::Green, Color::Black),
    ]
    .iter()
    .map(|(c1, c2)| Constraint::SumCount(*c1, *c2));
    let below = ALL_COLORS.iter().map(|c| Constraint::Below(*c));
    let above = ALL_COLORS.iter().map(|c| Constraint::Above(*c));
    let greater_than = [
        (Color::White, Color::Green),
        (Color::Green, Color::Orange),
        (Color::Orange, Color::Black),
        (Color::Black, Color::Blue),
        (Color::Blue, Color::White),
    ]
    .iter()
    .map(|(c1, c2)| Constraint::GreaterThan(*c1, *c2));
    return adjacent
        .chain(not_adjacent)
        .chain(count)
        .chain(sum_count)
        .chain(below)
        .chain(above)
        .chain(greater_than);
}

lazy_static! {
    static ref VALID_CONSTRAINTS: HashSet<Constraint> = {
        let mut s: HashSet<Constraint> = HashSet::new();
        for c in get_valid_constraints() {
            s.insert(c);
        }
        s
    };
}

impl Constraint {
    // Parses a comma separated list of constraints.
    // TODO(trevorm): consider altering return type to String and noting the failed
    // constraint in the parse.
    pub fn parse_list(s: &str) -> Result<Vec<Constraint>, &'static str> {
        if s.is_empty() {
            return Ok(Vec::new());
        }
        let constraints: Vec<Constraint> = s
            .split(",")
            .map(|e| Constraint::from_str(e))
            .try_collect()?;
        if constraints.iter().unique().count() == constraints.len() {
            Ok(constraints)
        } else {
            Err("all constraints in list must be unique")
        }
    }

    fn is_valid(&self) -> bool {
        VALID_CONSTRAINTS.contains(self)
    }

    // Returns the constraint or its canonicalized form, or None if constraint isn't valid.
    fn canonicalize(&self) -> Option<Constraint> {
        if self.is_valid() {
            return Some(*self);
        }

        match self {
            // Adjacent and NotAdjacent are valid for all color combinations.
            // The operations are commutative so swap to canonical order.
            Constraint::Adjacent(c1, c2) => Some(Constraint::Adjacent(*c2, *c1)),
            Constraint::NotAdjacent(c1, c2) => Some(Constraint::NotAdjacent(*c2, *c1)),
            // Not all color combinations are valid. Try swapping to see if it is canonical
            // otherwise fail.
            Constraint::SumCount(c1, c2) => {
                let c = Constraint::SumCount(*c2, *c1);
                if c.is_valid() {
                    Some(c)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // Weight for use in scoring. Typically 1, but 2 for Count(3, C) since it's always a
    // combination of Count(1, C) and Count(2, C).
    fn weight(&self) -> usize {
        match self {
            Constraint::Count(n, _) => {
                if *n == 3 {
                    2
                } else {
                    1
                }
            }
            _ => 1,
        }
    }

    // Return a Constraint that can be used as a map key to find conflicts. This covers direct
    // conflicts (inverse condition or same condition with different parameters). This does not
    // cover indirect conflicts. A non-exhaustive list of such conflicts:
    // * GreaterThan(C, D) -- Count(3, D)
    // * Adjacent(C, C) -- Count(1, C)
    // Both of these are well-handled by existing optimizations.
    fn conflict_key(&self) -> Constraint {
        match self {
            Constraint::NotAdjacent(c1, c2) => Constraint::Adjacent(*c1, *c2),
            Constraint::Count(_, c) => Constraint::Count(0, *c),
            Constraint::Above(c) => Constraint::Below(*c),
            _ => *self,
        }
    }
}

fn parse_two_colors(one: u8, two: u8) -> Result<(Color, Color), &'static str> {
    let c1 = Color::try_from(one)?;
    let c2 = Color::try_from(two)?;
    return Ok((c1, c2));
}

impl FromStr for Constraint {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("constraint may not be empty");
        }

        let v: Vec<u8> = s.bytes().collect();
        if v.len() == 1 {
            let c = Color::try_from(v[0])?;
            return Ok(Constraint::Count(1, c));
        }

        if v.len() == 2 {
            let (c1, c2) = parse_two_colors(v[0], v[1])?;
            if c1 == c2 {
                return Ok(Constraint::Count(2, c1));
            } else {
                return Constraint::SumCount(c1, c2)
                    .canonicalize()
                    .ok_or("Not a valid constraint");
            }
        }

        let c = match v[1] {
            b'|' => {
                let (c1, c2) = parse_two_colors(v[0], v[2])?;
                Ok(Constraint::Adjacent(c1, c2))
            }
            b'x' => {
                let (c1, c2) = parse_two_colors(v[0], v[2])?;
                Ok(Constraint::NotAdjacent(c1, c2))
            }
            b'/' => {
                if v[0] == b'*' {
                    let c = Color::try_from(v[2])?;
                    Ok(Constraint::Below(c))
                } else if v[2] == b'*' {
                    let c = Color::try_from(v[0])?;
                    Ok(Constraint::Above(c))
                } else {
                    Err("below/above constraints must have exactly one wildcard [*]")
                }
            }
            b'>' => {
                let (c1, c2) = parse_two_colors(v[0], v[2])?;
                Ok(Constraint::GreaterThan(c1, c2))
            }
            _ => Err("unknown constraint operator"),
        }?;
        c.canonicalize().ok_or("Not a valid constraint")
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::Adjacent(c1, c2) => write!(f, "{}|{}", char::from(*c1), char::from(*c2)),
            Constraint::NotAdjacent(c1, c2) => write!(f, "{}x{}", char::from(*c1), char::from(*c2)),
            Constraint::Count(n, c) => {
                write!(
                    f,
                    "{}",
                    std::iter::repeat(char::from(*c))
                        .take(*n)
                        .collect::<String>()
                )
            }
            Constraint::SumCount(c1, c2) => write!(f, "{}{}", char::from(*c1), char::from(*c2)),
            Constraint::Below(c) => write!(f, "*/{}", char::from(*c)),
            Constraint::Above(c) => write!(f, "{}/*", char::from(*c)),
            Constraint::GreaterThan(c1, c2) => write!(f, "{}>{}", char::from(*c1), char::from(*c2)),
        }
    }
}

pub struct ConstraintSet {
    scoring: Vec<Constraint>,
    dropped: Vec<Constraint>,
    total_weight: usize,
    max_weight: usize,
}

// TODO(trevorm): convenience method? from_str()?
impl ConstraintSet {
    // REQUIRES: no duplicates within constraints. This is true if constraints were generated
    // using Constraint::parse_list().
    pub fn with_constraints(constraints: &[Constraint]) -> ConstraintSet {
        let mut conflict_map: HashMap<Constraint, Vec<Constraint>> =
            HashMap::with_capacity(constraints.len());
        for c in constraints {
            conflict_map
                .entry(c.conflict_key())
                .or_insert(Vec::new())
                .push(*c);
        }

        let mut scoring: Vec<Constraint> = Vec::with_capacity(constraints.len());
        let mut dropped: Vec<Constraint> = Vec::new();
        for (conflict_key, mut conflict_constraints) in conflict_map.into_iter() {
            if conflict_constraints.len() == 1 {
                scoring.push(conflict_constraints[0]);
                continue;
            }

            // Based on Constraint.conflict_key() and disallowing duplicates it should not be
            // possible to have more than two constraints.
            debug_assert!(conflict_constraints.len() == 2);
            conflict_constraints.sort();
            match conflict_key {
                Constraint::Count(_, c) => scoring.push(Constraint::Count(3, c)),
                _ => {
                    scoring.push(conflict_constraints[0]);
                    dropped.push(conflict_constraints[1]);
                }
            }
        }
        scoring.sort();
        dropped.sort();
        let dropped_weight = dropped.iter().map(|c| c.weight()).sum::<usize>();
        ConstraintSet {
            scoring: scoring,
            dropped: dropped,
            total_weight: constraints.len(),
            max_weight: constraints.len() - dropped_weight,
        }
    }

    pub fn scoring_constraints(&self) -> &[Constraint] {
        &self.scoring
    }

    pub fn dropped_constraints(&self) -> &[Constraint] {
        &self.dropped
    }

    // Compute BoardScore for these constraints with scoring output.
    pub fn compute_score(
        &self,
        num_spheres: usize,
        matching_weight: usize,
        has_all_colors: bool,
    ) -> BoardScore {
        let penalty = (self.total_weight - matching_weight) * 2;
        BoardScore {
            score: if num_spheres > penalty {
                num_spheres - penalty
            } else {
                0
            },
            flag: penalty == 0 && has_all_colors,
        }
    }

    // Compute maximum possible BoardScore given these constraints.
    pub fn max_score(&self, num_spheres: usize) -> BoardScore {
        self.compute_score(num_spheres, self.max_weight, true)
    }
}

#[derive(Debug)]
pub struct ColorMix {
    // Pack everything into a single u32 containing 5 4-bit values.
    // 4 bits is enough as the input will not have more than 11 input values.
    // This is a form of vectorization that allows us to quickly compute
    // "all values <= 3" and "all values > 0".
    rep: u32,
}

impl ColorMix {
    pub fn with_colors(colors: &[Color]) -> Result<ColorMix, &'static str> {
        if colors.len() > NUM_POSITIONS {
            return Err("ColorMix may not have more than 11 colors");
        }
        let mut mix = ColorMix { rep: 0 };
        // Most of the cost of this method is branching in the loop. 11 is the common case
        // so we unroll it here.
        if colors.len() == 11 {
            mix.add(colors[0]);
            mix.add(colors[1]);
            mix.add(colors[2]);
            mix.add(colors[3]);
            mix.add(colors[4]);
            mix.add(colors[5]);
            mix.add(colors[6]);
            mix.add(colors[7]);
            mix.add(colors[8]);
            mix.add(colors[9]);
            mix.add(colors[10]);
        } else {
            for c in colors {
                mix.add(*c);
            }
        }
        if mix.rep & 0x33333 == mix.rep {
            Ok(mix)
        } else {
            Err("ColorMix may not have more than 3 of any color")
        }
    }

    fn add(&mut self, c: Color) {
        self.rep += 1 << (c as usize * 4)
    }

    fn count(&self, c: Color) -> usize {
        (self.rep as usize >> (c as usize * 4)) & 0xf
    }

    pub fn num_spheres(&self) -> usize {
        self.count(Color::Black)
            + self.count(Color::White)
            + self.count(Color::Blue)
            + self.count(Color::Green)
            + self.count(Color::Orange)
    }

    pub fn has_all_colors(&self) -> bool {
        (self.rep & 0x11111 | (self.rep >> 1) & 0x11111).count_ones() == 5
    }

    // Returns an upper bound of the number of matching constraints based entirely on the
    // mix of colors used.
    pub fn approximate_matching_constraints(&self, constraints: &[Constraint]) -> usize {
        return constraints
            .iter()
            .map(|&c| self.matches_constraint(&c))
            .sum();
    }

    fn matches_constraint(&self, constraint: &Constraint) -> usize {
        let m: bool = match constraint {
            Constraint::Adjacent(color1, color2) => {
                // If both colors are placed they must be adjacent. Any board with just one color
                // passes; if the colors are the same there must be 0 or 2+
                *color1 != *color2 || self.count(*color1) != 1
            }
            Constraint::Count(count, color) => self.count(*color) == *count,
            Constraint::SumCount(color1, color2) => self.count(*color1) + self.count(*color2) == 4,
            Constraint::GreaterThan(color1, color2) => self.count(*color1) > self.count(*color2),
            // Approximations of these are too pessimistic given that we are trying
            // to compute an upper bound score. The best approximations only return true
            // when a targeted color is 0, where the flag cannot be set.
            Constraint::NotAdjacent(_, _) => true,
            Constraint::Below(_) => true,
            Constraint::Above(_) => true,
        };
        if m {
            constraint.weight()
        } else {
            0
        }
    }

    pub fn approximate_score(&self, constraints: &ConstraintSet) -> BoardScore {
        constraints.compute_score(
            self.num_spheres(),
            self.approximate_matching_constraints(constraints.scoring_constraints()),
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
    fn count(self) -> usize {
        self.positions.count_ones() as usize
    }

    fn add(&mut self, position: usize) {
        self.positions |= 1 << position;
        self.adjacent |= BOARD_GRAPH[position].adjacent;
    }
}

impl Default for ColorInfo {
    fn default() -> ColorInfo {
        ColorInfo {
            positions: 0,
            adjacent: 0,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
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

impl fmt::Display for BoardScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={} flag={}", self.score, self.flag)
    }
}

type Positions = [Option<Color>; NUM_POSITIONS];
#[derive(Debug, PartialEq)]
pub struct BoardState {
    color_info: [ColorInfo; NUM_SPHERE_COLORS],
    positions: u16,
    below: u16,
    has_all_colors: bool,
}

impl BoardState {
    // Create a BoardState with the input positions. Returns None if the board is not valid.
    pub fn with_positions(positions: &[Option<Color>]) -> Option<BoardState> {
        BoardState::with_positions_and_meta(&positions, None)
    }

    // Create a BoardState with the input positions. Returns None if the board is not valid.
    pub fn with_positions_and_meta(
        mut positions: &[Option<Color>],
        has_all_colors: Option<bool>,
    ) -> Option<BoardState> {
        if positions.len() > NUM_POSITIONS {
            positions = &positions[0..NUM_POSITIONS];
        }
        let mut board = BoardState::default();
        // Unroll the loop in the common case.
        if positions.len() == NUM_POSITIONS {
            board.add_position(positions[0], 0);
            board.add_position(positions[1], 1);
            board.add_position(positions[2], 2);
            board.add_position(positions[3], 3);
            board.add_position(positions[4], 4);
            board.add_position(positions[5], 5);
            board.add_position(positions[6], 6);
            board.add_position(positions[7], 7);
            board.add_position(positions[8], 8);
            board.add_position(positions[9], 9);
            board.add_position(positions[10], 10);
        } else {
            for (i, p) in positions.iter().enumerate() {
                board.add_position(*p, i);
            }
        }
        if board.is_valid() {
            board.has_all_colors =
                has_all_colors.unwrap_or_else(|| board.color_info.iter().all(|&c| c.positions > 0));
            Some(board)
        } else {
            None
        }
    }

    fn add_position(&mut self, color: Option<Color>, index: usize) {
        match color {
            Some(c) => {
                self.color_info[c as usize].add(index);
                self.positions |= 1 << index;
                self.below |= BOARD_GRAPH[index].below;
            }
            None => {}
        }
    }

    // Produces all BoardState permutation from a single ColorMix.
    pub fn permutations_from_color_mix(mix: &ColorMix) -> impl Iterator<Item = BoardState> {
        let positions: Vec<Option<Color>> =
            BoardState::from(mix).positions().iter().copied().collect();
        // If we have less than 11 spheres we may be able to run fewer permutations on the
        // beginning of the array ("lower" spheres), which will skip permutations that
        // produce invalid board states.
        let p = match mix.num_spheres() {
            0..=3 => 7,   // need at least 4 spheres to stack anything, ~5K permutations.
            4..=10 => 10, // top sphere cannot be stacked, ~3.6M permutations.
            _ => 11,      // ~40M permutations.
        };
        positions
            .into_iter()
            .permutations(p)
            .as_board_state(mix.has_all_colors())
    }

    pub fn num_spheres(&self) -> usize {
        self.positions.count_ones() as usize
    }

    fn fill_positions_from_bitmap(&self, c: Color, mut bitmap: u16, p: &mut Positions) {
        while bitmap > 0 {
            let idx = bitmap.trailing_zeros();
            p[idx as usize] = Some(c);
            bitmap &= !(1 << idx);
        }
    }

    fn positions(&self) -> Positions {
        let mut p: Positions = [None; NUM_POSITIONS];
        self.fill_positions_from_bitmap(Color::Black, self.color_info[0].positions, &mut p);
        self.fill_positions_from_bitmap(Color::White, self.color_info[1].positions, &mut p);
        self.fill_positions_from_bitmap(Color::Blue, self.color_info[2].positions, &mut p);
        self.fill_positions_from_bitmap(Color::Green, self.color_info[3].positions, &mut p);
        self.fill_positions_from_bitmap(Color::Orange, self.color_info[4].positions, &mut p);
        p
    }

    // Returns false is this configuration is not valid (physically possible).
    // TODO(trevorm): return Result<(), &'static str> to get better error messages.
    fn is_valid(&self) -> bool {
        self.positions & self.below == self.below && self.color_info.iter().all(|&c| c.count() <= 3)
    }

    pub fn score(&self, constraints: &ConstraintSet) -> BoardScore {
        // Factory functions do not allow construction of invalid boards.
        debug_assert!(self.is_valid());
        let matching = constraints
            .scoring_constraints()
            .iter()
            .map(|c| self.matches_constraint(c))
            .sum();
        constraints.compute_score(self.num_spheres(), matching, self.has_all_colors)
    }

    fn color(&self, c: Color) -> ColorInfo {
        self.color_info[c as usize]
    }

    fn matches_constraint(&self, constraint: &Constraint) -> usize {
        let f: bool = match constraint {
            Constraint::Adjacent(color1, color2) => {
                let c1 = self.color(*color1);
                let c2 = self.color(*color2);
                c1.positions == 0
                    || c2.positions == 0
                    || (c1.positions & c2.adjacent == c1.positions
                        && c2.positions & c1.adjacent == c2.positions)
            }
            Constraint::NotAdjacent(color1, color2) => {
                let c1 = self.color(*color1);
                let c2 = self.color(*color2);
                c1.positions & !c2.adjacent == c1.positions
                    && c2.positions & !c1.adjacent == c2.positions
            }
            Constraint::Count(count, color) => self.color(*color).count() == *count,
            Constraint::SumCount(color1, color2) => {
                self.color(*color1).count() + self.color(*color2).count() == 4
            }
            Constraint::Below(color) => {
                let color_positions = self.color(*color).positions;
                color_positions & self.below == color_positions
            }
            Constraint::Above(color) => self.color(*color).positions & self.below == 0,
            Constraint::GreaterThan(color1, color2) => {
                self.color(*color1).count() > self.color(*color2).count()
            }
        };
        if f {
            constraint.weight()
        } else {
            0
        }
    }
}

impl Default for BoardState {
    fn default() -> BoardState {
        BoardState {
            color_info: [ColorInfo::default(); NUM_SPHERE_COLORS],
            positions: 0,
            below: 0,
            has_all_colors: false,
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
        let mut values: Positions = [None; NUM_POSITIONS];
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
        for (i, p) in self.positions().iter().enumerate() {
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
        let mut board = BoardState::default();
        let k = mix.count(Color::Black);
        for i in 0..k {
            board.add_position(Some(Color::Black), i)
        }
        let mut sum = k;
        let w = mix.count(Color::White);
        for i in sum..(sum + w) {
            board.add_position(Some(Color::White), i)
        }
        sum += w;
        let b = mix.count(Color::Blue);
        for i in sum..(sum + b) {
            board.add_position(Some(Color::Blue), i)
        }
        sum += b;
        let g = mix.count(Color::Green);
        for i in sum..(sum + g) {
            board.add_position(Some(Color::Green), i)
        }
        sum += g;
        let o = mix.count(Color::Orange);
        for i in sum..(sum + o) {
            board.add_position(Some(Color::Orange), i)
        }
        board.has_all_colors = mix.has_all_colors();
        // ColorMix guarantees that we won't have more than 3 of each color, and this arrangement
        // without gaps ensures that the board will be valid.
        board
    }
}

pub struct BoardStateAdaptor<I: Iterator> {
    iter: I,
    has_all_colors: bool,
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
                    let board = BoardState::with_positions_and_meta(&p, Some(self.has_all_colors));
                    if board.is_some() {
                        return board;
                    }
                }
            }
        }
    }
}

pub trait BoardStateIterator: Iterator {
    fn as_board_state(self, has_all_colors: bool) -> BoardStateAdaptor<Self>
    where
        Self: Sized + Iterator<Item = Vec<Option<Color>>>,
    {
        BoardStateAdaptor {
            iter: self,
            has_all_colors: has_all_colors,
        }
    }
}

impl<I: Iterator<Item = Vec<Option<Color>>>> BoardStateIterator for I {}

mod test {
    #[allow(unused_imports)]
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

    mod constraint {
        #[allow(unused_imports)]
        use super::*;

        mod canonicalize {
            #[allow(unused_imports)]
            use super::*;

            macro_rules! test {
                ($name: ident, $input: expr) => {
                    test!($name, $input, Some($input));
                };
                ($name: ident, $input: expr, $output: expr) => {
                    #[test]
                    fn $name() {
                        assert_eq!($input.canonicalize(), $output);
                    }
                };
            }

            test!(adjacent, Constraint::Adjacent(Color::Blue, Color::Green));
            test!(
                adjacent_non_canonical,
                Constraint::Adjacent(Color::Green, Color::Blue),
                Some(Constraint::Adjacent(Color::Blue, Color::Green))
            );
            test!(
                not_adjacent,
                Constraint::NotAdjacent(Color::Blue, Color::Green)
            );
            test!(
                not_adjacent_non_canonical,
                Constraint::NotAdjacent(Color::Green, Color::Blue),
                Some(Constraint::NotAdjacent(Color::Blue, Color::Green))
            );
            test!(sum_count, Constraint::SumCount(Color::Blue, Color::Green));
            test!(
                sum_count_non_canonical,
                Constraint::SumCount(Color::Green, Color::Blue),
                Some(Constraint::SumCount(Color::Blue, Color::Green))
            );
            test!(
                sum_count_invalid,
                Constraint::SumCount(Color::Green, Color::White),
                None
            );
            test!(
                greater_than,
                Constraint::GreaterThan(Color::Blue, Color::White)
            );
            test!(
                greater_than_invalid,
                Constraint::GreaterThan(Color::White, Color::Blue),
                None
            );
            test!(count, Constraint::Count(1, Color::Blue));
            test!(below, Constraint::Below(Color::Blue));
            test!(above, Constraint::Above(Color::Blue));
        }

        mod serde {
            #[allow(unused_imports)]
            use super::*;

            macro_rules! test {
                ($name: ident, $s: expr, $c: expr) => {
                    test!($name, $s, $c, $s);
                };
                ($name: ident, $s: expr, $c: expr, $es: expr) => {
                    #[test]
                    fn $name() {
                        let constraint = Constraint::from_str($s);
                        assert_eq!(constraint, $c);
                        if constraint.is_ok() {
                            assert_eq!(format!("{}", constraint.unwrap()), $es);
                        }
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
                "O|G",
                Ok(Constraint::Adjacent(Color::Green, Color::Orange)),
                "G|O"
            );
            test!(
                not_adjacent1,
                "KxW",
                Ok(Constraint::NotAdjacent(Color::Black, Color::White))
            );
            test!(
                not_adjacent2,
                "OxG",
                Ok(Constraint::NotAdjacent(Color::Green, Color::Orange)),
                "GxO"
            );
            test!(count1, "K", Ok(Constraint::Count(1, Color::Black)));
            test!(count2, "WW", Ok(Constraint::Count(2, Color::White)));
            test!(
                count1_invalid,
                "X",
                Err("Not a valid Color. Must be in [KWBOG]")
            );
            test!(
                count2_invalid,
                "KX",
                Err("Not a valid Color. Must be in [KWBOG]")
            );
            test!(
                sum_count1,
                "BG",
                Ok(Constraint::SumCount(Color::Blue, Color::Green))
            );
            test!(
                sum_count2,
                "GB",
                Ok(Constraint::SumCount(Color::Blue, Color::Green)),
                "BG"
            );
            test!(sum_count_invalid, "KO", Err("Not a valid constraint"));
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

        mod parse_list {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            fn parse() {
                let constraints = Constraint::parse_list("K,KK,O|G,OW,GxK,*/W").unwrap();
                assert_eq!(
                    constraints,
                    &[
                        Constraint::Count(1, Color::Black),
                        Constraint::Count(2, Color::Black),
                        Constraint::Adjacent(Color::Green, Color::Orange),
                        Constraint::SumCount(Color::White, Color::Orange),
                        Constraint::NotAdjacent(Color::Black, Color::Green),
                        Constraint::Below(Color::White),
                    ]
                );
            }

            #[test]
            fn empty() {
                assert_eq!(Constraint::parse_list(""), Ok(Vec::new()));
            }

            #[test]
            fn duplicate() {
                assert_eq!(
                    Constraint::parse_list("O|G,G|O"),
                    Err("all constraints in list must be unique")
                );
            }

            #[test]
            fn invalid_constraint() {
                assert_eq!(
                    Constraint::parse_list("O|G,GXO"),
                    Err("unknown constraint operator")
                );
            }
        }

        mod weight {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            fn count() {
                assert_eq!(Constraint::Count(1, Color::Blue).weight(), 1);
                assert_eq!(Constraint::Count(2, Color::Blue).weight(), 1);
                assert_eq!(Constraint::Count(3, Color::Blue).weight(), 2);
            }

            #[test]
            fn other() {
                assert_eq!(Constraint::Adjacent(Color::Blue, Color::Blue).weight(), 1);
            }
        }

        mod conflict_key {
            #[allow(unused_imports)]
            use super::*;

            macro_rules! test {
                ($name: ident, $input: expr) => {
                    test!($name, $input, Constraint::from_str($input).expect($input));
                };
                ($name: ident, $input: expr, $expected: expr) => {
                    #[test]
                    fn $name() {
                        let input = Constraint::from_str($input).expect($input);
                        assert_eq!(input.conflict_key(), $expected);
                    }
                };
            }

            test!(adjacent, "O|G");
            test!(
                not_adjacent,
                "OxG",
                Constraint::Adjacent(Color::Green, Color::Orange)
            );
            test!(count, "B", Constraint::Count(0, Color::Blue));
            test!(below, "*/B");
            test!(above, "B/*", Constraint::Below(Color::Blue));
            test!(sum_count, "KW");
            test!(greater_than, "W>G");
        }
    }

    mod constraint_set {
        #[allow(unused_imports)]
        use super::*;

        #[allow(dead_code)]
        fn make(constraints: &str) -> ConstraintSet {
            let c = Constraint::parse_list(constraints).expect(constraints);
            ConstraintSet::with_constraints(&c)
        }

        mod with_constraints {
            #[allow(unused_imports)]
            use super::*;

            macro_rules! test {
                ($name: ident, $input: expr, $scoring: expr) => {
                    test!($name, $input, $scoring, []);
                };
                ($name: ident, $input: expr, $scoring: expr, $dropped: expr) => {
                    #[test]
                    fn $name() {
                        let cs = make($input);
                        assert_eq!(cs.scoring_constraints(), &$scoring);
                        assert_eq!(cs.dropped_constraints(), &$dropped);
                    }
                };
            }

            test!(
                basic,
                "K,GG,O|G,OW,GxK,*/W",
                [
                    Constraint::Adjacent(Color::Green, Color::Orange),
                    Constraint::NotAdjacent(Color::Black, Color::Green),
                    Constraint::Count(1, Color::Black),
                    Constraint::Count(2, Color::Green),
                    Constraint::SumCount(Color::White, Color::Orange),
                    Constraint::Below(Color::White),
                ]
            );

            test!(count_conflict, "B,BB", [Constraint::Count(3, Color::Blue)]);
            test!(
                adjacent_conflict,
                "O|G,GxO",
                [Constraint::Adjacent(Color::Green, Color::Orange)],
                [Constraint::NotAdjacent(Color::Green, Color::Orange)]
            );
            test!(
                below_conflict,
                "B/*,*/B",
                [Constraint::Below(Color::Blue)],
                [Constraint::Above(Color::Blue)]
            );
        }

        mod compute_score {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            fn perfect() {
                assert_eq!(
                    make("K,GG,O|G,OW,GxK,*/W").compute_score(11, 6, true),
                    BoardScore::new(11, true)
                );
            }

            #[test]
            fn no_flag() {
                assert_eq!(
                    make("K,GG,O|G,OW,GxK,*/W").compute_score(11, 6, false),
                    BoardScore::new(11, false)
                );
            }

            #[test]
            fn penalty() {
                assert_eq!(
                    make("K,GG,O|G,OW,GxK,*/W").compute_score(11, 5, true),
                    BoardScore::new(9, false)
                );
            }

            #[test]
            fn wipeout_penalty() {
                assert_eq!(
                    make("K,GG,O|G,OW,GxK,*/W").compute_score(11, 0, true),
                    BoardScore::new(0, false)
                );
            }

            #[test]
            fn max_score() {
                assert_eq!(
                    make("K,GG,O|G,OW,GxK,*/W").max_score(11),
                    BoardScore::new(11, true)
                );
            }

            #[test]
            fn max_score_conflict() {
                assert_eq!(
                    make("K,GG,O|G,OW,GxO,*/W").max_score(11),
                    BoardScore::new(9, false)
                );
            }
        }
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
                    let mix = ColorMix::from_str($s).expect($s);
                    let constraints =
                        ConstraintSet::with_constraints(&Constraint::parse_list($c).expect($c));
                    assert_eq!(mix.approximate_score(&constraints).score, $score);
                }
            };
            ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                #[test]
                fn $name() {
                    let mix = ColorMix::from_str($s).expect($s);
                    let constraints =
                        ConstraintSet::with_constraints(&Constraint::parse_list($c).expect($c));
                    assert_eq!(
                        mix.approximate_score(&constraints),
                        BoardScore {
                            score: $score,
                            flag: $flag
                        }
                    );
                }
            };
        }

        // Passes because there are no white spheres.
        test!(adjacent1, "KOBG", "K|W", 4);
        // Passes because there are no black spheres.
        test!(adjacent2, "WOBG", "K|W", 4);
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
        test!(count1, "B", "BB", 0);
        test!(count2, "BK", "BB", 0);
        test!(count3, "BKK", "BB", 1);
        test!(count4, "BKKB", "BB", 4);
        test!(count5, "BKKBB", "BB", 3);
        test!(count6, "BKKBB", "B,BB", 5);
        test!(count7, "BKKB", "B,BB", 0);
        test!(sum_count1, "B", "BO", 0);
        test!(sum_count2, "BOBO", "BO", 4);
        test!(sum_count3, "BOBOW", "BO", 5);
        test!(sum_count4, "BOWB", "BO", 2);
        test!(sum_count5, "BOWO", "BO", 2);
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
        test!(flag_constraint, "KWBOG", "K", 5, true);
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
                    assert_eq!(BoardState::from_str($s).unwrap().positions(), positions, "input={} actual={:?}", $s, positions);
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
                    let state = BoardState::from_str($s).expect($s);
                    let constraints =
                        ConstraintSet::with_constraints(&Constraint::parse_list($c).expect($c));
                    assert_eq!(state.score(&constraints).score, $score);
                }
            };
            ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                #[test]
                fn $name() {
                    let state = BoardState::from_str($s).expect($s);
                    let constraints =
                        ConstraintSet::with_constraints(&Constraint::parse_list($c).expect($c));
                    assert_eq!(
                        state.score(&constraints),
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
        test!(adjacent5, "GGGKKKOOOBB", "K|W", 11);
        test!(not_adjacent1, ".KWKWKW", "KxW", 4);
        test!(not_adjacent2, ".KWKWKW", "WxK", 4);
        test!(not_adjacent3, "GKK.WW", "KxW", 5);
        test!(not_adjacent4, "GKK.WW", "WxK", 5);
        test!(count1, "B", "BB", 0);
        test!(count2, "BK", "BB", 0);
        test!(count3, "BKK", "BB", 1);
        test!(count4, "BKKB", "BB", 4);
        test!(count5, "BKKBB", "BB", 3);
        test!(sum_count1, "B", "BO", 0);
        test!(sum_count2, "BOBO", "BO", 4);
        test!(sum_count3, "BOBOW", "BO", 5);
        test!(sum_count4, "BOWB", "BO", 2);
        test!(sum_count5, "BOWO", "BO", 2);
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
        test!(flag_constraint, "KWBOG", "K", 5, true);
    }
}
