#[macro_use]
extern crate itertools;
#[macro_use]
extern crate lazy_static;

pub mod permutations;

use crate::permutations::{FixedArray, Iterators};
use itertools::Itertools;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::iter;
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

#[derive(Debug)]
pub struct ConstraintSet {
    scoring: Vec<Constraint>,
    dropped: Vec<Constraint>,
    total_weight: usize,
    max_weight: usize,
}

impl ConstraintSet {
    // REQUIRES: no duplicates within constraints. This is true if constraints were generated
    // using ConstraintSet::from_str().
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

    // Gets the upper bound score for each board size.
    fn get_upper_bound_scores(&self) -> Vec<(usize, BoardScore)> {
        let mut scores: Vec<(usize, BoardScore)> = Vec::with_capacity(NUM_POSITIONS);
        for k in 1..=NUM_POSITIONS {
            let max_score = self.max_score(k);
            let mut high_score = BoardScore::default();

            for m in ColorMix::all_combinations(k) {
                let score = m.approximate_score(self);
                if score > high_score {
                    high_score = score
                }
                if high_score == max_score {
                    break;
                }
            }
            scores.push((k, high_score))
        }
        // Return in descending order by score.
        scores.sort_by(|(_, a), (_, b)| a.cmp(&b).reverse());
        scores
    }

    // Returns a BoardState that achieves the highest possible score for these constraints.
    pub fn solve(&self) -> (BoardState, BoardScore) {
        let mut best_board = BoardState::default();
        let mut best_score = BoardScore::default();

        for (k, max_score) in self.get_upper_bound_scores() {
            if max_score < best_score {
                break;
            }

            // Iterate over all ColorMixes whose approximate_score() matches max_score, then iterate
            // over those permutations to compute final scores.
            for mix in
                ColorMix::all_combinations(k).filter(|m| m.approximate_score(self) == max_score)
            {
                for board in BoardState::permutations_from_color_mix(&mix) {
                    let score = board.score(self);
                    if score > best_score {
                        best_board = board;
                        best_score = score;
                        if best_score == max_score {
                            break;
                        }
                    }
                }

                if best_score == max_score {
                    break;
                }
            }
        }
        (best_board, best_score)
    }
}

impl FromStr for ConstraintSet {
    type Err = &'static str;

    // TODO(trevorm): consider altering return type to String and noting the failed
    // constraint in the parse.
    fn from_str(s: &str) -> Result<ConstraintSet, Self::Err> {
        if s.is_empty() {
            return Ok(ConstraintSet::with_constraints(&[]));
        }
        let constraints: Vec<Constraint> = s
            .split(",")
            .map(|e| Constraint::from_str(e))
            .try_collect()?;
        if constraints.iter().unique().count() == constraints.len() {
            Ok(ConstraintSet::with_constraints(&constraints))
        } else {
            Err("all constraints in list must be unique")
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColorMix {
    // Pack everything into a single u32 containing 5 4-bit values, one entry for each color.
    // 4 bits is enough as we will not process more than 11 input entries. This is a form of
    // vectorization that allows us to quickly compute "all values <= 3" and "all values > 0".
    rep: u32,
    // Compute the sum of all color counts as it is used during scoring.
    count: u32,
}

impl ColorMix {
    pub fn with_colors(colors: &[Color]) -> Result<ColorMix, &'static str> {
        if colors.len() > NUM_POSITIONS {
            return Err("ColorMix may not have more than 11 colors");
        }
        let mut mix = ColorMix { rep: 0, count: 0 };
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
        // Interestingly this is ~15% slower written as mix.rep & 0xCCCCC == 0
        if mix.rep & 0x33333 == mix.rep {
            Ok(mix)
        } else {
            Err("ColorMix may not have more than 3 of any color")
        }
    }

    pub fn all_combinations(num_spheres: usize) -> impl Iterator<Item = ColorMix> {
        ALL_COLORS
            .iter()
            .copied()
            .flat_map(|c| iter::repeat(c).take(3))
            .combinations(num_spheres)
            .as_color_mix()
            .unique()
    }

    fn add(&mut self, c: Color) {
        self.rep += 1 << (c as usize * 4);
        self.count += 1;
    }

    fn count(&self, c: Color) -> usize {
        (self.rep as usize >> (c as usize * 4)) & 0xf
    }

    pub fn num_spheres(&self) -> usize {
        self.count as usize
    }

    pub fn has_all_colors(&self) -> bool {
        (self.rep & 0x11111 | (self.rep >> 1) & 0x11111).count_ones() == 5
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Color> + 'a {
        ALL_COLORS
            .iter()
            .copied()
            .flat_map(move |c| iter::repeat(c).take(self.count(c)))
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
            Constraint::Above(color) => {
                // If a color must be above everything and there are 11 spheres it must be on top
                // and there must be no other spheres of that color.
                self.num_spheres() < 11 || self.count(*color) == 1
            }
            // Approximations of these are too pessimistic given that we are trying
            // to compute an upper bound score. The best approximations only return true
            // when a targeted color is 0, where the flag cannot be set.
            Constraint::NotAdjacent(_, _) => true,
            Constraint::Below(_) => true,
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
            let c = Color::try_from(b)?;
            colors.push(c);
        }
        ColorMix::with_colors(&colors)
    }
}

impl fmt::Display for ColorMix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.iter().map(|c| char::from(c)).collect::<String>()
        )
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

// Represents an array of 5 11-bit values, one for each color.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ColorBoardArray {
    rep: u64,
}

impl ColorBoardArray {
    fn get(self, c: Color) -> u16 {
        (self.rep >> (c as usize * 12)) as u16 & 0xfff
    }

    fn count(self, c: Color) -> usize {
        ((self.rep >> (c as usize * 12)) & 0xfff).count_ones() as usize
    }

    fn set_bit(&mut self, c: Color, i: usize) {
        debug_assert!(i <= 12);
        self.rep |= 1 << (c as usize * 12) + i
    }

    fn or_equal(&mut self, c: Color, v: u16) {
        debug_assert!(v <= 0xfff);
        self.rep |= (v as u64) << (c as usize * 12);
    }
}

impl Default for ColorBoardArray {
    fn default() -> ColorBoardArray {
        ColorBoardArray { rep: 0 }
    }
}

// There is an alternative representation for cpositions as a u128
// * 1 byte per position
// * Set the hi bit for every set position.
// * Set 1 << color as usize depending on the color set.
// This would have a few advantages over the current representation:
// * The representation itself would be easy to permute.
// * Validation of the physical board (below op) would be faster.
// * Creating an object from the representation would probably be faster.
// There isn't much point to doing this given the speed of the solver.
type Positions = [Option<Color>; NUM_POSITIONS];
#[derive(Debug, Eq, PartialEq)]
pub struct BoardState {
    cpositions: ColorBoardArray,
    cadjacency: ColorBoardArray,
    positions: u16,
    below: u16,
    has_all_colors: bool,
}

impl BoardState {
    // Create a BoardState with the input positions. Returns None if the board is not valid.
    pub fn with_positions(positions: &[Option<Color>]) -> Result<BoardState, &'static str> {
        // Computing ColorMix checks that spheres per color <=3
        let mix = ColorMix::with_colors(&positions.iter().filter_map(|c| *c).collect_vec())?;
        BoardState::with_positions_and_mix(&positions, &mix)
    }

    // Create a BoardState with the input positions. Returns None if the board is not valid.
    pub fn with_positions_and_mix(
        positions: &[Option<Color>],
        mix: &ColorMix,
    ) -> Result<BoardState, &'static str> {
        let mut board = BoardState::default();
        board.init(positions.iter());
        if board.positions & board.below == board.below {
            board.has_all_colors = mix.has_all_colors();
            Ok(board)
        } else {
            Err("Board has empty positions below occupied positions")
        }
    }

    fn init<'a, I>(&mut self, positions: I)
    where
        I: Iterator<Item = &'a Option<Color>>,
    {
        for (i, p, g) in izip!((0..NUM_POSITIONS), positions, BOARD_GRAPH.iter()) {
            match p {
                Some(c) => {
                    self.cpositions.set_bit(*c, i);
                    self.cadjacency.or_equal(*c, g.adjacent);
                    self.positions |= 1 << i;
                    self.below |= g.below;
                }
                None => {}
            }
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
            0..=3 => 7,   // need at least 4 spheres place any above L0
            4..=10 => 10, // L2 can only be stacked if there are 11 spheres.
            _ => 11,
        };
        positions
            .into_iter()
            .unique_permutations(p)
            .as_board_state(*mix)
    }

    pub fn num_spheres(&self) -> usize {
        self.positions.count_ones() as usize
    }

    fn fill_positions_from_bitmap(&self, c: Color, p: &mut Positions) {
        let mut bitmap = self.cpositions.get(c);
        while bitmap > 0 {
            let idx = bitmap.trailing_zeros();
            p[idx as usize] = Some(c);
            bitmap &= !(1 << idx);
        }
    }

    fn positions(&self) -> Positions {
        let mut p: Positions = [None; NUM_POSITIONS];
        for c in ALL_COLORS.iter() {
            self.fill_positions_from_bitmap(*c, &mut p);
        }
        p
    }

    pub fn score(&self, constraints: &ConstraintSet) -> BoardScore {
        let matching = constraints
            .scoring_constraints()
            .iter()
            .map(|c| self.matches_constraint(c))
            .sum();
        constraints.compute_score(self.num_spheres(), matching, self.has_all_colors)
    }

    fn matches_constraint(&self, constraint: &Constraint) -> usize {
        let f: bool = match constraint {
            Constraint::Adjacent(color1, color2) => {
                let c1_pos = self.cpositions.get(*color1);
                let c2_pos = self.cpositions.get(*color2);
                c1_pos == 0
                    || c2_pos == 0
                    || (c1_pos & self.cadjacency.get(*color2) == c1_pos
                        && c2_pos & self.cadjacency.get(*color1) == c2_pos)
            }
            Constraint::NotAdjacent(color1, color2) => {
                let c1_pos = self.cpositions.get(*color1);
                let c2_pos = self.cpositions.get(*color2);
                c1_pos & !self.cadjacency.get(*color2) == c1_pos
                    && c2_pos & !self.cadjacency.get(*color1) == c2_pos
            }
            Constraint::Count(count, color) => self.cpositions.count(*color) == *count,
            Constraint::SumCount(color1, color2) => {
                self.cpositions.count(*color1) + self.cpositions.count(*color2) == 4
            }
            Constraint::Below(color) => {
                let pos = self.cpositions.get(*color);
                pos & self.below == pos
            }
            Constraint::Above(color) => self.cpositions.get(*color) & self.below == 0,
            Constraint::GreaterThan(color1, color2) => {
                self.cpositions.count(*color1) > self.cpositions.count(*color2)
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
            cpositions: ColorBoardArray::default(),
            cadjacency: ColorBoardArray::default(),
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

        BoardState::with_positions(&values)
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
        let mut pos: Positions = [None; NUM_POSITIONS];
        for (i, o) in mix.iter().zip(pos.iter_mut()) {
            *o = Some(i)
        }
        let mut board = BoardState::default();
        board.init(pos.iter());
        board.has_all_colors = mix.has_all_colors();
        // ColorMix guarantees that we won't have more than 3 of each color, and this arrangement
        // without gaps ensures that the board will be valid.
        board
    }
}

pub struct BoardStateAdaptor<I: Iterator> {
    iter: I,
    mix: ColorMix,
}

impl<I> Iterator for BoardStateAdaptor<I>
where
    I: Iterator<Item = FixedArray<Option<Color>, 16>>,
{
    type Item = BoardState;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                None => return None,
                Some(p) => {
                    if let Ok(board) = BoardState::with_positions_and_mix(&p, &self.mix) {
                        return Some(board);
                    }
                }
            }
        }
    }
}

pub trait BoardStateIterator: Iterator {
    fn as_board_state(self, mix: ColorMix) -> BoardStateAdaptor<Self>
    where
        Self: Sized + Iterator<Item = FixedArray<Option<Color>, 16>>,
    {
        BoardStateAdaptor {
            iter: self,
            mix: mix,
        }
    }
}

impl<I: Iterator<Item = FixedArray<Option<Color>, 16>>> BoardStateIterator for I {}

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
            ConstraintSet::from_str(constraints).expect(constraints)
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

        // These would be pretty slow for normal tests but are much worse in a debug build.
        mod solve {
            #[allow(unused_imports)]
            use super::*;

            macro_rules! test {
                ($name: ident, $constraints: literal, $board: literal, $score: expr) => {
                    #[test]
                    fn $name() {
                        let (actual_board, actual_score) = ConstraintSet::from_str($constraints)
                            .expect($constraints)
                            .solve();
                        assert_eq!($score, actual_score);
                        assert_eq!(
                            BoardState::from_str($board).expect($board),
                            actual_board,
                            "{}",
                            actual_board
                        );
                    }
                };
            }

            test!(
                typical,
                "OO,KW,G/*,G|B,WxW,K|O",
                "KKKWOBBOBG.",
                BoardScore::new(10, true)
            );
            test!(
                adjacent_conflict,
                "G|G,GxG",
                "KKKWWWBBBGG",
                BoardScore::new(9, false)
            );
            test!(
                level_conflict,
                "G/*,*/G",
                "KKKWWWBBGGB",
                BoardScore::new(9, false)
            );
            test!(
                count_greater_than_soft_conflict,
                "K>B,B,BB",
                "KKKWWWBBBGG",
                BoardScore::new(9, false)
            );
            test!(
                above_soft_conflict,
                "W/*,B/*",
                "KKKGGGOWWB.",
                BoardScore::new(10, true)
            );
            test!(
                complex_adjacency,
                "G|G,B|K,O|W,WxK,BxB,GxK",
                "OKBGGWBKBOK",
                BoardScore::new(11, true)
            );
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
                    let constraints = ConstraintSet::from_str($c).expect($c);
                    assert_eq!(mix.approximate_score(&constraints).score, $score);
                }
            };
            ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                #[test]
                fn $name() {
                    let mix = ColorMix::from_str($s).expect($s);
                    let constraints = ConstraintSet::from_str($c).expect($c);
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

    mod board_state {
        #[allow(unused_imports)]
        use super::*;

        mod from_str {
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
                    Err("Board has empty positions below occupied positions")
                )
            }

            #[test]
            fn missing_l1() {
                assert_eq!(
                    BoardState::from_str("OKKKWWW...G"),
                    Err("Board has empty positions below occupied positions")
                )
            }

            #[test]
            fn color_limit3() {
                assert_eq!(
                    BoardState::from_str("OKKKWWWBBBK"),
                    Err("ColorMix may not have more than 3 of any color")
                )
            }
        }

        mod format {
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

        mod score {
            #[allow(unused_imports)]
            use super::*;
            macro_rules! test {
                ($name:ident, $s:expr, $c:expr, $score:expr) => {
                    #[test]
                    fn $name() {
                        let state = BoardState::from_str($s).expect($s);
                        let constraints = ConstraintSet::from_str($c).expect($c);
                        assert_eq!(state.score(&constraints).score, $score);
                    }
                };
                ($name:ident, $s:expr, $c:expr, $score:expr, $flag:expr) => {
                    #[test]
                    fn $name() {
                        let state = BoardState::from_str($s).expect($s);
                        let constraints = ConstraintSet::from_str($c).expect($c);
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
}
