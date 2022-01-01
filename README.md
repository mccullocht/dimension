# Dimension Scorer and Solver

This crate contains a scorer and solver for the board game [Dimension](https://boardgamegeek.com/boardgame/153318/dimension).

## Color Inputs

Both board state and constraints use the same notation to indicate the color of a sphere:

* `K` for black
* `W` for white
* `B` for blue
* `G` for green
* `O` for orange

Additionally `.` may be used to indicate empty positions on the board.

## Constraints

Constraints are input using the `--constraint` flag as a comma-separated list. This program only accepts the 60 cards included in the same as input:
* *Below*: this color must always be below another sphere. Written as below wildcard ex `'*/W'`
* *Above*: this color must not have any other spheres on top. Written as above wildcard ex `'W/*'`
* *Count1*: board must have exactly 1 sphere of this color. Written as a single sphere ex `'W'`
* *Count2*: board must have exactly 2 spheres of this color. Written as two spheres of the same color ex `'WW'`
* *SumCount*: board must have 4 of any combination of these two colors. Written as two spheres of different colors ex `'KW'`.
* *GreaterThan*: board must have more spheres of the first colar than the second. Written as greater than ex `'W>G'`.
* *Adjacent*: one (or both) colors may be missing or every sphere of these colors must touch. Written with a pipe ex `'B|W'`
* *NotAdjacent*: every sphere of these colors must not touch. Written with x ex `'BxW'`

This flag must be provided. When only constraints are specified the program solves for a board state that produces the highest score.

## Board State

Board state can be input using the `--board` flag to provide a list of 11 characters reprenting the position of each sphere. When provided in addition to `--constraints` the program produces the score given the constraints. Empty positions at the end of the board state may be omitted. Positions are numbered from the center and proceed clockwise on each level, ascending. Here's a crude ASCII art illustration of the order of positions:

```
Describe the relationship of each position on the board to all other positions.
L0      L1                L2
  1                1
 /|\              /
6 | 2   6--9-----7--2       9    7
|\|/|     / \   /            \  /
| 0 |    5   \ /              10
|/|\|         8--3             |
5 | 3          \               8
 \|/            4
  4
```

## Sample command

```
$ cargo run --release -- --constraints='G|G,B|K,O|W,WxK,BxB,GxK'
OKBGGWBKBOK score=11 flag=true
```

## Solver implementation notes

The solver is divided into two passes: an approximate scorer and a full scorer.

The approximate scorer accepts a mix of colors that appear on the board and scores against constraints that only use the color count, assuming that adjacency constraints match. This produces an upper bound score for each sphere count and substantially reduces the size of the search space.

The full scorer accepts board states -- permutations of spheres and empty positions -- and evaluates against all constraints. We run this on every permutation of the highest scoring color mixes. If any permutation can match the approximate score we get to exit early. A special iterator is used to generate permutations while ignoring swaps of identical elements to substantially reduce the number of permutations. We also track *conflicting* constraints that cannot both be satisfied simultaneously and drop one in order to hit our early exit condition more easily.