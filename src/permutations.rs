use itertools::Itertools;
use std::collections::VecDeque;

// TODO(trevorm): enum representation for initial + in flight.
#[derive(Clone, Debug)]
pub struct UniquePermutations<I: Iterator> {
    state: Vec<VecDeque<I::Item>>,
}

// TODO(trevorm): make this all a little less panick-y.
impl<I> UniquePermutations<I>
where
    I: Iterator,
    I::Item: Copy + Ord,
{
    pub fn new(iter: I, k: usize) -> UniquePermutations<I> {
        let mut values = iter.take(k).collect_vec();
        values.sort();
        debug_assert!(k <= values.len());
        let mut p = UniquePermutations {
            state: vec![VecDeque::with_capacity(k); k],
        };
        p.fill_state(0, &values);
        p
    }

    fn fill_state(&mut self, start: usize, mut values: &[I::Item]) {
        debug_assert_eq!(values.len(), self.state.len() - start);
        for i in start..self.state.len() {
            self.state[i].clear();
            for v in values.iter() {
                if self.state[i].back() != Some(v) {
                    self.state[i].push_back(*v)
                }
            }
            values = &values[1..];
        }
    }

    fn value(&self) -> Option<Vec<I::Item>> {
        self.partial_value(0)
    }

    fn partial_value(&self, start: usize) -> Option<Vec<I::Item>> {
        let mut v = Vec::with_capacity(self.state.len() - start);
        for p in self.state.iter().skip(start) {
            let item = p.front()?;
            v.push(*item);
        }
        Some(v)
    }

    fn advance(&mut self) {
        let start = match self.state.iter().rposition(|s| s.len() > 1) {
            Some(p) => p,
            None => {
                // This is the termination condition -- if all the state elements have a single
                // entry then no more permutations can be generated.
                self.state[0].clear();
                return;
            }
        };
        debug_assert!(start < self.state.len() - 1);
        debug_assert!(self.state[start].len() >= 2);
        let mut partial = self.partial_value(start).unwrap();
        self.state[start].pop_front();
        let new = self.state[start].front().unwrap();
        let swapp = partial.iter().position(|d| d == new).unwrap();
        debug_assert!(swapp != 0);
        partial.swap(0, swapp);
        let _ = &partial[1..].sort();
        self.fill_state(start + 1, &partial[1..]);
    }
}

impl<I> Iterator for UniquePermutations<I>
where
    I: Iterator,
    I::Item: Copy + Ord,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.value()?;
        self.advance();
        Some(v)
    }
}

pub trait Iterators: Iterator {
    fn unique_permutations(self, k: usize) -> UniquePermutations<Self>
    where
        Self: Sized,
        Self::Item: Copy + Ord,
    {
        UniquePermutations::<Self>::new(self, k)
    }
}

impl<I: Sized> Iterators for I where I: Iterator {}

mod test {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn permute0001() {
        assert_eq!(
            UniquePermutations::new([0, 0, 0, 1].iter().copied(), 4).collect_vec(),
            &[[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        );
    }
    #[test]
    fn permute0011() {
        assert_eq!(
            UniquePermutations::new([0, 0, 1, 1].iter().copied(), 4).collect_vec(),
            &[
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0]
            ]
        );
    }

    #[test]
    fn permute01112() {
        assert_eq!(
            UniquePermutations::new([0, 1, 1, 1, 2].iter().copied(), 5).collect_vec(),
            &[
                [0, 1, 1, 1, 2],
                [0, 1, 1, 2, 1],
                [0, 1, 2, 1, 1],
                [0, 2, 1, 1, 1],
                [1, 0, 1, 1, 2],
                [1, 0, 1, 2, 1],
                [1, 0, 2, 1, 1],
                [1, 1, 0, 1, 2],
                [1, 1, 0, 2, 1],
                [1, 1, 1, 0, 2],
                [1, 1, 1, 2, 0],
                [1, 1, 2, 0, 1],
                [1, 1, 2, 1, 0],
                [1, 2, 0, 1, 1],
                [1, 2, 1, 0, 1],
                [1, 2, 1, 1, 0],
                [2, 0, 1, 1, 1],
                [2, 1, 0, 1, 1],
                [2, 1, 1, 0, 1],
                [2, 1, 1, 1, 0],
            ]
        );
    }
}
