use itertools::Itertools;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug, Eq, PartialEq)]
enum FixedArrayRep<T: Sized, const N: usize> {
    Inline([T; N]),
    Heap(Box<[T]>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedArray<T: Copy + Default + Sized, const N: usize> {
    rep: FixedArrayRep<T, N>,
    len: usize,
}

impl<T: Copy + Default + Sized, const N: usize> FixedArray<T, N> {
    pub fn new(k: usize) -> FixedArray<T, N> {
        if k > N {
            FixedArray {
                rep: FixedArrayRep::Heap(vec![T::default(); k].into_boxed_slice()),
                len: k,
            }
        } else {
            FixedArray {
                rep: FixedArrayRep::Inline([T::default(); N]),
                len: k,
            }
        }
    }
}

impl<T: Copy + Default + Sized, const N: usize> Deref for FixedArray<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.rep {
            FixedArrayRep::Heap(a) => &a[0..self.len],
            FixedArrayRep::Inline(a) => &a[0..self.len],
        }
    }
}

impl<T: Copy + Default + Sized, const N: usize> DerefMut for FixedArray<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.rep {
            FixedArrayRep::Heap(a) => &mut a[0..self.len],
            FixedArrayRep::Inline(a) => &mut a[0..self.len],
        }
    }
}

impl<T: PartialEq<U> + Copy + Default + Sized, U, const M: usize, const N: usize>
    PartialEq<[U; M]> for FixedArray<T, N>
{
    fn eq(&self, other: &[U; M]) -> bool {
        if self.len == M {
            &self.deref() == other
        } else {
            false
        }
    }
}

// TODO(trevorm): enum representation for initial + in flight.
// TODO(trevorm): work with indices instead of values. We may be able to flatten
// the VecDeque into a BitSet64.
#[derive(Clone, Debug)]
pub struct UniquePermutations<I: Iterator> {
    state: Vec<VecDeque<I::Item>>,
}

// TODO(trevorm): make this all a little less panick-y.
impl<I> UniquePermutations<I>
where
    I: Iterator,
    I::Item: Copy + Default + Ord,
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

    fn value(&self) -> Option<FixedArray<I::Item, 16>> {
        let mut v = FixedArray::new(self.state.len());
        for (p, out) in self.state.iter().zip(v.iter_mut()) {
            let item = p.front()?;
            *out = *item;
        }
        Some(v)
    }

    fn fill_partial_value(&self, start: usize, values: &mut [I::Item]) {
        for (p, o) in self.state.iter().skip(start).zip(values.iter_mut()) {
            *o = *p.front().unwrap();
        }
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
        let mut buf = FixedArray::<I::Item, 16>::new(self.state.len() - (start + 1));
        self.fill_partial_value(start + 1, &mut buf);
        let partial = &mut buf[..(self.state.len() - (start + 1))];
        let old = self.state[start].pop_front().unwrap();
        let new = self.state[start].front().unwrap();
        partial[partial.iter().position(|d| d == new).unwrap()] = old;
        partial.sort();
        self.fill_state(start + 1, &partial);
    }
}

impl<I> Iterator for UniquePermutations<I>
where
    I: Iterator,
    I::Item: Copy + Default + Ord + Sized,
{
    type Item = FixedArray<I::Item, 16>;

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
        Self::Item: Copy + Default + Ord,
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
