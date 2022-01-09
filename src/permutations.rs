use itertools::Itertools;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Debug)]
pub struct BitSet64(u64);

impl BitSet64 {
    #[inline]
    pub fn front(&self) -> Option<usize> {
        if self.0 > 0 {
            Some(self.0.trailing_zeros() as usize)
        } else {
            None
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    #[inline]
    pub fn pop_front(&mut self) -> Option<usize> {
        let idx = self.front()?;
        self.0 ^= 1 << idx;
        Some(idx)
    }

    #[inline]
    pub fn set(&mut self, idx: usize) {
        self.0 |= 1 << idx;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

impl Default for BitSet64 {
    #[inline]
    fn default() -> BitSet64 {
        BitSet64(0)
    }
}

impl From<u64> for BitSet64 {
    #[inline]
    fn from(v: u64) -> BitSet64 {
        BitSet64(v)
    }
}

impl Iterator for BitSet64 {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 != 0 {
            let i = self.0.trailing_zeros();
            self.0 ^= 1 << i;
            Some(i as usize)
        } else {
            None
        }
    }
}

// TODO(trevorm): replace this with SmallVec
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

impl<T: PartialEq<U> + Copy + Default + Sized, U, const M: usize, const N: usize> PartialEq<[U; M]>
    for FixedArray<T, N>
{
    fn eq(&self, other: &[U; M]) -> bool {
        if self.len == M {
            &self.deref() == other
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
enum UniquePermutationRep {
    Inflight,
    Done,
}

#[derive(Clone, Debug)]
pub struct UniquePermutations<I: Iterator> {
    rep: UniquePermutationRep,
    unique_values: Vec<I::Item>,
    index_state: Vec<BitSet64>,
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

        let mut unique_values = Vec::with_capacity(values.len());
        unique_values.push(values[0]);
        let mut indices = vec![0; values.len()];
        for (v, i) in values.iter().zip(indices.iter_mut()).skip(1) {
            if unique_values.last().unwrap() != v {
                unique_values.push(*v);
            }
            *i = unique_values.len() - 1;
        }

        let mut p = UniquePermutations {
            rep: UniquePermutationRep::Inflight,
            unique_values: unique_values,
            index_state: vec![BitSet64::default(); values.len()],
        };
        p.fill_index_state(0, &indices);
        p
    }

    fn fill_index_state(&mut self, start: usize, indices: &[usize]) {
        debug_assert_eq!(indices.len(), self.index_state.len() - start);
        for (i, s) in self.index_state.iter_mut().skip(start).enumerate() {
            s.clear();
            for idx in indices.iter().skip(i) {
                s.set(*idx);
            }
        }
    }

    fn value(&self) -> Option<FixedArray<I::Item, 16>> {
        match &self.rep {
            UniquePermutationRep::Inflight => {
                let mut v = FixedArray::new(self.index_state.len());
                for (i, o) in self.index_state.iter().zip(v.iter_mut()) {
                    *o = self.unique_values[i.front().unwrap()];
                }
                Some(v)
            }
            UniquePermutationRep::Done => None,
        }
    }

    fn advance(&mut self) {
        let start = match self.index_state.iter().rposition(|s| s.len() > 1) {
            Some(p) => p,
            None => {
                self.rep = UniquePermutationRep::Done;
                return;
            }
        };
        debug_assert!(start < self.index_state.len() - 1);
        debug_assert!(self.index_state[start].len() >= 2);
        let mut partial = FixedArray::<usize, 16>::new(self.index_state.len() - (start + 1));
        for (i, o) in self.index_state.iter().skip(start + 1).zip(partial.iter_mut()) {
            *o = i.front().unwrap();
        }
        let old = self.index_state[start].pop_front().unwrap();
        let new = self.index_state[start].front().unwrap();
        let pos = partial.iter().position(|d| *d == new).unwrap();
        partial[pos] = old;
        partial.sort();
        self.fill_index_state(start + 1, &partial);
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
