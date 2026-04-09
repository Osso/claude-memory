use atomic::Atomic;
use byte_slice_cast::{AsByteSlice, AsMutByteSlice, ToByteSlice, ToMutByteSlice};
use log::info;
use std::{
    io::{Read, Write},
    iter::FromIterator,
    mem::{ManuallyDrop, MaybeUninit},
    sync::atomic::Ordering::Acquire,
    time::Instant,
};

use rayon::prelude::*;

use crate::{
    compat::*,
    index::Idx,
    input::{edgelist::Edges, Direction},
    Error, SharedMut, Target,
};

/// Defines how the neighbor list of individual nodes are organized within the
/// CSR target array.
#[derive(Default, Clone, Copy, Debug)]
pub enum CsrLayout {
    /// Neighbor lists are sorted and may contain duplicate target ids. This is
    /// the default representation.
    Sorted,
    /// Neighbor lists are not in any particular order.
    #[default]
    Unsorted,
    /// Neighbor lists are sorted and do not contain duplicate target ids.
    /// Self-loops, i.e., edges in the form of `(u, u)` are removed.
    Deduplicated,
}

/// A Compressed-Sparse-Row data structure to represent sparse graphs.
///
/// The data structure is composed of two arrays: `offsets` and `targets`. For a
/// graph with node count `n` and edge count `m`, `offsets` has exactly `n + 1`
/// and `targets` exactly `m` entries.
///
/// For a given node `u`, `offsets[u]` stores the start index of the neighbor
/// list of `u` in `targets`. The degree of `u`, i.e., the length of the
/// neighbor list is defined by `offsets[u + 1] - offsets[u]`. The neighbor list
/// of `u` is defined by the slice `&targets[offsets[u]..offsets[u + 1]]`.
#[derive(Debug)]
pub struct Csr<Index: Idx, NI, EV> {
    offsets: Box<[Index]>,
    targets: Box<[Target<NI, EV>]>,
}

impl<Index: Idx, NI, EV> Csr<Index, NI, EV> {
    pub(crate) fn new(offsets: Box<[Index]>, targets: Box<[Target<NI, EV>]>) -> Self {
        Self { offsets, targets }
    }

    #[inline]
    pub(crate) fn node_count(&self) -> Index {
        Index::new(self.offsets.len() - 1)
    }

    #[inline]
    pub(crate) fn edge_count(&self) -> Index {
        Index::new(self.targets.len())
    }

    #[inline]
    pub(crate) fn degree(&self, i: Index) -> Index {
        let from = self.offsets[i.index()];
        let to = self.offsets[(i + Index::new(1)).index()];

        to - from
    }

    #[inline]
    pub(crate) fn targets_with_values(&self, i: Index) -> &[Target<NI, EV>] {
        let from = self.offsets[i.index()];
        let to = self.offsets[(i + Index::new(1)).index()];

        &self.targets[from.index()..to.index()]
    }
}

impl<Index: Idx, NI> Csr<Index, NI, ()> {
    #[inline]
    pub(crate) fn targets(&self, i: Index) -> &[NI] {
        assert_eq!(
            std::mem::size_of::<Target<NI, ()>>(),
            std::mem::size_of::<NI>()
        );
        assert_eq!(
            std::mem::align_of::<Target<NI, ()>>(),
            std::mem::align_of::<NI>()
        );
        let from = self.offsets[i.index()];
        let to = self.offsets[(i + Index::new(1)).index()];

        let len = (to - from).index();

        let targets = &self.targets[from.index()..to.index()];

        // SAFETY: len is within bounds as it is calculated above as `to - from`.
        //         The types Target<T, ()> and T are verified to have the same
        //         size and alignment.
        unsafe { std::slice::from_raw_parts(targets.as_ptr() as *const _, len) }
    }
}

pub trait SwapCsr<Index: Idx, NI, EV> {
    fn swap_csr(&mut self, csr: Csr<Index, NI, EV>) -> &mut Self;
}

impl<NI, EV, E> From<(&'_ E, NI, Direction, CsrLayout)> for Csr<NI, NI, EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from(
        (edge_list, node_count, direction, csr_layout): (&'_ E, NI, Direction, CsrLayout),
    ) -> Self {
        let start = Instant::now();
        let degrees = edge_list.degrees(node_count, direction);
        info!("Computed degrees in {:?}", start.elapsed());

        let start = Instant::now();
        let offsets = prefix_sum_atomic(degrees);
        info!("Computed prefix sum in {:?}", start.elapsed());

        let start = Instant::now();
        let edge_count = offsets[node_count.index()].load(Acquire).index();
        let mut targets = build_target_array(edge_list, direction, &offsets, edge_count);
        info!("Computed target array in {:?}", start.elapsed());

        let start = Instant::now();
        let offsets = finalize_offsets(offsets);
        info!("Finalized offset array in {:?}", start.elapsed());

        let (offsets, targets) = apply_csr_layout(csr_layout, offsets, &mut targets);
        Csr {
            offsets: offsets.into_boxed_slice(),
            targets: targets.into_boxed_slice(),
        }
    }
}

fn build_target_array<NI, EV, E>(
    edge_list: &E,
    direction: Direction,
    offsets: &[Atomic<NI>],
    edge_count: usize,
) -> Vec<Target<NI, EV>>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    let mut targets = Vec::<Target<NI, EV>>::with_capacity(edge_count);
    let targets_ptr = SharedMut::new(targets.as_mut_ptr());
    populate_targets(edge_list, direction, offsets, &targets_ptr);

    unsafe {
        targets.set_len(edge_count);
    }
    targets
}

fn populate_targets<NI, EV, E>(
    edge_list: &E,
    direction: Direction,
    offsets: &[Atomic<NI>],
    targets_ptr: &SharedMut<Target<NI, EV>>,
) where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    if matches!(direction, Direction::Outgoing | Direction::Undirected) {
        write_targets(edge_list, offsets, targets_ptr, |source, _| source, |_, target| target);
    }

    if matches!(direction, Direction::Incoming | Direction::Undirected) {
        write_targets(edge_list, offsets, targets_ptr, |_, target| target, |source, _| source);
    }
}

fn write_targets<NI, EV, E, F, G>(
    edge_list: &E,
    offsets: &[Atomic<NI>],
    targets_ptr: &SharedMut<Target<NI, EV>>,
    offset_node: F,
    target_node: G,
) where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
    F: Fn(NI, NI) -> NI + Send + Sync,
    G: Fn(NI, NI) -> NI + Send + Sync,
{
    edge_list.edges().for_each(|(source, target, value)| {
        let offset = NI::get_and_increment(&offsets[offset_node(source, target).index()], Acquire);
        unsafe {
            targets_ptr
                .add(offset.index())
                .write(Target::new(target_node(source, target), value));
        }
    });
}

fn finalize_offsets<NI: Idx>(offsets: Vec<Atomic<NI>>) -> Vec<NI> {
    let mut offsets = ManuallyDrop::new(offsets);
    let (ptr, len, cap) = (offsets.as_mut_ptr(), offsets.len(), offsets.capacity());
    let mut offsets = unsafe { Vec::from_raw_parts(ptr as *mut _, len, cap) };
    offsets.rotate_right(1);
    offsets[0] = NI::zero();
    offsets
}

fn apply_csr_layout<NI: Idx, EV: Copy + Send>(
    csr_layout: CsrLayout,
    offsets: Vec<NI>,
    targets: &mut Vec<Target<NI, EV>>,
) -> (Vec<NI>, Vec<Target<NI, EV>>) {
    match csr_layout {
        CsrLayout::Unsorted => (offsets, std::mem::take(targets)),
        CsrLayout::Sorted => {
            let start = Instant::now();
            sort_targets(&offsets, targets);
            info!("Sorted targets in {:?}", start.elapsed());
            (offsets, std::mem::take(targets))
        }
        CsrLayout::Deduplicated => {
            let start = Instant::now();
            let offsets_targets = sort_and_deduplicate_targets(&offsets, &mut targets[..]);
            info!("Sorted and deduplicated targets in {:?}", start.elapsed());
            offsets_targets
        }
    }
}

unsafe impl<NI, EV> ToByteSlice for Target<NI, EV>
where
    NI: ToByteSlice,
    EV: ToByteSlice,
{
    fn to_byte_slice<S: AsRef<[Self]> + ?Sized>(slice: &S) -> &[u8] {
        let slice = slice.as_ref();
        let len = std::mem::size_of_val(slice);
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
    }
}

unsafe impl<NI, EV> ToMutByteSlice for Target<NI, EV>
where
    NI: ToMutByteSlice,
    EV: ToMutByteSlice,
{
    fn to_mut_byte_slice<S: AsMut<[Self]> + ?Sized>(slice: &mut S) -> &mut [u8] {
        let slice = slice.as_mut();
        let len = std::mem::size_of_val(slice);
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, len) }
    }
}

impl<NI, EV> Csr<NI, NI, EV>
where
    NI: Idx + ToByteSlice,
    EV: ToByteSlice,
{
    fn serialize<W: Write>(&self, output: &mut W) -> Result<(), Error> {
        let type_name = std::any::type_name::<NI>().as_bytes();
        output.write_all([type_name.len()].as_byte_slice())?;
        output.write_all(type_name)?;

        let node_count = self.node_count();
        let edge_count = self.edge_count();
        let meta = [node_count, edge_count];
        output.write_all(meta.as_byte_slice())?;

        output.write_all(self.offsets.as_byte_slice())?;
        output.write_all(self.targets.as_byte_slice())?;

        Ok(())
    }
}

impl<NI, EV> Csr<NI, NI, EV>
where
    NI: Idx + ToMutByteSlice,
    EV: ToMutByteSlice,
{
    fn deserialize<R: Read>(read: &mut R) -> Result<Csr<NI, NI, EV>, Error> {
        let mut type_name_len = [0_usize; 1];
        read.read_exact(type_name_len.as_mut_byte_slice())?;
        let [type_name_len] = type_name_len;

        let mut type_name = vec![0_u8; type_name_len];
        read.read_exact(type_name.as_mut_byte_slice())?;
        let type_name = String::from_utf8(type_name).expect("could not read type name");

        let expected_type_name = std::any::type_name::<NI>().to_string();

        if type_name != expected_type_name {
            return Err(Error::InvalidIdType {
                expected: expected_type_name,
                actual: type_name,
            });
        }

        let mut meta = [NI::zero(); 2];
        read.read_exact(meta.as_mut_byte_slice())?;

        let [node_count, edge_count] = meta;

        let mut offsets = Box::new_uninit_slice_compat(node_count.index() + 1);
        let offsets_ptr = offsets.as_mut_ptr() as *mut NI;
        let offsets_ptr =
            unsafe { std::slice::from_raw_parts_mut(offsets_ptr, node_count.index() + 1) };
        read.read_exact(offsets_ptr.as_mut_byte_slice())?;

        let mut targets = Box::new_uninit_slice_compat(edge_count.index());
        let targets_ptr = targets.as_mut_ptr() as *mut Target<NI, EV>;
        let targets_ptr =
            unsafe { std::slice::from_raw_parts_mut(targets_ptr, edge_count.index()) };
        read.read_exact(targets_ptr.as_mut_byte_slice())?;

        let offsets = unsafe { offsets.assume_init_compat() };
        let targets = unsafe { targets.assume_init_compat() };

        Ok(Csr::new(offsets, targets))
    }
}

pub struct NodeValues<NV>(pub(crate) Box<[NV]>);

impl<NV> NodeValues<NV> {
    pub fn new(node_values: Vec<NV>) -> Self {
        Self(node_values.into_boxed_slice())
    }
}

impl<NV> FromIterator<NV> for NodeValues<NV> {
    fn from_iter<T: IntoIterator<Item = NV>>(iter: T) -> Self {
        Self(iter.into_iter().collect::<Vec<_>>().into_boxed_slice())
    }
}

impl<NV> NodeValues<NV>
where
    NV: ToByteSlice,
{
    fn serialize<W: Write>(&self, output: &mut W) -> Result<(), Error> {
        let node_count = self.0.len();
        let meta = [node_count];
        output.write_all(meta.as_byte_slice())?;
        output.write_all(self.0.as_byte_slice())?;
        Ok(())
    }
}

impl<NV> NodeValues<NV>
where
    NV: ToMutByteSlice,
{
    fn deserialize<R: Read>(read: &mut R) -> Result<Self, Error> {
        let mut meta = [0_usize; 1];
        read.read_exact(meta.as_mut_byte_slice())?;
        let [node_count] = meta;

        let mut node_values = Box::new_uninit_slice_compat(node_count);
        let node_values_ptr = node_values.as_mut_ptr() as *mut NV;
        let node_values_slice =
            unsafe { std::slice::from_raw_parts_mut(node_values_ptr, node_count.index()) };
        read.read_exact(node_values_slice.as_mut_byte_slice())?;

        let offsets = unsafe { node_values.assume_init_compat() };

        Ok(NodeValues(offsets))
    }
}

mod graphs;
pub use graphs::{DirectedCsrGraph, UndirectedCsrGraph};

fn prefix_sum_atomic<NI: Idx>(degrees: Vec<Atomic<NI>>) -> Vec<Atomic<NI>> {
    let mut last = degrees.last().unwrap().load(Acquire);
    let mut sums = degrees
        .into_iter()
        .scan(NI::zero(), |total, degree| {
            let value = *total;
            *total += degree.into_inner();
            Some(Atomic::new(value))
        })
        .collect::<Vec<_>>();

    last += sums.last().unwrap().load(Acquire);
    sums.push(Atomic::new(last));

    sums
}

pub(crate) fn prefix_sum<NI: Idx>(degrees: Vec<NI>) -> Vec<NI> {
    let mut last = *degrees.last().unwrap();
    let mut sums = degrees
        .into_iter()
        .scan(NI::zero(), |total, degree| {
            let value = *total;
            *total += degree;
            Some(value)
        })
        .collect::<Vec<_>>();
    last += *sums.last().unwrap();
    sums.push(last);
    sums
}

pub(crate) fn sort_targets<NI, T, EV>(offsets: &[NI], targets: &mut [Target<T, EV>])
where
    NI: Idx,
    T: Copy + Send + Ord,
    EV: Send,
{
    to_mut_slices(offsets, targets)
        .par_iter_mut()
        .for_each(|list| list.sort_unstable());
}

fn sort_and_deduplicate_targets<NI, EV>(
    offsets: &[NI],
    targets: &mut [Target<NI, EV>],
) -> (Vec<NI>, Vec<Target<NI, EV>>)
where
    NI: Idx,
    EV: Copy + Send,
{
    let node_count = offsets.len() - 1;

    let mut new_degrees = Vec::with_capacity(node_count);
    let mut target_slices = to_mut_slices(offsets, targets);

    target_slices
        .par_iter_mut()
        .enumerate()
        .map(|(node, slice)| {
            slice.sort_unstable();
            // deduplicate
            let (dedup, _) = slice.partition_dedup_compat();
            let mut new_degree = dedup.len();
            // remove self loops .. there is at most once occurence of node inside dedup
            if let Ok(idx) = dedup.binary_search_by_key(&NI::new(node), |t| t.target) {
                dedup[idx..].rotate_left(1);
                new_degree -= 1;
            }
            NI::new(new_degree)
        })
        .collect_into_vec(&mut new_degrees);

    let new_offsets = prefix_sum(new_degrees);
    debug_assert_eq!(new_offsets.len(), node_count + 1);

    let edge_count = new_offsets[node_count].index();
    let mut new_targets: Vec<Target<NI, EV>> = Vec::with_capacity(edge_count);
    let new_target_slices = to_mut_slices(&new_offsets, new_targets.spare_capacity_mut());

    target_slices
        .into_par_iter()
        .zip(new_target_slices.into_par_iter())
        .for_each(|(old_slice, new_slice)| {
            MaybeUninit::write_slice_compat(new_slice, &old_slice[..new_slice.len()]);
        });

    // SAFETY: We copied all (potentially shortened) target ids from the old
    // target list to the new one.
    unsafe {
        new_targets.set_len(edge_count);
    }

    (new_offsets, new_targets)
}

fn to_mut_slices<'targets, NI: Idx, T>(
    offsets: &[NI],
    targets: &'targets mut [T],
) -> Vec<&'targets mut [T]> {
    let node_count = offsets.len() - 1;
    let mut target_slices = Vec::with_capacity(node_count);
    let mut tail = targets;
    let mut prev_offset = offsets[0];

    for &offset in &offsets[1..] {
        let (list, remainder) = tail.split_at_mut((offset - prev_offset).index());
        target_slices.push(list);
        tail = remainder;
        prev_offset = offset;
    }

    target_slices
}

#[cfg(test)]
mod tests;
