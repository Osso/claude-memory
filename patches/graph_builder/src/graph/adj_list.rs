use crate::{
    index::Idx, prelude::Direction, prelude::Edges, prelude::NodeValues as NodeValuesTrait,
    CsrLayout, DirectedDegrees, DirectedNeighbors, DirectedNeighborsWithValues, Graph, Target,
    UndirectedDegrees, UndirectedNeighbors, UndirectedNeighborsWithValues,
};
use crate::{EdgeMutation, EdgeMutationWithValues};

use log::info;
use std::sync::{RwLock, RwLockReadGuard};
use std::time::Instant;

use crate::graph::csr::NodeValues;
use rayon::prelude::*;

#[derive(Debug)]
pub struct AdjacencyList<NI, EV> {
    edges: Vec<RwLock<Vec<Target<NI, EV>>>>,
    layout: CsrLayout,
}

const _: () = {
    const fn is_send<T: Send>() {}
    const fn is_sync<T: Sync>() {}

    is_send::<AdjacencyList<u64, ()>>();
    is_sync::<AdjacencyList<u64, ()>>();
};

impl<NI: Idx, EV> AdjacencyList<NI, EV> {
    pub fn new(edges: Vec<Vec<Target<NI, EV>>>) -> Self {
        Self::with_layout(edges, CsrLayout::Unsorted)
    }

    pub fn with_layout(edges: Vec<Vec<Target<NI, EV>>>, layout: CsrLayout) -> Self {
        let edges = edges.into_iter().map(RwLock::new).collect::<_>();
        Self { edges, layout }
    }

    #[inline]
    pub(crate) fn node_count(&self) -> NI {
        NI::new(self.edges.len())
    }

    #[inline]
    pub(crate) fn edge_count(&self) -> NI
    where
        NI: Send + Sync,
        EV: Send + Sync,
    {
        NI::new(self.edges.par_iter().map(|v| v.read().unwrap().len()).sum())
    }

    #[inline]
    pub(crate) fn degree(&self, node: NI) -> NI {
        NI::new(self.edges[node.index()].read().unwrap().len())
    }

    #[inline]
    pub(crate) fn insert(&self, source: NI, target: Target<NI, EV>) {
        let mut edges = self.edges[source.index()].write().unwrap();
        Self::apply_layout(self.layout, &mut edges, target);
    }

    #[inline]
    pub(crate) fn insert_mut(&mut self, source: NI, target: Target<NI, EV>) {
        let edges = self.edges[source.index()].get_mut().unwrap();
        Self::apply_layout(self.layout, edges, target);
    }

    #[inline]
    fn check_bounds(&self, node: NI) -> Result<(), crate::Error> {
        if node >= self.node_count() {
            return Err(crate::Error::MissingNode {
                node: format!("{}", node.index()),
            });
        };
        Ok(())
    }

    #[inline]
    fn apply_layout(layout: CsrLayout, edges: &mut Vec<Target<NI, EV>>, target: Target<NI, EV>) {
        match layout {
            CsrLayout::Sorted => match edges.binary_search(&target) {
                Ok(i) => edges.insert(i, target),
                Err(i) => edges.insert(i, target),
            },
            CsrLayout::Unsorted => edges.push(target),
            CsrLayout::Deduplicated => match edges.binary_search(&target) {
                Ok(_) => {}
                Err(i) => edges.insert(i, target),
            },
        };
    }
}

#[derive(Debug)]
pub struct Targets<'slice, NI: Idx> {
    targets: RwLockReadGuard<'slice, Vec<Target<NI, ()>>>,
}

impl<'slice, NI: Idx> Targets<'slice, NI> {
    pub fn as_slice(&self) -> &'slice [NI] {
        assert_eq!(
            std::mem::size_of::<Target<NI, ()>>(),
            std::mem::size_of::<NI>()
        );
        assert_eq!(
            std::mem::align_of::<Target<NI, ()>>(),
            std::mem::align_of::<NI>()
        );
        // SAFETY: The types Target<T, ()> and T are verified to have the same
        //         size and alignment.
        //         We can upcast the lifetime since the MutexGuard
        //         is not exposed, so it is not possible to deref mutable.
        unsafe { std::slice::from_raw_parts(self.targets.as_ptr().cast(), self.targets.len()) }
    }
}

pub struct TargetsIter<'slice, NI: Idx> {
    _targets: Targets<'slice, NI>,
    slice: std::slice::Iter<'slice, NI>,
}

impl<'slice, NI: Idx> TargetsIter<'slice, NI> {
    pub fn as_slice(&self) -> &'slice [NI] {
        self.slice.as_slice()
    }
}

impl<'slice, NI: Idx> IntoIterator for Targets<'slice, NI> {
    type Item = &'slice NI;

    type IntoIter = TargetsIter<'slice, NI>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = self.as_slice();
        TargetsIter {
            _targets: self,
            slice: slice.iter(),
        }
    }
}

impl<'slice, NI: Idx> Iterator for TargetsIter<'slice, NI> {
    type Item = &'slice NI;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next()
    }
}

impl<NI: Idx> AdjacencyList<NI, ()> {
    #[inline]
    pub(crate) fn targets(&self, node: NI) -> Targets<'_, NI> {
        let targets = self.edges[node.index()].read().unwrap();

        Targets { targets }
    }
}

#[derive(Debug)]
pub struct TargetsWithValues<'slice, NI: Idx, EV> {
    targets: RwLockReadGuard<'slice, Vec<Target<NI, EV>>>,
}

impl<'slice, NI: Idx, EV> TargetsWithValues<'slice, NI, EV> {
    pub fn as_slice(&self) -> &'slice [Target<NI, EV>] {
        // SAFETY: We can upcast the lifetime since the MutexGuard
        // is not exposed, so it is not possible to deref mutable.
        unsafe { std::slice::from_raw_parts(self.targets.as_ptr(), self.targets.len()) }
    }
}

pub struct TargetsWithValuesIter<'slice, NI: Idx, EV> {
    _targets: TargetsWithValues<'slice, NI, EV>,
    slice: std::slice::Iter<'slice, Target<NI, EV>>,
}

impl<'slice, NI: Idx, EV> TargetsWithValuesIter<'slice, NI, EV> {
    pub fn as_slice(&self) -> &'slice [Target<NI, EV>] {
        self.slice.as_slice()
    }
}

impl<'slice, NI: Idx, EV> IntoIterator for TargetsWithValues<'slice, NI, EV> {
    type Item = &'slice Target<NI, EV>;

    type IntoIter = TargetsWithValuesIter<'slice, NI, EV>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = self.as_slice();
        TargetsWithValuesIter {
            _targets: self,
            slice: slice.iter(),
        }
    }
}

impl<'slice, NI: Idx, EV> Iterator for TargetsWithValuesIter<'slice, NI, EV> {
    type Item = &'slice Target<NI, EV>;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next()
    }
}

impl<NI: Idx, EV> AdjacencyList<NI, EV> {
    #[inline]
    pub(crate) fn targets_with_values(&self, node: NI) -> TargetsWithValues<'_, NI, EV> {
        TargetsWithValues {
            targets: self.edges[node.index()].read().unwrap(),
        }
    }
}

impl<NI, EV, E> From<(&'_ E, NI, Direction, CsrLayout)> for AdjacencyList<NI, EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from(
        (edge_list, node_count, direction, csr_layout): (&'_ E, NI, Direction, CsrLayout),
    ) -> Self {
        let start = Instant::now();
        let thread_safe_vec = init_adjacency_buckets(edge_list, node_count, direction);
        info!("Initialized adjacency list in {:?}", start.elapsed());

        let start = Instant::now();
        populate_adjacency_buckets(edge_list, direction, &thread_safe_vec);
        info!("Grouped edge tuples in {:?}", start.elapsed());

        let start = Instant::now();
        let edges = finalize_adjacency_buckets(thread_safe_vec, csr_layout, node_count);

        info!(
            "Applied list layout and finalized edge list in {:?}",
            start.elapsed()
        );

        AdjacencyList::with_layout(edges, csr_layout)
    }
}

fn init_adjacency_buckets<NI, EV, E>(
    edge_list: &E,
    node_count: NI,
    direction: Direction,
) -> Vec<RwLock<Vec<Target<NI, EV>>>>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    edge_list
        .degrees(node_count, direction)
        .into_par_iter()
        .map(|degree| RwLock::new(Vec::with_capacity(degree.into_inner().index())))
        .collect()
}

fn populate_adjacency_buckets<NI, EV, E>(
    edge_list: &E,
    direction: Direction,
    buckets: &[RwLock<Vec<Target<NI, EV>>>],
) where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    edge_list.edges().for_each(|(source, target, value)| {
        push_target(buckets, direction, source, target, value);
        push_target(buckets, reverse_direction(direction), target, source, value);
    });
}

fn push_target<NI, EV>(
    buckets: &[RwLock<Vec<Target<NI, EV>>>],
    direction: Direction,
    source: NI,
    target: NI,
    value: EV,
) where
    NI: Idx,
    EV: Copy + Send + Sync,
{
    if !matches!(direction, Direction::Outgoing | Direction::Undirected) {
        return;
    }

    buckets[source.index()]
        .write()
        .unwrap()
        .push(Target::new(target, value));
}

fn reverse_direction(direction: Direction) -> Direction {
    match direction {
        Direction::Outgoing => Direction::Incoming,
        Direction::Incoming => Direction::Outgoing,
        Direction::Undirected => Direction::Undirected,
    }
}

fn finalize_adjacency_buckets<NI, EV>(
    buckets: Vec<RwLock<Vec<Target<NI, EV>>>>,
    csr_layout: CsrLayout,
    node_count: NI,
) -> Vec<Vec<Target<NI, EV>>>
where
    NI: Idx,
    EV: Copy + Send + Sync,
{
    let mut edges = Vec::with_capacity(node_count.index());
    buckets
        .into_par_iter()
        .map(|list| {
            let mut list = list.into_inner().unwrap();
            apply_adjacency_layout(&mut list, csr_layout);
            list
        })
        .collect_into_vec(&mut edges);
    edges
}

fn apply_adjacency_layout<NI: Idx, EV>(list: &mut Vec<Target<NI, EV>>, csr_layout: CsrLayout) {
    match csr_layout {
        CsrLayout::Sorted => list.sort_unstable_by_key(|target| target.target),
        CsrLayout::Unsorted => {}
        CsrLayout::Deduplicated => {
            list.sort_unstable_by_key(|target| target.target);
            list.dedup_by_key(|target| target.target);
        }
    }
}

pub struct DirectedALGraph<NI: Idx, NV = (), EV = ()> {
    node_values: NodeValues<NV>,
    al_out: AdjacencyList<NI, EV>,
    al_inc: AdjacencyList<NI, EV>,
}

impl<NI: Idx, NV, EV> DirectedALGraph<NI, NV, EV>
where
    NV: Send + Sync,
    EV: Send + Sync,
{
    pub fn new(
        node_values: NodeValues<NV>,
        al_out: AdjacencyList<NI, EV>,
        al_inc: AdjacencyList<NI, EV>,
    ) -> Self {
        let g = Self {
            node_values,
            al_out,
            al_inc,
        };

        info!(
            "Created directed graph (node_count = {:?}, edge_count = {:?})",
            g.node_count(),
            g.edge_count()
        );

        g
    }
}

impl<NI: Idx, NV, EV> Graph<NI> for DirectedALGraph<NI, NV, EV>
where
    NV: Send + Sync,
    EV: Send + Sync,
{
    delegate::delegate! {
        to self.al_out {
            fn node_count(&self) -> NI;
            fn edge_count(&self) -> NI;
        }
    }
}

impl<NI: Idx, NV, EV> NodeValuesTrait<NI, NV> for DirectedALGraph<NI, NV, EV> {
    fn node_value(&self, node: NI) -> &NV {
        &self.node_values.0[node.index()]
    }
}

impl<NI: Idx, NV, EV> DirectedDegrees<NI> for DirectedALGraph<NI, NV, EV> {
    fn out_degree(&self, node: NI) -> NI {
        self.al_out.degree(node)
    }

    fn in_degree(&self, node: NI) -> NI {
        self.al_inc.degree(node)
    }
}

impl<NI: Idx, NV> DirectedNeighbors<NI> for DirectedALGraph<NI, NV, ()> {
    type NeighborsIterator<'a>
        = TargetsIter<'a, NI>
    where
        NV: 'a;

    fn out_neighbors(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.al_out.targets(node).into_iter()
    }

    fn in_neighbors(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.al_inc.targets(node).into_iter()
    }
}

impl<NI: Idx, NV, EV> DirectedNeighborsWithValues<NI, EV> for DirectedALGraph<NI, NV, EV> {
    type NeighborsIterator<'a>
        = TargetsWithValuesIter<'a, NI, EV>
    where
        NV: 'a,
        EV: 'a;

    fn out_neighbors_with_values(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.al_out.targets_with_values(node).into_iter()
    }

    fn in_neighbors_with_values(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.al_inc.targets_with_values(node).into_iter()
    }
}

impl<NI: Idx, NV> EdgeMutation<NI> for DirectedALGraph<NI, NV> {
    fn add_edge(&self, source: NI, target: NI) -> Result<(), crate::Error> {
        self.add_edge_with_value(source, target, ())
    }

    fn add_edge_mut(&mut self, source: NI, target: NI) -> Result<(), crate::Error> {
        self.add_edge_with_value_mut(source, target, ())
    }
}

impl<NI: Idx, NV, EV: Copy> EdgeMutationWithValues<NI, EV> for DirectedALGraph<NI, NV, EV> {
    fn add_edge_with_value(&self, source: NI, target: NI, value: EV) -> Result<(), crate::Error> {
        self.al_out.check_bounds(source)?;
        self.al_inc.check_bounds(target)?;
        self.al_out.insert(source, Target::new(target, value));
        self.al_inc.insert(target, Target::new(source, value));

        Ok(())
    }

    fn add_edge_with_value_mut(
        &mut self,
        source: NI,
        target: NI,
        value: EV,
    ) -> Result<(), crate::Error> {
        self.al_out.check_bounds(source)?;
        self.al_inc.check_bounds(target)?;
        self.al_out.insert_mut(source, Target::new(target, value));
        self.al_inc.insert_mut(target, Target::new(source, value));

        Ok(())
    }
}

impl<NI, EV, E> From<(E, CsrLayout)> for DirectedALGraph<NI, (), EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((edge_list, csr_layout): (E, CsrLayout)) -> Self {
        info!("Creating directed graph");
        let node_count = edge_list.max_node_id() + NI::new(1);
        let node_values = NodeValues::new(vec![(); node_count.index()]);

        let start = Instant::now();
        let al_out = AdjacencyList::from((&edge_list, node_count, Direction::Outgoing, csr_layout));
        info!("Created outgoing adjacency list in {:?}", start.elapsed());

        let start = Instant::now();
        let al_inc = AdjacencyList::from((&edge_list, node_count, Direction::Incoming, csr_layout));
        info!("Created incoming adjacency list in {:?}", start.elapsed());

        DirectedALGraph::new(node_values, al_out, al_inc)
    }
}

impl<NI, NV, EV, E> From<(NodeValues<NV>, E, CsrLayout)> for DirectedALGraph<NI, NV, EV>
where
    NI: Idx,
    NV: Send + Sync,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((node_values, edge_list, csr_layout): (NodeValues<NV>, E, CsrLayout)) -> Self {
        info!("Creating directed graph");
        let node_count = edge_list.max_node_id() + NI::new(1);

        let start = Instant::now();
        let al_out = AdjacencyList::from((&edge_list, node_count, Direction::Outgoing, csr_layout));
        info!("Created outgoing adjacency list in {:?}", start.elapsed());

        let start = Instant::now();
        let al_inc = AdjacencyList::from((&edge_list, node_count, Direction::Incoming, csr_layout));
        info!("Created incoming adjacency list in {:?}", start.elapsed());

        DirectedALGraph::new(node_values, al_out, al_inc)
    }
}

pub struct UndirectedALGraph<NI: Idx, NV = (), EV = ()> {
    node_values: NodeValues<NV>,
    al: AdjacencyList<NI, EV>,
}

impl<NI: Idx, NV, EV> UndirectedALGraph<NI, NV, EV>
where
    NV: Send + Sync,
    EV: Send + Sync,
{
    pub fn new(node_values: NodeValues<NV>, al: AdjacencyList<NI, EV>) -> Self {
        let g = Self { node_values, al };

        info!(
            "Created undirected graph (node_count = {:?}, edge_count = {:?})",
            g.node_count(),
            g.edge_count()
        );

        g
    }
}

impl<NI: Idx, NV, EV> Graph<NI> for UndirectedALGraph<NI, NV, EV>
where
    NV: Send + Sync,
    EV: Send + Sync,
{
    fn node_count(&self) -> NI {
        self.al.node_count()
    }

    fn edge_count(&self) -> NI {
        self.al.edge_count() / NI::new(2)
    }
}

impl<NI: Idx, NV, EV> NodeValuesTrait<NI, NV> for UndirectedALGraph<NI, NV, EV> {
    fn node_value(&self, node: NI) -> &NV {
        &self.node_values.0[node.index()]
    }
}

impl<NI: Idx, NV, EV> UndirectedDegrees<NI> for UndirectedALGraph<NI, NV, EV> {
    fn degree(&self, node: NI) -> NI {
        self.al.degree(node)
    }
}

impl<NI: Idx, NV> UndirectedNeighbors<NI> for UndirectedALGraph<NI, NV, ()> {
    type NeighborsIterator<'a>
        = TargetsIter<'a, NI>
    where
        NV: 'a;

    fn neighbors(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.al.targets(node).into_iter()
    }
}

impl<NI: Idx, NV, EV> UndirectedNeighborsWithValues<NI, EV> for UndirectedALGraph<NI, NV, EV> {
    type NeighborsIterator<'a>
        = TargetsWithValuesIter<'a, NI, EV>
    where
        NV: 'a,
        EV: 'a;

    fn neighbors_with_values(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.al.targets_with_values(node).into_iter()
    }
}

impl<NI: Idx, NV> EdgeMutation<NI> for UndirectedALGraph<NI, NV, ()> {
    fn add_edge(&self, source: NI, target: NI) -> Result<(), crate::Error> {
        self.add_edge_with_value(source, target, ())
    }

    fn add_edge_mut(&mut self, source: NI, target: NI) -> Result<(), crate::Error> {
        self.add_edge_with_value_mut(source, target, ())
    }
}

impl<NI: Idx, NV, EV: Copy> EdgeMutationWithValues<NI, EV> for UndirectedALGraph<NI, NV, EV> {
    fn add_edge_with_value(&self, source: NI, target: NI, value: EV) -> Result<(), crate::Error> {
        self.al.check_bounds(source)?;
        self.al.check_bounds(target)?;
        self.al.insert(source, Target::new(target, value));
        self.al.insert(target, Target::new(source, value));

        Ok(())
    }

    fn add_edge_with_value_mut(
        &mut self,
        source: NI,
        target: NI,
        value: EV,
    ) -> Result<(), crate::Error> {
        self.al.check_bounds(source)?;
        self.al.check_bounds(target)?;
        self.al.insert_mut(source, Target::new(target, value));
        self.al.insert_mut(target, Target::new(source, value));

        Ok(())
    }
}

impl<NI, EV, E> From<(E, CsrLayout)> for UndirectedALGraph<NI, (), EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((edge_list, csr_layout): (E, CsrLayout)) -> Self {
        info!("Creating undirected graph");
        let node_count = edge_list.max_node_id() + NI::new(1);
        let node_values = NodeValues::new(vec![(); node_count.index()]);

        let start = Instant::now();
        let al = AdjacencyList::from((&edge_list, node_count, Direction::Undirected, csr_layout));
        info!("Created adjacency list in {:?}", start.elapsed());

        UndirectedALGraph::new(node_values, al)
    }
}

impl<NI, NV, EV, E> From<(NodeValues<NV>, E, CsrLayout)> for UndirectedALGraph<NI, NV, EV>
where
    NI: Idx,
    NV: Send + Sync,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((node_values, edge_list, csr_layout): (NodeValues<NV>, E, CsrLayout)) -> Self {
        info!("Creating undirected graph");
        let node_count = edge_list.max_node_id() + NI::new(1);

        let start = Instant::now();
        let al = AdjacencyList::from((&edge_list, node_count, Direction::Undirected, csr_layout));
        info!("Created adjacency list in {:?}", start.elapsed());

        UndirectedALGraph::new(node_values, al)
    }
}

#[cfg(test)]
mod tests;
