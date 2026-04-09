use super::{Csr, CsrLayout, NodeValues, SwapCsr};
use byte_slice_cast::{ToByteSlice, ToMutByteSlice};
use log::info;
use rayon::prelude::*;
use std::{
    convert::TryFrom,
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
};

use crate::{
    graph_ops::{DeserializeGraphOp, SerializeGraphOp, ToUndirectedOp},
    index::Idx,
    input::{edgelist::Edges, Direction},
    DirectedDegrees, DirectedNeighbors, DirectedNeighborsWithValues, Error, Graph,
    NodeValues as NodeValuesTrait, Target, UndirectedDegrees, UndirectedNeighbors,
    UndirectedNeighborsWithValues,
};

#[cfg(feature = "dotgraph")]
use crate::input::DotGraph;
#[cfg(feature = "dotgraph")]
use std::hash::Hash;

pub struct DirectedCsrGraph<NI: Idx, NV = (), EV = ()> {
    node_values: NodeValues<NV>,
    csr_out: Csr<NI, NI, EV>,
    csr_inc: Csr<NI, NI, EV>,
}

impl<NI: Idx, NV, EV> DirectedCsrGraph<NI, NV, EV> {
    pub fn new(
        node_values: NodeValues<NV>,
        csr_out: Csr<NI, NI, EV>,
        csr_inc: Csr<NI, NI, EV>,
    ) -> Self {
        let g = Self {
            node_values,
            csr_out,
            csr_inc,
        };
        info!(
            "Created directed graph (node_count = {:?}, edge_count = {:?})",
            g.node_count(),
            g.edge_count()
        );

        g
    }
}

impl<NI, NV, EV> ToUndirectedOp for DirectedCsrGraph<NI, NV, EV>
where
    NI: Idx,
    NV: Clone + Send + Sync,
    EV: Copy + Send + Sync,
{
    type Undirected = UndirectedCsrGraph<NI, NV, EV>;

    fn to_undirected(&self, layout: impl Into<Option<CsrLayout>>) -> Self::Undirected {
        let node_values = NodeValues::new(self.node_values.0.to_vec());
        let layout = layout.into().unwrap_or_default();
        let edges = ToUndirectedEdges { g: self };

        UndirectedCsrGraph::from((node_values, edges, layout))
    }
}

struct ToUndirectedEdges<'g, NI: Idx, NV, EV> {
    g: &'g DirectedCsrGraph<NI, NV, EV>,
}

impl<NI, NV, EV> Edges for ToUndirectedEdges<'_, NI, NV, EV>
where
    NI: Idx,
    NV: Send + Sync,
    EV: Copy + Send + Sync,
{
    type NI = NI;
    type EV = EV;
    type EdgeIter<'a>
        = ToUndirectedEdgesIter<'a, NI, NV, EV>
    where
        Self: 'a;

    fn edges(&self) -> Self::EdgeIter<'_> {
        ToUndirectedEdgesIter { g: self.g }
    }

    fn max_node_id(&self) -> Self::NI {
        self.g.node_count() - NI::new(1)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        unimplemented!("This type is not used in tests")
    }
}

struct ToUndirectedEdgesIter<'g, NI: Idx, NV, EV> {
    g: &'g DirectedCsrGraph<NI, NV, EV>,
}

impl<NI: Idx, NV: Send + Sync, EV: Copy + Send + Sync> ParallelIterator
    for ToUndirectedEdgesIter<'_, NI, NV, EV>
{
    type Item = (NI, NI, EV);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let par_iter = (0..self.g.node_count().index())
            .into_par_iter()
            .flat_map_iter(|n| {
                let n = NI::new(n);
                self.g
                    .out_neighbors_with_values(n)
                    .map(move |t| (n, t.target, t.value))
            });
        par_iter.drive_unindexed(consumer)
    }
}

impl<NI: Idx, NV, EV> Graph<NI> for DirectedCsrGraph<NI, NV, EV> {
    delegate::delegate! {
        to self.csr_out {
            fn node_count(&self) -> NI;
            fn edge_count(&self) -> NI;
        }
    }
}

impl<NI: Idx, NV, EV> NodeValuesTrait<NI, NV> for DirectedCsrGraph<NI, NV, EV> {
    fn node_value(&self, node: NI) -> &NV {
        &self.node_values.0[node.index()]
    }
}

impl<NI: Idx, NV, EV> DirectedDegrees<NI> for DirectedCsrGraph<NI, NV, EV> {
    fn out_degree(&self, node: NI) -> NI {
        self.csr_out.degree(node)
    }

    fn in_degree(&self, node: NI) -> NI {
        self.csr_inc.degree(node)
    }
}

impl<NI: Idx, NV> DirectedNeighbors<NI> for DirectedCsrGraph<NI, NV, ()> {
    type NeighborsIterator<'a>
        = std::slice::Iter<'a, NI>
    where
        NV: 'a;

    fn out_neighbors(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.csr_out.targets(node).iter()
    }

    fn in_neighbors(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.csr_inc.targets(node).iter()
    }
}

impl<NI: Idx, NV, EV> DirectedNeighborsWithValues<NI, EV> for DirectedCsrGraph<NI, NV, EV> {
    type NeighborsIterator<'a>
        = std::slice::Iter<'a, Target<NI, EV>>
    where
        NV: 'a,
        EV: 'a;

    fn out_neighbors_with_values(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.csr_out.targets_with_values(node).iter()
    }

    fn in_neighbors_with_values(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.csr_inc.targets_with_values(node).iter()
    }
}

impl<NI, EV, E> From<(E, CsrLayout)> for DirectedCsrGraph<NI, (), EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((edge_list, csr_option): (E, CsrLayout)) -> Self {
        info!("Creating directed graph");
        let node_count = edge_list.max_node_id() + NI::new(1);
        let node_values = NodeValues::new(vec![(); node_count.index()]);

        let csr_out = Csr::from((&edge_list, node_count, Direction::Outgoing, csr_option));
        let csr_inc = Csr::from((&edge_list, node_count, Direction::Incoming, csr_option));

        DirectedCsrGraph::new(node_values, csr_out, csr_inc)
    }
}

impl<NI, NV, EV, E> From<(NodeValues<NV>, E, CsrLayout)> for DirectedCsrGraph<NI, NV, EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((node_values, edge_list, csr_option): (NodeValues<NV>, E, CsrLayout)) -> Self {
        let node_count = NI::new(node_values.0.len());
        let node_count_from_edge_list = edge_list.max_node_id() + NI::new(1);

        assert!(
            node_count >= node_count_from_edge_list,
            "number of node values ({}) does not match node count of edge list ({})",
            node_count.index(),
            node_count_from_edge_list.index()
        );

        let csr_out = Csr::from((&edge_list, node_count, Direction::Outgoing, csr_option));
        let csr_inc = Csr::from((&edge_list, node_count, Direction::Incoming, csr_option));

        DirectedCsrGraph::new(node_values, csr_out, csr_inc)
    }
}

#[cfg(feature = "dotgraph")]
impl<NI, Label> From<(DotGraph<NI, Label>, CsrLayout)> for DirectedCsrGraph<NI, ()>
where
    NI: Idx,
    Label: Idx + Hash,
{
    fn from((dot_graph, csr_layout): (DotGraph<NI, Label>, CsrLayout)) -> Self {
        let DotGraph { edge_list, .. } = dot_graph;
        DirectedCsrGraph::from((edge_list, csr_layout))
    }
}

#[cfg(feature = "dotgraph")]
impl<NI, Label> From<(DotGraph<NI, Label>, CsrLayout)> for DirectedCsrGraph<NI, Label>
where
    NI: Idx,
    Label: Idx + Hash,
{
    fn from((dot_graph, csr_layout): (DotGraph<NI, Label>, CsrLayout)) -> Self {
        let DotGraph {
            edge_list, labels, ..
        } = dot_graph;
        let node_values = NodeValues::new(labels);

        DirectedCsrGraph::from((node_values, edge_list, csr_layout))
    }
}

impl<W, NI, NV, EV> SerializeGraphOp<W> for DirectedCsrGraph<NI, NV, EV>
where
    W: Write,
    NI: Idx + ToByteSlice,
    NV: ToByteSlice,
    EV: ToByteSlice,
{
    fn serialize(&self, mut output: W) -> Result<(), Error> {
        let DirectedCsrGraph {
            node_values,
            csr_out,
            csr_inc,
        } = self;

        node_values.serialize(&mut output)?;
        csr_out.serialize(&mut output)?;
        csr_inc.serialize(&mut output)?;
        Ok(())
    }
}

impl<R, NI, NV, EV> DeserializeGraphOp<R, Self> for DirectedCsrGraph<NI, NV, EV>
where
    R: Read,
    NI: Idx + ToMutByteSlice,
    NV: ToMutByteSlice,
    EV: ToMutByteSlice,
{
    fn deserialize(mut read: R) -> Result<Self, Error> {
        let node_values: NodeValues<NV> = NodeValues::deserialize(&mut read)?;
        let csr_out: Csr<NI, NI, EV> = Csr::deserialize(&mut read)?;
        let csr_inc: Csr<NI, NI, EV> = Csr::deserialize(&mut read)?;
        Ok(DirectedCsrGraph::new(node_values, csr_out, csr_inc))
    }
}

impl<NI, EV> TryFrom<(PathBuf, CsrLayout)> for DirectedCsrGraph<NI, EV>
where
    NI: Idx + ToMutByteSlice,
    EV: ToMutByteSlice,
{
    type Error = Error;

    fn try_from((path, _): (PathBuf, CsrLayout)) -> Result<Self, Self::Error> {
        let reader = BufReader::new(File::open(path)?);
        DirectedCsrGraph::deserialize(reader)
    }
}

pub struct UndirectedCsrGraph<NI: Idx, NV = (), EV = ()> {
    node_values: NodeValues<NV>,
    csr: Csr<NI, NI, EV>,
}

impl<NI: Idx, EV> From<Csr<NI, NI, EV>> for UndirectedCsrGraph<NI, (), EV> {
    fn from(csr: Csr<NI, NI, EV>) -> Self {
        UndirectedCsrGraph::new(NodeValues::new(vec![(); csr.node_count().index()]), csr)
    }
}

impl<NI: Idx, NV, EV> UndirectedCsrGraph<NI, NV, EV> {
    pub fn new(node_values: NodeValues<NV>, csr: Csr<NI, NI, EV>) -> Self {
        let g = Self { node_values, csr };
        info!(
            "Created undirected graph (node_count = {:?}, edge_count = {:?})",
            g.node_count(),
            g.edge_count()
        );

        g
    }
}

impl<NI: Idx, NV, EV> Graph<NI> for UndirectedCsrGraph<NI, NV, EV> {
    fn node_count(&self) -> NI {
        self.csr.node_count()
    }

    fn edge_count(&self) -> NI {
        self.csr.edge_count() / NI::new(2)
    }
}

impl<NI: Idx, NV, EV> NodeValuesTrait<NI, NV> for UndirectedCsrGraph<NI, NV, EV> {
    fn node_value(&self, node: NI) -> &NV {
        &self.node_values.0[node.index()]
    }
}

impl<NI: Idx, NV, EV> UndirectedDegrees<NI> for UndirectedCsrGraph<NI, NV, EV> {
    fn degree(&self, node: NI) -> NI {
        self.csr.degree(node)
    }
}

impl<NI: Idx, NV> UndirectedNeighbors<NI> for UndirectedCsrGraph<NI, NV> {
    type NeighborsIterator<'a>
        = std::slice::Iter<'a, NI>
    where
        NV: 'a;

    fn neighbors(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.csr.targets(node).iter()
    }
}

impl<NI: Idx, NV, EV> UndirectedNeighborsWithValues<NI, EV> for UndirectedCsrGraph<NI, NV, EV> {
    type NeighborsIterator<'a>
        = std::slice::Iter<'a, Target<NI, EV>>
    where
        NV: 'a,
        EV: 'a;

    fn neighbors_with_values(&self, node: NI) -> Self::NeighborsIterator<'_> {
        self.csr.targets_with_values(node).iter()
    }
}

impl<NI: Idx, NV, EV> SwapCsr<NI, NI, EV> for UndirectedCsrGraph<NI, NV, EV> {
    fn swap_csr(&mut self, mut csr: Csr<NI, NI, EV>) -> &mut Self {
        std::mem::swap(&mut self.csr, &mut csr);
        self
    }
}

impl<NI, EV, E> From<(E, CsrLayout)> for UndirectedCsrGraph<NI, (), EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((edge_list, csr_option): (E, CsrLayout)) -> Self {
        info!("Creating undirected graph");
        let node_count = edge_list.max_node_id() + NI::new(1);
        let node_values = NodeValues::new(vec![(); node_count.index()]);
        let csr = Csr::from((&edge_list, node_count, Direction::Undirected, csr_option));

        UndirectedCsrGraph::new(node_values, csr)
    }
}

impl<NI, NV, EV, E> From<(NodeValues<NV>, E, CsrLayout)> for UndirectedCsrGraph<NI, NV, EV>
where
    NI: Idx,
    EV: Copy + Send + Sync,
    E: Edges<NI = NI, EV = EV>,
{
    fn from((node_values, edge_list, csr_option): (NodeValues<NV>, E, CsrLayout)) -> Self {
        let node_count = NI::new(node_values.0.len());
        let node_count_from_edge_list = edge_list.max_node_id() + NI::new(1);

        assert!(
            node_count >= node_count_from_edge_list,
            "number of node values ({}) does not match node count of edge list ({})",
            node_count.index(),
            node_count_from_edge_list.index()
        );

        let csr = Csr::from((&edge_list, node_count, Direction::Undirected, csr_option));
        UndirectedCsrGraph::new(node_values, csr)
    }
}

#[cfg(feature = "dotgraph")]
impl<NI, Label> From<(DotGraph<NI, Label>, CsrLayout)> for UndirectedCsrGraph<NI, ()>
where
    NI: Idx,
    Label: Idx + Hash,
{
    fn from((dot_graph, csr_layout): (DotGraph<NI, Label>, CsrLayout)) -> Self {
        let DotGraph { edge_list, .. } = dot_graph;
        UndirectedCsrGraph::from((edge_list, csr_layout))
    }
}

#[cfg(feature = "dotgraph")]
impl<NI, Label> From<(DotGraph<NI, Label>, CsrLayout)> for UndirectedCsrGraph<NI, Label>
where
    NI: Idx,
    Label: Idx + Hash,
{
    fn from((dot_graph, csr_layout): (DotGraph<NI, Label>, CsrLayout)) -> Self {
        let DotGraph {
            edge_list, labels, ..
        } = dot_graph;
        let node_values = NodeValues::new(labels);

        UndirectedCsrGraph::from((node_values, edge_list, csr_layout))
    }
}

impl<W, NI, NV, EV> SerializeGraphOp<W> for UndirectedCsrGraph<NI, NV, EV>
where
    W: Write,
    NI: Idx + ToByteSlice,
    NV: ToByteSlice,
    EV: ToByteSlice,
{
    fn serialize(&self, mut output: W) -> Result<(), Error> {
        let UndirectedCsrGraph { node_values, csr } = self;

        node_values.serialize(&mut output)?;
        csr.serialize(&mut output)?;
        Ok(())
    }
}

impl<R, NI, NV, EV> DeserializeGraphOp<R, Self> for UndirectedCsrGraph<NI, NV, EV>
where
    R: Read,
    NI: Idx + ToMutByteSlice,
    NV: ToMutByteSlice,
    EV: ToMutByteSlice,
{
    fn deserialize(mut read: R) -> Result<Self, Error> {
        let node_values = NodeValues::deserialize(&mut read)?;
        let csr: Csr<NI, NI, EV> = Csr::deserialize(&mut read)?;
        Ok(UndirectedCsrGraph::new(node_values, csr))
    }
}

impl<NI, EV> TryFrom<(PathBuf, CsrLayout)> for UndirectedCsrGraph<NI, EV>
where
    NI: Idx + ToMutByteSlice,
    EV: ToMutByteSlice,
{
    type Error = Error;

    fn try_from((path, _): (PathBuf, CsrLayout)) -> Result<Self, Self::Error> {
        let reader = BufReader::new(File::open(path)?);
        UndirectedCsrGraph::deserialize(reader)
    }
}
