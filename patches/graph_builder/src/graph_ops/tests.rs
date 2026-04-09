use crate::{
    UndirectedNeighbors, builder::GraphBuilder, graph::csr::UndirectedCsrGraph,
    graph_ops::unzip_degrees_and_nodes,
};

use super::*;

#[test]
fn split_by_partition_3_parts() {
    let partition = vec![0..2, 2..5, 5..10];
    let mut slice = (0..10).collect::<Vec<_>>();
    let splits = split_by_partition(&partition, &mut slice);

    assert_eq!(splits.len(), partition.len());
    for (s, p) in splits.into_iter().zip(partition) {
        assert_eq!(s, p.into_iter().collect::<Vec<usize>>());
    }
}

#[test]
fn split_by_partition_8_parts() {
    let partition = vec![0..1, 1..2, 2..3, 3..4, 4..6, 6..7, 7..8, 8..10];
    let mut slice = (0..10).collect::<Vec<_>>();
    let splits = split_by_partition(&partition, &mut slice);

    assert_eq!(splits.len(), partition.len());
    for (s, p) in splits.into_iter().zip(partition) {
        assert_eq!(s, p.into_iter().collect::<Vec<usize>>());
    }
}

#[test]
fn greedy_node_map_partition_1_part() {
    let partitions = greedy_node_map_partition::<usize, _>(|_| 1_usize, 10, 10, 99999);
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0], 0..10);
}

#[test]
fn greedy_node_map_partition_2_parts() {
    let partitions = greedy_node_map_partition::<usize, _>(|x| x % 2_usize, 10, 4, 99999);
    assert_eq!(partitions.len(), 2);
    assert_eq!(partitions[0], 0..8);
    assert_eq!(partitions[1], 8..10);
}

#[test]
fn greedy_node_map_partition_6_parts() {
    let partitions = greedy_node_map_partition::<usize, _>(|x| x, 10, 6, 99999);
    assert_eq!(partitions.len(), 6);
    assert_eq!(partitions[0], 0..4);
    assert_eq!(partitions[1], 4..6);
    assert_eq!(partitions[2], 6..7);
    assert_eq!(partitions[3], 7..8);
    assert_eq!(partitions[4], 8..9);
    assert_eq!(partitions[5], 9..10);
}

#[test]
fn greedy_node_map_partition_max_batches() {
    let partitions = greedy_node_map_partition::<usize, _>(|x| x, 10, 6, 3);
    assert_eq!(partitions.len(), 3);
    assert_eq!(partitions[0], 0..4);
    assert_eq!(partitions[1], 4..6);
    assert_eq!(partitions[2], 6..10);
}

#[test]
fn sort_by_degree_test() {
    let graph: UndirectedCsrGraph<_> = GraphBuilder::new()
        .edges::<u32, _>(vec![
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 2),
        ])
        .build();

    assert_eq!(
        sort_by_degree_desc(&graph),
        vec![(5, 2), (4, 3), (4, 1), (3, 0)]
    );
}

#[test]
fn unzip_degrees_and_nodes_test() {
    let degrees_and_nodes = vec![(5, 2), (4, 3), (4, 1), (3, 0)];

    let (degrees, nodes) = unzip_degrees_and_nodes::<u32>(degrees_and_nodes);

    assert_eq!(degrees, vec![5, 4, 4, 3]);
    assert_eq!(nodes, vec![3, 2, 0, 1]);
}

#[test]
fn relabel_by_degree_test() {
    let mut graph: UndirectedCsrGraph<_> = GraphBuilder::new()
        .edges::<u32, _>(vec![
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 2),
        ])
        .build();

    graph.make_degree_ordered();

    assert_eq!(graph.node_count(), graph.node_count());
    assert_eq!(graph.edge_count(), graph.edge_count());
    assert_eq!(graph.degree(0), 5);
    assert_eq!(graph.degree(1), 4);
    assert_eq!(graph.degree(2), 4);
    assert_eq!(graph.degree(3), 3);
    assert_eq!(graph.neighbors(0).as_slice(), &[1, 1, 2, 2, 3]);
    assert_eq!(graph.neighbors(1).as_slice(), &[0, 0, 2, 3]);
    assert_eq!(graph.neighbors(2).as_slice(), &[0, 0, 1, 3]);
    assert_eq!(graph.neighbors(3).as_slice(), &[0, 1, 2]);
}
