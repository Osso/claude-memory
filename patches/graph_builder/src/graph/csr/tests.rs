use std::{
    io::{Seek, SeekFrom},
    sync::atomic::Ordering::SeqCst,
};

use rayon::ThreadPoolBuilder;

use crate::builder::GraphBuilder;

use super::*;

#[test]
fn to_mut_slices_test() {
    let offsets = &[0, 2, 5, 5, 8];
    let targets = &mut [0, 1, 2, 3, 4, 5, 6, 7];
    let slices = to_mut_slices::<usize, usize>(offsets, targets);

    assert_eq!(
        slices,
        vec![vec![0, 1], vec![2, 3, 4], vec![], vec![5, 6, 7]]
    );
}

fn t<T>(t: T) -> Target<T, ()> {
    Target::new(t, ())
}

#[test]
fn sort_targets_test() {
    let offsets = &[0, 2, 5, 5, 8];
    let mut targets = vec![t(1), t(0), t(4), t(2), t(3), t(5), t(6), t(7)];
    sort_targets::<usize, _, _>(offsets, &mut targets);

    assert_eq!(
        targets,
        vec![t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7)]
    );
}

#[test]
fn sort_and_deduplicate_targets_test() {
    let offsets = &[0, 3, 7, 7, 10];
    // 0: [1, 1, 0]    => [1] (removed duplicate and self loop)
    // 1: [4, 2, 3, 2] => [2, 3, 4] (removed duplicate)
    let mut targets = vec![t(1), t(1), t(0), t(4), t(2), t(3), t(2), t(5), t(6), t(7)];
    let (offsets, targets) = sort_and_deduplicate_targets::<usize, _>(offsets, &mut targets);

    assert_eq!(offsets, vec![0, 1, 4, 4, 7]);
    assert_eq!(targets, vec![t(1), t(2), t(3), t(4), t(5), t(6), t(7)]);
}

#[test]
fn prefix_sum_test() {
    let degrees = vec![42, 0, 1337, 4, 2, 0];
    let prefix_sum = prefix_sum::<usize>(degrees);

    assert_eq!(prefix_sum, vec![0, 42, 42, 1379, 1383, 1385, 1385]);
}

#[test]
fn prefix_sum_atomic_test() {
    let degrees = vec![42, 0, 1337, 4, 2, 0]
        .into_iter()
        .map(Atomic::<usize>::new)
        .collect::<Vec<_>>();

    let prefix_sum = prefix_sum_atomic(degrees)
        .into_iter()
        .map(|n| n.load(SeqCst))
        .collect::<Vec<_>>();

    assert_eq!(prefix_sum, vec![0, 42, 42, 1379, 1383, 1385, 1385]);
}

#[test]
fn serialize_directed_usize_graph_test() {
    let mut file = tempfile::tempfile().unwrap();

    let g0: DirectedCsrGraph<usize> = GraphBuilder::new()
        .edges(vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)])
        .build();

    assert!(g0.serialize(&file).is_ok());

    file.seek(SeekFrom::Start(0)).unwrap();
    let g1 = DirectedCsrGraph::<usize>::deserialize(file).unwrap();

    assert_eq!(g0.node_count(), g1.node_count());
    assert_eq!(g0.edge_count(), g1.edge_count());
    assert_eq!(g0.out_neighbors(0).as_slice(), g1.out_neighbors(0).as_slice());
    assert_eq!(g0.out_neighbors(1).as_slice(), g1.out_neighbors(1).as_slice());
    assert_eq!(g0.out_neighbors(2).as_slice(), g1.out_neighbors(2).as_slice());
    assert_eq!(g0.out_neighbors(3).as_slice(), g1.out_neighbors(3).as_slice());
    assert_eq!(g0.in_neighbors(0).as_slice(), g1.in_neighbors(0).as_slice());
    assert_eq!(g0.in_neighbors(1).as_slice(), g1.in_neighbors(1).as_slice());
    assert_eq!(g0.in_neighbors(2).as_slice(), g1.in_neighbors(2).as_slice());
    assert_eq!(g0.in_neighbors(3).as_slice(), g1.in_neighbors(3).as_slice());
}

#[test]
fn serialize_undirected_usize_graph_test() {
    let mut file = tempfile::tempfile().unwrap();

    let g0: UndirectedCsrGraph<usize> = GraphBuilder::new()
        .edges(vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)])
        .build();

    assert!(g0.serialize(&file).is_ok());

    file.seek(SeekFrom::Start(0)).unwrap();
    let g1 = UndirectedCsrGraph::<usize>::deserialize(file).unwrap();

    assert_eq!(g0.node_count(), g1.node_count());
    assert_eq!(g0.edge_count(), g1.edge_count());
    assert_eq!(g0.neighbors(0).as_slice(), g1.neighbors(0).as_slice());
    assert_eq!(g0.neighbors(1).as_slice(), g1.neighbors(1).as_slice());
    assert_eq!(g0.neighbors(2).as_slice(), g1.neighbors(2).as_slice());
    assert_eq!(g0.neighbors(3).as_slice(), g1.neighbors(3).as_slice());
}

#[test]
fn serialize_directed_u32_graph_test() {
    let mut file = tempfile::tempfile().unwrap();

    let g0: DirectedCsrGraph<u32> = GraphBuilder::new()
        .edges(vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)])
        .build();

    assert!(g0.serialize(&file).is_ok());

    file.seek(SeekFrom::Start(0)).unwrap();
    let g1 = DirectedCsrGraph::<u32>::deserialize(file).unwrap();

    assert_eq!(g0.node_count(), g1.node_count());
    assert_eq!(g0.edge_count(), g1.edge_count());
    assert_eq!(g0.out_neighbors(0).as_slice(), g1.out_neighbors(0).as_slice());
    assert_eq!(g0.out_neighbors(1).as_slice(), g1.out_neighbors(1).as_slice());
    assert_eq!(g0.out_neighbors(2).as_slice(), g1.out_neighbors(2).as_slice());
    assert_eq!(g0.out_neighbors(3).as_slice(), g1.out_neighbors(3).as_slice());
    assert_eq!(g0.in_neighbors(0).as_slice(), g1.in_neighbors(0).as_slice());
    assert_eq!(g0.in_neighbors(1).as_slice(), g1.in_neighbors(1).as_slice());
    assert_eq!(g0.in_neighbors(2).as_slice(), g1.in_neighbors(2).as_slice());
    assert_eq!(g0.in_neighbors(3).as_slice(), g1.in_neighbors(3).as_slice());
}

#[test]
fn serialize_undirected_u32_graph_test() {
    let mut file = tempfile::tempfile().unwrap();

    let g0: UndirectedCsrGraph<u32> = GraphBuilder::new()
        .edges(vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)])
        .build();

    assert!(g0.serialize(&file).is_ok());

    file.seek(SeekFrom::Start(0)).unwrap();
    let g1 = UndirectedCsrGraph::<u32>::deserialize(file).unwrap();

    assert_eq!(g0.node_count(), g1.node_count());
    assert_eq!(g0.edge_count(), g1.edge_count());
    assert_eq!(g0.neighbors(0).as_slice(), g1.neighbors(0).as_slice());
    assert_eq!(g0.neighbors(1).as_slice(), g1.neighbors(1).as_slice());
    assert_eq!(g0.neighbors(2).as_slice(), g1.neighbors(2).as_slice());
    assert_eq!(g0.neighbors(3).as_slice(), g1.neighbors(3).as_slice());
}

#[test]
fn serialize_invalid_id_size() {
    let mut file = tempfile::tempfile().unwrap();

    let g0: UndirectedCsrGraph<u32> = GraphBuilder::new()
        .edges(vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)])
        .build();

    assert!(g0.serialize(&file).is_ok());

    file.seek(SeekFrom::Start(0)).unwrap();

    let res: Result<UndirectedCsrGraph<usize>, Error> =
        UndirectedCsrGraph::<usize>::deserialize(file);

    assert!(res.is_err());

    let _expected = Error::InvalidIdType {
        expected: String::from("usize"),
        actual: String::from("u32"),
    };

    assert!(matches!(res, _expected));
}

#[test]
fn test_to_undirected() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let g: DirectedCsrGraph<u32> = GraphBuilder::new()
            .edges(vec![(0, 1), (3, 0), (0, 3), (7, 0), (0, 42), (21, 0)])
            .build();

        let ug = g.to_undirected(None);
        assert_eq!(ug.degree(0), 6);
        assert_eq!(ug.neighbors(0).as_slice(), &[1, 3, 42, 3, 7, 21]);

        let ug = g.to_undirected(CsrLayout::Unsorted);
        assert_eq!(ug.degree(0), 6);
        assert_eq!(ug.neighbors(0).as_slice(), &[1, 3, 42, 3, 7, 21]);

        let ug = g.to_undirected(CsrLayout::Sorted);
        assert_eq!(ug.degree(0), 6);
        assert_eq!(ug.neighbors(0).as_slice(), &[1, 3, 3, 7, 21, 42]);

        let ug = g.to_undirected(CsrLayout::Deduplicated);
        assert_eq!(ug.degree(0), 5);
        assert_eq!(ug.neighbors(0).as_slice(), &[1, 3, 7, 21, 42]);
    });
}

#[test]
fn directed_from_node_values_exceeding_edge_list_max_id() {
    let g0: DirectedCsrGraph<u32, u32> = GraphBuilder::new()
        .edges(vec![(0, 1), (1, 2)])
        .node_values(vec![0, 1, 2, 3])
        .build();

    assert_eq!(g0.node_count(), 4);
    for node in 0..4 {
        assert_eq!(g0.node_value(node), &node);
    }

    assert_eq!(g0.out_degree(0), 1);
    assert_eq!(g0.out_degree(1), 1);
    assert_eq!(g0.out_degree(2), 0);
    assert_eq!(g0.out_degree(3), 0);
}

#[test]
fn undirected_from_node_values_exceeding_edge_list_max_id() {
    let g0: UndirectedCsrGraph<u32, u32> = GraphBuilder::new()
        .edges(vec![(0, 1), (1, 2)])
        .node_values(vec![0, 1, 2, 3])
        .build();

    assert_eq!(g0.node_count(), 4);
    for node in 0..4 {
        assert_eq!(g0.node_value(node), &node);
    }

    assert_eq!(g0.degree(0), 1);
    assert_eq!(g0.degree(1), 2);
    assert_eq!(g0.degree(2), 1);
    assert_eq!(g0.degree(3), 0);
}
