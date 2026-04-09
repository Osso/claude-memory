use super::*;
use crate::prelude::EdgeList;
use crate::GraphBuilder;
use tap::prelude::*;

#[test]
fn empty_list() {
    let list = AdjacencyList::<u32, u32>::new(vec![]);
    assert_eq!(list.node_count(), 0);
    assert_eq!(list.edge_count(), 0);
}

#[test]
fn degree() {
    let list = AdjacencyList::<u32, u32>::new(vec![
        /* node 0 */ vec![Target::new(1, 42)],
        /* node 1 */ vec![Target::new(0, 1337)],
    ]);
    assert_eq!(list.node_count(), 2);
    assert_eq!(list.edge_count(), 2);
    assert_eq!(list.degree(0), 1);
    assert_eq!(list.degree(1), 1);
}

#[test]
fn targets_with_values() {
    let list = AdjacencyList::<u32, u32>::new(vec![
        /* node 0 */ vec![Target::new(1, 42)],
        /* node 1 */ vec![Target::new(0, 1337)],
    ]);

    assert_eq!(
        list.targets_with_values(0).as_slice(),
        &[Target::new(1, 42)]
    );
    assert_eq!(
        list.targets_with_values(1).as_slice(),
        &[Target::new(0, 1337)]
    );
}

#[test]
fn targets() {
    let list = AdjacencyList::<u32, ()>::new(vec![
        /* node 0 */ vec![Target::new(1, ())],
        /* node 1 */ vec![Target::new(0, ())],
    ]);

    assert_eq!(list.targets(0).as_slice(), &[1]);
    assert_eq!(list.targets(1).as_slice(), &[0]);
}

#[test]
fn from_edges_outgoing() {
    let edges = vec![(0, 1, 42), (0, 2, 1337), (1, 0, 43), (2, 0, 1338)];
    let edges = EdgeList::new(edges);
    let list =
        AdjacencyList::<u32, u32>::from((&edges, 3, Direction::Outgoing, CsrLayout::Unsorted));

    assert_eq!(
        list.targets_with_values(0)
            .into_iter()
            .collect::<Vec<_>>()
            .tap_mut(|v| v.sort_by_key(|t| t.target)),
        &[&Target::new(1, 42), &Target::new(2, 1337)]
    );
    assert_eq!(list.targets_with_values(1).as_slice(), &[Target::new(0, 84)]);
    assert_eq!(
        list.targets_with_values(2).as_slice(),
        &[Target::new(0, 1337)]
    );
}

#[test]
fn from_edges_incoming() {
    let edges = vec![(0, 1, 42), (0, 2, 1337), (1, 0, 43), (2, 0, 1338)];
    let edges = EdgeList::new(edges);
    let list =
        AdjacencyList::<u32, u32>::from((&edges, 3, Direction::Incoming, CsrLayout::Unsorted));

    assert_eq!(
        list.targets_with_values(0)
            .into_iter()
            .collect::<Vec<_>>()
            .tap_mut(|v| v.sort_by_key(|t| t.target)),
        &[&Target::new(1, 42), &Target::new(2, 1337)]
    );
    assert_eq!(list.targets_with_values(1).as_slice(), &[Target::new(0, 42)]);
    assert_eq!(
        list.targets_with_values(2).as_slice(),
        &[Target::new(0, 1337)]
    );
}

#[test]
fn from_edges_undirected() {
    let edges = vec![(0, 1, 42), (0, 2, 1337), (1, 0, 43), (2, 0, 1338)];
    let edges = EdgeList::new(edges);
    let list = AdjacencyList::<u32, u32>::from((&edges, 3, Direction::Undirected, CsrLayout::Unsorted));

    assert_eq!(
        list.targets_with_values(0)
            .into_iter()
            .collect::<Vec<_>>()
            .tap_mut(|v| v.sort_by_key(|t| t.target)),
        &[
            &Target::new(1, 42),
            &Target::new(1, 43),
            &Target::new(2, 1337),
            &Target::new(2, 1338)
        ]
    );
    assert_eq!(
        list.targets_with_values(1)
            .into_iter()
            .collect::<Vec<_>>()
            .tap_mut(|v| v.sort_by_key(|t| t.target)),
        &[&Target::new(0, 42), &Target::new(0, 43)]
    );
    assert_eq!(
        list.targets_with_values(2)
            .into_iter()
            .collect::<Vec<_>>()
            .tap_mut(|v| v.sort_by_key(|t| t.target)),
        &[&Target::new(0, 1337), &Target::new(0, 1338)]
    );
}

#[test]
fn from_edges_sorted() {
    let edges = vec![
        (0, 1, ()),
        (0, 3, ()),
        (0, 2, ()),
        (1, 3, ()),
        (1, 2, ()),
        (1, 0, ()),
    ];
    let edges = EdgeList::new(edges);
    let list = AdjacencyList::<u32, ()>::from((&edges, 3, Direction::Outgoing, CsrLayout::Sorted));

    assert_eq!(list.targets(0).as_slice(), &[1, 2, 3]);
    assert_eq!(list.targets(1).as_slice(), &[0, 2, 3]);
}

#[test]
fn from_edges_deduplicated() {
    let edges = vec![
        (0, 1, ()),
        (0, 3, ()),
        (0, 3, ()),
        (0, 2, ()),
        (0, 2, ()),
        (1, 3, ()),
        (1, 3, ()),
        (1, 2, ()),
        (1, 2, ()),
        (1, 0, ()),
        (1, 0, ()),
    ];
    let edges = EdgeList::new(edges);
    let list =
        AdjacencyList::<u32, ()>::from((&edges, 3, Direction::Outgoing, CsrLayout::Deduplicated));

    assert_eq!(list.targets(0).as_slice(), &[1, 2, 3]);
    assert_eq!(list.targets(1).as_slice(), &[0, 2, 3]);
}

#[test]
fn directed_al_graph() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 1), (0, 2), (1, 2)])
        .build::<DirectedALGraph<u32, ()>>();

    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 3);
    assert_eq!(g.out_degree(0), 2);
    assert_eq!(g.out_neighbors(0).as_slice(), &[1, 2]);
    assert_eq!(g.in_degree(2), 2);
    assert_eq!(g.in_neighbors(2).as_slice(), &[0, 1]);
}

#[test]
fn directed_al_graph_with_node_values() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 1), (0, 2), (1, 2)])
        .node_values(vec!["foo", "bar", "baz"])
        .build::<DirectedALGraph<u32, &str>>();

    assert_eq!(g.node_value(0), &"foo");
    assert_eq!(g.node_value(1), &"bar");
    assert_eq!(g.node_value(2), &"baz");
}

#[test]
fn directed_al_graph_add_edge_unsorted() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges([(0, 2), (1, 2)])
        .build::<DirectedALGraph<u32>>();

    assert_eq!(g.out_neighbors(0).as_slice(), &[2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[2, 1]);
    g.add_edge(0, 2).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[2, 1, 2]);
}

#[test]
fn directed_al_graph_add_edge_sorted() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 2), (1, 2)])
        .build::<DirectedALGraph<u32>>();

    assert_eq!(g.out_neighbors(0).as_slice(), &[2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[1, 2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[1, 1, 2]);
}

#[test]
fn directed_al_graph_add_edge_deduplicated() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Deduplicated)
        .edges([(0, 2), (1, 2), (1, 3)])
        .build::<DirectedALGraph<u32>>();

    assert_eq!(g.out_neighbors(0).as_slice(), &[2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[1, 2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[1, 2]);
    g.add_edge(0, 3).expect("add edge failed");
    assert_eq!(g.out_neighbors(0).as_slice(), &[1, 2, 3]);
}

#[test]
fn directed_al_graph_add_edge_with_value() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges_with_values([(0, 2, 4.2), (1, 2, 13.37)])
        .build::<DirectedALGraph<u32, (), f32>>();

    assert_eq!(
        g.out_neighbors_with_values(0).as_slice(),
        &[Target::new(2, 4.2)]
    );
    g.add_edge_with_value(0, 1, 19.84).expect("add edge failed");
    assert_eq!(
        g.out_neighbors_with_values(0).as_slice(),
        &[Target::new(2, 4.2), Target::new(1, 19.84)]
    );
    g.add_edge_with_value(0, 2, 1.23).expect("add edge failed");
    assert_eq!(
        g.out_neighbors_with_values(0).as_slice(),
        &[
            Target::new(2, 4.2),
            Target::new(1, 19.84),
            Target::new(2, 1.23)
        ]
    );
}

#[test]
fn directed_al_graph_add_edge_missing_node() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges([(0, 2), (1, 2)])
        .build::<DirectedALGraph<u32>>();

    let err = g.add_edge(0, 3).unwrap_err();
    assert!(matches!(err, crate::Error::MissingNode { node } if node == "3" ));
}

#[test]
fn directed_al_graph_add_edge_parallel() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges([(0, 1), (0, 2), (0, 3)])
        .build::<DirectedALGraph<u32>>();

    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| g.add_edge(0, 1));
        }
    });

    assert_eq!(g.edge_count(), 7);
}

#[test]
fn undirected_al_graph_add_edge_unsorted() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges([(0, 2), (1, 2)])
        .build::<UndirectedALGraph<u32>>();

    assert_eq!(g.neighbors(0).as_slice(), &[2]);
    assert_eq!(g.neighbors(1).as_slice(), &[2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[2, 1]);
    assert_eq!(g.neighbors(1).as_slice(), &[2, 0]);
    g.add_edge(0, 2).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[2, 1, 2]);
}

#[test]
fn undirected_al_graph_add_edge_sorted() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 2), (1, 2)])
        .build::<UndirectedALGraph<u32>>();

    assert_eq!(g.neighbors(0).as_slice(), &[2]);
    assert_eq!(g.neighbors(1).as_slice(), &[2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[1, 2]);
    assert_eq!(g.neighbors(1).as_slice(), &[0, 2]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[1, 1, 2]);
}

#[test]
fn undirected_al_graph_add_edge_deduplicated() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Deduplicated)
        .edges([(0, 2), (1, 2), (1, 3)])
        .build::<UndirectedALGraph<u32>>();

    assert_eq!(g.neighbors(0).as_slice(), &[2]);
    assert_eq!(g.neighbors(1).as_slice(), &[2, 3]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[1, 2]);
    assert_eq!(g.neighbors(1).as_slice(), &[0, 2, 3]);
    g.add_edge(0, 1).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[1, 2]);
    assert_eq!(g.neighbors(1).as_slice(), &[0, 2, 3]);
    g.add_edge(0, 3).expect("add edge failed");
    assert_eq!(g.neighbors(0).as_slice(), &[1, 2, 3]);
}

#[test]
fn undirected_al_graph_add_edge_with_value() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges_with_values([(0, 2, 4.2), (1, 2, 13.37)])
        .build::<UndirectedALGraph<u32, (), f32>>();

    assert_eq!(
        g.neighbors_with_values(0).as_slice(),
        &[Target::new(2, 4.2)]
    );
    assert_eq!(
        g.neighbors_with_values(1).as_slice(),
        &[Target::new(2, 13.37)]
    );
    g.add_edge_with_value(0, 1, 19.84).expect("add edge failed");
    assert_eq!(
        g.neighbors_with_values(0).as_slice(),
        &[Target::new(2, 4.2), Target::new(1, 19.84)]
    );
    assert_eq!(
        g.neighbors_with_values(1).as_slice(),
        &[Target::new(2, 13.37), Target::new(0, 19.84)]
    );
    g.add_edge_with_value(0, 2, 1.23).expect("add edge failed");
    assert_eq!(
        g.neighbors_with_values(0).as_slice(),
        &[
            Target::new(2, 4.2),
            Target::new(1, 19.84),
            Target::new(2, 1.23)
        ]
    );
}

#[test]
fn undirected_al_graph_add_edge_missing_node() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges([(0, 2), (1, 2)])
        .build::<UndirectedALGraph<u32>>();

    let err = g.add_edge(0, 3).unwrap_err();
    assert!(matches!(err, crate::Error::MissingNode { node } if node == "3" ));
}

#[test]
fn undirected_al_graph_add_edge_parallel() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Unsorted)
        .edges([(0, 1), (0, 2), (0, 3)])
        .build::<UndirectedALGraph<u32>>();

    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| g.add_edge(0, 1));
        }
    });

    assert_eq!(g.edge_count(), 7);
}

#[test]
fn undirected_al_graph() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 1), (0, 2), (1, 2)])
        .build::<UndirectedALGraph<u32, ()>>();

    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 3);
    assert_eq!(g.degree(0), 2);
    assert_eq!(g.degree(2), 2);
    assert_eq!(g.neighbors(0).as_slice(), &[1, 2]);
    assert_eq!(g.neighbors(2).as_slice(), &[0, 1]);
}

#[test]
fn undirected_al_graph_cycle() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 1), (1, 0)])
        .build::<UndirectedALGraph<u32, ()>>();

    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 2);
    assert_eq!(g.degree(0), 2);
    assert_eq!(g.degree(1), 2);
    assert_eq!(g.neighbors(0).as_slice(), &[1, 1]);
    assert_eq!(g.neighbors(1).as_slice(), &[0, 0]);
}

#[test]
fn undirected_al_graph_with_node_values() {
    let g = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .edges([(0, 1), (0, 2), (1, 2)])
        .node_values(vec!["foo", "bar", "baz"])
        .build::<UndirectedALGraph<u32, &str>>();

    assert_eq!(g.node_value(0), &"foo");
    assert_eq!(g.node_value(1), &"bar");
    assert_eq!(g.node_value(2), &"baz");
}
