import logging

import numpy as np

from graphs_on_grids.structure import Graph, GraphList, StaticGraphDataset


def create_graph_dataset(
    num_graphs,
    num_features,
    num_nodes,
    create_edge_features=True,
    num_features_edge=None,
) -> StaticGraphDataset:
    if create_edge_features:
        if num_features_edge is None:
            num_features_edge = num_features
        edge_feature_names = [str(num) for num in range(num_features_edge)]

    feature_names = [str(num) for num in range(num_features)]
    edge_list = []
    edge_generation_count = 0
    while not edge_list:
        edge_list = _create_edge_list(num_nodes)
        edge_generation_count += 1
        if edge_generation_count > 10:
            logging.warning("Edge generation taking longer than expected")
    graphs = (
        GraphList(
            node_feature_names=feature_names,
            num_nodes=num_nodes,
            edge_feature_names=edge_feature_names,
            num_edges=len(edge_list),
        )
        if create_edge_features
        else GraphList(node_feature_names=feature_names, num_nodes=num_nodes)
    )

    for i in range(num_graphs):
        node_features = np.random.randint(
            low=1, high=50, size=(num_nodes, num_features)
        )
        if create_edge_features:
            edge_features = np.random.randint(
                low=1, high=50, size=(len(edge_list), num_features_edge)
            )
            graphs.append(
                Graph(
                    ID=i,
                    node_features=node_features,
                    node_feature_names=feature_names,
                    edge_features=edge_features,
                    edge_feature_names=edge_feature_names,
                    n_edges=len(edge_list),
                )
            )
        else:
            graphs.append(
                Graph(
                    ID=i,
                    node_features=node_features,
                    node_feature_names=feature_names,
                    n_edges=len(edge_list),
                )
            )
    dataset = StaticGraphDataset(edge_list=edge_list, graphs=graphs)
    return dataset


def create_test_graph(num_features, num_nodes, num_edges=0, num_edge_features=0):
    feature_names = [str(num) for num in range(num_features)]
    node_features = np.random.randint(low=1, high=50, size=(num_nodes, num_features))
    graph_id = np.random.randint(-50, 50)
    if num_edge_features > 0 and num_edges > 0:
        edge_features = np.random.randint(
            low=0, high=50, size=(num_edges, num_edge_features)
        )
        return Graph(
            graph_id,
            node_features,
            feature_names,
            edge_features,
            feature_names,
            n_edges=num_edges,
        )
    return Graph(graph_id, node_features, feature_names, n_edges=num_edges)


def create_test_graph_sequence(
    num_graphs, num_features, num_nodes, num_edges=0, num_edge_features=0
):
    graphs = [
        create_test_graph(num_features, num_nodes, num_edges, num_edge_features)
        for _ in range(num_graphs)
    ]
    graph_list = GraphList(
        num_nodes=num_nodes,
        node_feature_names=graphs[-1].node_feature_names,
        num_edges=num_edges,
        edge_feature_names=graphs[-1].edge_feature_names,
        strict_checks=False,
    )
    graph_list.extend(graphs)
    return graph_list


def _create_edge_list(num_nodes):
    edges = []
    edges_per_node = 2
    for i in range(num_nodes):
        edge_targets = np.random.randint(low=0, high=num_nodes - 1, size=edges_per_node)
        contains_back_dir = False
        reset_counter = 0
        if i > 0:
            contains_back_dir = contains_back_direction(
                i, edge_targets, edges, edges_per_node
            )
        while (
            i in edge_targets
            or len(set(edge_targets)) != len(edge_targets)
            or contains_back_dir
            and reset_counter < 10
        ):
            edge_targets = np.random.randint(
                low=0, high=num_nodes - 1, size=edges_per_node
            )
            if i > 0:
                contains_back_dir = contains_back_direction(
                    i, edge_targets, edges, edges_per_node
                )
            reset_counter += 1
        if reset_counter >= 10:
            return []
        for edge in edge_targets:
            edges.append([i, edge])
    edges = edges + np.fliplr(edges).tolist()
    return edges


def contains_back_direction(current_idx, targets, edges, edges_per_node):
    for target in targets:
        edges_tmp = edges[target * edges_per_node : target * edges_per_node + 3]
        if edges_tmp:
            edges_tmp = np.array(edges_tmp)[:, 1]
            if current_idx in edges_tmp:
                return True
    return False
