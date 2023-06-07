import numpy as np

from gog.structure.graph import GraphList, Graph, StaticGraphDataset


def create_graph_dataset(
    num_graphs, num_features, num_nodes, create_edge_features=True
) -> StaticGraphDataset:
    feature_names = [str(num) for num in range(num_features)]
    edge_list = np.unique(
        np.random.randint(low=0, high=num_nodes - 1, size=(int(num_nodes * 3), 2)),
        axis=0,
    ).tolist()
    graphs = (
        GraphList(
            node_feature_names=feature_names,
            num_nodes=num_nodes,
            edge_feature_names=feature_names,
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
                low=1, high=50, size=(len(edge_list), num_features)
            )
            graphs.append(
                Graph(
                    ID=i,
                    node_features=node_features,
                    node_feature_names=feature_names,
                    edge_features=edge_features,
                    edge_feature_names=feature_names,
                )
            )
        else:
            graphs.append(
                Graph(
                    ID=i, node_features=node_features, node_feature_names=feature_names
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
            graph_id, node_features, feature_names, edge_features, feature_names
        )
    return Graph(graph_id, node_features, feature_names)
