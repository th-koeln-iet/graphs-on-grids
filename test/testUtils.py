import numpy as np

from gog.structure.graph import GraphList, Graph, StaticGraphDataset


def create_graph_dataset(num_graphs, num_features, num_nodes) -> StaticGraphDataset:
    feature_names = [str(num) for num in range(num_features)]
    graphs = GraphList(features=feature_names, num_nodes=num_nodes)

    for i in range(num_graphs):
        data = np.random.randint(low=1, high=50, size=(num_nodes, num_features))
        graphs.append(Graph(data, feature_names))
    edge_list = np.unique(np.random.randint(low=0, high=num_nodes - 1, size=(int(num_nodes * 3), 2)), axis=0).tolist()
    dataset = StaticGraphDataset(edge_list=edge_list, graphs=graphs)
    return dataset


def create_test_graph(num_features, num_nodes):
    feature_names = [str(num) for num in range(num_features)]
    data = np.random.randint(low=0, high=50, size=(num_nodes, num_features))
    return Graph(data, feature_names)
