from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from gog.structure.graph import StaticGraphDataset, Graph


def create_train_test_split(dataset: StaticGraphDataset, train_size=0.8, random_state=None, shuffle=True):
    graph_list = dataset.graphs
    num_graphs = len(graph_list)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(graph_list)
    last_train_index = int(num_graphs * train_size)
    train, test = graph_list[0:last_train_index], graph_list[last_train_index:num_graphs + 1]
    dataset.set_splits(train=train, test=test)
    return train, test


def apply_scaler(dataset: StaticGraphDataset, method="zero_mean"):
    train, _, test = dataset.get_splits()
    train, test = dataset.graphs_to_df(train), dataset.graphs_to_df(test)
    scaler = None
    if method == "zero_mean":
        scaler = StandardScaler()
    elif method == "min_max":
        scaler = MinMaxScaler()
    if scaler is None:
        raise ValueError(f"Invalid method={method} provided. Only 'zero_mean' or 'min_max' are valid'")

    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    dataset.scaler = scaler
    num_nodes = dataset.adjacency_matrix.shape[0]
    _replace_node_features(num_nodes, dataset.train, train)
    _replace_node_features(num_nodes, dataset.test, test)
    return dataset.train, dataset.test


def _replace_node_features(num_nodes, graph_list, scaled_split: List[Graph]):
    index = 0
    num_nodes = num_nodes
    for graph in graph_list:
        graph.node_features = scaled_split[index:index + num_nodes]
        index += num_nodes


def mask_labels(X_train: List[Graph], X_test: List[Graph], targets: List[str], nodes: List | np.ndarray,
                method: str = "zeros"):
    X_train_mask, X_test_mask = [graph.__copy__() for graph in X_train].copy(), [graph.__copy__() for graph in
                                                                                 X_test].copy()
    return _mask_split(X_train_mask, targets, nodes, method), _mask_split(X_test_mask, targets, nodes, method)


def _mask_split(split: List[Graph], targets: List[str], nodes: List, method: str):
    for instance in split:
        feature_matrix = instance.node_features
        feature_indices = [instance.node_feature_names.index(feature_name) for feature_name in targets]
        for i, row in enumerate(feature_matrix):
            if i in nodes:
                if method == "zeros":
                    row[feature_indices] = 0
    return split
