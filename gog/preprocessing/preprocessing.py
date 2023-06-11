import logging
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from gog.structure.graph import StaticGraphDataset, GraphList, Graph


def create_train_test_split(
    dataset: StaticGraphDataset, train_size=0.8, random_state=None, shuffle=True
):
    if not isinstance(dataset, StaticGraphDataset):
        raise ValueError(
            f"Expected input to be of type {StaticGraphDataset.type()}. Received type {type(dataset)}"
        )

    graph_list = dataset.graphs
    num_graphs = len(graph_list)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(graph_list)
    last_train_index = int(num_graphs * train_size)
    train, test = (
        graph_list[0:last_train_index],
        graph_list[last_train_index : num_graphs + 1],
    )
    dataset.set_splits(train=train, test=test)
    return train, test


def create_windowed_train_test_split(
    dataset: StaticGraphDataset,
    window_size,
    len_labels=1,
    step=1,
    start=0,
    train_size=0.8,
    random_state=None,
    shuffle=False,
):
    if not isinstance(dataset, StaticGraphDataset):
        raise ValueError(
            f"Expected input to be of type {StaticGraphDataset.type()}. Received type {type(dataset)}"
        )
    graphs = dataset.graphs
    windows = GraphList(
        num_nodes=graphs.num_nodes,
        node_feature_names=graphs.node_feature_names,
        num_edges=graphs.num_edges,
        edge_feature_names=graphs.edge_feature_names,
        strict_checks=False,
    )
    labels = GraphList(
        num_nodes=graphs.num_nodes,
        node_feature_names=graphs.node_feature_names,
        num_edges=graphs.num_edges,
        edge_feature_names=graphs.edge_feature_names,
        strict_checks=False,
    )
    num_graphs = len(graphs)
    while start + window_size + len_labels < num_graphs:
        end = start + window_size
        current_window = graphs[start:end]
        label_window = graphs[end + 1 : end + 1 + len_labels]
        if len(current_window) == window_size:
            labels.append(label_window)
            windows.append(current_window)
        start = start + step
    if num_graphs != len(windows) * window_size:
        logging.warning(
            f"Dataset of size {num_graphs}, cannot be cleanly divided with window size {window_size}. Discarded {num_graphs - len(windows) * window_size} graph instances."
        )
    dataset.graphs = windows
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(windows)
        np.random.shuffle(labels)

    num_instances = len(windows)
    last_train_index = int(num_instances * train_size)
    X_train, X_test, y_train, y_test = (
        windows[0:last_train_index],
        windows[last_train_index : num_instances + 1],
        labels[0:last_train_index],
        labels[last_train_index : num_instances + 1],
    )
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def apply_scaler(
    dataset: StaticGraphDataset, method="zero_mean", target="node"
) -> Tuple[GraphList, GraphList]:
    if not isinstance(dataset, StaticGraphDataset):
        raise ValueError(
            f"Expected input to be of type {StaticGraphDataset.type()}. Received type {type(dataset)}"
        )

    if not dataset.train or not dataset.test:
        raise ValueError(
            f"The dataset has not yet been split into a training or test set. Did you already call 'create_train_test_split' on this dataset?"
        )

    train, _, test = dataset.get_splits()
    train, test = train.to_pandas(), test.to_pandas()

    if isinstance(train, list) and isinstance(test, list):
        if target == "node":
            train, test = train[0], test[0]
        elif target == "edge":
            train, test = train[1], test[1]
        else:
            raise ValueError(
                f"Expected target to be either 'node' or 'edge'. Received {target}"
            )
    else:
        if target == "edge":
            raise ValueError(
                f"Expected dataset to contain edge features with for target 'edge'"
            )

    scaler = None
    if method == "zero_mean":
        scaler = StandardScaler()
    elif method == "min_max":
        scaler = MinMaxScaler()
    if scaler is None:
        raise ValueError(
            f"Invalid method={method} provided. Only 'zero_mean' or 'min_max' are valid'"
        )

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    if target == "node":
        num_nodes = dataset.adjacency_matrix.shape[0]
        dataset.node_scaler = scaler
        _replace_node_features(num_nodes, dataset.train, train)
        _replace_node_features(num_nodes, dataset.test, test)
    elif target == "edge":
        num_edges = np.count_nonzero(dataset.adjacency_matrix == 1)
        dataset.edge_scaler = scaler
        _replace_edge_features(num_edges, dataset.train, train)
        _replace_edge_features(num_edges, dataset.test, test)

    return dataset.train, dataset.test


def _replace_node_features(num_nodes, graph_list, scaled_split: GraphList):
    index = 0
    for graph in graph_list:
        graph.node_features = scaled_split[index : index + num_nodes]
        index += num_nodes


def _replace_edge_features(num_edges, graph_list, scaled_split: GraphList):
    index = 0
    for graph in graph_list:
        graph.edge_features = scaled_split[index : index + num_edges]
        index += num_edges


def _get_feature_indices(
    X_train: GraphList, X_test: GraphList, targets: List[str], num_nodes: int
):
    if len(X_train) != 0:
        reference_graph = X_train[-1]
    elif len(X_test) != 0:
        reference_graph = X_test[-1]
    else:
        return []
    if isinstance(reference_graph, GraphList):
        reference_graph = reference_graph[-1]
    feature_indices = [
        reference_graph.node_feature_names.index(feature_name)
        for feature_name in targets
    ]
    feature_indices = np.reshape(
        np.repeat(feature_indices, num_nodes), newshape=(len(targets), num_nodes)
    )
    return feature_indices


def mask_labels(
    X_train: GraphList,
    X_test: GraphList,
    targets: List[str],
    nodes: List | np.ndarray,
    method: str = "zeros",
):
    if not isinstance(X_train, GraphList) or not isinstance(X_test, GraphList):
        raise ValueError(
            f"Expected both inputs for X_train and X_test to be of type {type(GraphList)}. Received types {type(X_train), type(X_test)}"
        )

    feature_indices = _get_feature_indices(X_train, X_test, targets, len(nodes))

    X_train_mask = GraphList(
        [
            _mask_split(graph.__copy__(), targets, nodes, method, feature_indices)
            for graph in X_train
        ],
        X_train.num_nodes,
        X_train.node_feature_names,
        num_edges=X_train.num_edges,
        edge_feature_names=X_train.edge_feature_names,
        strict_checks=X_train.strict_checks,
    )
    X_test_mask = GraphList(
        [
            _mask_split(graph.__copy__(), targets, nodes, method, feature_indices)
            for graph in X_test
        ],
        num_nodes=X_test.num_nodes,
        node_feature_names=X_test.node_feature_names,
        num_edges=X_test.num_edges,
        edge_feature_names=X_test.edge_feature_names,
        strict_checks=X_test.strict_checks,
    )

    return X_train_mask, X_test_mask


def _mask_split(
    graph: Graph | GraphList,
    targets: List[str],
    nodes: List,
    method: str,
    feature_indices: List,
):
    if isinstance(graph, GraphList):
        return _mask_split_dynamic(graph, targets, nodes, method, feature_indices)
    feature_matrix = graph.node_features

    if method == "zeros":
        feature_matrix[nodes, feature_indices] = 0

    return graph


def _mask_split_dynamic(
    graph_sequence: GraphList,
    targets: List[str],
    nodes: List,
    method: str,
    feature_indices: List,
):
    lst = GraphList(
        data=[
            _mask_split(
                graph.__copy__(), targets, nodes, method, feature_indices
            ).__copy__()
            for graph in graph_sequence
        ],
        num_nodes=graph_sequence.num_nodes,
        node_feature_names=graph_sequence.node_feature_names,
        num_edges=graph_sequence.num_edges,
        edge_feature_names=graph_sequence.edge_feature_names,
        strict_checks=graph_sequence.strict_checks,
    )
    return lst


def create_validation_set(
    X: GraphList, y: GraphList, validation_size=0.2
) -> Tuple[GraphList, GraphList, GraphList, GraphList]:
    if not isinstance(X, GraphList) or not isinstance(y, GraphList):
        raise ValueError(
            f"Expected both inputs for X and y to be of type {type(GraphList())}. Received types {type(X), type(y)}"
        )
    if len(X) != len(y):
        raise ValueError(
            f"Expected same number of instances in X and y. Received {len(X), len(y)}"
        )
    if isinstance(X[-1], GraphList):
        return _create_time_series_validation_set(X, y, validation_size)

    split_idx = int(len(X) * validation_size)
    X_train, X_val = X[split_idx:], X[:split_idx]
    split_idx = int(len(y) * validation_size)
    y_train, y_val = y[split_idx:], y[:split_idx]
    return X_train, X_val, y_train, y_val


def _create_time_series_validation_set(X: GraphList, y: GraphList, validation_size=0.2):
    num_instances = len(X)
    last_train_index = int(num_instances * (1 - validation_size))
    X_train, X_val, y_train, y_val = (
        X[0:last_train_index],
        X[last_train_index : num_instances + 1],
        y[0:last_train_index],
        y[last_train_index : num_instances + 1],
    )
    return X_train, X_val, y_train, y_val
