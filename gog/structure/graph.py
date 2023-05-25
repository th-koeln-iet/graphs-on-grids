from collections import UserList
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


class Graph:

    def __init__(self, node_features, node_feature_names, edge_features=None, edge_feature_names=None):
        self.edge_features = None
        self.edge_feature_names: List = edge_feature_names
        self.node_features: np.ndarray = self._set_node_features(node_features, node_feature_names)
        self.node_feature_names: List = node_feature_names
        self.set_edge_features(edge_features, edge_feature_names) if edge_features is not None else None

    @staticmethod
    def _set_node_features(node_features: pd.DataFrame | np.ndarray, node_feature_names: List[str]) -> np.ndarray:
        if type(node_features) == np.ndarray:
            return node_features
        return node_features[node_feature_names].to_numpy()

    def set_edge_features(self, edge_features: pd.DataFrame | np.ndarray, edge_feature_names: List[str]):
        self.edge_feature_names = edge_feature_names
        if type(edge_features) == np.ndarray:
            self.edge_features = edge_features
            return
        self.edge_features = edge_features[edge_feature_names].to_numpy()

    def __copy__(self):
        copy = type(self)(self.node_features, self.node_feature_names, self.edge_features, self.edge_feature_names)
        self.node_features = self.node_features.copy()
        self.node_feature_names = self.node_feature_names.copy()
        self.edge_features = self.edge_features.copy() if self.edge_features is not None else None
        self.node_feature_names = self.edge_feature_names.copy() if self.edge_feature_names else None
        return copy


class GraphList(UserList[Graph]):
    def __init__(self, data=None, num_nodes: int = 0, node_feature_names: List[str] = None, num_edges: int = 0,
                 edge_feature_names: List[str] = None):
        super().__init__(data)
        self.node_feature_names: List[str] = node_feature_names
        self.num_nodes: int = num_nodes
        self.num_edges: int = num_edges
        self.edge_feature_names: List[str] = edge_feature_names

    def append(self, item: Graph) -> None:
        if set(item.node_feature_names) != set(self.node_feature_names):
            raise ValueError(f"Node features do not match other graphs. Expected features {self.node_feature_names}")
        if self.data and item.node_features.shape != self.data[-1].node_features.shape:
            raise ValueError(
                f"Different node feature dimensions provided. Expected {self.data[-1].node_features.shape} but received {item.node_features.shape}")
        if item.edge_feature_names and self.edge_feature_names and set(item.edge_feature_names) != set(
                self.edge_feature_names):
            raise ValueError(f"Edge features do not match other graphs. Expected features {self.node_feature_names}")

        # edge features exist and are of the same shape
        if self.data and item.edge_features and self.data[-1].edge_features and item.edge_features.shape != self.data[
            -1].edge_features.shape:
            raise ValueError(
                f"Different edge feature dimensions provided. Expected {self.data[-1].edge_features.shape} but received {item.edge_features.shape}")
        super().append(item)

    def __getitem__(self, item):
        res = self.data[item]
        if type(res) == list:
            return self.__class__(res, self.num_nodes, self.node_feature_names, self.num_edges, self.edge_feature_names)
        else:
            return res

    def copy(self):
        return GraphList(data=self.data.copy(), num_nodes=self.num_nodes, node_feature_names=self.node_feature_names,
                         num_edges=self.num_edges, edge_feature_names=self.edge_feature_names)

    def to_numpy(self):
        if self.edge_feature_names:
            return [np.array([graph.node_features for graph in self]),
                    np.array([graph.edge_features for graph in self])]
        return np.array([graph.node_features for graph in self])

    def to_pandas(self):
        if self.edge_feature_names:
            df_node_features = pd.DataFrame(np.vstack(self.to_numpy()[0]), columns=self.node_feature_names)
            df_edge_features = pd.DataFrame(np.vstack(self.to_numpy()[1]), columns=self.edge_feature_names)
            return pd.concat([df_node_features, df_edge_features], axis=0, ignore_index=True)
        return pd.DataFrame(np.vstack(self.to_numpy()), columns=self.node_feature_names)


class StaticGraphDataset:

    def __init__(self, edge_list, graphs: GraphList):
        self.test = None
        self.val = None
        self.train = None
        self.scaler = None
        self.adjacency_matrix = self._edge_list_to_adj(edge_list)
        self.graphs = graphs
        self._validate_node_features()

    @staticmethod
    def pandas_to_graphs(df_node_features: pd.DataFrame, num_nodes: int, node_feature_names: List[str],
                         df_edge_features: pd.DataFrame = None, num_edges: int = 0,
                         edge_feature_names=None):
        edge_feature_names = edge_feature_names if not edge_feature_names else list(df_edge_features.columns.values)
        graph_list = GraphList(num_nodes=num_nodes, node_feature_names=node_feature_names, num_edges=num_edges,
                               edge_feature_names=edge_feature_names)
        for i in tqdm(range(0, df_node_features.shape[0], num_nodes), desc="Creating graph dataset"):
            df_tmp = df_node_features.iloc[i:i + num_nodes].copy()
            graph_list.append(Graph(df_tmp, node_feature_names))
        if df_edge_features is not None:
            graph_idx = 0
            for i in tqdm(range(0, df_edge_features.shape[0], num_edges), desc="Transferring edge features"):
                df_tmp = df_edge_features.iloc[i:i + num_edges].copy()
                current_graph = graph_list[graph_idx]
                current_graph.set_edge_features(df_tmp, edge_feature_names)
                graph_idx += 1
        return graph_list

    def _validate_node_features(self):
        if self.graphs:
            first_graph = self.graphs[0]
            features = first_graph.node_feature_names
            feature_shape = first_graph.node_features.shape
            for graph in self.graphs:
                if features != graph.node_feature_names or feature_shape != graph.node_features.shape:
                    raise KeyError("Invalid list of graphs given. Different node features are present")
            self.graph_feature_names = features

    def _edge_list_to_adj(self, edge_list) -> np.ndarray:
        size = len(set([n for e in edge_list for n in e]))
        adj = np.zeros((size, size), dtype=np.float32)
        for sink, source in edge_list:
            adj[sink - 1][source - 1] = 1
        return adj

    def set_train_split(self, train: GraphList):
        self.train = train

    def set_validation_split(self, val: GraphList):
        self.val = val

    def set_test_split(self, test: GraphList):
        self.test = test

    def set_splits(self, train, test, val=None):
        self.set_train_split(train)
        self.set_validation_split(val)
        self.set_test_split(test)

    def get_splits(self):
        return self.train, self.val, self.test
