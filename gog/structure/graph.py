from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


class Graph:

    def __init__(self, data, node_feature_names):
        self.node_features: np.ndarray = self._set_node_features(data, node_feature_names)
        self.node_feature_names: List = node_feature_names

    @staticmethod
    def _set_node_features(data: pd.DataFrame | np.ndarray, node_feature_names: List[str]) -> np.ndarray:
        if type(data) == np.ndarray:
            return data
        return data[node_feature_names].to_numpy()

    def __copy__(self):
        copy = type(self)(self.node_features, self.node_feature_names)
        self.node_features = self.node_features.copy()
        self.node_feature_names = self.node_feature_names.copy()
        return copy


class StaticGraphDataset:

    def __init__(self, edge_list: List, graphs: List[Graph]):
        self.test = None
        self.val = None
        self.train = None
        self.scaler = None
        self.adjacency_matrix = self._edge_list_to_adj(edge_list)
        self.graphs = graphs
        self._validate_node_features()

    @staticmethod
    def pandas_to_graphs(df: pd.DataFrame, num_nodes: int, features: List[str]):
        graph_list = []
        for i in tqdm(range(0, df.shape[0], num_nodes)):
            df_tmp = df.iloc[i:i + num_nodes].copy()
            graph_list.append(Graph(df_tmp, features))
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

    def graphs_to_pandas(self, graphs: List[Graph]):
        dataframes = [pd.DataFrame(graph.node_features, columns=self.graph_feature_names) for graph in graphs]
        return pd.concat(dataframes, ignore_index=True)

    def graphs_to_numpy(self, graphs: List[Graph]):
        return np.array([graph.node_features for graph in graphs])

    def set_train_split(self, train: List[Graph]):
        self.train = train

    def set_validation_split(self, val: List[Graph]):
        self.val = val

    def set_test_split(self, test: List[Graph]):
        self.test = test

    def set_splits(self, train, test, val=None):
        self.set_train_split(train)
        self.set_validation_split(val)
        self.set_test_split(test)

    def get_splits(self):
        return self.train, self.val, self.test
