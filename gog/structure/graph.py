from typing import List

import numpy as np
import pandas as pd


def _edge_list_to_adj(edge_list) -> np.ndarray:
    size = len(set([n for e in edge_list for n in e]))
    adj = np.zeros((size, size))
    for sink, source in edge_list:
        adj[sink - 1][source - 1] = 1
    return adj


class Graph:

    def __init__(self, data, node_feature_names):
        self.node_features = self._set_node_features(data, node_feature_names)
        self.node_feature_names: List = node_feature_names

    @staticmethod
    def _set_node_features(data: pd.DataFrame | np.ndarray, node_feature_names: List[str]):
        if type(data) == np.ndarray:
            return data
        data.sort_values(by="Node")
        return data[node_feature_names].to_numpy()

    def __copy__(self):
        copy = type(self)(self.node_features, self.node_feature_names)
        self.node_features = self.node_features.copy()
        return copy


class StaticGraphs:

    def __init__(self, edge_list: List, graphs: List[Graph]):
        self.adjacency_matrix = _edge_list_to_adj(edge_list)
        self.graphs = graphs
        self._validate_node_features()

    def _validate_node_features(self):
        if self.graphs:
            first_graph = self.graphs[0]
            features = first_graph.node_feature_names
            feature_shape = first_graph.node_features.shape
            for graph in self.graphs:
                if features != graph.node_feature_names or feature_shape != graph.node_features.shape:
                    raise KeyError("Invalid list of graphs given. Different node features are present")
