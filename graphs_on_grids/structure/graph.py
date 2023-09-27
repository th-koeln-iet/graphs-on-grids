from collections import UserList
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


class Graph:
    """
    Basic container class for graph data
    """

    def __init__(
        self,
        ID: int,
        node_features: np.ndarray,
        node_feature_names: list,
        edge_features: np.ndarray = None,
        edge_feature_names: list = None,
        n_edges: int = None,
    ) -> None:
        """

        :param ID: Identifier of graph
        :param node_features: node feature matrix of graph instance
        :param node_feature_names: node feature names
        :param edge_features: edge feature matrix of graph instance
        :param edge_feature_names: edge feature names
        :param n_edges: number of edges in the graph
        """
        self.ID = ID
        self.n_edges = n_edges
        self.edge_features = None
        self.edge_feature_names: List = edge_feature_names
        self.node_features: np.ndarray = self._set_node_features(
            node_features, node_feature_names
        )
        self.node_feature_names: List = node_feature_names
        self.set_edge_features(
            edge_features, edge_feature_names
        ) if edge_features is not None else None

    def _set_node_features(
        self, node_features: pd.DataFrame | np.ndarray, node_feature_names: List[str]
    ) -> np.ndarray:
        if type(node_features) == np.ndarray:
            return node_features
        return node_features[node_feature_names].to_numpy()

    def set_edge_features(
        self, edge_features: pd.DataFrame | np.ndarray, edge_feature_names: List[str]
    ):
        self.edge_feature_names = edge_feature_names
        if type(edge_features) == np.ndarray:
            self.edge_features = edge_features
            return
        self.edge_features = edge_features[edge_feature_names].to_numpy()

    @property
    def n_nodes(self):
        if self.node_features:
            return self.node_features.shape[0]

    @property
    def n_edges(self):
        if self.edge_features:
            return self.edge_features.shape[0]

    @n_edges.setter
    def n_edges(self, value):
        self._n_edges = value

    def __copy__(self):
        copy = type(self)(
            self.ID,
            self.node_features,
            self.node_feature_names,
            self.edge_features,
            self.edge_feature_names,
        )
        self.node_features = self.node_features.copy()
        self.node_feature_names = self.node_feature_names.copy()
        self.edge_features = (
            self.edge_features.copy() if self.edge_features is not None else None
        )
        self.edge_feature_names = (
            self.edge_feature_names.copy() if self.edge_feature_names else None
        )
        return copy


class GraphList(UserList):
    """
    Wrapper class for lists of `gog.structure.Graph` instances. Includes methods to convert to numpy arrays
    or pandas dataframes. GraphLists include validation for Graphs that are appended to it including checks for
    equal features and node/edge dimensions. Otherwise, behaves like regular python lists.
    """

    def __init__(
        self,
        data: Graph | list = None,
        num_nodes: int = 0,
        node_feature_names: List[str] = None,
        num_edges: int = 0,
        edge_feature_names: List[str] = None,
        strict_checks: bool = True,
    ) -> None:
        """

        :param data: GraphList data
        :param num_nodes: number of nodes in graphs
        :param node_feature_names: node feature names used in graphs
        :param num_edges: number of edges in graphs
        :param edge_feature_names: edge feature names used in graphs
        :param strict_checks: whether to perform validation on appended graphs
        """
        super().__init__(data)
        self.node_feature_names: List[str] = node_feature_names
        self.num_nodes: int = num_nodes
        self.num_edges: int = num_edges
        self.edge_feature_names: List[str] = edge_feature_names
        self.strict_checks = strict_checks

    def append(self, item: Graph) -> None:
        if self.strict_checks:
            self.validate_graph_structure(item)
        else:
            self.validate_graph_sequence_structure(item)
        super().append(item)

    def validate_graph_sequence_structure(self, item):
        if self.data and len(item) != len(self.data[-1]):
            raise ValueError(
                f"Different window size provided. Expected {len(self.data[-1])} but received {len(item)}"
            )
        for graph in item:
            self.validate_graph_structure(graph)

    def validate_graph_structure(self, item):
        compare_to = self.data[-1] if self.data else None
        if isinstance(compare_to, GraphList):
            compare_to = compare_to[-1]
        if set(item.node_feature_names) != set(self.node_feature_names):
            raise ValueError(
                f"Node features do not match other graphs. Expected features {self.node_feature_names}"
            )
        if self.data and item.node_features.shape != compare_to.node_features.shape:
            raise ValueError(
                f"Different node feature dimensions provided. Expected {compare_to.node_features.shape} but received {item.node_features.shape}"
            )
        if (
            item.edge_feature_names
            and self.edge_feature_names
            and set(item.edge_feature_names) != set(self.edge_feature_names)
        ):
            raise ValueError(
                f"Edge features do not match other graphs. Expected features {self.node_feature_names}"
            )
        # edge features exist and are of the same shape
        if (
            self.data
            and isinstance(item.edge_features, np.ndarray)
            and isinstance(compare_to.edge_features, np.ndarray)
            and item.edge_features.shape != compare_to.edge_features.shape
        ):
            raise ValueError(
                f"Different edge feature dimensions provided. Expected {compare_to.edge_features.shape} but received {item.edge_features.shape}"
            )

    def __getitem__(self, item):
        res = self.data[item]
        if type(res) == list:
            return self.__class__(
                res,
                num_nodes=self.num_nodes,
                node_feature_names=self.node_feature_names,
                num_edges=self.num_edges,
                edge_feature_names=self.edge_feature_names,
                strict_checks=self.strict_checks,
            )
        else:
            return res

    def type(self):
        return self.__class__.__name__

    def copy(self):
        return GraphList(
            data=self.data.copy(),
            num_nodes=self.num_nodes,
            node_feature_names=self.node_feature_names,
            num_edges=self.num_edges,
            edge_feature_names=self.edge_feature_names,
            strict_checks=self.strict_checks,
        )

    def to_numpy(self) -> List[np.ndarray] | np.ndarray:
        """
        Converts `GraphList` instance to

            - ndarray of shape (n_graphs, n_nodes, n_node_features) if graphs only include node features and are
              not windowed
            - list of ndarrays of shapes (n_graphs, n_nodes, n_node_features) and (n_graphs, n_edges, n_edge_features).
              Index 0 of the list contains the node features, index 1 contains the edge features
            - nparray of shape (n_graphs, window_size, n_nodes, n_node_features) if graphs only include node features
              and graphs are windowed
            - list of ndarrays of shapes (n_graphs, window_size, n_nodes, n_node_features) and
              (n_graphs, window_size, n_edges, n_edge_features). Index 0 of the list contains the node features,
              index 1 contains the edge features
        :return: either ndarray of the node features or list of ndarrays containing node and edge features
        """
        if not self.strict_checks:
            return self._to_numpy_dynamic()
        if self.edge_feature_names:
            return [
                np.array([graph.node_features for graph in self]),
                np.array([graph.edge_features for graph in self]),
            ]
        return np.array([graph.node_features for graph in self])

    def _to_numpy_dynamic(self):
        if self.edge_feature_names:
            node_and_edge_features = [
                graph_sequence.to_numpy() for graph_sequence in self
            ]
            node_feature_list, edge_feature_list = [], []
            for node_features, edge_features in node_and_edge_features:
                node_feature_list.append(node_features)
                edge_feature_list.append(edge_features)
            return [np.array(node_feature_list), np.array(edge_feature_list)]
        return np.array([graph_sequence.to_numpy() for graph_sequence in self])

    def _create_timeseries_columns(self, feature_names, seq_len):
        column_names = []
        for i in range(seq_len):
            columns = feature_names.copy()
            for col_idx, column in enumerate(columns):
                columns[col_idx] = f"{column}_{i}"
            column_names += columns
        return column_names

    def _to_pandas_dynamic(self):
        to_numpy = self.to_numpy()
        if self.edge_feature_names:
            node_columns = self._create_timeseries_columns(
                self.node_feature_names, to_numpy[0].shape[1]
            )
            edge_columns = self._create_timeseries_columns(
                self.edge_feature_names, to_numpy[1].shape[1]
            )
            df_node_features = pd.DataFrame(
                np.hstack(np.hstack(self.to_numpy()[0])), columns=node_columns
            )
            df_edge_features = pd.DataFrame(
                np.hstack(np.hstack(self.to_numpy()[1])), columns=edge_columns
            )
            return [df_node_features, df_edge_features]
        columns = self._create_timeseries_columns(
            self.node_feature_names, to_numpy.shape[1]
        )
        return pd.DataFrame(np.hstack(np.hstack(to_numpy)), columns=columns)

    def to_pandas(self):
        """
        Converts `GraphList` instance to

            - `pd.DataFrame` of shape (n_graphs * n_nodes, n_node_features) if graphs only include node features and are
              not windowed.
            - list of `pd.DataFrame`s of shapes (n_graphs * n_nodes, n_node_features) and (n_graphs * n_edges,
              n_edge_features).
              Index 0 of the list contains the node features, index 1 contains the edge features
            - `pd.DataFrame` of shape (n_graphs * n_nodes, window_size * n_node_features) if graphs only include node
              features and graphs are windowed. In this case, features are indexed by their position in the sequence and
              represented as extra columns.
            - list of `pd.DataFrame`s of shapes (n_graphs * n_nodes, window_size * n_node_features) and
              (n_graphs * n_edges,  window_size * n_edge_features). Index 0 of the list contains the node features,
              index 1 contains the edge features. In this case, features are indexed by their position in the sequence
              and represented as extra columns.
        :return: either `pd.DataFrame` of the node features or list of `pd.DataFrames` containing node and edge features
        """
        if not self.strict_checks:
            return self._to_pandas_dynamic()
        if self.edge_feature_names:
            df_node_features = pd.DataFrame(
                np.vstack(self.to_numpy()[0]), columns=self.node_feature_names
            )
            df_edge_features = pd.DataFrame(
                np.vstack(self.to_numpy()[1]), columns=self.edge_feature_names
            )
            return [df_node_features, df_edge_features]
        return pd.DataFrame(np.vstack(self.to_numpy()), columns=self.node_feature_names)


class StaticGraphDataset:
    """
    Main container to store graph datasets for graphs with fixed dimensions and feature dimension for the whole dataset.
    Additionally, works together with the provided preprocessing methods in `preprocessing.py` to simplify the data
    preparation process.
    """

    def __init__(self, edge_list, graphs: GraphList):
        """
        Initializes and validates provided graph data
        :param edge_list: A list of node pairs. The list must include both directions for each edge.
        :param graphs: A `GraphList` instance of all graphs in the dataset. This list should be created with the static
        method `pandas_to_graphs()`.
        """
        self.test = None
        self.val = None
        self.train = None
        self.node_scaler = None
        self.edge_scaler = None
        self.adjacency_matrix = self._edge_list_to_adj(edge_list)
        self.graphs = graphs
        self.edge_list = edge_list
        self._validate_features()

    @staticmethod
    def numpy_to_graphs(
        node_features: np.ndarray,
        num_nodes: int,
        node_feature_names: List[str],
        edge_features: np.ndarray = None,
        num_edges: int = 0,
        edge_feature_names: List[str] = None,
    ) -> GraphList:
        """
        Converts a dataset from a `np.ndarray` to a `GraphList` that can be used to initialize an instance of
        `StaticGraphDataset`.
        :param node_features: `np.ndarray` containing node feature data. The array needs to be of shape (n_graphs,
        n_nodes, n_features).
        :param num_nodes: The number of nodes for graphs in the dataset
        :param node_feature_names: The node features to extract from df_node_features.
        :param edge_features: Same as df_node_features but for edge features. Only needed if edge features are
        present in the graphs. The array needs to be of shape (n_graphs, n_edges, n_features).
        :param num_edges: The number of edges for graphs in the dataset
        :param edge_feature_names: The edge features to extract from the df_edge_features
        :return: A `GraphList` instance that contains `Graph`-objects containing all relevant data.
        """
        if edge_features is not None and not edge_feature_names:
            raise ValueError(
                "Cannot process edge features when their names are not passed via the `edge_feature_names` variable"
            )
        graph_list = GraphList(
            num_nodes=num_nodes,
            node_feature_names=node_feature_names,
            num_edges=num_edges,
            edge_feature_names=edge_feature_names,
        )
        for i in tqdm(
            range(0, node_features.shape[0]),
            desc="Creating graph dataset",
        ):
            tmp_node_features = node_features[i].copy()
            if edge_features is not None:
                tmp_edge_features = edge_features[i].copy()
                graph_list.append(
                    Graph(
                        ID=i,
                        node_features=tmp_node_features,
                        node_feature_names=node_feature_names,
                        edge_features=tmp_edge_features,
                        edge_feature_names=edge_feature_names,
                        n_edges=num_edges,
                    )
                )
            else:
                graph_list.append(
                    Graph(
                        ID=i,
                        node_features=tmp_node_features,
                        node_feature_names=node_feature_names,
                        n_edges=num_edges,
                    )
                )
        return graph_list

    @staticmethod
    def pandas_to_graphs(
        df_node_features: pd.DataFrame,
        num_nodes: int,
        node_feature_names: List[str],
        df_edge_features: pd.DataFrame = None,
        num_edges: int = 0,
        edge_feature_names: List[str] = None,
    ) -> GraphList:
        """
        Converts a tabular dataset from a `pd.DataFrame` to a `GraphList` that can be used to initialize an instance of
        `StaticGraphDataset`.
        :param df_node_features: `pd.DataFrame` containing node feature data. This function expects all graphs to follow
        the same node order and graphs to be adjacent to each other within the table.<br>

        **Example**: Each graph contains 40 nodes and there are 100 graphs present. This means that the DataFrame must
        contain 4000 rows, where graph 0 starts at index 0 and ends at index 39, graph 1 starts at index 40 and ends at
        index 79, etc. The nodes for each graph are in the same order, e.g. starting at 0 and ending at 39. Other orders
        are possible but need to be consistent across the dataset.
        :param num_nodes: The number of nodes for graphs in the dataset
        :param node_feature_names: The node features to extract from df_node_features.
        :param df_edge_features: Same as df_node_features but for edge features. Only needed if edge features are
        present in the graphs
        :param num_edges: The number of edges for graphs in the dataset
        :param edge_feature_names: The edge features to extract from the df_edge_features
        :return: A `GraphList` instance that contains `Graph`-objects containing all relevant data.
        """
        node_features_numpy = (
            df_node_features[node_feature_names]
            .to_numpy()
            .reshape((-1, num_nodes, len(node_feature_names)))
        )
        del df_node_features
        edge_features_numpy = None
        if df_edge_features is not None:
            edge_features_numpy = (
                df_edge_features[edge_feature_names]
                .to_numpy()
                .reshape((-1, num_edges, len(edge_feature_names)))
            )
            del df_edge_features
        graph_list = StaticGraphDataset.numpy_to_graphs(
            node_features_numpy,
            num_nodes,
            node_feature_names,
            edge_features_numpy,
            num_edges,
            edge_feature_names,
        )
        return graph_list

    def _validate_features(self):
        if self.graphs:
            first_graph = self.graphs[0]
            node_feature_names = first_graph.node_feature_names
            node_feature_shape = first_graph.node_features.shape

            check_edge_features = False
            if first_graph.edge_feature_names:
                edge_feature_names = first_graph.edge_feature_names
                edge_feature_shape = first_graph.edge_features.shape
                check_edge_features = True

            for graph in self.graphs:
                if (
                    node_feature_names != graph.node_feature_names
                    or node_feature_shape != graph.node_features.shape
                ):
                    raise ValueError(
                        "Invalid list of graphs given. Different node features are present"
                    )
                if check_edge_features and (
                    graph.edge_feature_names != edge_feature_names
                    or edge_feature_shape != graph.edge_features.shape
                ):
                    raise ValueError(
                        "Invalid list of graphs given. Different edge features are present"
                    )
            self.node_feature_names = node_feature_names
            if first_graph.edge_feature_names:
                self.edge_feature_names = edge_feature_names

    def _edge_list_to_adj(self, edge_list) -> np.ndarray:
        size = len(set([n for e in edge_list for n in e]))
        adj = np.zeros((size, size), dtype=np.float32)
        for sink, source in edge_list:
            adj[sink - 1][source - 1] = 1

        is_symmetric = np.allclose(adj, adj.T, rtol=1e-05, atol=1e-08)
        main_diag = np.diag(adj)
        if main_diag.any():
            raise ValueError(f"Self-loops are currently not supported.")
        upper_triangle = np.triu(adj, k=1)
        lower_triangle = np.tril(adj, k=-1)

        if not upper_triangle.any() or not lower_triangle.any():
            raise ValueError(
                f"Incomplete edge list provided. Both edge directions need to be provided"
            )

        main_diag.any()
        if not is_symmetric:
            raise ValueError(
                f"Adjacency matrix is not symmetric. Please provide edges for both directions."
            )
        return adj

    def set_train_split(self, train: GraphList | List[GraphList]):
        self.train = train

    def set_validation_split(self, val: GraphList | List[GraphList]):
        self.val = val

    def set_test_split(self, test: GraphList | List[GraphList]):
        self.test = test

    def set_splits(self, train, test, val=None):
        self.set_train_split(train)
        self.set_validation_split(val)
        self.set_test_split(test)

    def get_splits(self):
        return self.train, self.val, self.test

    @classmethod
    def type(cls):
        return cls.__name__

    def __len__(self):
        return len(self.graphs)
