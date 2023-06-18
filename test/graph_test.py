import pytest

from gog import create_train_test_split_windowed
from test.testUtils import (
    create_graph_dataset,
    create_test_graph,
    create_test_graph_sequence,
)


class TestGraphs:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 5
        cls.n_features_node = 2
        cls.n_features_edge = 3
        cls.n_nodes = 40
        cls.dataset_node_edge = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features_node,
            num_nodes=cls.n_nodes,
            num_features_edge=cls.n_features_edge,
        )
        cls.dataset_node = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features_node,
            num_nodes=cls.n_nodes,
            create_edge_features=False,
        )
        cls.feature_names = cls.dataset_node.node_feature_names

    def test_append_different_node_features_in_graph_list(self):
        graphs = self.dataset_node.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(self.n_features_node + 1, self.n_nodes))

    def test_append_different_node_count_in_graph_list(self):
        graphs = self.dataset_node.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(self.n_features_node, self.n_nodes + 1))

    def test_append_different_edge_features_in_graph_list(self):
        graphs = self.dataset_node_edge.graphs
        num_edges = graphs.num_edges
        with pytest.raises(ValueError):
            graphs.append(
                create_test_graph(
                    self.n_features_node,
                    self.n_nodes,
                    num_edges,
                    self.n_features_node + 1,
                )
            )

    def test_append_different_edge_count_in_graph_list(self):
        graphs = self.dataset_node_edge.graphs
        with pytest.raises(ValueError):
            graphs.append(
                create_test_graph(
                    self.n_features_node,
                    self.n_nodes,
                    graphs.num_edges + 1,
                    self.n_features_node,
                )
            )

    def test_to_pandas_node_features(self):
        graphs = self.dataset_node.graphs
        df = graphs.to_pandas()
        assert df.shape == (self.n_graphs * self.n_nodes, self.n_features_node)

    def test_to_pandas_node_and_edge_features(self):
        graphs = self.dataset_node_edge.graphs
        dfs = graphs.to_pandas()

        assert isinstance(dfs, list)
        df_node = dfs[0]
        df_edge = dfs[1]
        assert df_node.shape == (self.n_graphs * self.n_nodes, self.n_features_node)
        assert df_edge.shape == (self.n_graphs * graphs.num_edges, self.n_features_edge)

    def test_to_numpy_node_features(self):
        graphs = self.dataset_node.graphs
        arr = graphs.to_numpy()
        assert arr.shape == (self.n_graphs, self.n_nodes, self.n_features_node)

    def test_to_numpy_node_and_edge_features(self):
        graphs = self.dataset_node_edge.graphs
        arr_list = graphs.to_numpy()

        assert isinstance(arr_list, list)
        arr_node = arr_list[0]
        arr_edge = arr_list[1]
        assert arr_node.shape == (self.n_graphs, self.n_nodes, self.n_features_node)
        assert arr_edge.shape == (
            self.n_graphs,
            graphs.num_edges,
            self.n_features_edge,
        )


class TestGraphsTimeSeries:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 50
        cls.n_features_node = 2
        cls.n_features_edge = 3
        cls.n_nodes = 40
        cls.dataset_node_edge = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features_node,
            num_nodes=cls.n_nodes,
            num_features_edge=cls.n_features_edge,
        )
        cls.dataset_node = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features_node,
            num_nodes=cls.n_nodes,
            create_edge_features=False,
        )
        cls.feature_names = cls.dataset_node.node_feature_names
        cls.window_size = 5
        cls.len_labels = 1
        (
            cls.X_train_node,
            _,
            _,
            _,
        ) = create_train_test_split_windowed(
            cls.dataset_node, window_size=cls.window_size, len_labels=cls.len_labels
        )
        (
            cls.X_train_node_edge,
            _,
            _,
            _,
        ) = create_train_test_split_windowed(
            cls.dataset_node_edge,
            window_size=cls.window_size,
            len_labels=cls.len_labels,
        )

    def test_append_different_node_features_in_graph_list(self):
        with pytest.raises(ValueError):
            new_sequence = create_test_graph_sequence(
                self.window_size,
                self.n_features_node + 1,
                self.n_nodes,
            )
            self.X_train_node.append(new_sequence)

    def test_append_different_node_count_in_graph_list(self):
        with pytest.raises(ValueError):
            new_sequence = create_test_graph_sequence(
                self.window_size,
                self.n_features_node,
                self.n_nodes + 1,
            )
            self.X_train_node.append(new_sequence)

    def test_append_different_edge_features_in_graph_list(self):
        graphs = self.dataset_node_edge.graphs
        num_edges = graphs.num_edges
        with pytest.raises(ValueError):
            self.X_train_node_edge.append(
                create_test_graph_sequence(
                    self.window_size,
                    self.n_features_node,
                    self.n_nodes,
                    num_edges,
                    self.n_features_edge + 1,
                )
            )

    def test_append_different_edge_count_in_graph_list(self):
        graphs = self.dataset_node_edge.graphs
        with pytest.raises(ValueError):
            self.X_train_node_edge.append(
                create_test_graph_sequence(
                    self.window_size,
                    self.n_features_node,
                    self.n_nodes,
                    graphs.num_edges + 1,
                    self.n_features_node,
                )
            )

    def test_append_different_sequence_length_in_graph_list(self):
        graphs = self.dataset_node.graphs
        with pytest.raises(ValueError):
            new_sequence = create_test_graph_sequence(
                self.window_size + 1,
                self.n_features_node,
                self.n_nodes,
                graphs.num_edges,
                self.n_features_node,
            )
            graphs.append(new_sequence)

    def test_to_pandas_node_features(self):
        graphs = self.dataset_node.graphs
        df = graphs.to_pandas()
        assert df.shape == (
            len(graphs) * self.n_nodes,
            self.n_features_node * self.window_size,
        )

    def test_to_pandas_node_and_edge_features(self):
        graphs = self.dataset_node_edge.graphs
        dfs = graphs.to_pandas()

        assert isinstance(dfs, list)
        df_node = dfs[0]
        df_edge = dfs[1]
        assert df_node.shape == (
            len(graphs) * self.n_nodes,
            self.n_features_node * self.window_size,
        )
        assert df_edge.shape == (
            len(graphs) * graphs.num_edges,
            self.n_features_edge * self.window_size,
        )

    def test_to_numpy_node_features(self):
        graphs = self.dataset_node.graphs
        arr = graphs.to_numpy()
        assert arr.shape == (
            len(graphs),
            self.window_size,
            self.n_nodes,
            self.n_features_node,
        )

    def test_to_numpy_node_and_edge_features(self):
        graphs = self.dataset_node_edge.graphs
        arr_list = graphs.to_numpy()

        assert isinstance(arr_list, list)
        arr_node = arr_list[0]
        arr_edge = arr_list[1]
        assert arr_node.shape == (
            len(graphs),
            self.window_size,
            self.n_nodes,
            self.n_features_node,
        )
        assert arr_edge.shape == (
            len(graphs),
            self.window_size,
            graphs.num_edges,
            self.n_features_edge,
        )
