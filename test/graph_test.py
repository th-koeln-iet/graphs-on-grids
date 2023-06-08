import pytest

from test.testUtils import create_graph_dataset, create_test_graph


class TestGraphs:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 5
        cls.n_features = 2
        cls.n_nodes = 40
        cls.dataset = create_graph_dataset(
            num_graphs=cls.n_graphs, num_features=cls.n_features, num_nodes=cls.n_nodes
        )
        cls.feature_names = cls.dataset.node_feature_names

    def test_append_different_node_features_in_graph_list(self):
        graphs = self.dataset.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(self.n_features + 1, self.n_nodes))

    def test_append_different_node_count_in_graph_list(self):
        graphs = self.dataset.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(self.n_features, self.n_nodes + 1))

    def test_append_different_edge_features_in_graph_list(self):
        graphs = self.dataset.graphs
        num_edges = graphs.num_edges
        with pytest.raises(ValueError):
            graphs.append(
                create_test_graph(
                    self.n_features, self.n_nodes, num_edges, self.n_features + 1
                )
            )

    def test_append_different_edge_count_in_graph_list(self):
        graphs = self.dataset.graphs
        with pytest.raises(ValueError):
            graphs.append(
                create_test_graph(
                    self.n_features, self.n_nodes, graphs.num_edges + 1, self.n_features
                )
            )
