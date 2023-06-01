import pytest

from test.testUtils import create_graph_dataset, create_test_graph


class TestGraphs:
    @classmethod
    def setup_method(cls):
        cls.dataset = create_graph_dataset(num_graphs=5, num_features=2, num_nodes=4)
        cls.feature_names = cls.dataset.graph_feature_names

    def test_append_different_node_features_in_graph_list(self):
        graphs = self.dataset.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(3, 4))

    def test_append_different_node_count_in_graph_list(self):
        graphs = self.dataset.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(2, 5))

    def test_append_different_edge_features_in_graph_list(self):
        graphs = self.dataset.graphs
        num_edges = graphs.num_edges
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(2, 4, num_edges, 5))

    def test_append_different_edge_count_in_graph_list(self):
        graphs = self.dataset.graphs
        with pytest.raises(ValueError):
            graphs.append(create_test_graph(2, 4, 24, 2))
