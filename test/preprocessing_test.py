import numpy as np
import pytest

from gog import create_train_test_split_windowed
from gog.preprocessing import (
    create_train_test_split,
    mask_labels,
    create_validation_set,
    apply_scaler,
)
from test.testUtils import create_graph_dataset, create_test_graph


class TestPreprocessing:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 5
        cls.n_features = 2
        cls.n_nodes = 40
        cls.dataset = create_graph_dataset(
            num_graphs=cls.n_graphs, num_features=cls.n_features, num_nodes=cls.n_nodes
        )
        cls.feature_names = cls.dataset.node_feature_names

    def test_create_train_test_split(self):
        train, test = create_train_test_split(self.dataset)
        assert len(train) == 4
        assert len(test) == 1

        assert None not in train
        assert None not in test

        for inst in train:
            assert inst not in test

    def test_create_train_test_split_wrong_input_type(self):
        dataset = None
        with pytest.raises(ValueError):
            create_train_test_split(dataset)

    def test_mask_labels(self):
        train, test = create_train_test_split(self.dataset)
        targets = self.feature_names
        nodes_to_mask = [1, 2, 3]
        train_masked, test_masked = mask_labels(train, test, targets, nodes_to_mask)

        # assert correct length
        assert len(train_masked) == 4
        assert len(test_masked) == 1

        # masked features in selected nodes are actually masked
        for graph in train_masked:
            masked_nodes = graph.node_features[nodes_to_mask]
            for node in masked_nodes:
                assert node[0] == 0.0 and node[1] == 0.0

        # original feature matrix is unaffected
        for graph in train:
            masked_nodes = graph.node_features[nodes_to_mask]
            for node in masked_nodes:
                assert node[0] != 0.0 and node[1] != 0.0

    def test_mask_labels_wrong_input(self):
        train, test = [], []
        targets = ["0"]
        nodes_to_mask = [1, 2]

        with pytest.raises(ValueError):
            mask_labels(train, test, targets, nodes_to_mask)

    def test_create_validation_set(self):
        graphs = self.dataset.graphs
        X_train, X_val, y_train, y_val = create_validation_set(
            graphs, graphs, validation_size=0.4
        )

        assert len(X_train) == len(y_train) == 3
        assert len(X_val) == len(y_val) == 2

        for inst in X_train:
            assert inst not in X_val

        for inst in y_train:
            assert inst not in y_val

    def test_create_validation_set_different_size(self):
        graphs = self.dataset.graphs
        graphs_larger = graphs.copy()
        graphs_larger.append(
            create_test_graph(num_nodes=self.n_nodes, num_features=self.n_features)
        )
        with pytest.raises(ValueError):
            create_validation_set(graphs, graphs_larger)

    def test_create_validation_set_wrong_input_type(self):
        X = np.array([])
        y = np.array([])
        with pytest.raises(ValueError):
            create_validation_set(X, y)

    def test_apply_scaler_wrong_input_type(self):
        dataset = None
        with pytest.raises(ValueError):
            apply_scaler(dataset)

    def test_apply_scaler_zero_mean(self):
        dataset = self.dataset
        # need to initialize train, test split in dataset object to correctly apply scaling
        train, test = create_train_test_split(dataset)
        train_scaled, test_scaled = apply_scaler(dataset)

        assert len(train) == len(train_scaled)
        assert len(test) == len(test_scaled)

        from sklearn.preprocessing import StandardScaler

        assert isinstance(dataset.node_scaler, StandardScaler)

    def test_apply_scaler_min_max(self):
        dataset = self.dataset
        # need to initialize train, test split in dataset object to correctly apply scaling
        train, test = create_train_test_split(dataset)
        train_scaled, test_scaled = apply_scaler(dataset, method="min_max")

        assert len(train) == len(train_scaled)
        assert len(test) == len(test_scaled)

        from sklearn.preprocessing import MinMaxScaler

        assert isinstance(dataset.node_scaler, MinMaxScaler)

    def test_apply_scaler_edge_zero_mean(self):
        dataset = self.dataset
        # need to initialize train, test split in dataset object to correctly apply scaling
        train, test = create_train_test_split(dataset)
        train_scaled, test_scaled = apply_scaler(dataset, target="edge")

        assert len(train) == len(train_scaled)
        assert len(test) == len(test_scaled)

        from sklearn.preprocessing import StandardScaler

        assert isinstance(dataset.edge_scaler, StandardScaler)

    def test_apply_scaler_edge_min_max(self):
        dataset = self.dataset
        # need to initialize train, test split in dataset object to correctly apply scaling
        train, test = create_train_test_split(dataset)
        train_scaled, test_scaled = apply_scaler(
            dataset, method="min_max", target="edge"
        )

        assert len(train) == len(train_scaled)
        assert len(test) == len(test_scaled)

        from sklearn.preprocessing import MinMaxScaler

        assert isinstance(dataset.edge_scaler, MinMaxScaler)

    def test_apply_scaler_wrong_method(self):
        dataset = self.dataset
        # need to initialize train, test split in dataset object to correctly apply scaling
        create_train_test_split(dataset)
        with pytest.raises(ValueError):
            apply_scaler(dataset, method="magic")

    def test_apply_scaler_wrong_target(self):
        dataset = self.dataset
        # need to initialize train, test split in dataset object to correctly apply scaling
        create_train_test_split(dataset)
        with pytest.raises(ValueError):
            apply_scaler(dataset, target="magic")

    def test_apply_scaler_edge_scaling_without_edge_features(self):
        dataset = create_graph_dataset(
            num_graphs=self.n_graphs,
            num_features=self.n_features,
            num_nodes=self.n_nodes,
            create_edge_features=False,
        )
        create_train_test_split(dataset)
        with pytest.raises(ValueError) as err:
            apply_scaler(dataset, target="edge")
        assert "contain edge features" in str(err.value)


class TestPreprocessingTimeSeries:
    @classmethod
    def setup_method(cls):
        cls.n_graphs = 50
        cls.n_features = 2
        cls.n_nodes = 40
        cls.dataset = create_graph_dataset(
            num_graphs=cls.n_graphs, num_features=cls.n_features, num_nodes=cls.n_nodes
        )
        cls.feature_names = cls.dataset.node_feature_names
        cls.n_edges = cls.dataset.graphs.num_edges

    def test_create_train_test_split(self):
        window_size = 5
        len_labels = 1
        X_train, X_test, y_train, y_test = create_train_test_split_windowed(
            self.dataset, window_size=window_size, len_labels=len_labels
        )
        expected_test_len = 0.2 * self.n_graphs - 1
        assert len(X_test) == expected_test_len
        assert (
            len(X_train) == self.n_graphs - expected_test_len - window_size - len_labels
        )

        assert None not in X_train
        assert None not in X_test

    def test_apply_scaler_node(self):
        window_size = 5
        len_labels = 1
        create_train_test_split_windowed(
            self.dataset, window_size=window_size, len_labels=len_labels
        )
        X_train, X_test, y_train, y_test = apply_scaler(self.dataset)

        assert (
            X_train.shape[0] == len(self.dataset.train[0]) * self.n_nodes
            and X_train.shape[1] == self.n_features * window_size
        )
        assert (
            X_test.shape[0] == len(self.dataset.test[0]) * self.n_nodes
            and X_test.shape[1] == self.n_features * window_size
        )
        assert (
            y_train.shape[0] == len(self.dataset.train[1]) * self.n_nodes
            and y_train.shape[1] == self.n_features * len_labels
        )
        assert (
            y_test.shape[0] == len(self.dataset.test[1]) * self.n_nodes
            and y_test.shape[1] == self.n_features * len_labels
        )

    def test_apply_scaler_edge(self):
        window_size = 5
        len_labels = 1
        create_train_test_split_windowed(
            self.dataset, window_size=window_size, len_labels=len_labels
        )
        X_train, X_test, y_train, y_test = apply_scaler(self.dataset, target="edge")

        assert (
            X_train.shape[0] == len(self.dataset.train[0]) * self.n_edges
            and X_train.shape[1] == self.n_features * window_size
        )
        assert (
            X_test.shape[0] == len(self.dataset.test[0]) * self.n_edges
            and X_test.shape[1] == self.n_features * window_size
        )
        assert (
            y_train.shape[0] == len(self.dataset.train[1]) * self.n_edges
            and y_train.shape[1] == self.n_features * len_labels
        )
        assert (
            y_test.shape[0] == len(self.dataset.test[1]) * self.n_edges
            and y_test.shape[1] == self.n_features * len_labels
        )

    def test_mask_labels(self):
        window_size = 5
        len_labels = 3
        X_train, X_test, y_train, y_test = create_train_test_split_windowed(
            self.dataset, window_size=window_size, len_labels=len_labels
        )
        train_len = len(X_train)
        test_len = len(X_test)
        targets = self.feature_names
        nodes_to_mask = [1, 2, 3]
        train_masked, test_masked = mask_labels(X_train, X_test, targets, nodes_to_mask)

        # assert correct length
        assert len(train_masked) == train_len
        assert len(test_masked) == test_len

        # masked features in selected nodes are actually masked
        for graph_sequence in train_masked:
            for graph in graph_sequence:
                masked_nodes = graph.node_features[nodes_to_mask]
                for node in masked_nodes:
                    assert node[0] == 0.0 and node[1] == 0.0

        # original feature matrix is unaffected
        for graph_sequence in y_train:
            for graph in graph_sequence:
                masked_nodes = graph.node_features[nodes_to_mask]
                for node in masked_nodes:
                    assert node[0] != 0.0 and node[1] != 0.0

    def test_create_validation_set(self):
        graphs = self.dataset.graphs
        X_train, X_val, y_train, y_val = create_validation_set(
            graphs, graphs, validation_size=0.4
        )

        assert len(X_train) == len(y_train) == 30
        assert len(X_val) == len(y_val) == 20

        for inst in X_train:
            assert inst not in X_val

        for inst in y_train:
            assert inst not in y_val
