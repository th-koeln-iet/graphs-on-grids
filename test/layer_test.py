import numpy as np
import tensorflow as tf
from tensorflow import keras

import gog
from gog.preprocessing import create_train_test_split, mask_labels
from test.testUtils import create_graph_dataset


class TestLayers:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 64
        cls.n_features = 2
        cls.n_nodes = 40
        cls.dataset = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features,
            num_nodes=cls.n_nodes,
            create_edge_features=False,
        )
        cls.feature_names = cls.dataset.node_feature_names
        train, test = create_train_test_split(cls.dataset, shuffle=False)
        masked_train, masked_test = mask_labels(
            train, test, cls.dataset.node_feature_names, np.arange(0, cls.n_nodes // 2)
        )
        cls.X_train = masked_train.to_numpy()
        cls.y_train = train.to_numpy()
        cls.X_test = masked_test.to_numpy()

        # Model/Training parameters
        cls.EPOCHS = 2
        cls.BATCH_SIZE = 16
        cls.LR = 0.001
        cls.loss_fn = tf.keras.losses.MeanSquaredError()

    @classmethod
    def setup_method(cls):
        cls.optimizer = tf.keras.optimizers.Adam(learning_rate=cls.LR)

    def test_graph_base(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.n_nodes, self.n_features)),
                gog.layers.GraphBase(adj, embedding_size),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                gog.layers.GraphBase(adj, embedding_size),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(self.n_features),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features

    def test_graph_base_with_mlp(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.n_nodes, self.n_features)),
                gog.layers.GraphBase(adj, embedding_size, hidden_units_node=[8, 8, 4]),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                gog.layers.GraphBase(adj, embedding_size, hidden_units_node=[8, 8, 4]),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(self.n_features),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features

    def test_graph_conv(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.n_nodes, self.n_features)),
                gog.layers.GraphConvolution(adj, embedding_size),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                gog.layers.GraphConvolution(adj, embedding_size),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(self.n_features),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features

    def test_graph_attn(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.n_nodes, self.n_features)),
                gog.layers.GraphAttention(adj, embedding_size),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                gog.layers.GraphAttention(adj, embedding_size),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(self.n_features),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features

    def test_graph_mh_attn(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.n_nodes, self.n_features)),
                gog.layers.MultiHeadGraphAttention(adj, embedding_size, num_heads=2),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                gog.layers.MultiHeadGraphAttention(adj, embedding_size, num_heads=2),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(self.n_features),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features


class TestTemporalLayers:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 5
        cls.num_features = 2
        cls.n_nodes = 4
        cls.dataset = create_graph_dataset(num_graphs=64, num_features=2, num_nodes=4)
        cls.feature_names = cls.dataset.node_feature_names

        # Model/Training parameters
        cls.EPOCHS = 2
        cls.BATCH_SIZE = 16
        cls.LR = 0.001
        cls.optimizer = tf.keras.optimizers.Adam(learning_rate=cls.LR)
        cls.loss_fn = tf.keras.losses.MeanSquaredError()
