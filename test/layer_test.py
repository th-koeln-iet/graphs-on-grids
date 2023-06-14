import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import gog
from gog import create_windowed_train_test_split
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
        self._execute_layer_test(model)

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
        self._execute_layer_test(model)

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
        self._execute_layer_test(model)

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
        self._execute_layer_test(model)

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
        self._execute_layer_test(model)

    def _execute_layer_test(self, model):
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features


class TestLayersWithEdgeFeatures:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 64
        cls.n_features = 2
        cls.n_nodes = 40
        cls.dataset = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features,
            num_nodes=cls.n_nodes,
            create_edge_features=True,
        )
        cls.n_edges = cls.dataset.graphs.num_edges
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
        model = self._create_multi_input_model(gog.GraphBase, adj, embedding_size)
        self._execute_layer_test(model)

    def test_wrong_embedding_size(self):
        adj = self.dataset.adjacency_matrix
        gog.GraphLayer(adj, 4)
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, -1)
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 0)

    def test_wrong_hidden_layer_definition(self):
        adj = self.dataset.adjacency_matrix
        gog.GraphLayer(adj, 4, hidden_units_node=[4, 3])
        gog.GraphLayer(adj, 4, hidden_units_edge=[4, 3])
        gog.GraphLayer(adj, 4, hidden_units_node=(4, 3))
        gog.GraphLayer(adj, 4, hidden_units_edge=(4, 3))
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 4, hidden_units_node=3)
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 4, hidden_units_edge=3)

    def test_wrong_dropout_range(self):
        adj = self.dataset.adjacency_matrix
        gog.GraphLayer(adj, 4, dropout_rate=0)
        gog.GraphLayer(adj, 4, dropout_rate=1)
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 4, dropout_rate=-0.1)
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 4, dropout_rate=1.1)

    def test_wrong_activation_function(self):
        adj = self.dataset.adjacency_matrix
        gog.GraphLayer(adj, 4, activation="relu")
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 4, activation="magic")

    def test_asymmetric_adj_matrix(self):
        adj = self.dataset.adjacency_matrix.copy()
        adj[0, 1] = 2
        with pytest.raises(ValueError):
            gog.GraphLayer(adj, 4)

    def test_graph_base_with_mlp(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3

        # test list and tuple as input types
        model = self._create_multi_input_model(
            gog.GraphBase,
            adj,
            embedding_size,
            hidden_units_node=(8, 4),
            hidden_units_edge=[8, 4],
        )
        self._execute_layer_test(model)

    def test_graph_conv(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = self._create_multi_input_model(
            gog.GraphConvolution, adj, embedding_size
        )
        self._execute_layer_test(model)

    def test_graph_attn(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = self._create_multi_input_model(gog.GraphAttention, adj, embedding_size)
        self._execute_layer_test(model)

    def test_graph_attn_with_mlp(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = self._create_multi_input_model(
            gog.GraphAttention,
            adj,
            embedding_size,
            hidden_units_node=[8, 4],
            hidden_units_edge=[8, 4],
        )
        self._execute_layer_test(model)

    def test_graph_mh_attn(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = self._create_multi_input_model(
            gog.MultiHeadGraphAttention, adj, embedding_size
        )
        self._execute_layer_test(model)

    def _execute_layer_test(self, model):
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 3
        assert y_pred.shape[1] == self.n_nodes
        assert y_pred.shape[2] == self.n_features

    def _create_multi_input_model(
        self,
        graph_layer,
        adj,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
    ):
        if hidden_units_node is None:
            hidden_units_node = []
        if hidden_units_edge is None:
            hidden_units_edge = []

        input_layer = keras.layers.Input((self.n_nodes, self.n_features))
        input_layer_edge = keras.layers.Input((self.n_edges, self.n_features))
        gnn = graph_layer(adj, embedding_size, hidden_units_node, hidden_units_edge)(
            [input_layer, input_layer_edge]
        )
        gnn = keras.layers.BatchNormalization()(gnn)
        gnn = keras.layers.ReLU()(gnn)
        gnn = graph_layer(adj, embedding_size, hidden_units_node, hidden_units_edge)(
            [gnn, input_layer_edge]
        )
        gnn = keras.layers.BatchNormalization()(gnn)
        gnn = keras.layers.ReLU()(gnn)
        gnn = graph_layer(adj, embedding_size, hidden_units_node, hidden_units_edge)(
            [gnn, input_layer_edge]
        )
        out = tf.keras.layers.Dense(self.n_features)(gnn)
        return keras.models.Model(inputs=[input_layer, input_layer_edge], outputs=out)


class TestTemporalLayers:
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

        # Model/Training parameters
        cls.EPOCHS = 2
        cls.BATCH_SIZE = 16
        cls.LR = 0.001
        cls.optimizer = tf.keras.optimizers.Adam(learning_rate=cls.LR)
        cls.loss_fn = tf.keras.losses.MeanSquaredError()

        cls.window_size = 5
        cls.len_labels = 3

        X_train, X_test, y_train, y_test = create_windowed_train_test_split(
            cls.dataset, window_size=cls.window_size, len_labels=cls.len_labels
        )
        masked_train, masked_test = mask_labels(
            X_train,
            X_test,
            cls.dataset.node_feature_names,
            np.arange(0, cls.n_nodes // 2),
        )
        cls.X_train = masked_train.to_numpy()
        cls.y_train = y_train.to_numpy()

        cls.X_test = masked_test.to_numpy()

    @classmethod
    def setup_method(cls):
        cls.optimizer = tf.keras.optimizers.Adam(learning_rate=cls.LR)

    def test_graph_base_temporal_conv(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphBaseTemporalConv(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphBaseTemporalConv(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.ConvOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_base_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphBaseLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphBaseLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_base_conv_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphBaseConvLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphBaseConvLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_base_gru(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphBaseGRU(adj, embedding_size, output_seq_len=3),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphBaseGRU(adj, embedding_size, output_seq_len=3),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_conv_temporal_conv(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphConvTemporalConv(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphConvTemporalConv(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.ConvOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_conv_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphConvolutionLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphConvolutionLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_conv_conv_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphConvolutionConvLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphConvolutionConvLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_conv_gru(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphConvolutionGRU(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphConvolutionGRU(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_attn_temporal_conv(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphAttentionTemporalConv(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphAttentionTemporalConv(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.ConvOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_attn_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphAttentionLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphAttentionLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_attn_conv_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphAttentionConvLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphAttentionConvLSTM(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def test_graph_attn_gru(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 3
        model = tf.keras.models.Sequential(
            [
                keras.layers.Input((self.window_size, self.n_nodes, self.n_features)),
                gog.layers.temporal.GraphAttentionGRU(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.GraphAttentionGRU(
                    adj, embedding_size, output_seq_len=3
                ),
                keras.layers.ReLU(),
                gog.layers.temporal.RecurrentOutputBlock(
                    output_seq_len=3, units=self.n_features
                ),
            ]
        )
        self._execute_temporal_layer_test(model)

    def _execute_temporal_layer_test(self, model):
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 4
        assert y_pred.shape[1] == self.len_labels
        assert y_pred.shape[2] == self.n_nodes
        assert y_pred.shape[3] == self.n_features


class TestTemporalLayersWithEdgeFeatures:
    @classmethod
    def setup_class(cls):
        cls.n_graphs = 64
        cls.n_features = 2
        cls.n_nodes = 40
        cls.dataset = create_graph_dataset(
            num_graphs=cls.n_graphs,
            num_features=cls.n_features,
            num_nodes=cls.n_nodes,
            create_edge_features=True,
        )
        cls.feature_names = cls.dataset.node_feature_names
        cls.n_edges = cls.dataset.graphs.num_edges

        # Model/Training parameters
        cls.EPOCHS = 2
        cls.BATCH_SIZE = 16
        cls.LR = 0.001
        cls.optimizer = tf.keras.optimizers.Adam(learning_rate=cls.LR)
        cls.loss_fn = tf.keras.losses.MeanSquaredError()

        cls.window_size = 5
        cls.len_labels = 3

        X_train, X_test, y_train, y_test = create_windowed_train_test_split(
            cls.dataset, window_size=cls.window_size, len_labels=cls.len_labels
        )
        masked_train, masked_test = mask_labels(
            X_train,
            X_test,
            cls.dataset.node_feature_names,
            np.arange(0, cls.n_nodes // 2),
        )
        cls.X_train = masked_train.to_numpy()
        cls.y_train = y_train.to_numpy()

        cls.X_test = masked_test.to_numpy()

    @classmethod
    def setup_method(cls):
        cls.optimizer = tf.keras.optimizers.Adam(learning_rate=cls.LR)

    def test_graph_base_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 2
        model = self._create_multi_input_temporal_model(
            gog.GraphConvolutionLSTM,
            gog.RecurrentOutputBlock,
            adj,
            embedding_size,
            self.len_labels,
        )
        self._execute_temporal_layer_test(model)

    def test_graph_base_gru(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 2
        model = self._create_multi_input_temporal_model(
            gog.GraphAttentionGRU,
            gog.RecurrentOutputBlock,
            adj,
            embedding_size,
            self.len_labels,
        )
        self._execute_temporal_layer_test(model)

    def test_graph_base_conv_lstm(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 2
        model = self._create_multi_input_temporal_model(
            gog.GraphBaseConvLSTM,
            gog.RecurrentOutputBlock,
            adj,
            embedding_size,
            self.len_labels,
        )
        self._execute_temporal_layer_test(model)

    def test_graph_base_temporal_conv(self):
        adj = self.dataset.adjacency_matrix
        embedding_size = self.n_features * 2
        model = self._create_multi_input_temporal_model(
            gog.GraphBaseTemporalConv,
            gog.ConvOutputBlock,
            adj,
            embedding_size,
            self.len_labels,
        )
        self._execute_temporal_layer_test(model)

    def _execute_temporal_layer_test(self, model):
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        model.fit(
            self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        y_pred = model.predict(self.X_test)
        assert len(y_pred.shape) == 4
        assert y_pred.shape[1] == self.len_labels
        assert y_pred.shape[2] == self.n_nodes
        assert y_pred.shape[3] == self.n_features

    def _create_multi_input_temporal_model(
        self,
        graph_layer,
        output_layer,
        adj,
        embedding_size,
        output_seq_len,
        hidden_units_node=None,
        hidden_units_edge=None,
    ):
        if hidden_units_node is None:
            hidden_units_node = []
        if hidden_units_edge is None:
            hidden_units_edge = []

        input_layer = keras.layers.Input(
            (self.window_size, self.n_nodes, self.n_features)
        )
        input_layer_edge = keras.layers.Input(
            (self.window_size, self.n_edges, self.n_features)
        )
        gnn = graph_layer(
            adj, embedding_size, output_seq_len, hidden_units_node, hidden_units_edge
        )([input_layer, input_layer_edge])
        gnn = keras.layers.ReLU()(gnn)
        gnn = graph_layer(
            adj, embedding_size, output_seq_len, hidden_units_node, hidden_units_edge
        )([gnn, input_layer_edge])
        gnn = keras.layers.ReLU()(gnn)
        gnn = graph_layer(
            adj, embedding_size, output_seq_len, hidden_units_node, hidden_units_edge
        )([gnn, input_layer_edge])
        out = output_layer(self.len_labels, self.n_features)(gnn)
        return keras.models.Model(inputs=[input_layer, input_layer_edge], outputs=out)
