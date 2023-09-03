import numpy as np
import tensorflow as tf
from tensorflow import keras


class TemporalConv(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        output_seq_len,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer: str
        | keras.initializers.Initializer
        | None = "glorot_uniform",
        weight_regularizer: str | keras.regularizers.Regularizer | None = None,
        bias_initializer: str | keras.initializers.Initializer | None = "zeros",
    ):
        super(TemporalConv, self).__init__()
        self.output_seq_len = output_seq_len
        self.adjacency_matrix = adjacency_matrix
        self.embedding_size = embedding_size
        self.hidden_units_node = hidden_units_node
        self.hidden_units_edge = hidden_units_edge
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.bias_initializer = bias_initializer

        self.graph_layer = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            batch_size, seq_len, num_nodes, num_node_features = input_shape
        else:
            batch_size, seq_len, num_nodes, num_node_features = input_shape[0]
            _, _, num_edges, num_edge_features = input_shape[1]

        self.tmp_conv1 = keras.layers.Conv2D(
            filters=seq_len,
            kernel_size=(1, num_node_features),
            padding="same",
            activation="relu",
            name="conv_1",
        )
        self.tmp_conv2 = keras.layers.Conv2D(
            filters=self.output_seq_len,
            kernel_size=(1, self.embedding_size),
            padding="same",
            activation="relu",
            name="conv_2",
        )
        if isinstance(input_shape, list):
            self.tmp_conv_edge = keras.layers.Conv2D(
                filters=seq_len,
                kernel_size=(1, num_edge_features),
                padding="same",
                activation="relu",
                name="conv_edge",
            )

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            node_features, edge_features = inputs
            _, seq_len, num_nodes, num_node_features = inputs[0].get_shape().as_list()
            _, _, num_edges, num_edge_features = inputs[1].get_shape().as_list()
            edge_features = tf.transpose(edge_features, perm=[0, 2, 3, 1])
            # batch_size, num_edges, num_edge_features, seq_len
            temporal_conv_edge = self.tmp_conv_edge(edge_features)

            # batch_size, seq_len, num_edges, num_edge_features
            conv_edge_transposed = tf.transpose(temporal_conv_edge, perm=[0, 3, 1, 2])
            reshaped_edge = tf.reshape(
                conv_edge_transposed, shape=(-1, num_edges, num_edge_features)
            )
        else:
            _, seq_len, num_nodes, num_node_features = inputs.get_shape().as_list()
            node_features = inputs
        node_features = tf.transpose(node_features, perm=[0, 2, 3, 1])
        # batch_size, num_nodes, num_features, seq_len
        temporal_conv_1 = self.tmp_conv1(node_features)

        # batch_size, seq_len, num_nodes, num_features
        conv_1_transposed = tf.transpose(temporal_conv_1, perm=[0, 3, 1, 2])
        reshaped_node = tf.reshape(
            conv_1_transposed, shape=(-1, num_nodes, num_node_features)
        )
        graph_output = (
            self.graph_layer(reshaped_node)
            if not isinstance(inputs, list)
            else self.graph_layer([reshaped_node, reshaped_edge])
        )
        graph_output_reshaped = tf.reshape(
            graph_output, shape=(-1, seq_len, num_nodes, self.embedding_size)
        )
        transposed = tf.transpose(graph_output_reshaped, perm=[0, 2, 3, 1])
        temporal_conv_2 = self.tmp_conv2(transposed)
        conv_2_transposed = tf.transpose(temporal_conv_2, perm=[0, 3, 1, 2])
        return conv_2_transposed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adjacency_matrix": self.adjacency_matrix,
                "embedding_size": self.embedding_size,
                "output_seq_len": self.output_seq_len,
                "hidden_units_node": self.hidden_units_node,
                "hidden_units_edge": self.hidden_units_edge,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "activation": self.activation,
                "weight_initializer": self.weight_initializer,
                "weight_regularizer": self.weight_regularizer,
                "bias_initializer": self.bias_initializer,
                "graph_layer": self.graph_layer,
            }
        )
        return config


class GraphLSTM(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer: str
        | keras.initializers.Initializer
        | None = "glorot_uniform",
        weight_regularizer: str | keras.regularizers.Regularizer | None = None,
        bias_initializer: str | keras.initializers.Initializer | None = "zeros",
    ):
        super(GraphLSTM, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.embedding_size = embedding_size
        self.hidden_units_node = hidden_units_node
        self.hidden_units_edge = hidden_units_edge
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.bias_initializer = bias_initializer
        self.graph_layer = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            batch_size, seq_len, num_nodes, num_node_features = input_shape
        else:
            batch_size, seq_len, num_nodes, num_node_features = input_shape[0]
            _, _, num_edges, num_edge_features = input_shape[1]

        self.lstm_in = keras.layers.LSTM(
            units=num_node_features,
            return_sequences=True,
        )

        self.lstm_out = keras.layers.LSTM(
            units=num_nodes * self.embedding_size,
            return_sequences=True,
        )

        if isinstance(input_shape, list):
            self.lstm_edge = keras.layers.LSTM(
                units=num_edge_features,
                return_sequences=True,
            )

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            (
                batch_size,
                edge_features,
                edge_seq_len,
                node_features,
                num_edge_features,
                num_edges,
                num_node_features,
                num_nodes,
                seq_len,
            ) = init_shape_variables(inputs)

            reshaped_edge = self.compute_edge_features_temporal(
                batch_size,
                edge_features,
                edge_seq_len,
                num_edge_features,
                num_edges,
                seq_len,
            )
        else:
            node_features = inputs
            node_feature_shape = tf.shape(inputs)
            batch_size, seq_len, num_nodes, num_node_features = (
                node_feature_shape[0],
                node_feature_shape[1],
                node_feature_shape[2],
                node_feature_shape[3],
            )

        node_features = tf.transpose(node_features, [2, 0, 1, 3])
        lstm_input = tf.reshape(
            node_features, shape=(batch_size * num_nodes, seq_len, num_node_features)
        )
        lstm_in_output = self.lstm_in(lstm_input)

        # reshape to (batch_size * seq_len, num_nodes, num_features
        reshaped = tf.reshape(
            lstm_in_output, shape=(num_nodes, batch_size, seq_len, num_node_features)
        )
        reshaped = tf.transpose(reshaped, perm=[1, 2, 0, 3])
        reshaped_node = tf.reshape(reshaped, shape=(-1, num_nodes, num_node_features))
        output = (
            self.graph_layer(reshaped_node)
            if not isinstance(inputs, list)
            else self.graph_layer([reshaped_node, reshaped_edge])
        )

        # reshape to (batch_size, seq_len, num_nodes, embedding_size)
        output_reshaped = tf.reshape(
            output, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )

        # reshape to (batch_size, seq_len, num_nodes * embedding_size)
        output_reshaped = tf.reshape(
            output_reshaped,
            shape=(batch_size, seq_len, num_nodes * self.embedding_size),
        )

        # output dimension (batch_size, seq_len, num_nodes * embedding_size)
        output_lstm = self.lstm_out(output_reshaped)

        output_lstm_reshaped = tf.reshape(
            output_lstm, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )
        return output_lstm_reshaped

    def compute_edge_features_temporal(
        self,
        batch_size,
        edge_features,
        edge_seq_len,
        num_edge_features,
        num_edges,
        seq_len,
    ):
        edge_features = tf.transpose(edge_features, [2, 0, 1, 3])
        lstm_edge = tf.reshape(
            edge_features,
            shape=(batch_size * num_edges, edge_seq_len, num_edge_features),
        )
        lstm_edge_output = self.lstm_edge(lstm_edge)

        reshaped = tf.reshape(
            lstm_edge_output,
            shape=(num_edges, batch_size, edge_seq_len, num_edge_features),
        )
        reshaped = tf.transpose(reshaped, perm=[1, 2, 0, 3])
        reshaped_edge = tf.reshape(
            reshaped, shape=(batch_size * seq_len, num_edges, num_edge_features)
        )
        return reshaped_edge

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adjacency_matrix": self.adjacency_matrix,
                "embedding_size": self.embedding_size,
                "hidden_units_node": self.hidden_units_node,
                "hidden_units_edge": self.hidden_units_edge,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "activation": self.activation,
                "weight_initializer": self.weight_initializer,
                "weight_regularizer": self.weight_regularizer,
                "bias_initializer": self.bias_initializer,
                "graph_layer": self.graph_layer,
            }
        )
        return config


class GraphGRU(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer: str
        | keras.initializers.Initializer
        | None = "glorot_uniform",
        weight_regularizer: str | keras.regularizers.Regularizer | None = None,
        bias_initializer: str | keras.initializers.Initializer | None = "zeros",
    ):
        super(GraphGRU, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.embedding_size = embedding_size
        self.hidden_units_node = hidden_units_node
        self.hidden_units_edge = hidden_units_edge
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.bias_initializer = bias_initializer
        self.graph_layer = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            batch_size, seq_len, num_nodes, num_node_features = input_shape
        else:
            batch_size, seq_len, num_nodes, num_node_features = input_shape[0]
            _, _, num_edges, num_edge_features = input_shape[1]

        self.gru_in = keras.layers.GRU(
            units=num_node_features,
            return_sequences=True,
        )

        self.gru_out = keras.layers.GRU(
            units=self.embedding_size,
            return_sequences=True,
        )

        if isinstance(input_shape, list):
            self.gru_edge = keras.layers.GRU(
                units=num_edge_features,
                return_sequences=True,
            )

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            (
                batch_size,
                edge_features,
                edge_seq_len,
                node_features,
                num_edge_features,
                num_edges,
                num_node_features,
                num_nodes,
                seq_len,
            ) = init_shape_variables(inputs)

            reshaped_edge = self.compute_edge_features_temporal(
                batch_size,
                edge_features,
                edge_seq_len,
                num_edge_features,
                num_edges,
                seq_len,
            )
        else:
            node_features = inputs
            node_feature_shape = tf.shape(inputs)
            batch_size, seq_len, num_nodes, num_node_features = (
                node_feature_shape[0],
                node_feature_shape[1],
                node_feature_shape[2],
                node_feature_shape[3],
            )

        node_features = tf.transpose(node_features, [2, 0, 1, 3])
        gru_input = tf.reshape(
            node_features, shape=(batch_size * num_nodes, seq_len, num_node_features)
        )
        gru_in_output = self.gru_in(gru_input)

        # reshape to (batch_size * seq_len, num_nodes, num_features
        reshaped = tf.reshape(
            gru_in_output, shape=(num_nodes, batch_size, seq_len, num_node_features)
        )
        reshaped = tf.transpose(reshaped, perm=[1, 2, 0, 3])
        reshaped_node = tf.reshape(reshaped, shape=(-1, num_nodes, num_node_features))
        output = (
            self.graph_layer(reshaped_node)
            if not isinstance(inputs, list)
            else self.graph_layer([reshaped_node, reshaped_edge])
        )
        output_reshaped = tf.reshape(
            output, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )
        output_reshaped = tf.transpose(output_reshaped, [2, 0, 1, 3])
        output_reshaped = tf.reshape(
            output_reshaped, shape=(batch_size * num_nodes, seq_len, self.embedding_size)
        )

        # # reshape to (batch_size, seq_len, num_nodes, embedding_size)
        # output_reshaped = tf.reshape(
        #     output, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        # )
        #
        # # reshape to (batch_size, seq_len, num_nodes * embedding_size)
        # output_reshaped = tf.reshape(
        #     output_reshaped,
        #     shape=(batch_size, seq_len, num_nodes * self.embedding_size),
        # )

        # output dimension (batch_size, seq_len, num_nodes * embedding_size)
        output_gru = self.gru_out(output_reshaped)
        output_gru_reshaped = tf.reshape(
            output_gru, shape=(num_nodes, batch_size, seq_len, self.embedding_size)
        )
        output_gru_reshaped = tf.transpose(output_gru_reshaped, perm=[1, 2, 0, 3])

        output_gru_reshaped = tf.reshape(
            output_gru_reshaped, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )
        return output_gru_reshaped

    def compute_edge_features_temporal(
        self,
        batch_size,
        edge_features,
        edge_seq_len,
        num_edge_features,
        num_edges,
        seq_len,
    ):
        edge_features = tf.transpose(edge_features, [2, 0, 1, 3])
        gru_edge = tf.reshape(
            edge_features,
            shape=(batch_size * num_edges, edge_seq_len, num_edge_features),
        )
        gru_edge_output = self.gru_edge(gru_edge)

        reshaped = tf.reshape(
            gru_edge_output,
            shape=(num_edges, batch_size, edge_seq_len, num_edge_features),
        )
        reshaped = tf.transpose(reshaped, perm=[1, 2, 0, 3])
        reshaped_edge = tf.reshape(
            reshaped, shape=(batch_size * seq_len, num_edges, num_edge_features)
        )
        return reshaped_edge

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adjacency_matrix": self.adjacency_matrix,
                "embedding_size": self.embedding_size,
                "hidden_units_node": self.hidden_units_node,
                "hidden_units_edge": self.hidden_units_edge,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "activation": self.activation,
                "weight_initializer": self.weight_initializer,
                "weight_regularizer": self.weight_regularizer,
                "bias_initializer": self.bias_initializer,
                "graph_layer": self.graph_layer,
            }
        )
        return config


class GraphConvLSTM(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer: str
        | keras.initializers.Initializer
        | None = "glorot_uniform",
        weight_regularizer: str | keras.regularizers.Regularizer | None = None,
        bias_initializer: str | keras.initializers.Initializer | None = "zeros",
    ):
        super(GraphConvLSTM, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.embedding_size = embedding_size
        self.hidden_units_node = hidden_units_node
        self.hidden_units_edge = hidden_units_edge
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.bias_initializer = bias_initializer
        self.graph_layer = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            batch_size, seq_len, num_nodes, num_node_features = input_shape
        else:
            batch_size, seq_len, num_nodes, num_node_features = input_shape[0]
            _, _, num_edges, num_edge_features = input_shape[1]

        self.lstm_in = keras.layers.ConvLSTM2D(
            filters=1,
            kernel_size=(1, num_node_features),
            padding="same",
            activation="relu",
            return_sequences=True,
        )

        self.lstm_out = keras.layers.ConvLSTM2D(
            filters=1,
            kernel_size=(1, self.embedding_size),
            padding="same",
            activation="relu",
            return_sequences=True,
        )

        if isinstance(input_shape, list):
            self.lstm_edge = keras.layers.ConvLSTM2D(
                filters=1,
                kernel_size=(1, num_node_features),
                padding="same",
                activation="relu",
                return_sequences=True,
            )

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            (
                batch_size,
                edge_features,
                edge_seq_len,
                node_features,
                num_edge_features,
                num_edges,
                num_node_features,
                num_nodes,
                seq_len,
            ) = init_shape_variables(inputs)

            reshaped_edge = self.compute_edge_features_temporal(
                batch_size,
                edge_features,
                edge_seq_len,
                num_edge_features,
                num_edges,
                seq_len,
            )
        else:
            node_features = inputs
            node_feature_shape = tf.shape(inputs)
            batch_size, seq_len, num_nodes, num_node_features = (
                node_feature_shape[0],
                node_feature_shape[1],
                node_feature_shape[2],
                node_feature_shape[3],
            )

        # add empty channel dimension
        lstm_input = tf.expand_dims(node_features, axis=-1)

        # remove channel dimension before graph layer
        # output dimension(batch_size, seq_len, num_nodes, num_features)
        lstm_in_output = tf.squeeze(self.lstm_in(lstm_input), axis=-1)

        # reshape to (batch_size * seq_len, num_nodes, num_features
        reshaped = tf.reshape(
            lstm_in_output, shape=(num_nodes, batch_size, seq_len, num_node_features)
        )
        reshaped = tf.transpose(reshaped, perm=[1, 2, 0, 3])
        reshaped_node = tf.reshape(reshaped, shape=(-1, num_nodes, num_node_features))
        output = (
            self.graph_layer(reshaped_node)
            if not isinstance(inputs, list)
            else self.graph_layer([reshaped_node, reshaped_edge])
        )

        # reshape to (batch_size, seq_len, num_nodes, embedding_size)
        output_reshaped = tf.reshape(
            output, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )

        # add empty channel dimension
        output_reshaped = tf.expand_dims(output_reshaped, axis=-1)
        # output dimension (batch_size, seq_len, num_nodes, embedding_size)
        output_lstm = tf.squeeze(self.lstm_out(output_reshaped), axis=-1)

        output_lstm_reshaped = tf.reshape(
            output_lstm, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )

        return output_lstm_reshaped

    def compute_edge_features_temporal(
        self,
        batch_size,
        edge_features,
        edge_seq_len,
        num_edge_features,
        num_edges,
        seq_len,
    ):
        edge_features = tf.transpose(edge_features, [2, 0, 1, 3])
        edge_features = tf.expand_dims(edge_features, axis=-1)
        lstm_edge_output = tf.squeeze(self.lstm_edge(edge_features), axis=-1)

        reshaped = tf.reshape(
            lstm_edge_output,
            shape=(num_edges, batch_size, edge_seq_len, num_edge_features),
        )
        reshaped = tf.transpose(reshaped, perm=[1, 2, 0, 3])
        reshaped_edge = tf.reshape(
            reshaped, shape=(batch_size * seq_len, num_edges, num_edge_features)
        )
        return reshaped_edge

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adjacency_matrix": self.adjacency_matrix,
                "embedding_size": self.embedding_size,
                "hidden_units_node": self.hidden_units_node,
                "hidden_units_edge": self.hidden_units_edge,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "activation": self.activation,
                "weight_initializer": self.weight_initializer,
                "weight_regularizer": self.weight_regularizer,
                "bias_initializer": self.bias_initializer,
                "graph_layer": self.graph_layer,
            }
        )
        return config


def init_shape_variables(inputs):
    node_features, edge_features = inputs
    node_feature_shape = tf.shape(node_features)
    batch_size, seq_len, num_nodes, num_node_features = (
        node_feature_shape[0],
        node_feature_shape[1],
        node_feature_shape[2],
        node_feature_shape[3],
    )
    edge_feature_shape = tf.shape(edge_features)
    edge_seq_len, num_edges, num_edge_features = (
        edge_feature_shape[1],
        edge_feature_shape[2],
        edge_feature_shape[3],
    )
    return (
        batch_size,
        edge_features,
        edge_seq_len,
        node_features,
        num_edge_features,
        num_edges,
        num_node_features,
        num_nodes,
        seq_len,
    )
