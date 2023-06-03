import keras.layers
import numpy as np
import tensorflow as tf

from gog.layers import GraphBase


class GraphBaseConv(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphBaseConv, self).__init__()
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

    def build(self, input_shape):
        batch_size, seq_len, num_nodes, num_features = input_shape
        self.tmp_conv1 = keras.layers.Conv2D(
            filters=seq_len,
            kernel_size=(num_features, 1),
            padding="same",
            activation="relu",
        )
        self.graph_base = GraphBase(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_edge=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )
        self.tmp_conv2 = keras.layers.Conv2D(
            filters=seq_len,
            kernel_size=(self.embedding_size, 1),
            padding="same",
            activation="relu",
        )

    def call(self, inputs, *args, **kwargs):
        _, seq_len, num_nodes, num_features = inputs.get_shape().as_list()
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        # batch_size, num_nodes, num_features, seq_len
        temporal_conv_1 = self.tmp_conv1(inputs)

        # batch_size, seq_len, num_nodes, num_features
        conv_1_transposed = tf.transpose(temporal_conv_1, perm=[0, 3, 1, 2])
        reshaped = tf.reshape(conv_1_transposed, shape=(-1, num_nodes, num_features))
        graph_output = self.graph_base(reshaped)
        graph_output_reshaped = tf.reshape(
            graph_output, shape=(-1, seq_len, num_nodes, self.embedding_size)
        )
        transposed = tf.transpose(graph_output_reshaped, perm=[0, 2, 3, 1])
        temporal_conv_2 = self.tmp_conv2(transposed)
        conv_2_transposed = tf.transpose(temporal_conv_2, perm=[0, 3, 1, 2])
        return conv_2_transposed


class GraphBaseLSTM(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphBaseLSTM, self).__init__()
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

    def build(self, input_shape):
        batch_size, seq_len, num_nodes, num_features = input_shape

        self.lstm_in = keras.layers.LSTM(
            units=num_features,
            return_sequences=True,
        )
        self.graph_layer = GraphBase(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_edge=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )
        self.lstm_out = keras.layers.LSTM(
            units=self.embedding_size,
            return_sequences=True,
        )

    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        batch_size, seq_len, num_nodes, num_features = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        lstm_input = tf.reshape(
            inputs, shape=(batch_size * num_nodes, seq_len, num_features)
        )
        lstm_in_output = self.lstm_in(lstm_input)

        # reshape to (batch_size * seq_len, num_nodes, num_features
        reshaped = tf.reshape(lstm_in_output, shape=(-1, num_nodes, num_features))
        output = self.graph_layer(reshaped)

        # reshape to (batch_size, seq_len, num_nodes, embedding_size)
        output_reshaped = tf.reshape(
            output, shape=(batch_size * num_nodes, seq_len, self.embedding_size)
        )

        # output dimension (batch_size * num_nodes, seq_len, embedding_size)
        output_lstm = self.lstm_out(output_reshaped)

        output_lstm_reshaped = tf.reshape(
            output_lstm, shape=(batch_size, seq_len, num_nodes, self.embedding_size)
        )
        return output_lstm_reshaped
