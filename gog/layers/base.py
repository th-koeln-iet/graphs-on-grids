import numpy as np
import tensorflow as tf
from tensorflow import keras

from gog.layers.graph_layer import GraphLayer


class GraphBase(GraphLayer):
    def __init__(self, adjacency_matrix: np.ndarray,
                 embedding_size,
                 hidden_units_node=None,
                 hidden_units_edge=None,
                 dropout_rate=0,
                 use_bias=True,
                 activation=None,
                 aggregation_method="sum",
                 weight_initializer="glorot_uniform",
                 weight_regularizer=None,
                 bias_initializer="zeros"):

        super(GraphBase, self).__init__()
        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")
        if aggregation_method != "sum" and aggregation_method != "mean":
            raise ValueError(
                f"Received invalid aggregation method={aggregation_method}. Valid options are 'sum' or 'mean'.")
        if not isinstance(hidden_units_node, (list, type(None))) or not isinstance(hidden_units_edge,
                                                                                   (list, type(None))):
            raise ValueError(
                f"Received invalid type for hidden units parameters. Hidden units need to be of type 'list'")

        self.hidden_units_node = [] if hidden_units_node is None else hidden_units_node
        self.hidden_units_edge = [] if hidden_units_edge is None else hidden_units_edge

        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(
                f"Received invalid value for dropout_rate parameter. Only values between 0 and 1 are valid")
        self.dropout_rate = dropout_rate
        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.aggregation_method = aggregation_method
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.weight_regularizer = keras.regularizers.get(weight_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        # Degree vector of adjacency matrix
        self._D = tf.reduce_sum(self._A_tilde, axis=1)

        # Edge list
        rows, cols = np.where(self.adjacency_matrix == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

        self.node_feature_MLP = self.create_node_mlp()

    def build(self, input_shape):
        if len(input_shape) == 2:
            self.edge_feature_MLP = self.create_edge_mlp()

    def call(self, inputs, *args, **kwargs):
        # if edge features are present
        if type(inputs) == list:
            node_features, edge_features = inputs

            gather = tf.gather(node_features, self.edges, axis=1)
            node_feature_shape = tf.shape(node_features)
            reshape = tf.reshape(gather,
                                 (node_feature_shape[0], tf.shape(self.edges)[0], 2 * node_feature_shape[2]))
            node_node_edge_features = tf.concat([reshape, edge_features], axis=2)

            edge_weights = self.edge_feature_MLP(node_node_edge_features)

            edge_weights = tf.squeeze(edge_weights)

            # calculate mean edge weights over batch
            edge_weights_avg = tf.math.reduce_mean(edge_weights, axis=0)
            weighted_adj_matrix = tf.tensor_scatter_nd_update(self.adjacency_matrix, self.edges,
                                                              edge_weights_avg)
            self._A_tilde = tf.math.add(weighted_adj_matrix, tf.eye(self.adjacency_matrix.shape[0]))
        else:
            node_features = inputs

        masked_feature_matrix = tf.matmul(self._A_tilde, node_features)

        if self.aggregation_method == "mean":
            masked_feature_matrix = tf.divide(masked_feature_matrix, self._D[:, None])

        output = self.node_feature_MLP(masked_feature_matrix)

        if self.activation is not None:
            output = self.activation(output)

        return output
