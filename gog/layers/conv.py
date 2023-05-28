import numpy as np
import tensorflow as tf
from tensorflow import keras

from gog.layers.graph_layer import GraphLayer


class GraphConvolution(GraphLayer):

    def __init__(self, adjacency_matrix: np.ndarray,
                 embedding_size,
                 hidden_units_node=None,
                 hidden_units_edge=None,
                 dropout_rate=0,
                 use_bias=True,
                 activation=None,
                 weight_initializer="glorot_uniform",
                 weight_regularizer=None,
                 bias_initializer="zeros"):
        super(GraphConvolution, self).__init__()
        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")

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
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.weight_regularizer = keras.regularizers.get(weight_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        # Degree matrix of adjacency matrix
        D = tf.zeros_like(self._A_tilde)
        D = tf.linalg.set_diag(D, tf.reduce_sum(self._A_tilde, axis=1))

        # Inverse of square root of degree matrix
        self._D_mod = tf.linalg.inv(tf.linalg.sqrtm(D))

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

        A_hat = tf.matmul(tf.matmul(self._D_mod, self._A_tilde), self._D_mod)
        masked_feature_matrix = tf.matmul(A_hat, node_features)

        output = self.node_feature_MLP(masked_feature_matrix)

        if self.activation is not None:
            output = self.activation(output)

        return output
