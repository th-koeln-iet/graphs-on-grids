import numpy as np
import tensorflow as tf

from gog.layers.graph_layer import GraphLayer


class GraphConvolution(GraphLayer):
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
        super(GraphConvolution, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_edge,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

        # Degree matrix of adjacency matrix
        D = tf.zeros_like(self._A_tilde)
        D = tf.linalg.set_diag(D, tf.reduce_sum(self._A_tilde, axis=1))

        # Inverse of square root of degree matrix
        self._D_mod = tf.linalg.inv(tf.linalg.sqrtm(D))

    def build(self, input_shape):
        if len(input_shape) == 2:
            self.edge_feature_MLP = self.create_edge_mlp()

    def call(self, inputs, *args, **kwargs):
        # if edge features are present
        if type(inputs) == list:
            node_features, edge_features = inputs

            gather = tf.gather(node_features, self.edges, axis=1)
            node_feature_shape = tf.shape(node_features)
            reshape = tf.reshape(
                gather,
                (
                    node_feature_shape[0],
                    tf.shape(self.edges)[0],
                    2 * node_feature_shape[2],
                ),
            )
            node_node_edge_features = tf.concat([reshape, edge_features], axis=2)

            edge_weights = self.edge_feature_MLP(node_node_edge_features)

            edge_weights = tf.squeeze(edge_weights)

            # calculate mean edge weights over batch
            edge_weights_avg = tf.math.reduce_mean(edge_weights, axis=0)
            weighted_adj_matrix = tf.tensor_scatter_nd_update(
                self.adjacency_matrix, self.edges, edge_weights_avg
            )
            self._A_tilde = tf.math.add(
                weighted_adj_matrix, tf.eye(self.adjacency_matrix.shape[0])
            )
        else:
            node_features = inputs

        A_hat = tf.matmul(tf.matmul(self._D_mod, self._A_tilde), self._D_mod)
        masked_feature_matrix = tf.matmul(A_hat, node_features)

        output = self.node_feature_MLP(masked_feature_matrix)

        if self.activation is not None:
            output = self.activation(output)

        return output
