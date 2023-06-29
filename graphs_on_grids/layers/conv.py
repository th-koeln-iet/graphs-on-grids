import numpy as np
import tensorflow as tf
from tensorflow import keras

from graphs_on_grids.layers.graph_layer import GraphLayer


class GraphConvolution(GraphLayer):
    r"""
    Graph convolution layer as shown in the [original paper](http://arxiv.org/abs/1609.02907)

    $$
        \textbf{H}^{(t+1)} = \sigma \biggl( \tilde{D}^{-{1\over2}}\tilde{A}\tilde{D}^{-{1\over2}} H^{(t)}W^{(t)}\biggr)
    $$ where where \( \hat{A} = A + I \) is the adjacency matrix with added self-loops
    and \( \tilde{D} \) is its degree matrix.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size: int,
        hidden_units_node: list | tuple = None,
        hidden_units_edge: list | tuple = None,
        dropout_rate: int | float = 0,
        use_bias: bool = True,
        activation: str | None = None,
        weight_initializer: str
        | keras.initializers.Initializer
        | None = "glorot_uniform",
        weight_regularizer: str | keras.regularizers.Regularizer | None = None,
        bias_initializer: str | keras.initializers.Initializer | None = "zeros",
    ):
        """
        :param adjacency_matrix: adjacency matrix of the graphs to be passed to the model
        :param embedding_size: the output dimensionality of the node feature vector
        :param hidden_units_node: list or tuple of neuron counts in the hidden layers used in the MLP for processing
        node features
        :param hidden_units_edge: list or tuple of neuron counts in the hidden layers used in the MLP for processing
        edge features
        :param dropout_rate: The dropout rate used after each dense layer in the node- or edge-MLPs
        :param use_bias: Whether to use bias in the hidden layers in the node- and edge-MLPs
        :param activation: Activation function to be used within the layer
        :param weight_initializer: Weight initializer to be used within the layer
        :param weight_regularizer: Weight regularizer to be used within the layer
        :param bias_initializer: Bias initializer to be used within the layer
        """
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
            node_features_expanded = tf.reshape(
                gather,
                (
                    node_feature_shape[0],
                    tf.shape(self.edges)[0],
                    2 * node_feature_shape[2],
                ),
            )
            node_node_edge_features = self.combine_node_edge_features(
                edge_features, node_features, node_features_expanded
            )

            edge_weights = self.edge_feature_MLP(node_node_edge_features)

            edge_weights = tf.squeeze(edge_weights)

            # calculate mean edge weights over batch
            edge_weights_avg = tf.math.reduce_mean(edge_weights, axis=0)
            weighted_adj_matrix = tf.tensor_scatter_nd_update(
                self.adjacency_matrix, self.edges, edge_weights_avg
            )
            self._A_tilde = weighted_adj_matrix
        else:
            node_features = inputs

        A_hat = tf.matmul(tf.matmul(self._D_mod, self._A_tilde), self._D_mod)
        masked_feature_matrix = tf.matmul(A_hat, node_features)

        output = self.node_feature_MLP(masked_feature_matrix)

        if self.activation is not None:
            output = self.activation(output)

        return output
