import numpy as np
import tensorflow as tf
from tensorflow import keras


class GraphBase(keras.layers.Layer):
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
        self.hidden_units_node = [embedding_size] if hidden_units_node is None else hidden_units_node
        self.hidden_units_edge = [embedding_size] if hidden_units_edge is None else hidden_units_edge

        self.dropout_rate = dropout_rate
        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.aggregation_method = aggregation_method
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.weight_regularizer = keras.regularizers.get(weight_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.mlp_layer_index = 0

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        # Degree vector of adjacency matrix
        self._D = tf.reduce_sum(self._A_tilde, axis=1)

        # Edge list
        rows, cols = np.where(self.adjacency_matrix == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

        self.node_feature_MLP = self.create_MLP()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1] if len(input_shape) == 2 else input_shape[-1], self.embedding_size),
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer, trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(shape=(self.embedding_size,), initializer=self.bias_initializer,
                                     regularizer=self.weight_regularizer, trainable=True)

    def call(self, inputs, *args, **kwargs):
        # if edge features are present
        if len(inputs) == 2:
            node_features, edge_features = inputs
            # self.adjacency_matrix = transform_adjacency_matrix(self.MLP, self.adjacency_matrix, self.edges,
            #                                                    node_features, edge_features)
            gather = tf.gather(node_features, self.edges, axis=1)
            node_feature_shape = tf.shape(node_features)
            reshape = tf.reshape(gather, (node_feature_shape[0], tf.shape(self.edges)[0], 2 * node_feature_shape[2]))
            node_node_edge_features = tf.concat([reshape, edge_features], axis=2)

            edge_weights = self.node_feature_MLP(node_node_edge_features)

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

        output = tf.matmul(masked_feature_matrix, self.W)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def create_MLP(self):
        self.edge_mlp_layers = []
        for n_neurons in self.hidden_units_node:
            setattr(self, f"dense_{self.mlp_layer_index}",
                    keras.layers.Dense(n_neurons, use_bias=self.use_bias, kernel_initializer=self.weight_initializer,
                                       bias_initializer=self.bias_initializer,
                                       kernel_regularizer=self.weight_regularizer))
            self.edge_mlp_layers.append(getattr(self, f"dense_{self.mlp_layer_index}"))

            setattr(self, f"batchNorm_{self.mlp_layer_index}", keras.layers.BatchNormalization())
            self.edge_mlp_layers.append(getattr(self, f"batchNorm_{self.mlp_layer_index}"))

            setattr(self, f"dropout_{self.mlp_layer_index}", keras.layers.Dropout(self.dropout_rate))
            self.edge_mlp_layers.append(getattr(self, f"dropout_{self.mlp_layer_index}"))

            setattr(self, f"relu_{self.mlp_layer_index}", keras.layers.ReLU())
            self.edge_mlp_layers.append(getattr(self, f"relu_{self.mlp_layer_index}"))
            self.mlp_layer_index += 1
        self.edge_dense_out = keras.layers.Dense(1, activation=keras.activations.sigmoid)
        self.edge_mlp_layers.append(self.edge_dense_out)
        return keras.Sequential(self.edge_mlp_layers)
