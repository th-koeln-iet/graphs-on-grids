import numpy as np
import tensorflow as tf
from tensorflow import keras


class GNN(keras.layers.Layer):
    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, use_bias=True, activation=None,
                 aggregation_method="sum", weight_initializer="glorot_uniform", bias_initializer="zeros"):
        super(GNN, self).__init__()
        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")
        if aggregation_method != "sum" and aggregation_method != "mean":
            raise ValueError(
                f"Received invalid aggregation method={aggregation_method}. Valid options are 'sum' or 'mean'.")
        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.aggregation_method = aggregation_method
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        # Degree vector of adjacency matrix
        self._D = tf.reduce_sum(self._A_tilde, axis=1)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.embedding_size), initializer=self.weight_initializer, trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(shape=(self.embedding_size,), initializer=self.bias_initializer, trainable=True)

    def call(self, inputs, *args, **kwargs):
        masked_feature_matrix = tf.matmul(self._A_tilde, inputs)

        if self.aggregation_method == "mean":
            masked_feature_matrix = tf.divide(masked_feature_matrix, self._D[:, None])

        output = tf.matmul(masked_feature_matrix, self.W)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)

        if self.activation is not None:
            output = self.activation(output)

        return output
