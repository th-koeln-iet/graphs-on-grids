import numpy as np
import tensorflow as tf
from tensorflow import keras


class GNN(keras.layers.Layer):
    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, use_bias=True, activation=None,
                 neighborhood_normalization=False, weight_initializer="glorot_uniform", bias_initializer="zeros"):
        super(GNN, self).__init__()
        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")

        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.neighborhood_normalization = neighborhood_normalization
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_hat = tf.math.add(self.adjacency_matrix, np.identity(self.adjacency_matrix.shape[0]))
        # Degree vector of adjacency matrix
        self._D = tf.reduce_sum(self._A_hat, axis=1)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.embedding_size), initializer=self.weight_initializer, trainable=True
        )
        self.b = self.add_weight(shape=(self.embedding_size,), initializer=self.bias_initializer, trainable=True)

    def call(self, inputs, *args, **kwargs):
        masked_feature_matrix = tf.matmul(self._A_hat, inputs)
        if self.neighborhood_normalization:
            masked_feature_matrix = tf.divide(masked_feature_matrix, self._D[:, None])
        output = tf.matmul(masked_feature_matrix, self.W)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)

        if self.activation is not None:
            output = self.activation(output)

        return output
