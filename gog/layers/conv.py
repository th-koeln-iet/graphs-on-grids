import numpy as np
import tensorflow as tf
from tensorflow import keras


class GCN(keras.layers.Layer):

    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, use_bias=True, activation=None,
                 weight_initializer="glorot_uniform", bias_initializer="zeros"):
        super(GCN, self).__init__()
        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")

        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        # Degree matrix of adjacency matrix
        D = tf.zeros_like(self._A_tilde)
        D = tf.linalg.set_diag(D, tf.reduce_sum(self._A_tilde, axis=1))

        # Inverse of square root of degree matrix
        self._D_mod = tf.linalg.inv(tf.linalg.sqrtm(D))

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.embedding_size), initializer=self.weight_initializer, trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(shape=(self.embedding_size,), initializer=self.bias_initializer, trainable=True)

    def call(self, inputs, *args, **kwargs):
        A_hat = tf.matmul(tf.matmul(self._D_mod, self._A_tilde), self._D_mod)
        masked_feature_matrix = tf.matmul(A_hat, inputs)

        output = tf.matmul(masked_feature_matrix, self.W)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)

        if self.activation is not None:
            output = self.activation(output)

        return output
