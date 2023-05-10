import numpy as np
import tensorflow as tf
from tensorflow import keras


class GNN(keras.layers.Layer):
    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, num_nodes):
        super(GNN, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.embedding_size = embedding_size
        self.num_nodes = num_nodes

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.embedding_size), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(self.embedding_size,), initializer="zeros", trainable=True)

    def call(self, inputs, *args, **kwargs):
        A_hat = tf.math.add(self.adjacency_matrix, np.identity(self.adjacency_matrix.shape[0]))
        masked_feature_matrix = tf.matmul(A_hat, inputs)
        output = tf.matmul(masked_feature_matrix, self.W)
        return tf.nn.bias_add(output, self.b)
