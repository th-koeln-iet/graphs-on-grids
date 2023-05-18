import numpy as np
import tensorflow as tf
from tensorflow import keras


class GAT(keras.layers.Layer):

    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, use_bias=True, activation=None,
                 weight_initializer="glorot_uniform", kernel_regularizer=None, bias_initializer="zeros"):
        super(GAT, self).__init__()

        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")

        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        rows, cols = np.where(self._A_tilde == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.embedding_size), initializer=self.weight_initializer,
            regularizer=self.kernel_regularizer, trainable=True,
            name="weight_kernel"
        )
        if self.use_bias:
            self.b = self.add_weight(shape=(self.embedding_size,), initializer=self.bias_initializer,
                                     regularizer=self.kernel_regularizer, trainable=True,
                                     name="bias_vector")
        self.W_attn = self.add_weight(shape=(2 * self.embedding_size, 1),
                                      initializer=self.weight_initializer, regularizer=self.kernel_regularizer,
                                      trainable=True, name="attention_kernel")

    def call(self, inputs, *args, **kwargs):
        # Linearly transform node states
        node_states_transformed = tf.matmul(inputs, self.W)
        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, self.edges, axis=1)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(inputs)[0], tf.shape(self.edges)[0], 2 * self.embedding_size)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.W_attn)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores[0, :],
            segment_ids=self.edges[:, 0],
            num_segments=tf.reduce_max(self.edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(self.edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum
        # (3) apply attention scores and aggregate
        # first attention score can be taken since the attention values are equal for the whole batch
        # write normalized attention scores to position where adjacency_matrix equals one
        weighted_adj = tf.tensor_scatter_nd_update(self._A_tilde, self.edges, attention_scores_norm[0])
        output = tf.matmul(weighted_adj, node_states_transformed)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)

        if self.activation is not None:
            output = self.activation(output)

        return output
