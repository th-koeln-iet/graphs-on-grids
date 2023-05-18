import numpy as np
import tensorflow as tf
from tensorflow import keras


class GraphAttention(keras.layers.Layer):

    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, use_bias=True, activation=None,
                 weight_initializer="glorot_uniform", weight_regularizer=None, bias_initializer="zeros"):
        super(GraphAttention, self).__init__()

        self.embedding_size = int(embedding_size) if not isinstance(embedding_size, int) else embedding_size
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}")

        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.weight_regularizer = keras.regularizers.get(weight_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.math.add(self.adjacency_matrix, tf.eye(self.adjacency_matrix.shape[0]))

        rows, cols = np.where(self._A_tilde == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.embedding_size), initializer=self.weight_initializer,
            regularizer=self.weight_regularizer, trainable=True,
            name="weight_kernel"
        )
        if self.use_bias:
            self.b = self.add_weight(shape=(self.embedding_size,), initializer=self.bias_initializer,
                                     regularizer=self.weight_regularizer, trainable=True,
                                     name="bias_vector")
        self.W_attn = self.add_weight(shape=(2 * self.embedding_size, 1),
                                      initializer=self.weight_initializer, regularizer=self.weight_regularizer,
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
            data=attention_scores[-1, :],
            segment_ids=self.edges[:, 0],
            num_segments=tf.reduce_max(self.edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(self.edges[:, 0], tf.int32))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) apply attention scores and aggregate
        # last attention score can be taken since the attention values are equal for the whole batch
        # write normalized attention scores to position where adjacency_matrix equals one
        weighted_adj = tf.tensor_scatter_nd_update(self._A_tilde, self.edges, attention_scores_norm[-1])
        output = tf.matmul(weighted_adj, node_states_transformed)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)

        if self.activation is not None:
            output = self.activation(output)

        return output


class MultiHeadGraphAttention(keras.layers.Layer):
    def __init__(self, adjacency_matrix: np.ndarray, embedding_size, num_heads=3, use_bias=True, activation=None,
                 weight_initializer="glorot_uniform", weight_regularizer=None, bias_initializer="zeros",
                 concat_heads=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.concat_heads = concat_heads
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.attention_layers = [
            GraphAttention(adjacency_matrix=adjacency_matrix, embedding_size=embedding_size, use_bias=use_bias,
                           activation=activation, weight_initializer=weight_initializer,
                           weight_regularizer=weight_regularizer, bias_initializer=bias_initializer)
            for _ in range(num_heads)]

    def call(self, inputs, *args, **kwargs):

        # Obtain outputs from each attention head
        outputs = [
            attention_layer(inputs) for attention_layer in self.attention_layers
        ]

        # Concatenate or average the node states from each head
        if self.concat_heads:
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return tf.nn.relu(outputs)
