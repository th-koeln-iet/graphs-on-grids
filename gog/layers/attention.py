import numpy as np
import tensorflow as tf
from tensorflow import keras

from gog.layers.graph_layer import GraphLayer


class GraphAttention(GraphLayer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_attention=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphAttention, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_attention,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

        self.edge_feature_indices = self.calculate_edge_feature_indices()

        rows, cols = np.where(self._A_tilde == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

    def build(self, input_shape):
        self.attention_mlp = self.create_attention_mlp()

    def call(self, inputs, *args, **kwargs):
        # if edge features are present
        if type(inputs) == list:
            node_features, edge_features = inputs
        else:
            edge_features = None
            node_features = inputs

        # Update node embeddings through MLP
        node_states_transformed = self.node_feature_MLP(node_features)

        # Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, self.edges, axis=1)
        node_states_expanded = tf.reshape(
            node_states_expanded,
            (
                tf.shape(node_features)[0],
                tf.shape(self.edges)[0],
                2 * self.embedding_size,
            ),
        )

        if type(inputs) == list:
            node_states_expanded = self.append_edge_features(
                edge_features, node_features, node_states_expanded
            )

        attention_scores = self.attention_mlp(node_states_expanded)
        attention_scores = tf.squeeze(attention_scores, -1)

        # Normalize attention scores
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

        # apply attention scores and aggregate
        # last attention score can be taken since the attention values are equal for the whole batch
        # write normalized attention scores to position where adjacency_matrix equals one
        weighted_adj = tf.tensor_scatter_nd_update(
            self._A_tilde, self.edges, attention_scores_norm[-1]
        )
        output = tf.matmul(weighted_adj, node_states_transformed)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def append_edge_features(self, edge_features, node_features, node_states_expanded):
        edge_feature_shape = tf.shape(edge_features)
        # zero tensor of shape (Batch_size, |E| + |V|, |X_e|)
        zeros_edge_feature_matrix = tf.zeros(
            shape=(
                edge_feature_shape[0],
                edge_feature_shape[1] + tf.shape(node_features)[1],
                edge_feature_shape[2],
            ),
            dtype=tf.float32,
        )
        # computed edge positions in 'zeros_edge_feature_matrix'
        edge_feature_indices = tf.expand_dims(self.edge_feature_indices, axis=0)
        # repeated edge positions for batch computation
        batched_edge_feature_indices = tf.repeat(
            edge_feature_indices, edge_feature_shape[0], axis=0
        )
        # tensor containing batch indices, shape: (1, batch_size * |E|)
        batch_index_list = tf.expand_dims(
            tf.repeat(
                tf.range(edge_feature_shape[0], dtype=tf.int32),
                tf.shape(edge_feature_indices)[1],
            ),
            axis=0,
        )
        batch_index_list = tf.reshape(batch_index_list, (edge_feature_shape[0], -1))
        # reshaped to (batch_size, |E|, 1)
        batch_index_list = tf.expand_dims(batch_index_list, axis=2)
        # indices for update operation with shape (batch_size, |E|, 2).
        # Contains pairs of [batch_number, index_to_update]
        edge_feature_indices = tf.squeeze(
            tf.concat([batch_index_list, batched_edge_feature_indices], axis=2)
        )
        # batched update of zero tensor with edge features
        edge_features = tf.tensor_scatter_nd_update(
            tensor=zeros_edge_feature_matrix,
            indices=edge_feature_indices,
            updates=edge_features,
        )
        # contains concatenation of neighbor node pair features and corresponding edge features
        # if a pair contains a self-loop, the edge feature vector is zeroed.
        # Shape (batch_size, |E| + |V|, 2 * |X_v| + |E|)
        node_states_expanded = tf.concat([node_states_expanded, edge_features], axis=2)
        return node_states_expanded

    def calculate_edge_feature_indices(self):
        """
        Calculates position of edge features in list of all edges. This is necessary since self-edges are used for
        attention but cannot contain any edge features.
        :return: list of edge indices in list containing edges and self-loops
        """
        idx_list = []
        edge_index = 0
        adj = self._A_tilde.numpy()
        for i, j in np.ndindex(adj.shape):
            if i == j and adj[i, j] == 1:
                edge_index += 1
            if i != j and adj[i, j] == 1:
                idx_list.append([edge_index])
                edge_index += 1
        return tf.convert_to_tensor(idx_list, dtype=tf.int32)

    def create_attention_mlp(self):
        self.attention_mlp_layers = []
        self.attention_mlp_layers = self.create_hidden_layers(
            self.hidden_units_edge, False
        )
        self.attention_dense_out = keras.layers.Dense(
            1, activation=keras.layers.LeakyReLU(0.2)
        )
        self.attention_mlp_layers.append(self.attention_dense_out)
        return keras.Sequential(
            self.attention_mlp_layers, name="sequential_attention_scores"
        )


class MultiHeadGraphAttention(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        dropout_rate=0,
        num_heads=3,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
        concat_heads=True,
    ):
        super(MultiHeadGraphAttention, self).__init__()
        self.hidden_units_node = hidden_units_node
        self.concat_heads = concat_heads
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.attention_layers = [
            GraphAttention(
                adjacency_matrix=adjacency_matrix,
                hidden_units_node=None,
                embedding_size=embedding_size,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
                activation=activation,
                weight_initializer=weight_initializer,
                weight_regularizer=weight_regularizer,
                bias_initializer=bias_initializer,
            )
            for _ in range(num_heads)
        ]

    def call(self, inputs, *args, **kwargs):
        # Obtain outputs from each attention head
        outputs = [attention_layer(inputs) for attention_layer in self.attention_layers]

        # Concatenate or average the node states from each head
        if self.concat_heads:
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
