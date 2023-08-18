import numpy as np
import tensorflow as tf
from tensorflow import keras

from graphs_on_grids.layers.graph_layer import GraphLayer


class GraphAttention(GraphLayer):
    r"""
    Graph attention layer as shown in the [original paper](https://arxiv.org/pdf/1710.10903.pdf)

    $$
        \textbf{H}^{(t+1)} = \sigma \biggl( \tilde{A_\alpha} H^{(t)}W^{(t)}\biggr)
    $$ where \( \tilde{A_\alpha} \) is the adjacency matrix weighted by the attention scores \(\alpha\)
    and \(\alpha\) is computed by:
    $$
        \mathbf{\alpha}_{ij} =\frac{ \exp\left(\mathrm{LeakyReLU}\left(
        a^{\top} [(XW)_i \, \| \, (XW)_j]\right)\right)}{\sum\limits_{k
        \in \mathcal{N}(i) \cup \{ i \}} \exp\left(\mathrm{LeakyReLU}\left(
        a^{\top} [(XW)_i \, \| \, (XW)_k]\right)\right)}
    $$ for each node pair \((i,j)\) where \(a \in \mathbb{R}^{2F'}\) is a trainable attention kernel.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size: int,
        hidden_units_node: list | tuple = None,
        hidden_units_attention: list | tuple = None,
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
        :param hidden_units_attention: list or tuple of neuron counts in the hidden layers used in the MLP for
        computing attention scores
        :param dropout_rate: The dropout rate used after each dense layer in the node- or edge-MLPs
        :param use_bias: Whether to use bias in the hidden layers in the node- and edge-MLPs
        :param activation: Activation function to be used within the layer
        :param weight_initializer: Weight initializer to be used within the layer
        :param weight_regularizer: Weight regularizer to be used within the layer
        :param bias_initializer: Bias initializer to be used within the layer
        """
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
            node_states_expanded = self.combine_node_edge_features(
                edge_features, node_features, node_states_expanded
            )

        attention_scores = self.attention_mlp(node_states_expanded)
        attention_scores = tf.squeeze(attention_scores, -1)

        # Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=tf.reduce_mean(attention_scores, axis=0),
            segment_ids=self.edges[:, 0],
            num_segments=tf.reduce_max(self.edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(self.edges[:, 0], tf.int32))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # apply attention scores and aggregate
        # attention scores are averaged for the whole batch
        # write normalized attention scores to position where adjacency_matrix equals one
        weighted_adj = tf.tensor_scatter_nd_update(
            self._A_tilde, self.edges, tf.reduce_mean(attention_scores_norm, axis=0)
        )
        output = tf.matmul(weighted_adj, node_states_transformed)

        if self.activation is not None:
            output = self.activation(output)

        return output

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


class MultiHeadGraphAttention(GraphLayer):
    r"""
    Multi-head graph attention layer as shown in the [original paper](https://arxiv.org/pdf/1710.10903.pdf)

    Computes `num_heads` independent graph attention layers and combines them by concatenation or averaging
    depending on the `concat_heads` parameter.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size: int,
        hidden_units_node: list | tuple = None,
        hidden_units_attention: list | tuple = None,
        dropout_rate: int | float = 0,
        num_heads: int = 3,
        use_bias: bool = True,
        activation: str | None = None,
        weight_initializer: str
        | keras.initializers.Initializer
        | None = "glorot_uniform",
        weight_regularizer: str | keras.regularizers.Regularizer | None = None,
        bias_initializer: str | keras.initializers.Initializer | None = "zeros",
        concat_heads: bool = True,
    ):
        """
        :param adjacency_matrix: adjacency matrix of the graphs to be passed to the model
        :param embedding_size: the output dimensionality of the node feature vector
        :param hidden_units_node: list or tuple of neuron counts in the hidden layers used in the MLP for processing
        node features
        :param hidden_units_attention: list or tuple of neuron counts in the hidden layers used in the MLP for
        computing attention scores
        :param dropout_rate: The dropout rate used after each dense layer in the node- or edge-MLPs
        :param num_heads: Number of independent attention heads
        :param use_bias: Whether to use bias in the hidden layers in the node- and edge-MLPs
        :param activation: Activation function to be used within the layer
        :param weight_initializer: Weight initializer to be used within the layer
        :param weight_regularizer: Weight regularizer to be used within the layer
        :param bias_initializer: Bias initializer to be used within the layer
        :param concat_heads: Whether to concatenate (True) results from the attention heads or average (False) them.
        """
        super(MultiHeadGraphAttention, self).__init__(
            adjacency_matrix=adjacency_matrix,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_attention,
            embedding_size=embedding_size,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )
        self.concat_heads = concat_heads
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        # do not create weights for Sequential node feature MLP
        self.node_feature_MLP = None
        self.attention_layers = [
            GraphAttention(
                adjacency_matrix=adjacency_matrix,
                hidden_units_node=hidden_units_node,
                hidden_units_attention=hidden_units_attention,
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

    def get_config(self):
        config = super().get_config()
        config["num_heads"] = self.num_heads
        config["concat_heads"] = self.concat_heads
        return config
