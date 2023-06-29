import numpy as np

from tensorflow import keras
from graphs_on_grids.layers import GraphAttention
from graphs_on_grids.layers.temporal.temporal_layer import (
    TemporalConv,
    GraphLSTM,
    GraphGRU,
    GraphConvLSTM,
)


class GraphAttentionTemporalConv(TemporalConv):
    """
    Implementation of a `TemporalConv` layer using a `GraphAttention` layer as graph layer.
    See the [documentation](/usage/temporal_layers) on how temporal layers work.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size: int,
        output_seq_len: int,
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
        :param output_seq_len: number of graphs in the output sequence
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
        super(GraphAttentionTemporalConv, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            output_seq_len=output_seq_len,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_attention,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.graph_layer = GraphAttention(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_attention=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )


class GraphAttentionLSTM(GraphLSTM):
    """
    Implementation of a `GraphLSTM` layer using a `GraphAttention` layer as graph layer.
    See the [documentation](/usage/temporal_layers) on how temporal layers work.
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
        super(GraphAttentionLSTM, self).__init__(
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

    def build(self, input_shape):
        super().build(input_shape)
        self.graph_layer = GraphAttention(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_attention=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )


class GraphAttentionGRU(GraphGRU):
    """
    Implementation of a `GraphGRU` layer using a `GraphAttention` layer as graph layer.
    See the [documentation](/usage/temporal_layers) on how temporal layers work.
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
        super(GraphAttentionGRU, self).__init__(
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

    def build(self, input_shape):
        super().build(input_shape)
        self.graph_layer = GraphAttention(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_attention=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )


class GraphAttentionConvLSTM(GraphConvLSTM):
    """
    Implementation of a `GraphConvLSTM` layer using a `GraphAttention` layer as graph layer.
    See the [documentation](/usage/temporal_layers) on how temporal layers work.
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
        super(GraphAttentionConvLSTM, self).__init__(
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

    def build(self, input_shape):
        super().build(input_shape)
        self.graph_layer = GraphAttention(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_attention=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )
