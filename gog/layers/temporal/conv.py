import numpy as np

from gog.layers import GraphConvolution
from gog.layers.temporal.temporal_layer import (
    TemporalConv,
    GraphLSTM,
    GraphGRU,
    GraphConvLSTM,
)


class GraphConvTemporalConv(TemporalConv):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        output_seq_len,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphConvTemporalConv, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            output_seq_len=output_seq_len,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_edge,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.graph_layer = GraphConvolution(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_edge=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )


class GraphConvolutionLSTM(GraphLSTM):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        output_seq_len,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphConvolutionLSTM, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            output_seq_len=output_seq_len,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_edge,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.graph_layer = GraphConvolution(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_edge=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )


class GraphConvolutionGRU(GraphGRU):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        output_seq_len,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphConvolutionGRU, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            output_seq_len=output_seq_len,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_edge,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.graph_layer = GraphConvolution(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_edge=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )


class GraphConvolutionConvLSTM(GraphConvLSTM):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        output_seq_len,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
        super(GraphConvolutionConvLSTM, self).__init__(
            adjacency_matrix=adjacency_matrix,
            embedding_size=embedding_size,
            output_seq_len=output_seq_len,
            hidden_units_node=hidden_units_node,
            hidden_units_edge=hidden_units_edge,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            activation=activation,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer,
            bias_initializer=bias_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.graph_layer = GraphConvolution(
            adjacency_matrix=self.adjacency_matrix,
            embedding_size=self.embedding_size,
            hidden_units_node=self.hidden_units_node,
            hidden_units_edge=self.hidden_units_edge,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            activation=self.activation,
            weight_initializer=self.weight_initializer,
            weight_regularizer=self.weight_regularizer,
            bias_initializer=self.bias_initializer,
        )
