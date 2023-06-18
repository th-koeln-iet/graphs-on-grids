import numpy as np
import tensorflow as tf
from tensorflow import keras


class GraphLayer(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size: int,
        hidden_units_node: list | tuple = None,
        hidden_units_edge: list | tuple = None,
        dropout_rate: int | float = 0,
        use_bias: bool = True,
        activation: str | None = None,
        weight_initializer: str | None = "glorot_uniform",
        weight_regularizer: str | None = None,
        bias_initializer: str | None = "zeros",
    ) -> None:
        """
        :param adjacency_matrix: adjacency matrix of the graphs to be passed to the model
        :param embedding_size: the output dimensionality of the node feature vector
        :param hidden_units_node: list or tuple of neuron counts in the hidden layers used in the MLP for processing
        node features
        :param hidden_units_edge: list or tuple of neuron counts in the hidden layers used in the MLP for processing
        edge features
        :param dropout_rate: The dropout rate used after each dense layer in the node- or edge-MLPs
        :param use_bias: Whether to use bias in the hidden layers in the node- and edge-MLPs
        :param activation: Activation function to be used within the layer
        :param weight_initializer: Weight initializer to be used within the layer
        :param weight_regularizer: Weight regularizer to be used within the layer
        :param bias_initializer: Bias initializer to be used within the layer
        """
        super(GraphLayer, self).__init__()
        self.mlp_layer_index = 0
        self.embedding_size = (
            int(embedding_size)
            if not isinstance(embedding_size, int)
            else embedding_size
        )
        if embedding_size <= 0:
            raise ValueError(
                f"Received invalid embedding_size, expected positive integer. Received embedding_size={embedding_size}"
            )
        if not isinstance(
            hidden_units_node, (list, tuple, type(None))
        ) or not isinstance(hidden_units_edge, (list, tuple, type(None))):
            raise ValueError(
                f"Received invalid type for hidden units parameters. Hidden units need to be of type 'list'"
            )

        self.hidden_units_node = [] if hidden_units_node is None else hidden_units_node
        self.hidden_units_edge = [] if hidden_units_edge is None else hidden_units_edge

        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(
                f"Received invalid value for dropout_rate parameter. Only values between 0 and 1 are valid"
            )
        self.dropout_rate = dropout_rate

        if not np.allclose(
            adjacency_matrix, adjacency_matrix.T, rtol=1e-05, atol=1e-08
        ):
            raise ValueError(f"Expected symmetric adjacency matrix.")
        self.adjacency_matrix = adjacency_matrix
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.weight_regularizer = keras.regularizers.get(weight_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Adjacency matrix with self loops
        self._A_tilde = tf.linalg.set_diag(
            self.adjacency_matrix, tf.ones(self.adjacency_matrix.shape[0])
        )

        # Edge list
        rows, cols = np.where(self._A_tilde == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

        self.node_feature_MLP = self.create_node_mlp()
        self.edge_feature_indices = self.calculate_edge_feature_indices()

    def create_edge_mlp(self):
        self.edge_mlp_layers = []
        self.edge_mlp_layers = self.create_hidden_layers(self.hidden_units_edge)
        self.edge_dense_out = keras.layers.Dense(
            1, activation=keras.activations.sigmoid
        )
        self.edge_mlp_layers.append(self.edge_dense_out)
        return keras.Sequential(self.edge_mlp_layers, name="sequential_edge_features")

    def create_node_mlp(self):
        self.node_mlp_layers = []
        self.node_mlp_layers = self.create_hidden_layers(self.hidden_units_node)
        self.node_dense_out = keras.layers.Dense(self.embedding_size)
        self.node_mlp_layers.append(self.node_dense_out)
        return keras.Sequential(self.node_mlp_layers, name="sequential_node_features")

    def create_hidden_layers(self, hidden_units, use_batch_norm=True):
        mlp_layers = []
        for n_neurons in hidden_units:
            setattr(
                self,
                f"dense_{self.mlp_layer_index}",
                keras.layers.Dense(
                    n_neurons,
                    use_bias=self.use_bias,
                    kernel_initializer=self.weight_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.weight_regularizer,
                ),
            )
            mlp_layers.append(getattr(self, f"dense_{self.mlp_layer_index}"))

            if use_batch_norm:
                setattr(
                    self,
                    f"batchNorm_{self.mlp_layer_index}",
                    keras.layers.BatchNormalization(),
                )
                mlp_layers.append(getattr(self, f"batchNorm_{self.mlp_layer_index}"))

            setattr(
                self,
                f"dropout_{self.mlp_layer_index}",
                keras.layers.Dropout(self.dropout_rate),
            )
            mlp_layers.append(getattr(self, f"dropout_{self.mlp_layer_index}"))

            setattr(self, f"relu_{self.mlp_layer_index}", keras.layers.ReLU())
            mlp_layers.append(getattr(self, f"relu_{self.mlp_layer_index}"))
            self.mlp_layer_index += 1
        return mlp_layers

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

    def combine_node_edge_features(
        self, edge_features, node_features, node_states_expanded
    ):
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
