import numpy as np
import tensorflow as tf
from tensorflow import keras


class GraphLayer(keras.layers.Layer):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        embedding_size,
        hidden_units_node=None,
        hidden_units_edge=None,
        dropout_rate=0,
        use_bias=True,
        activation=None,
        weight_initializer="glorot_uniform",
        weight_regularizer=None,
        bias_initializer="zeros",
    ):
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
        rows, cols = np.where(self.adjacency_matrix == 1)
        self.edges = tf.convert_to_tensor(np.column_stack((rows, cols)), dtype=tf.int32)

        self.node_feature_MLP = self.create_node_mlp()

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
