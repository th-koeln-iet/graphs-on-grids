from tensorflow import keras


class GraphLayer(keras.layers.Layer):

    def __init__(self):
        super(GraphLayer, self).__init__()
        self.mlp_layer_index = 0

    def create_edge_mlp(self):
        self.edge_mlp_layers = []
        self.edge_mlp_layers = self.create_hidden_layers(self.hidden_units_edge)
        self.edge_dense_out = keras.layers.Dense(1, activation=keras.activations.sigmoid)
        self.edge_mlp_layers.append(self.edge_dense_out)
        return keras.Sequential(self.edge_mlp_layers, name="sequential_edge_features")

    def create_node_mlp(self):
        self.node_mlp_layers = []
        self.node_mlp_layers = self.create_hidden_layers(self.hidden_units_node)
        self.node_dense_out = keras.layers.Dense(self.embedding_size, activation=keras.activations.sigmoid)
        self.node_mlp_layers.append(self.node_dense_out)
        return keras.Sequential(self.node_mlp_layers, name="sequential_edge_features")

    def create_hidden_layers(self, hidden_units):
        mlp_layers = []
        for n_neurons in hidden_units:
            setattr(self, f"dense_{self.mlp_layer_index}",
                    keras.layers.Dense(n_neurons, use_bias=self.use_bias, kernel_initializer=self.weight_initializer,
                                       bias_initializer=self.bias_initializer,
                                       kernel_regularizer=self.weight_regularizer))
            mlp_layers.append(getattr(self, f"dense_{self.mlp_layer_index}"))

            setattr(self, f"batchNorm_{self.mlp_layer_index}", keras.layers.BatchNormalization())
            mlp_layers.append(getattr(self, f"batchNorm_{self.mlp_layer_index}"))

            setattr(self, f"dropout_{self.mlp_layer_index}", keras.layers.Dropout(self.dropout_rate))
            mlp_layers.append(getattr(self, f"dropout_{self.mlp_layer_index}"))

            setattr(self, f"relu_{self.mlp_layer_index}", keras.layers.ReLU())
            mlp_layers.append(getattr(self, f"relu_{self.mlp_layer_index}"))
            self.mlp_layer_index += 1
        return mlp_layers
