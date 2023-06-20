import tensorflow as tf
from tensorflow import keras


class ConvOutputBlock(keras.layers.Layer):
    """
    Convolutional output layer to be used with implementations of the `TemporalConv` layers.
    A 2D-Convolution is applied to create graphs according to the parameter `output_seq_len`. At the output,
    a dense layer adjusts the graphs to the embedding size given by the `units` parameter.
    """

    def __init__(self, output_seq_len: int, units: int, activation: str = None) -> None:
        """
        :param output_seq_len: number of graphs in the output sequence
        :param units: embedding size of the output graph(s)
        :param activation: activation function of the dense output layer
        """
        super(ConvOutputBlock, self).__init__()
        self.output_seq_len = output_seq_len
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        batch_size, seq_len, num_nodes, num_features = input_shape
        self.output_conv = keras.layers.Conv2D(
            filters=self.output_seq_len,
            kernel_size=(num_features, 1),
            padding="same",
            activation="relu",
        )
        self.dense = keras.layers.Dense(units=self.units, activation=self.activation)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        output_conv = self.output_conv(inputs)
        output_conv_transposed = tf.transpose(output_conv, perm=[0, 3, 1, 2])
        output = self.dense(output_conv_transposed)
        return output


class RecurrentOutputBlock(keras.layers.Layer):
    """
    Recurrent output layer to be used with implementations of the `GraphLSTM`, `GraphGRU` and `GraphConvLSTM` layers.
    The last output of the last temporal layer is taken and passed through a dense layer with `units` * `output_seq_len`
    neurons. The output is then reshaped to generate `output_seq_len` graphs with embedding size `units`.
    """

    def __init__(self, output_seq_len: int, units: int, activation: str = None) -> None:
        """
        :param output_seq_len: number of graphs in the output sequence
        :param units: embedding size of the output graph(s)
        :param activation: activation function of the dense output layer
        """
        super(RecurrentOutputBlock, self).__init__()
        self.output_seq_len = output_seq_len
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.out_dense = keras.layers.Dense(
            units=self.units * self.output_seq_len, activation=self.activation
        )

    def call(self, inputs, *args, **kwargs):
        # input: (batch_size, seq_len, num_nodes, num_features)
        shape = tf.shape(inputs)
        batch_size, seq_len, num_nodes, num_features = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        # take last output in LSTM-sequence
        last_lstm_output = inputs[:, -1, :, :]

        # create #output_seq_len new graphs from lstm prediction
        dense_out = self.out_dense(last_lstm_output)
        dense_out = tf.expand_dims(dense_out, axis=1)

        # reshape to (batch_size, output_seq_len, num_nodes, num_features)
        reshape = tf.reshape(
            dense_out, (batch_size, self.output_seq_len, num_nodes, self.units)
        )
        return reshape
