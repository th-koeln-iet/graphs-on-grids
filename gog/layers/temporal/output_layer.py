import tensorflow as tf
from tensorflow import keras


class ConvOutputBlock(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(ConvOutputBlock, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        batch_size, seq_len, num_nodes, num_features = input_shape
        self.output_conv = keras.layers.Conv2D(
            filters=1, kernel_size=(num_features, 1), padding="same", activation="relu"
        )
        self.dense = keras.layers.Dense(units=self.units, activation=self.activation)

    def call(self, inputs, *args, **kwargs):
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        output_conv = self.output_conv(inputs)
        output_conv_transposed = tf.transpose(output_conv, perm=[0, 3, 1, 2])
        reduced_conv = tf.squeeze(output_conv_transposed, axis=1)
        output = self.dense(reduced_conv)
        return output


class RecurrentOutputBlock(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(RecurrentOutputBlock, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.dense = keras.layers.Dense(units=self.units, activation=self.activation)

    def call(self, inputs, *args, **kwargs):
        # input: (batch_size, seq_len, num_nodes, num_features)
        # return last output in LSTM-sequence
        return self.dense(inputs[:, -1, :, :])
