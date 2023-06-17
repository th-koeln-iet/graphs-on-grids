import tensorflow as tf
from tensorflow import keras


class FlattenedDenseOutput(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        batch_size, num_nodes, embedding_size = input_shape
        self.dense_flat = keras.layers.Dense(
            num_nodes * self.units, activation=self.activation
        )

    def call(self, inputs, *args, **kwargs):
        _, num_nodes, embedding_size = inputs.get_shape().as_list()
        flattened = tf.reshape(inputs, (-1, num_nodes * embedding_size))
        flat_dense_output = self.dense_flat(flattened)
        output_reshaped = tf.reshape(flat_dense_output, (-1, num_nodes, self.units))
        return output_reshaped
