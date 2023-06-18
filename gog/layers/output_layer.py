import tensorflow as tf
from tensorflow import keras


class FlattenedDenseOutput(keras.layers.Layer):
    """
    A utility output layer that takes in a 2D feature matrix and flattens it before passing it through a regular
    dense layer. The output feature matrix is reshaped to be 2D again.
    """

    def __init__(self, units: int, activation: str = None) -> None:
        """

        :param units: dimensionality of the output node feature vector
        :param activation: activation function used in dense layer
        """
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
