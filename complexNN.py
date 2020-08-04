import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations

class ComplexDense(keras.layers.Layer):
    def __init__(self, input_dim, units, activation=None):
        super(ComplexDense, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=tf.cast(w_init(shape=(input_dim, units), dtype='float32'), dtype=tf.complex64),
            trainable=True,
        )
        self.activation =  activations.get(activation)
        
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))
