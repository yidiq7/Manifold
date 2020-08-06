import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
import numpy as np

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

def complex_hessians(ys,
                     xs,
                     gate_gradients=False,
                     aggregation_method=None,
                     name="hessians"):

    xs = gradients_util._AsList(xs)  # pylint: disable=protected-access
    kwargs = {
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method
    }

    hessians = []
    _gradients = tf.gradients(ys, xs, **kwargs) 
    
    def gradients_z(func, x):
        dx_real = tf.gradients(tf.math.real(func), x)
        dx_imag = tf.gradients(tf.math.imag(func), x) 
        return tf.math.conj(dx_real - dx_imag*tf.constant(1j, dtype=x.dtype)) / 2
       
    for gradient, x in zip(_gradients, xs):
        # change shape to one-dimension without graph branching
        gradient = array_ops.reshape(gradient, [-1])

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(x)
        loop_vars = [
            array_ops.constant(0, dtypes.int32),
            tensor_array_ops.TensorArray(x.dtype, n)
        ]
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
       
        _, hessian = control_flow_ops.while_loop(
            lambda j, _: j < n,
            lambda j, result: (j + 1,
                               result.write(j, gradients_z(gradient[j], x)[0] / 2)),
            loop_vars
        )

        _shape = array_ops.shape(x)
        _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                              array_ops.concat((_shape, _shape), 0))
        hessians.append(_reshaped_hessian)

    return hessians

def generate_dataset(patch):
    x = tf.convert_to_tensor(np.array(patch.points, dtype=np.complex64))
    y = tf.cast(patch.num_eta_tf('FS'), dtype=tf.complex64)

    # The Kahler metric calculated by complex_hessian will include the derivative of the norm_coordinate, 
    # here we transform the restriction so that the corresponding column and row will be ignored in the hessian
    trans_mat = np.delete(np.identity(patch.dimensions), patch.norm_coordinate, axis=0)
    trans_tensor = tf.convert_to_tensor(np.array(trans_mat, dtype=np.complex64))
    restriction = tf.matmul(patch.r_tf, trans_tensor) 

    dataset = tf.data.Dataset.from_tensor_slices((x, y, restriction))

    return dataset
