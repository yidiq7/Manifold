import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
import numpy as np 
import tensorflow.python.keras.backend as K
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Biholomorphic(keras.layers.Layer):
    '''A layer transform zi to zi*zjbar'''
    def __init__(self):
        super(Biholomorphic, self).__init__()
        
    def call(self, inputs):
        zzbar = tf.einsum('ai,aj->aij', inputs, tf.math.conj(inputs))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 25])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
       
        zzbar = tf.transpose(zzbar)
        intermediate_tensor = tf.reduce_sum(tf.abs(zzbar), 1)
        bool_mask = tf.squeeze(tf.math.logical_not(tf.math.less(intermediate_tensor, 1e-3)))
        zzbar = tf.boolean_mask(zzbar, bool_mask)
        zzbar = tf.transpose(zzbar)

        return zzbar

class Dense(keras.layers.Layer):
    def __init__(self, input_dim, units, activation=None, trainable=True):
        super(Dense, self).__init__()
        #w_init = tf.random_normal_initializer(mean=0.05, stddev=0.05)
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=trainable,
        )
        self.activation =  activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

class WidthOneDense(keras.layers.Layer):
    def __init__(self, activation=None):
        super(WidthOneDense, self).__init__()
        #w_init = tf.random_normal_initializer()
        w_init = tf.reshape(tf.concat([tf.constant([1,0,0,0,0,1,0,0,0,1,0,0,1,0,1], dtype=tf.float32), tf.zeros(10)], 0), [25, 1]) 
        self.w = tf.Variable(
            #initial_value=w_init(shape=(25, 1)),
            initial_value=w_init,
            trainable=True,
        )
        self.activation =  activations.get(activation)
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))


def gradients_zbar(func, x):
    dx_real = tf.gradients(tf.math.real(func), x)
    dx_imag = tf.gradients(tf.math.imag(func), x)
    return (dx_real + dx_imag*tf.constant(1j, dtype=x.dtype)) / 2


def complex_hessian(func, x):
    # Take a real function and calculate dzdzbar(f)
    #grad = gradients_z(func, x)
    grad = tf.math.conj(tf.gradients(func, x))
    hessian = tf.stack([gradients_zbar(tmp[0], x)[0]
                        for tmp in tf.unstack(grad, axis=2)],
                       axis = 1) / 2.0
 
    return hessian 

def generate_dataset(HS):
    dataset = None
    for patch in HS.patches:
        for subpatch in patch.patches:
            new_dataset = dataset_on_patch(subpatch)
            if dataset is None:
                dataset = new_dataset
            else:
                dataset = dataset.concatenate(new_dataset)
    return dataset

def dataset_on_patch(patch):

    # So that you don't need to invoke set_k()
    patch.s_tf_1, patch.J_tf_1 = patch.num_s_J_tf(k=1)
    patch.omega_omegabar = patch.get_omega_omegabar(lambdify=True)
    patch.restriction = patch.get_restriction(lambdify=True)
    patch.r_tf = patch.num_restriction_tf()

    x = tf.convert_to_tensor(np.array(patch.points, dtype=np.complex64))
    y = tf.cast(patch.num_Omega_Omegabar_tf(), dtype=tf.float32)

    mass = y / tf.cast(patch.num_FS_volume_form_tf('identity', k=1), dtype=tf.float32)

    # The Kahler metric calculated by complex_hessian will include the derivative of the norm_coordinate, 
    # here we transform the restriction so that the corresponding column and row will be ignored in the hessian
    trans_mat = np.delete(np.identity(patch.dimensions), patch.norm_coordinate, axis=0)
    trans_tensor = tf.convert_to_tensor(np.array(trans_mat, dtype=np.complex64))
    restriction = tf.matmul(patch.r_tf, trans_tensor) 

    dataset = tf.data.Dataset.from_tensor_slices((x, y, mass, restriction))

    return dataset

def weighted_MAPE(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(K.abs(y_true - y_pred) / y_true * weights)

def weighted_MSE(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(tf.square(y_pred / y_true - 1) * weights)

def max_error(y_true, y_pred, mass):
    return tf.math.reduce_max(K.abs(y_true - y_pred) / y_true)

def MAPE_plus_max_error(y_true, y_pred, mass):
    return 0.1*max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)

def cal_total_loss(dataset, loss_function):

    total_loss = 0
    total_mass = 0

    for step, (points, omega_omegabar, mass, restriction) in enumerate(dataset):
        omega = volume_form(points, omega_omegabar, mass, restriction)
        mass_sum = tf.reduce_sum(mass)
        total_loss += loss_function(omega_omegabar, omega, mass) * mass_sum
        total_mass += mass_sum

    total_loss = total_loss / total_mass

    return total_loss.numpy()

def cal_max_error(dataset):
    '''
    find max|eta - 1| over the whole dataset: calculate the error on each batch then compare.
    '''
    max_error_tmp = 0
    for step, (points, omega_omegabar, mass, restriction) in enumerate(dataset):
        omega = volume_form(points, omega_omegabar, mass, restriction)
        error = max_error(omega_omegabar, omega, mass).numpy()
        if error > max_error_tmp:
            max_error_tmp = error

    return max_error_tmp
