from tensorflow import keras
from tensorflow.python.keras import activations
import numpy as np 
import tensorflow.python.keras.backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class Biholomorphic(keras.layers.Layer):
    '''A layer transform zi to zi*zjbar'''
    def __init__(self):
        super(Biholomorphic, self).__init__()
        
    def call(self, inputs):
        zzbar = tf.einsum('ai,aj->aij', inputs, tf.math.conj(inputs))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 25])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)
        
class Biholomorphic_k2(keras.layers.Layer):
    '''A layer transform zi to symmetrized zi1*zi2, then to zi1*zi2 * zi1zi2bar'''
    def __init__(self):
        super(Biholomorphic_k2, self).__init__()
        
    def call(self, inputs):
        # zi to zi1*zi2 
        zz = tf.einsum('ai,aj->aij', inputs, inputs)
        zz = tf.linalg.band_part(zz, 0, -1) # zero below upper triangular
        zz = tf.reshape(zz, [-1, 5**2])
        zz = tf.reshape(remove_zero_entries(zz), [-1, 15])
     
        # zi1*zi2 to zzbar
        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 15**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)


class Biholomorphic_k3(keras.layers.Layer):
    '''A layer transform zi to symmetrized zi1*zi2*zi3, then to zzbar'''
    def __init__(self):
        super(Biholomorphic_k3, self).__init__()
        
    def call(self, inputs):
        zz = tf.einsum('ai,aj,ak->aijk', inputs, inputs, inputs)
        zz = tf.linalg.band_part(zz, 0, -1)
        zz = tf.transpose(zz, perm=[0, 3, 1, 2])
        zz = tf.linalg.band_part(zz, 0, -1)
        zz = tf.transpose(zz, perm=[0, 2, 3, 1])
        zz = tf.reshape(zz, [-1, 5**3]) 
        zz = tf.reshape(remove_zero_entries(zz,35), [-1, 35])

        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 35**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)

class Biholomorphic_k4(keras.layers.Layer):
    '''A layer transform zi to symmetrized zi1*zi2*zi3*zi4, then to zzbar'''
    def __init__(self):
        super(Biholomorphic_k4, self).__init__()
        
    def call(self, inputs):
        zz = tf.einsum('ai,aj,ak,al->aijkl', inputs, inputs, inputs, inputs)
        zz = tf.linalg.band_part(zz, 0, -1) 
        zz = tf.transpose(zz, perm=[0, 4, 1, 2, 3])
        zz = tf.linalg.band_part(zz, 0, -1) 
        zz = tf.transpose(zz, perm=[0, 4, 1, 2, 3]) # 3412
        zz = tf.linalg.band_part(zz, 0, -1) 
        #zz = tf.transpose(zz, perm=[0, 4, 2, 3, 1]) # 2413
        #zz = tf.linalg.band_part(zz, 0, -1) 
        zz = tf.transpose(zz, perm=[0, 3, 4, 1, 2]) # 1324
        #zz = tf.linalg.band_part(zz, 0, -1) 
        #zz = tf.transpose(zz, perm=[0, 1, 3, 2, 4]) # Transfrom it back 
        zz = tf.reshape(zz, [-1, 5**4]) 
        zz = tf.reshape(remove_zero_entries(zz,70), [-1, 70])

        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 70**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)

class Biholomorphic_k8(keras.layers.Layer):
    '''A layer transform zi to symmetrized rank 8, then to zzbar'''
    def __init__(self):
        super(Biholomorphic_k8, self).__init__()
        
    def call(self, inputs):
        zz = tf.einsum('ai,aj,ak,al,am,an,ao,ap->aijklmnop', inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs)
        zz = generalized_upper_triangular( zz, 9 )
        zz = tf.reshape(zz, [-1, 5**8]) 
        zz = tf.reshape(remove_zero_entries(zz,495), [-1, 495])

        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 495**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)

def generalized_upper_triangular(x, n):
    perm1 = [0, n-1]
    perm1.extend(range(1,n-1))
    perm2 = [0, n-2, n-1]
    perm2.extend(range(1,n-2))
    for i in range(n-3):
        x = tf.linalg.band_part(x, 0, -1) 
        x = tf.transpose(x, perm=perm1)
    x = tf.linalg.band_part(x, 0, -1) 
    x = tf.transpose(x, perm=perm2)
    return x

def remove_zero_entries(x,n=-1):
    x = tf.transpose(x)
    intermediate_tensor = tf.reduce_sum(tf.abs(x), 1)
    bool_mask = tf.squeeze(tf.math.logical_not(tf.math.less(intermediate_tensor, 1e-4)))
    x = tf.boolean_mask(x, bool_mask)
    x = tf.transpose(x)
    if n>=0:
        tf.debugging.assert_shapes([(x,('',n))])
    return x

class Dense(keras.layers.Layer):
    def __init__(self, input_dim, units, activation=None, trainable=True):
        super(Dense, self).__init__()
        #w_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=trainable,
        )
        self.activation =  activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

class WidthOneDense(keras.layers.Layer):
    '''
    Usage: layer = WidthOneDense(n**2, 1)
           where n is the number of sections for different ks
           n = 5 for k = 1
           n = 15 for k = 2
           n = 35 for k = 3
    This layer is used directly after Biholomorphic_k layers to sum over all 
    the terms in the previous layer. The weights are initialized so that the h
    matrix is a real identity matrix. The training does not work if they are randomly
    initialized.
    '''
    def __init__(self, input_dim, units, activation=None, trainable=True):
        super(WidthOneDense, self).__init__()
        dim = int(np.sqrt(input_dim))
        mask = tf.cast(tf.linalg.band_part(tf.ones([dim, dim]),0,-1), dtype=tf.bool)
        upper_tri = tf.boolean_mask(tf.eye(dim), mask)
        w_init = tf.reshape(tf.concat([upper_tri, tf.zeros(input_dim - len(upper_tri))], axis=0), [-1, 1])
        self.w = tf.Variable(
            initial_value=w_init,
            trainable=trainable,
        )
        self.activation =  activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))
'''
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

class OuterProduct(keras.layers.Layer):
    # Outer product complex, then take real and img only
    def __init__(self, k):
        super(OuterProduct, self).__init__()
        self.k = k 
        w_init = tf.random_normal_initializer(mean=1.0, stddev=0.05)
        if k == 1:
            # input_dim = 15 * 2 (real and imag)
            input_dim = 30
        else:
            # for k = 2, input_dim = 15*(15+1)/2 * 2
            input_dim = 16 * 15**(k-1)

        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, 1), dtype='float32'),
            trainable=True,
        )

    def get_upper_triangular(self, matrix):
        #upper_tri = tf.linalg.band_part(matrix, 0, -1)
        ones = tf.ones_like(upper_tri)
        mask = tf.cast(tf.linalg.band_part(ones, 0, -1), dtype=tf.bool)
        upper_tri = tf.reshape(tf.boolean_mask(upper_tri, mask), [tf.shape(upper_tri)[0], -1])
        return upper_tri

    def call(self, inputs):
        zzbar = tf.einsum('ai,aj->aij', inputs, tf.math.conj(inputs))
        zzbar = self.get_upper_triangular(zzbar)
        i = 1
        zzbar_k = zzbar

        while i < self.k:  
            zzbar_k = tf.einsum('ai,aj->aij', zzbar_k, zzbar)
            if i == 1:
                zzbar_k = self.get_upper_triangular(zzbar_k) 
            else:
                zzbar_k = tf.reshape(zzbar_k, [tf.shape(inputs)[0], -1])
            i = i + 1

        zzbar_k = tf.concat([tf.math.real(zzbar_k), tf.math.imag(zzbar_k)], axis=1)

        # It is possible to delete all of the zeros but then it will be harder to calculate
        # the number of dimensions 
        #zzbar = tf.transpose(zzbar)
        #intermediate_tensor = tf.reduce_sum(tf.abs(zzbar), 1)
        #bool_mask = tf.squeeze(tf.math.logical_not(tf.math.less(intermediate_tensor, 1e-3)))
        #zzbar = tf.boolean_mask(zzbar, bool_mask)
        #zzbar = tf.transpose(zzbar)

        return tf.matmul(zzbar_k, self.w)     
'''
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
    return 1*max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)

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
