import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from hypersurface_tf import *
from biholoNN import *
import tensorflow as tf
import numpy as np
import time
import sys
import configparser
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = configparser.RawConfigParser()
saved_path  = sys.argv[1]
model_name = sys.argv[2]
full_path = saved_path + model_name + '.txt'
config.read(full_path)
seed = int(config.get('Results', 'seed'))
psi = float(config.get('Results', 'psi'))
n_pairs = int(config.get('Results', 'n_pairs'))
n_points = int(config.get('Results', 'n_points'))
batch_size = int(config.get('Results', 'batch_size'))
layers = config.get('Results', 'layers')
loss_func = config.get('Results', 'loss function')
function_mappings = {'weighted_MAPE': weighted_MAPE, 'weighted_MSE': weighted_MSE}
loss_func = function_mappings[loss_func]

n_epochs = int(config.get('Results', 'n_epochs'))
train_time = float(config.get('Results', 'train_time'))

n_units = layers.split('_')
for i in range(0, len(n_units)):
    n_units[i] = int(n_units[i])


np.random.seed(seed)

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + psi*z0*z1*z2*z3*z4
HS = Hypersurface(Z, f, n_pairs)
HS_test = Hypersurface(Z, f, n_pairs)

train_set = generate_dataset(HS)
test_set = generate_dataset(HS_test)

train_set = train_set.shuffle(HS.n_points).batch(batch_size)
test_set = test_set.shuffle(HS_test.n_points).batch(batch_size)

class KahlerPotential(tf.keras.Model):

    def __init__(self):
        super(KahlerPotential, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25       , n_unit[0], activation=tf.square)
        self.layer2 = Dense(n_unit[0], n_unit[1], activation=tf.square)
        self.layer3 = Dense(n_unit[1], n_unit[2], activation=tf.square)
        self.layer4 = Dense(n_unit[2], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.math.log(x)
        return x

#model = KahlerPotential()
model = tf.keras.models.load_model(saved_path + model_name, compile=False)

@tf.function
def volume_form(x, Omega_Omegabar, mass, restriction):

    kahler_metric = complex_hessian(tf.math.real(model(x)), x)
    volume_form = tf.math.real(tf.linalg.det(tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
    return volume_form / factor

def cal_total_loss(dataset, loss_function):
    
    total_loss = 0
    total_mass = 0
    
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(dataset):
        omega = volume_form(points, Omega_Omegabar, mass, restriction)
        mass_sum = tf.reduce_sum(mass)
        total_loss += loss_function(Omega_Omegabar, omega, mass) * mass_sum
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
    
sigma_train = cal_total_loss(train_set, weighted_MAPE) 
sigma_test = cal_total_loss(test_set, weighted_MAPE) 
E_train = cal_total_loss(train_set, weighted_MSE) 
E_test = cal_total_loss(test_set, weighted_MSE) 
E_max_train = cal_max_error(train_set) 
E_max_test = cal_max_error(test_set) 

#######################################################################
# Calculate delta_sigma

def delta_sigma_square_train(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum((K.abs(y_true - y_pred) / y_true - sigma_train)**2 * weights)

def delta_sigma_square_test(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum((K.abs(y_true - y_pred) / y_true - sigma_test)**2 * weights)

def delta_E_square_train(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(((y_pred / y_true - 1)**2 - E_train)**2 * weights)

def delta_E_square_test(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(((y_pred / y_true - 1)**2 - E_test)**2 * weights)

delta_sigma_train = math.sqrt(cal_total_loss(train_set, delta_sigma_square_train) / HS.n_points)
delta_sigma_test = math.sqrt(cal_total_loss(test_set, delta_sigma_square_test) / HS.n_points)
delta_E_train = math.sqrt(cal_total_loss(train_set, delta_E_square_train) / HS.n_points)
delta_E_test = math.sqrt(cal_total_loss(test_set, delta_E_square_test) / HS.n_points)

#####################################################################
# Write to file

with open(saved_path + model_name + ".txt", "w") as f:
    f.write('[Results] \n')
    f.write('model_name = {} \n'.format(model_name))
    f.write('seed = {} \n'.format(seed))
    f.write('psi = {} \n'.format(psi))
    f.write('n_pairs = {} \n'.format(n_pairs))
    f.write('n_points = {} \n'.format(HS.n_points))
    f.write('batch_size = {} \n'.format(batch_size))
    f.write('layers = {} \n'.format(layers)) 
    f.write('loss function = {} \n'.format(loss_func.__name__))
    f.write('\n')
    f.write('n_epochs = {} \n'.format(n_epochs))
    f.write('train_time = {:.6g} \n'.format(train_time))
    f.write('sigma_train = {:.6g} \n'.format(sigma_train))
    f.write('sigma_test = {:.6g} \n'.format(sigma_test))
    f.write('delta_sigma_train = {:.6g} \n'.format(delta_sigma_train))
    f.write('delta_sigma_test = {:.6g} \n'.format(delta_sigma_test))
    f.write('E_train = {:.6g} \n'.format(E_train))
    f.write('E_test = {:.6g} \n'.format(E_test))
    f.write('delta_E_train = {:.6g} \n'.format(delta_sigma_train))
    f.write('delta_E_test = {:.6g} \n'.format(delta_sigma_test))
    f.write('E_max_train = {:.6g} \n'.format(E_max_train))
    f.write('E_max_test = {:.6g} \n'.format(E_max_test))

with open(saved_path + "summary.txt", "w") as f:
    f.write('{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(model_name, loss_func.__name__, psi, n_pairs, train_time, sigma_train, sigma_test, E_train, E_test, E_max_train, E_max_test))
