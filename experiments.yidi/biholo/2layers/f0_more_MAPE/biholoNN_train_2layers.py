import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from hypersurface_tf import *
from biholoNN import *
import tensorflow as tf
import numpy as np
import time
import sys
import math
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = int(sys.argv[1])
psi = 0.5
n_pairs = 100000
batch_size = 5000
layers = sys.argv[2]
#layers = '500_500_500_2000_1'
max_epochs = 2000000
#loss_func = weighted_MAPE
loss_func = weighted_MAPE
early_stopping = True

# Gradient clipping
grad_clipping = False
clip_threshold = 0.05

n_units = layers.split('_')
for i in range(0, len(n_units)):
    n_units[i] = int(n_units[i])

load_path = 'experiments.yidi/biholo/2layers/f0_more/'
saved_path = 'experiments.yidi/biholo/2layers/f0_more_MAPE/'
model_name = layers + '_seed' + str(seed) 
#model_name = 'f2_' + layers + '_seed' + str(seed) + '_threshold' + str(clip_threshold) 

np.random.seed(seed)
tf.random.set_seed(seed)

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + psi*z0*z1*z2*z3*z4
#g = z0**4 + z1**4 + z2**4 + z3**4 + 0.1*z0*z1*z2*z3
#h = z0**4 + z1**4 + z2**4 + z4**4 + 0.2*z0*z1*z2*z4
#f = z3*g + z4*h
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
        self.layer1 = Dense(25, n_units[0], activation=tf.square)
        self.layer2 = Dense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = Dense(n_units[1], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = tf.math.log(x)
        return x

#model = KahlerPotential()
model = tf.keras.models.load_model(load_path + model_name, compile=False)

@tf.function
def volume_form(x, Omega_Omegabar, mass, restriction):

    kahler_metric = complex_hessian(tf.math.real(model(x)), x)
    volume_form = tf.math.real(tf.linalg.det(tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
    #factor = tf.constant(35.1774, dtype=tf.complex64)
    return volume_form / factor

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

# Training
optimizer = tf.keras.optimizers.Adam()

#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = saved_path + 'logs/' + model_name + '/train'
test_log_dir = saved_path + 'logs/' + model_name + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

log_file = open(saved_path + model_name + '.log', 'w')

start_time = time.time()

stop = False
loss_old = 100000
epoch = 0

while epoch < max_epochs and stop is False:
    epoch = epoch + 1
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
        with tf.GradientTape() as tape:
        
            omega = volume_form(points, Omega_Omegabar, mass, restriction)
            loss = loss_func(Omega_Omegabar, omega, mass)  
            grads = tape.gradient(loss, model.trainable_weights)
            if grad_clipping is True:
                grads = [tf.clip_by_value(grad, -clip_threshold, clip_threshold) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #if step % 500 == 0:
        #    print("step %d: loss = %.4f" % (step, loss))

    E_max_train = cal_max_error(train_set) 
    E_max_test = cal_max_error(test_set) 
    
    train_loss = cal_total_loss(train_set, loss_func)
    test_loss = cal_total_loss(test_set, loss_func)

    print("train_loss:", loss.numpy())
    print("test_loss:", test_loss)

    log_file.write("train_loss: {:.6g} \n".format(loss))
    log_file.write("test_loss: {:.6g} \n".format(test_loss))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=epoch)
        tf.summary.scalar('max_error', E_max_train, step=epoch)
        if loss_func.__name__ != "weighted_MAPE":
            tf.summary.scalar('MAPE', cal_total_loss(train_set, weighted_MAPE), step=epoch)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss, step=epoch)
        tf.summary.scalar('max_error', E_max_test, step=epoch)
        if loss_func.__name__ != "weighted_MAPE":
            tf.summary.scalar('MAPE', cal_total_loss(test_set, weighted_MAPE), step=epoch)    # Early stopping 

    if early_stopping is True and epoch > 400:
        if epoch % 5 == 0:
            if train_loss > loss_old:
                stop = True 
            loss_old = train_loss 

train_time = time.time() - start_time

log_file.close()
model.save(saved_path + model_name)

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

#print(delta_sigma_train)
#print(delta_sigma_test)

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
    f.write('n_parameters = {} \n'.format(model.count_params())) 
    f.write('loss function = {} \n'.format(loss_func.__name__))
    f.write('grad_clipping = {} \n'.format(grad_clipping))
    if grad_clipping is True:
        f.write('clip_threshold = {} \n'.format(clip_threshold))
    f.write('\n')
    f.write('n_epochs = {} \n'.format(epoch))
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

with open(saved_path + "summary.txt", "a") as f:
    f.write('{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(model_name, loss_func.__name__, psi, n_pairs, train_time, sigma_train, sigma_test, E_train, E_test, E_max_train, E_max_test))
    #f.write('%s %g %d %f %f %f %f %f %f %f\n' % (model_name, psi, n_pairs, train_time, train_loss, test_loss, E_train, E_test, E_max_train, E_max_test))

