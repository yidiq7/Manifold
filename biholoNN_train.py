import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hypersurface_tf import *
from biholoNN import *
from biholoNN_lbfgs import *
import tensorflow as tf
import numpy as np
import time
import sys
import math
from models import *
import argparse
import tensorflow_probability as tfp

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]

parser = argparse.ArgumentParser()
# Data generation
parser.add_argument('--seed', type=int)
parser.add_argument('--n_pairs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--function')
parser.add_argument('--psi', type=float)
parser.add_argument('--phi', type=float)
parser.add_argument('--alpha', type=float)

# Network
parser.add_argument('--OuterProductNN_k', type=int)
parser.add_argument('--layers')
parser.add_argument('--load_model')
parser.add_argument('--save_dir')
parser.add_argument('--save_name')

# Training
parser.add_argument('--max_epochs', type=int)
parser.add_argument('--loss_func')
parser.add_argument('--clip_threshold', type=float)
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--learning_rate', type=float, default=0.001)

args = parser.parse_args()
print("Processing model: " + args.save_name)
# Data generation 
seed = args.seed
n_pairs = args.n_pairs
batch_size = args.batch_size
psi = args.psi

f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + psi*z0*z1*z2*z3*z4
if args.function == 'f1':
    phi = args.phi
    f = f + phi*(z3*z4**4 + z3**2*z4**3 + z3**3*z4**2 + z3**4*z4)
elif args.function == 'f2':
    alpha = args.alpha
    f = f + alpha*(z2*z0**4 + z0*z4*z1**3 + z0*z2*z3*z4**2 + z3**2*z1**3 + z4*z1**2*z2**2 + z0*z1*z2*z3**2 +
                   z2*z4*z3**3 + z0*z1**4 + z0*z4**2*z2**2 + z4**3*z1**2 + z0*z2*z3**3 + z3*z4*z0**3 + z1**3*z4**2 +
                   z0*z2*z4*z1**2 + z1**2*z3**3 + z1*z4**4 + z1*z2*z0**3 + z2**2*z4**3 + z4*z2**4 + z1*z3**4)

np.random.seed(seed)
tf.random.set_seed(seed)
HS = Hypersurface(Z, f, n_pairs)
HS_test = Hypersurface(Z, f, n_pairs)

train_set = generate_dataset(HS)
test_set = generate_dataset(HS_test)

if batch_size is None or args.optimizer == 'lbfgs':
    batch_size = HS.n_points

train_set = train_set.shuffle(HS.n_points).batch(batch_size)
test_set = test_set.shuffle(HS_test.n_points).batch(batch_size)

# Network 
if args.OuterProductNN_k is not None:
    k = args.OuterProductNN_k
else:
    layers = args.layers
    n_units = layers.split('_')
    for i in range(0, len(n_units)):
        n_units[i] = int(n_units[i])
    n_hidden = len(n_units) - 1
    k = 2**n_hidden

#model_name = layers + '_seed' + str(seed) 
load_path = args.load_model
if load_path is not None:
    model = tf.keras.models.load_model(load_path, compile=False)
elif args.OuterProductNN_k is not None:
    if k == 2:
        model = OuterProductNN_k2()  
    elif k == 3:
        model = OuterProductNN_k3()  
    elif k == 4:
        model = OuterProductNN_k4()  
    else:
        raise Exception("Only k = 2,3,4 are supported now")
    #model = OuterProductNN(k)  
elif n_hidden == 0:
    model = zerolayer(n_units)
elif n_hidden == 1:
    model = onelayer(n_units)
elif n_hidden == 2:
    model = twolayers(n_units) 
elif n_hidden == 3:
    model = threelayers(n_units) 
elif n_hidden == 4:
    model = fourlayers(n_units) 
elif n_hidden == 5:
    model = fivelayers(n_units) 

max_epochs = args.max_epochs
func_dict = {"weighted_MAPE": weighted_MAPE, "weighted_MSE": weighted_MSE, "max_error":max_error,
             "MAPE_plus_max_error": MAPE_plus_max_error}
loss_func = func_dict[args.loss_func]
#early_stopping = False
clip_threshold = args.clip_threshold
#saved_path = 'experiments.yidi/biholo/4layers/f0/'
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = args.save_name

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
start_time = time.time()
if args.optimizer == 'lbfgs':
    # iter+1 everytime f is evoked, which will also be invoked when calculationg the hessian, etc
    # So the true max_epochs will be 3 times user's input
    max_epochs = int(max_epochs/3)
    for step, (points, omega_omegabar, mass, restriction) in enumerate(train_set):
        train_func = function_factory(model, loss_func, points, omega_omegabar, mass, restriction)
    init_params = tf.dynamic_stitch(train_func.idx, model.trainable_variables)
    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=train_func, initial_position=init_params, max_iterations=max_epochs)
    train_func.assign_new_model_parameters(results.position)

else:
    if args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(args.learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    train_log_dir = save_dir + '/logs/' + save_name + '/train'
    test_log_dir = save_dir + '/logs/' + save_name + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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
                if clip_threshold is not None:
                    grads = [tf.clip_by_value(grad, -clip_threshold, clip_threshold) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            #if step % 500 == 0:
            #    print("step %d: loss = %.4f" % (step, loss))
        if epoch % 10 == 0:
            sigma_max_train = cal_max_error(train_set) 
            sigma_max_test = cal_max_error(test_set) 

            E_train = cal_total_loss(train_set, weighted_MSE)
            E_test = cal_total_loss(test_set, weighted_MSE)
        
            sigma_train = cal_total_loss(train_set, weighted_MAPE)
            sigma_test  = cal_total_loss(test_set, weighted_MAPE)

            def delta_sigma_square_train(y_true, y_pred, mass):
                weights = mass / K.sum(mass)
                return K.sum((K.abs(y_true - y_pred) / y_true - sigma_train)**2 * weights)

            def delta_sigma_square_test(y_true, y_pred, mass):
                weights = mass / K.sum(mass)
                return K.sum((K.abs(y_true - y_pred) / y_true - sigma_test)**2 * weights)

            delta_sigma_train = math.sqrt(cal_total_loss(train_set, delta_sigma_square_train) / HS.n_points)
            delta_sigma_test = math.sqrt(cal_total_loss(test_set, delta_sigma_square_test) / HS.n_points)

            print("train_loss:", loss.numpy())
            print("test_loss:", cal_total_loss(test_set, loss_func))

            with train_summary_writer.as_default():
                tf.summary.scalar('max_error', sigma_max_train, step=epoch)
                tf.summary.scalar('delta_sigma', delta_sigma_train, step=epoch)
                tf.summary.scalar('E', E_train, step=epoch)
                tf.summary.scalar('sigma', sigma_train , step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('max_error', sigma_max_test, step=epoch)
                tf.summary.scalar('delta_sigma', delta_sigma_test, step=epoch)
                tf.summary.scalar('E', E_test, step=epoch)
                tf.summary.scalar('sigma', sigma_test, step=epoch)    # Early stopping 

       # if early_stopping is True and epoch > 800:
       #     if epoch % 5 == 0:
       #         if train_loss > loss_old:
       #             stop = True 
       #         loss_old = train_loss 

train_time = time.time() - start_time

model.save(save_dir + '/' + save_name)

sigma_train = cal_total_loss(train_set, weighted_MAPE) 
sigma_test = cal_total_loss(test_set, weighted_MAPE) 
E_train = cal_total_loss(train_set, weighted_MSE) 
E_test = cal_total_loss(test_set, weighted_MSE) 
sigma_max_train = cal_max_error(train_set) 
sigma_max_test = cal_max_error(test_set) 

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

with open(save_dir + save_name + ".txt", "w") as f:
    f.write('[Results] \n')
    f.write('model_name = {} \n'.format(save_name))
    f.write('seed = {} \n'.format(seed))
    f.write('n_pairs = {} \n'.format(n_pairs))
    f.write('n_points = {} \n'.format(HS.n_points))
    f.write('batch_size = {} \n'.format(batch_size))
    f.write('function = {} \n'.format(args.function))
    f.write('psi = {} \n'.format(psi))
    if args.function == 'f1':
        f.write('phi = {} \n'.format(phi))
    elif args.function == 'f2':
        f.write('alpha = {} \n'.format(alpha)) 
    f.write('k = {} \n'.format(k)) 
    f.write('n_parameters = {} \n'.format(model.count_params())) 
    f.write('loss function = {} \n'.format(loss_func.__name__))
    if clip_threshold is not None:
        f.write('clip_threshold = {} \n'.format(clip_threshold))
    f.write('\n')
    f.write('n_epochs = {} \n'.format(max_epochs))
    f.write('train_time = {:.6g} \n'.format(train_time))
    f.write('sigma_train = {:.6g} \n'.format(sigma_train))
    f.write('sigma_test = {:.6g} \n'.format(sigma_test))
    f.write('delta_sigma_train = {:.6g} \n'.format(delta_sigma_train))
    f.write('delta_sigma_test = {:.6g} \n'.format(delta_sigma_test))
    f.write('E_train = {:.6g} \n'.format(E_train))
    f.write('E_test = {:.6g} \n'.format(E_test))
    f.write('delta_E_train = {:.6g} \n'.format(delta_E_train))
    f.write('delta_E_test = {:.6g} \n'.format(delta_E_test))
    f.write('sigma_max_train = {:.6g} \n'.format(sigma_max_train))
    f.write('sigma_max_test = {:.6g} \n'.format(sigma_max_test))

with open(save_dir + "summary.txt", "a") as f:
    if args.function == 'f0':  
        f.write('{} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(save_name, args.function, psi, train_time, sigma_train, sigma_test, E_train, E_test, sigma_max_train, sigma_max_test))
    elif args.function == 'f1':  
        f.write('{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(save_name, args.function, psi, phi, train_time, sigma_train, sigma_test, E_train, E_test, sigma_max_train, sigma_max_test))
    elif args.function == 'f2': 
        f.write('{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(save_name, args.function, psi, alpha, train_time, sigma_train, sigma_test, E_train, E_test, sigma_max_train, sigma_max_test))
