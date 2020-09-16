import os
from hypersurface_tf import *
from generate_h import *
from biholoNN import *
import tensorflow as tf
import numpy as np
import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = int(sys.argv[2])
psi = float(sys.argv[3])
alpha = float(sys.argv[4])
n_pairs = 100000
batch_size = 1000
layers = '500_500_100_1'
max_epochs = 100000
loss_func = weighted_MAPE
loss_old_init = 10

saved_path = sys.argv[1]
model_name = layers + '_seed' + str(seed) 

np.random.seed(seed)
tf.random.set_seed(seed)

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 - psi*z0*z1*z2*z3*z4 + alpha*(z2*z0**4 + z0*z4*z1**3  + z0*z2*z3*z4**2 + (z3**2)*(z1**3) +  z4*(z1**2)*(z2**2) + z0*z1*z2*z3**2 +  z2*z4*z3**3 + z0*z1**4 +  z0*(z4**2)*(z2**2) + (z4**3)*(z1**2) + z0*z2*z3**3 + z3*z4*z0**3 + (z1**3)*(z4**2) + z0*z2*z4*z1**2 + (z1**2)*(z3**3) + z1*z4**4 + z1*z2*z0**3 + (z2**2)*(z4**3) +  z4*z2**4 + z1*z3**4)
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
        self.layer1 = Dense(25,500, activation=tf.square)
        self.layer2 = Dense(500,500, activation=tf.square)
        self.layer3 = Dense(500,100, activation=tf.square)
        self.layer4 = Dense(100, 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.math.log(x)
        return x

model = KahlerPotential()
#model = tf.keras.models.load_model(saved_path + model_name, compile=False)

@tf.function
def volume_form(x, Omega_Omegabar, mass, restriction):

    kahler_metric = complex_hessian(tf.math.real(model(x)), x)
    volume_form = tf.math.real(tf.linalg.det(tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
    #factor = tf.constant(35.1774, dtype=tf.complex64)
    return volume_form / factor

optimizer = tf.keras.optimizers.Adam()

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

# Training
log_file = open(saved_path + model_name + '.log', 'w')

start_time = time.time()

stop = False
loss_old = loss_old_init
epoch = 0

while epoch < max_epochs and stop is False:
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
        with tf.GradientTape() as tape:
        
            omega = volume_form(points, Omega_Omegabar, mass, restriction)
            loss = loss_func(Omega_Omegabar, omega, mass)  
            grads = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #if step % 500 == 0:
        #    print("step %d: loss = %.4f" % (step, loss))
    
    test_loss = cal_total_loss(test_set, loss_func)
    print("train_loss:", loss.numpy())
    print("test_loss:", test_loss)

    log_file.write("train_loss: {:.6g} \n".format(loss))
    log_file.write("test_loss: {:.6g} \n".format(test_loss))
       
    # Early stopping 
    if epoch % 10 == 0:
        train_loss = cal_total_loss(train_set, loss_func)
        if train_loss > loss_old:
            stop = True 
        loss_old = train_loss 

    epoch = epoch + 1

train_time = time.time() - start_time

log_file.close()
model.save(saved_path + model_name)

sigma_train = cal_total_loss(train_set, weighted_MAPE) 
sigma_test = cal_total_loss(test_set, weighted_MAPE) 
E_train = cal_total_loss(train_set, weighted_MSE) 
E_test = cal_total_loss(test_set, weighted_MSE) 
E_max_train = cal_total_loss(train_set, max_error) 
E_max_test = cal_total_loss(test_set, max_error) 

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
    return K.sum((K.abs(y_true - y_pred) / y_true - E_train)**2 * weights)

def delta_E_square_test(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum((K.abs(y_true - y_pred) / y_true - E_test)**2 * weights)



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
    f.write('alpha = {} \n'.format(alpha))
    f.write('n_pairs = {} \n'.format(n_pairs))
    f.write('n_points = {} \n'.format(HS.n_points))
    f.write('batch_size = {} \n'.format(batch_size))
    f.write('layers = {} \n'.format(layers)) 
    f.write('loss function = {} \n'.format(loss_func.__name__))
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
    f.write('{} {} {} {} {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} {:.10g}\n'.format(model_name, loss_func.__name__, psi, alpha, n_pairs, train_time, sigma_train, sigma_test, E_train, E_test, E_max_train, E_max_test))
    #f.write('%s %g %d %f %f %f %f %f %f %f\n' % (model_name, psi, n_pairs, train_time, train_loss, test_loss, E_train, E_test, E_max_train, E_max_test))

#HEADER:   model_name loss_func psi alpha n_pairs train_time sigma_train sigma_test E_train E_test E_max_train E_max_test
