import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from hypersurface_tf import *
from generate_h import *
from biholoNN import *
import tensorflow as tf
import numpy as np
import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = int(sys.argv[1])
psi = float(sys.argv[2])
n_pairs = 100000
batch_size = 1000
layers = '100_500_1'
max_epochs = 10000

saved_path = 'experiments_nannu/biholo/2layers/f0'
model_name = layers + '_f0' + '_seed' + str(seed) + '_' + str(int(10*psi))



class KahlerPotential(tf.keras.Model):

    def __init__(self):
        super(KahlerPotential, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25,100, activation=tf.square)
        self.layer2 = Dense(100,500, activation=tf.square)
        self.layer3 = Dense(500, 1)
        #self.layer4 = Dense(100, 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        #x = tf.reduce_sum(x, 1)
        x = tf.math.log(x)
        return x

if __name__ =='__main__':

    z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0,z1,z2,z3,z4]
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 - psi*z0*z1*z2*z3*z4
    np.random.seed(seed)
    HS = Hypersurface(Z, f, n_pairs)
    HS_test = Hypersurface(Z, f, n_pairs)

    train_set = generate_dataset(HS)
    test_set = generate_dataset(HS_test)

    train_set = train_set.shuffle(HS.n_points).batch(batch_size)
    test_set = test_set.shuffle(HS_test.n_points).batch(batch_size)

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
    loss_old = 10
    epoch = 0

    while epoch < max_epochs and stop is False:
        for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
            with tf.GradientTape() as tape:
            
                omega = volume_form(points, Omega_Omegabar, mass, restriction)
                loss = weighted_MAPE(Omega_Omegabar, omega, mass)  
                grads = tape.gradient(loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        
        test_loss = cal_total_loss(test_set, weighted_MAPE)
        print("train_loss:", loss.numpy())
        print("test_loss:", test_loss)

        log_file.write("train_loss: %f \n" %loss)
        log_file.write("test_loss: %f \n" %test_loss)
           
        # Early stopping 
        if epoch % 10 == 0:
            train_loss = cal_total_loss(train_set, weighted_MAPE)
            if train_loss > loss_old:
                stop = True 
            loss_old = train_loss 

        epoch +=  1

    train_time = time.time() - start_time

    log_file.close()
    model.save(saved_path + model_name)

    #######################################################################
    # Calculate delta_sigma

    train_loss = cal_total_loss(train_set, weighted_MAPE)

    def delta_sigma_square_train(y_true, y_pred, mass):
        weights = mass / K.sum(mass)
        return K.sum((K.abs(y_true - y_pred) / y_true - train_loss)**2 * weights)

    def delta_sigma_square_test(y_true, y_pred, mass):
        weights = mass / K.sum(mass)
        return K.sum((K.abs(y_true - y_pred) / y_true - test_loss)**2 * weights)

    delta_sigma_train = math.sqrt(cal_total_loss(train_set, delta_sigma_square_train) / HS.n_points)
    delta_sigma_test = math.sqrt(cal_total_loss(test_set, delta_sigma_square_test) / HS.n_points)

    print(delta_sigma_train)
    print(delta_sigma_test)

    #####################################################################
    # Write to file

    with open(saved_path + model_name + ".txt", "w") as f:
        f.write('[Results] \n')
        f.write('model_name = %s \n' % model_name)
        f.write('seed = %d \n' % seed)
        f.write('psi = %g \n' % psi)
        f.write('n_pairs = %d \n' % n_pairs)
        f.write('n_points = %d \n' % HS.n_points)
        f.write('batch_size = %d \n' % batch_size)
        f.write('layers = %s \n' % layers) 
        f.write('\n')
        f.write('n_epochs = %d \n' % epoch)
        f.write('train_time = %f \n' % train_time)
        f.write('sigma_train = %f \n' % train_loss)
        f.write('sigma_test = %f \n' % test_loss)
        f.write('delta_sigma_train = %f \n' % delta_sigma_train)
        f.write('delta_sigma_test = %f \n' % delta_sigma_test)

    with open(saved_path + "summary.txt", "a") as f:
        f.write('%d %g %d %f %f %f %f %f \n' % (seed, psi, n_pairs, train_time, train_loss, test_loss, delta_sigma_train, delta_sigma_test))
