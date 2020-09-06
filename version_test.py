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

seed = 1234
psi = 0.5
n_pairs = 10000
batch_size = 1000
layers = '50_100_100'
max_epochs = 10000

saved_path = 'experiments.mrd/biholo/'
model_name = layers 

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + psi*z0*z1*z2*z3*z4
np.random.seed(seed)
HS = Hypersurface(Z, f, n_pairs)

train_set = generate_dataset(HS)

train_set = train_set.shuffle(HS.n_points).batch(batch_size)

class KahlerPotential(tf.keras.Model):

    def __init__(self):
        super(KahlerPotential, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = WidthOneDense()
        #self.layer1 = Dense(25,50, activation=tf.square)
        #self.layer2 = Dense(50,100, activation=tf.square)
        #self.layer3 = Dense(100,100, activation=tf.square)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = tf.reduce_sum(x, 1)
        x = tf.math.log(x)
        return x

model = KahlerPotential()
#model = tf.keras.models.load_model(saved_path + model_name, compile=False)
with open('test.log', 'w') as f:
    sys.stdout = f

    @tf.function
    def volume_form(x, Omega_Omegabar, mass, restriction):

        kahler_metric = complex_hessian(tf.math.real(model(x)), x)
        volume_form = tf.math.real(tf.linalg.det(tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
        weights = mass / tf.reduce_sum(mass)
        factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
        #factor = tf.constant(35.1774, dtype=tf.complex64)
        return volume_form / factor

    @tf.function
    def kahler_metric(x, Omega_Omegabar, mass, restriction):

        kahler_metric = complex_hessian(tf.math.real(model(x)), x)
        return kahler_metric

    def cal_total_loss(dataset, loss_function):
    
        total_loss = 0
        total_mass = 0
    
        for step, (points, Omega_Omegabar, mass, restriction) in enumerate(dataset):
            metric = kahler_metric(points, Omega_Omegabar, mass, restriction)
            omega = volume_form(points, Omega_Omegabar, mass, restriction)
            mass_sum = tf.reduce_sum(mass)
            total_loss += loss_function(Omega_Omegabar, omega, mass) * mass_sum
            total_mass += mass_sum
            if step == 0:
                print('points', points)
                print('metric', metric)
                print('omega', omega)
                print('total_mass', total_mass)
        total_loss = total_loss / total_mass
    
        return total_loss.numpy()

    train_loss = cal_total_loss(train_set, weighted_MAPE) 
    print('train_loss', train_loss)

