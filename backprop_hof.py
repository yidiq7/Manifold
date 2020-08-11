from hypersurface_tf import *
from generate_h import *
import sympy as sp
from pprint import pprint
from scipy.optimize import minimize
from sympy.utilities.iterables import flatten
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import sys
import pickle

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def train(HS, g, learning_rate, factor):
    with tf.GradientTape() as t:
        h = tf.matmul(g, g, adjoint_b=True)
        h = h/h[0][0]
        integration = HS.integrate(lambda patch: tf.abs(patch.num_eta_tf(h)/factor - 1), tensor=True)
    dg = t.gradient(integration, g)
    g.assign_sub(learning_rate * dg)


k = int(sys.argv[1])
n_points = int(sys.argv[2])
seed = int(sys.argv[3])
learning_rate = float(sys.argv[4])
n_epochs = int(sys.argv[5])
outfile = sys.argv[6]

np.random.seed(seed)

if __name__ =='__main__':

    z0, z1, z2, z3, z4= sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0,z1,z2,z3,z4]
    psi = 0.5
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + psi*z0*z1*z2*z3*z4 + (z3*(z4**4)) + ((z3**2)*(z4**3)) + ((z3**3)*(z4**2)) + ((z3**4)*z4)
    HS = Hypersurface(Z, f, n_points)
    HS.set_k(k)

    factor = HS.integrate(lambda patch: patch.num_eta_tf('FS'), holomorphic=True, tensor=True).numpy()
    h_sym = get_sym_info(HS)



    g0 = np.linalg.cholesky(param_to_matrix(initial_FS_param(HS, h_sym), h_sym))
    g0 = np.array(g0, dtype=np.complex64)
    with tf.device('/cpu:0'):
        g = tf.Variable(g0)

    HS_test = Hypersurface(Z, f, 10000)
    HS_test.set_k(k)

    test_old = 10

    epochs = range(n_epochs)
    for epoch in epochs:
        train(HS, g, learning_rate, factor)

        if epoch % 50 == 0:
            h = tf.matmul(g, g, adjoint_b=True)
            h = h/h[0][0]
            integration = HS.integrate(lambda patch: tf.abs(patch.num_eta_tf(h)/factor - 1), tensor=True).numpy()
            test = HS_test.integrate(lambda patch: tf.abs(patch.num_eta_tf(h)/factor - 1), tensor=True).numpy()
            print('train:', integration)
            print('test:', test)
            if test > test_old:
                break
            test_old = test

    h_minimal = tf.matmul(g, g, adjoint_b=True)
    h_minimal = h_minimal/h_minimal[0][0]

    train_sigma = HS.integrate(lambda patch: tf.abs(patch.num_eta_tf(h_minimal)/factor - 1), tensor=True).numpy()
    train_delta_sigma = math.sqrt(HS.integrate(lambda patch: (tf.abs(patch.num_eta_tf(h_minimal)/factor - 1) - train_sigma)**2, 
                                         tensor=True).numpy() / HS.n_points)

    test_sigma = HS_test.integrate(lambda patch: tf.abs(patch.num_eta_tf(h_minimal)/factor - 1), tensor=True).numpy()
    test_delta_sigma = math.sqrt(HS_test.integrate(lambda patch: (tf.abs(patch.num_eta_tf(h_minimal)/factor - 1) - test_sigma)**2, 
                                         tensor=True).numpy() / HS_test.n_points)

    print('delta sigma train:', train_delta_sigma)
    print('delta sigma test:', test_delta_sigma)
    pickle.dump( g, open( outfile+".dat", "wb" ) )
    with open( outfile+".txt", "w" ) as f:
        sys.stdout = f
        print( 'psi=', psi )
        print( 'k=', k )
        print( 'n_points=', n_points )
        print( 'seed=', seed )
        print( 'rate=', learning_rate, 'n_epochs=', n_epochs )
        print('train=', integration)
        print('test=', test)
        print('delta_sigma_train=', train_delta_sigma)
        print('delta_sigma_test=', test_delta_sigma)
