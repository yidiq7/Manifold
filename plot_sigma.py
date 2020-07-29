#from hypersurface_tf import *
#from generate_h import *
import os
import configparser
import numpy as np
import matplotlib.pyplot as plt
#import sympy as sp
#import math

n_points = 10000
seed = 123

'''
z0, z1, z2, z3, z4= sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + 0.5*z0*z1*z2*z3*z4
HS = Hypersurface(Z, f, n_points)
'''

if not os.path.exists('plot_complex64'):
    os.makedirs('plot_complex64')

config = configparser.RawConfigParser()
directory = './results_complex64'

sigma_list = []
delta_sigma_list = []
k_list = []

for filename in sorted(os.listdir(directory)):
    full_path = os.path.join(directory, filename)
    config.read(full_path)

    k = int(config.get('results', 'k'))
    n_pts_read = int(config.get('results', 'n_points'))
    seed_read = int(config.get('results', 'seed'))
    sigma = float(config.get('results', 'sigma'))
    delta_sigma = float(config.get('results', 'delta_sigma'))

    if n_pts_read == n_points and seed_read == seed:

        '''
        HS.set_k(k)
        h_sym = get_sym_info(HS)

        param_txt = config.get('results', 'param')
        param = np.fromstring(param_txt[1:-1], dtype=np.float, sep=',')
        h_minimal = np.array(param_to_matrix(param, h_sym), dtype=np.complex64)
        print(k, param, len(h_minimal))

        factor = HS.integrate(lambda patch: patch.num_eta_tf('FS'), holomorphic=True, tensor=True).numpy()
        sigma = HS.integrate(lambda patch: tf.abs(patch.num_eta_tf(h_minimal)/factor - 1), tensor=True).numpy()
        delta_sigma = math.sqrt(HS.integrate(lambda patch: (tf.abs(patch.num_eta_tf(h_minimal)/factor - 1) - sigma)**2, 
                                             tensor=True).numpy() / HS.n_points)
        '''

        k_list.append(k)
        sigma_list.append(sigma)
        delta_sigma_list.append(delta_sigma)

plt.figure()
sigma_plt = plt.subplot(111)
sigma_plt.plot(k_list, sigma_list)        
sigma_plt.get_figure().savefig('./plot_complex64/sigma.png')

plt.figure()
delta_sigma_plt = plt.subplot(111)
delta_sigma_plt.plot(k_list, delta_sigma_list)        
delta_sigma_plt.get_figure().savefig('./plot_complex64/delta_sigma.png')


