from hypersurface_tf import *
from generate_h import *
import sympy as sp
from pprint import pprint
from scipy.optimize import minimize
from sympy.utilities.iterables import flatten
import math
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

psi = float(sys.argv[1])
k = int(sys.argv[2])
n_points = int(sys.argv[3])
seed = int(sys.argv[4])

np.random.seed(seed)

z0, z1, z2, z3, z4= sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + 5*psi*z0*z1*z2*z3*z4
HS = Hypersurface(Z, f, n_points)
HS.set_k(k)

factor = HS.integrate(lambda patch: patch.num_eta_tf('FS'), holomorphic=True, tensor=True)

h_sym = get_sym_info(HS)

def integration(param): 
    h = param_to_matrix(param, h_sym)
    #h = np.matmul(g, np.conj(g.transpose()))
    
    integration = HS.integrate(lambda patch: tf.abs(patch.num_eta_tf(h)/factor - 1), tensor=True)
    #print(HS.integrate(lambda patch: patch.num_FS_volume_form_tf(h)/patch.num_FS_volume_form_tf('FS'), tensor=True))
    #print(integration)
    #print(param)
    #try:
    #    np.linalg.cholesky(h)
    #except:
    #    print("Not positive definite")
    return integration

g0 = initial_FS_param(HS, h_sym)

res = minimize(integration, g0, method='L-BFGS-B', options={'ftol': 1e-06, 'maxiter':200})
h_minimal = param_to_matrix(res.x, h_sym)


sigma = HS.integrate(lambda patch: tf.abs(patch.num_eta_tf(h_minimal)/factor - 1), tensor=True)

delta_sigma = math.sqrt(HS.integrate(lambda patch: (tf.abs(patch.num_eta_tf(h_minimal)/factor - 1) - sigma)**2, tensor=True) / HS.n_points)

print('psi =', psi ,', k =', k, ', n_points =', n_points, ', seed =', seed)
print('init =', g0)
print('param =', res.x)
print('sigma =', sigma)
print('delta_sigma =', delta_sigma)
