from hypersurface_tf import *
from generate_h import *
import sympy as sp
from pprint import pprint
from scipy.optimize import minimize
from sympy.utilities.iterables import flatten
import math
import sys

psi = float(sys.argv[1])
k = int(sys.argv[2])
n_points = int(sys.argv[3])
seed = int(sys.argv[4])

numpy.random.seed(seed)

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
    
    integration = HS.integrate(lambda patch, h_matrix=h: 
                                tf.abs(patch.num_eta_tf(h_matrix)/factor - 1), 
                                holomorphic=True, tensor=True)
    print(HS.integrate(lambda patch: patch.num_FS_volume_form_tf(h)/patch.num_FS_volume_form_tf('FS'), tensor=True))
    print(integration)
    print(param)
    try:
        np.linalg.cholesky(h)
    except:
        print("Not positive definite")
    return integration

g0 = initial_FS_param(HS, h_sym)

res = minimize(integration, g0, method='L-BFGS-B', options={'ftol': 1e-06, 'maxiter':200})
h_minimal = param_to_matrix(res.x)


sigma = HS.integrate(lambda patch, point, h_matrix=h_minimal:
                         np.absolute(patch.num_eta(h_matrix, point)/factor - 1).real,
                         holomorphic=True, numerical=True)

delta_sigma = math.sqrt(HS.integrate(lambda patch, point, h_matrix=h_minimal:
                         (np.absolute(patch.num_eta(h_matrix, point)/factor - 1).real - sigma)**2, 
                         holomorphic=True, numerical=True)/HS.n_points)

print('psi =', psi ,', k =', k, ', n_points =', n_points, ', seed =', seed)
print('init =', g0)
print('param =', res.x)
print('sigma =', sigma, 'delta_sigma =', delta_sigma)
