from hypersurface import *
import sympy as sp
from pprint import pprint
from scipy.optimize import minimize
from sympy.utilities.iterables import flatten
import math
import itertools
import sys

psi = float(sys.argv[1])
k = int(sys.argv[2])
n_points = int(sys.argv[3])

z0, z1, z2, z3, z4= sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + 5*psi*z0*z1*z2*z3*z4
HS = Hypersurface(Z, f, n_points)
HS.set_k(k)

factor = HS.integrate(lambda patch, point, h_matrix='FS': 
                            patch.num_eta(h_matrix, point), 
                            holomorphic=True, numerical=True)
def get_h_matrix(HS, k):
    # Get the power of the coordinates
    sec, ns = HS.get_sections(k)
    h_diag = []
    for expr in sec:
        power = []
        for i in range(len(Z)):
            power.append(expr.diff(Z[i])*Z[i]/expr)
        h_diag.append(power)

    h_matrix = []
    h_params = np.zeros((ns, ns))
    # make a pair for each matrix element
    for i in range(len(h_diag)):
        h_matrix_row = []
        pz = h_diag[i]
        for j in range(len(h_diag)):
            pzbar = h_diag[j]
            #if k >= len(Z):
            for l in range(len(Z)-1):
                if (pz[l]-pzbar[l]-pz[l+1]+pzbar[l+1]) % 5 != 0:
                    h_matrix_row.append([])
                    break
            else:
                h_matrix_row.append([pz,pzbar])
                h_params[i][j] = -1
                    #if (pz!=pzbar):
                    #print( "non-diagonal", pz, pzbar )
            #else:
                #if i == j:
                   # h_matrix_row.append([pz,pzbar])
                   # h_params[i][j] = -1
                #else:
               #     h_matrix_row.append([])
                    
        h_matrix.append(h_matrix_row)                  
    #print( h_matrix[2][2] )
    
    h_type = []
    param = 1
    for i in range(ns):
        for j in range(ns):
            if h_params[i][j] > -1:
                continue
            for m in range(i, ns):
                for n in range(ns):
                    if h_params[m][n] > -1:
                        continue
                    if sorted(h_matrix[m][n][0]) == sorted(h_matrix[i][j][0]):
                        for perm in itertools.permutations(range(5)):
                            if h_matrix[i][j] == [[h_matrix[m][n][0][p] for p in perm],[h_matrix[m][n][1][p] for p in perm]]:
                                h_params[m][n] = param
                                h_params[n][m] = param
                                break
            param += 1
            if i == 0 and j == 0:
                continue
            else:
                h_type.append(i != j)
            
    return (h_params, h_type)


(h_sym, h_complex) = get_h_matrix(HS, k)


def number_of_real_parameters(h_complex):
    return sum([2 if x else 1 for x in h_complex])

def param_to_matrix(param):
    h_matrix = np.array(h_sym,dtype='complex')
    i_cpx = len(h_complex)
    for i in range(len(h_complex)):
         if not h_complex[i]:
            x = exp(param[i])
            for m in range(len(h_sym)):
                if h_sym[m][m] == i+2:
                    h_matrix[m][m] = x
                    #print(h_matrix[m][m])
    for i in range(len(h_complex)):
        if h_complex[i]:
            x = exp(complex(-(param[i] - 1)**2, param[i_cpx]))
            for m in range(len(h_sym)):
                for n in range(m,len(h_sym)):
                    if h_sym[m][n] == i+2:
                        # these should all be related by symmetry - check?
                        xn = x * sqrt(h_matrix[m][m]*h_matrix[n][n])
                        h_matrix[m][n] = xn
                        h_matrix[n][m] = np.conj(xn)
            i_cpx += 1
    return h_matrix

def integration(param): 
    h = param_to_matrix(param)
    #h = np.matmul(g, np.conj(g.transpose()))
    
    integration = HS.integrate(lambda patch, point, h_matrix=h: 
                                np.absolute(patch.num_eta(h_matrix, point)/factor - 1).real, 
                                holomorphic=True, numerical=True)
    integration = integration.real
    #print(HS.integrate(lambda patch, point: patch.num_FS_volume_form(h, point)/patch.num_FS_volume_form('identity', point), numerical=True))
    #print(integration)
    #print(param)
    #try:
    #    np.linalg.cholesky(h)
    #except:
    #    print("Not positive definite")
    return integration


g0 = np.zeros(number_of_real_parameters(h_complex))
coeffs = sp.Poly(sp.expand((z0 + z1 + z2 + z3 + z4)**k), Z).coeffs()
for i in range(len(h_complex)):
    if h_complex[i] is False:
        # Found the corresponding element in h_sym 
        for j in range(len(h_sym)):
            if h_sym[j][j] == i + 2:
                g0[i] = math.log(coeffs[j]) 
h = param_to_matrix(g0)

res = minimize(integration, g0, method='L-BFGS-B', options={'ftol': 1e-06, 'maxiter':200})
h_minimal = param_to_matrix(res.x)


sigma = HS.integrate(lambda patch, point, h_matrix=h_minimal: 
                         np.absolute(patch.num_eta(h_matrix, point)/factor - 1).real, 
                         holomorphic=True, numerical=True)

delta_sigma = math.sqrt(HS.integrate(lambda patch, point, h_matrix=h_minimal: 
                         (np.absolute(patch.num_eta(h_matrix, point)/factor - 1).real - sigma)**2, 
                         holomorphic=True, numerical=True)/HS.n_points)

print('psi =', psi ,', k =', k, ', n_points =', n_points)
print('param =', res.x)
print('fun =', res.fun)
print('jac =', res.jac)
print('sigma =', sigma, 'delta_sigma =', delta_sigma)
