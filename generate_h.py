import itertools
import math
import numpy as np
import sympy as sp

def get_sym_info(HS):
    # Get the power of the coordinates
    k = HS.k
    sec, ns = HS.get_sections(k)
    dim = HS.dimensions
    h_diag = []
    for expr in sec:
        power = []
        for i in range(dim):
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
            for l in range(dim-1):
                if (pz[l]-pzbar[l]-pz[l+1]+pzbar[l+1]) % dim != 0:
                    h_matrix_row.append([])
                    break
            else:
                h_matrix_row.append([pz,pzbar])
                h_params[i][j] = -1
        h_matrix.append(h_matrix_row)

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
                        for perm in itertools.permutations(range(dim)):
                            if h_matrix[i][j] == [[h_matrix[m][n][0][p] for p in perm],[h_matrix[m][n][1][p] for p in perm]]:
                                h_params[m][n] = param
                                h_params[n][m] = param
                                break
            param += 1
            if i == 0 and j == 0:
                continue
            else:
                h_type.append(i!=j)

    h_sym = {'sym': h_params, 'complex': h_tpye}
    return h_sym


def number_of_real_parameters(h_sym):
    return sum([2 if x else 1 for x in h_sym['complex']])


def param_to_matrix(param, h_sym):
    h_matrix = np.array(h_sym['sym'], dtype='complex')
    i_cpx = len(h_sym['complex'])
    for i in range(len(h_sym['complex'])):
         if not h_sym['complex'][i]:
            x = exp(param[i])
            for m in range(len(h_sym['sym'])):
                if h_sym['sym'][m][m] == i+2:
                    h_matrix[m][m] = x
                    #print(h_matrix[m][m])
    for i in range(len(h_sym['complex'])):
        if h_sym['complex'][i]:
            x = exp(complex(-(param[i] - 1)**2, param[i_cpx]))
            for m in range(len(h_sym['sym'])):
                for n in range(m, len(h_sym['sym'])):
                    if h_sym['sym'][m][n] == i+2:
                        # these should all be related by symmetry - check?
                        xn = x * sqrt(h_matrix[m][m]*h_matrix[n][n])
                        h_matrix[m][n] = xn
                        h_matrix[n][m] = np.conj(xn)

            i_cpx += 1
    return h_matrix

def initial_FS_param(HS, h_sym):
    param = np.zeros(number_of_real_parameters(h_sym))
    coeffs = sp.Poly(sp.expand((sum(HS.coordiantes))**HS.k), HS.coordinates).coeffs()
    for i in range(len(h_sym['complex'])):
        if h_sym['complex'][i] is False:
            # Found the corresponding element in h_sym 
            for j in range(len(h_sym['sym'])):
                if h_sym['sym'][j][j] == i + 2:
                    param[i] = math.log(coeffs[j])
    return param
