from hypersurface import *
import sympy as sp
d = 3
Z = [sp.symbols('z_'+str(i+1)) for i in range(d)]
f = sum([k**3 for k in Z])
HS = Hypersurface(Z, f, d, 10)
#print(len(K3.points))
#print(K3.dimensions)
HS.list_patches()
HS.print_all_points()
print(' ')
HS.patches[0].print_all_points()
print('Holomorphic volume form on all patches')
print(HS.eval_holvolform())
print('Holomorphic volume form on patch 1:')
print(HS.patches[0].eval_holvolform())
