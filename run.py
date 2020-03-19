from hypersurface import *
import sympy as sp
d = 3
Z = [sp.symbols('z_'+str(i+1)) for i in range(d)]
f = sum([k**3 for k in Z])
K3 = Hypersurface(Z, f, d, 10)
print(len(K3.points))
print(K3.dimensions)


