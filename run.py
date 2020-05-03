from hypersurface import *
import sympy as sp
from pprint import pprint
import time

start_time = time.time()
z0, z1, z2, z3 = sp.symbols('z0, z1, z2, z3')
Z = [z0,z1,z2,z3]
f = z0**4 + z1**4 + z2**4 + z3**4
HS = Hypersurface(Z, f, 200)
mid_time = time.time()
print("mid_time", mid_time - start_time)
k = 4
sections, ns = HS.get_sections(k)
H = np.identity(ns, dtype=int)
#pprint(HS.patches[0].patches[0].points)

integral = 0
for patch in HS.patches:
    for subpatch in patch.patches:
        sub_integral = subpatch.integrate_lmd(subpatch.get_FS_volume_form(H, k))
        print(sub_integral)
        integral += sub_integral
print(integral)
elapsed_time = time.time() - mid_time
print("time:", elapsed_time)


