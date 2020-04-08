from hypersurface import *
import sympy as sp
d = 4
Z = [sp.symbols('z_'+str(i+1)) for i in range(d)]
f = sum([k**4 for k in Z])
HS = Hypersurface(Z, f, d, 2)
#print(len(K3.points))
#print(K3.dimensions)
HS.list_patches()
HS.print_all_points()
print(' ')
HS.patches[0].print_all_points()
print('Holomorphic volume form on all patches')
print(HS.holo_volume_form)
print('Holomorphic volume form on patch 1:')
print(HS.patches[0].holo_volume_form)
print("Evaluate holo_volume_form:")
print(HS.eval("holo_volume_form"))
print("Evaluate holo_volume_form on patch 0:")
print(HS.patches[0].eval("holo_volume_form"))
