from hypersurface import *
import sympy as sp
from pprint import pprint
d = 4
Z = [sp.symbols('z_'+str(i+1)) for i in range(d)]
f = sum([k**4 for k in Z])
HS = Hypersurface(Z, f, d, 2)
#print(len(K3.points))
#print(K3.dimensions)
HS.list_patches()
HS.patches[0].list_patches()
HS.patches[1].list_patches()
HS.patches[2].list_patches()
HS.patches[3].list_patches()
#HS.print_all_points()
print(' ')
#HS.patches[0].print_all_points()
#HS.patches[1].print_all_points()
#HS.patches[2].print_all_points()
#HS.patches[3].print_all_points()

#print('Holomorphic volume form on all patches')
pprint(HS.holo_volume_form)
pprint(HS.integrate(1))
#print('Holomorphic volume form on patch 1:')
#print(HS.patches[0].holo_volume_form)
#pprint("Evaluate holo_volume_form:")
#pprint(HS.eval_all("holo_volume_form"))
#print("Evaluate holo_volume_form on patch 0:")
#print(HS.patches[3].eval("holo_volume_form"))
