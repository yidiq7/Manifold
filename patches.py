import sympy as sp
import numpy as np
from hypersurface import *

class Patches():

    def __init__(self, coordinates, function, dimensions, points, norm_coordinate):
        self.hypersurface = Hypersurface()
        self.coordinates = coordinates
        self.function = function
        self.dimensions = dimensions
        self.points = points
        self.norm_coordinate = norm_coordinate
        self.n_points = len(self.points)
        self.patches = []
        self.grad = self.get_grad()
        self.holo_volume_form = self.get_holvolform()

    def print_all_points(self):
        print("All points on this patch:")
        print(self.points)

    def set_patch(self, points_on_patch, norm_coordinate=None):
        new_patch = Patches(self.coordinates, self.function, self.dimensions,
                            points_on_patch, norm_coordinate)
        self.patches.append(new_patch)

    def eval(self, expr_name):
        expr_array = np.array(getattr(self, expr_name))
        #print(expr_array)
        expr_array_evaluated = []
        for point in self.points:
            expr_evaluated = []
            for expr in np.nditer(expr_array, flags=['refs_ok']):
                expr = expr.item(0)
                expr = expr.subs([(self.coordinates[i], point[i])
                                   for i in range(self.dimensions)])
                expr_evaluated.append(sp.simplify(expr))
            expr_array_evaluated.append(expr_evaluated)
        expr_array_evaluated = np.array(expr_array_evaluated)
        return expr_array_evaluated

    def get_grad(self):
        return self.hypersurface.get_grad()
        #grad = []
        #for i in range(len(self.coordinates)):
        #    if i == self.norm_coordinate:
        #        continue
        #    grad_i = self.function.diff(self.coordinates[i])
        #    grad.append(grad_i)
        #return grad

    def get_holvolform(self):
        return self.hypersurface.get_holvolform()
        #holvolform = []
        #for i in range(len(self.coordinates)):
        #    if i == self.norm_coordinate:
        #        continue
        #    holvolform_i = 1/self.grad[i]
        #    holvolform.append(holvolform_i)
        #return holvolform

# Sub class of patches
# Redesign Eval function
# How to define sub patches?
