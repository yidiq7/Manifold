import sympy as sp
import numpy as np
class Patches():

    def __init__(self, coordinates, function, dimensions, points, norm_coordinate):
        self.coordinates = coordinates
        self.function = function
        self.dimensions = dimensions
        self.points = points
        self.norm_coordinate = norm_coordinate
        self.n_points = len(self.points)
        self.holo_volume_form = self.__get_holvolform()

    def print_all_points(self):
        print("All points on this patch:")
        print(self.points)

    def eval(self, expr_name):
        expr_array = np.array(getattr(self, expr_name))
        #print(expr_array)
        expr_array_evaluated = []
        for point in self.points:
            expr_evaluated = []
            for expr in np.nditer(expr_array, flags=['refs_ok']):
                #print(expr)
                expr = expr.item(0)
                expr = expr.subs([(self.coordinates[i], point[i])
                                   for i in range(self.dimensions)])
                expr_evaluated.append(sp.simplify(expr))
            expr_array_evaluated.append(expr_evaluated)
        expr_array_evaluated = np.array(expr_array_evaluated)
        return expr_array_evaluated

    def __get_holvolform(self):
        holvolform = []
        for i in range(len(self.coordinates)):
            if i == self.norm_coordinate:
                continue
            holvolform_i = 1/self.function.diff(self.coordinates[i])
            holvolform.append(holvolform_i)
        return holvolform

    # def eval_holvolform(self):
    #    holvolform = []
    #    holvolform_npt = []
    #    for n in range(self.n_points):
    #        for i in range(len(self.coordinates)):
    #            holvolform_i = 1/self.function.diff(self.coordinates[i])
    #            holvolform_eval = holvolform_i.subs([(self.coordinates[j], self.points[n][j]) for j in range(len(self.coordinates))])
    #            holvolform_npt.append(sp.simplify(holvolform_eval))
    #        holvolform.append(holvolform_npt)
    #    return holvolform

   # def eval_omegaomegabar():

