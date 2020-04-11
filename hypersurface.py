import numpy as np
import sympy as sp
from manifold import *
from patches import *

# In manifold and type
class Hypersurface(Manifold):

    def __init__(self, coordinates, function, dimensions, n_points):
        super().__init__(dimensions) # Add one more variable for dimension
        self.function = function
        self.coordinates = coordinates
        self.n_points = n_points
        self.__zpairs = self.__generate_random_pair()
        self.points = self.__solve_points()
        self.patches = []
        self.__autopatch()
        self.holo_volume_form = self.__get_holvolform()
        self.transition_function = self.__get_transition_function()
    #def HolVolForm(F, Z, j)


    def reset_patchwork(self):
        #self.patches = [None]*n_patches
        self.patches = []

    def set_patch(self, points_on_patch, norm_coordinate):
        #patch.append(point)
        #for points in points_on_patch:
        #    new_patch = Patches(self.coordinates, self.function, self.dimensions, points)
        #    self.patches.append(new_patch)
        new_patch = Patches(self.coordinates, self.function, self.dimensions,
                            points_on_patch, norm_coordinate)
        self.patches.append(new_patch)
        
    def list_patches(self):
        print("Number of Patches:", len(self.patches))
        i = 1
        for patch in self.patches:
            print("Points in patch", i, ":", len(patch.points))
            i = i + 1

    def normalize_point(self, point, norm_coordinate):
        point_normalized = []
        for coordinate in point:
            norm_coefficient = point[norm_coordinate]
            coordinate_normalized = sp.simplify(coordinate / norm_coefficient)
            point_normalized.append(coordinate_normalized)
        return point_normalized

    def print_all_points(self):
        print("All points on this hypersurface:")
        print(self.points)

    def eval(self, expr_name):
        expr_evaluated = []
        for patch in self.patches:
            expr_evaluated.append(patch.eval(expr_name))
        expr_evaluated = np.array(expr_evaluated)
        return expr_evaluated



    # Private:

    def __generate_random_pair(self):
        z_random_pair = []
        for i in range(self.n_points):
            zv = []
            for j in range(2):
                zv.append([complex(c[0],c[1]) for c in np.random.normal(0.0, 1.0, (self.dimensions, 2))])
            z_random_pair.append(zv)
        return(z_random_pair)

    def __solve_points(self):
        points = []
        for zpair in self.__zpairs:
            a = sp.symbols('a')
            line = [zpair[0][i]+(a*zpair[1][i]) for i in range(self.dimensions)]
            function_eval = self.function.subs([(self.coordinates[i], line[i])
                                                for i in range(self.dimensions)])
            #print(sp.expand(function_eval))
            #function_lambda = sp.lambdify(a, function_eval, ["scipy", "numpy"])
            #a_solved = fsolve(function_lambda, 1)
            a_solved = sp.polys.polytools.nroots(function_eval)
            #a_rational = sp.solvers.solve(sp.Eq(sp.nsimplify(function_eval, rational=True)),a)
            # print("Solution for a_lambda:", a_poly)
            # a_solved = sp.solvers.solve(sp.Eq(sp.expand(function_eval)),a)
            for pram_a in a_solved:
                points.append([zpair[0][i]+(pram_a*zpair[1][i])
                               for i in range(self.dimensions)])
        return points

    # def __autopatch(self):
    #    self.reset_patchwork()
    #    #self.reset_patchwork(self.dimensions)
    #    #for i in range(self.dimensions):
    #    #     self.patches[i] = []
    #     points_on_patch = [[] for i in range(self.dimensions)]
    #     for point in self.points:
    #         norms = np.absolute(point)
    #         for i in range(self.dimensions):
    #             if norms[i] == max(norms):
    #                 point_normalized = self.normalize_point(point, i)
    #                 points_on_patch[i].append(point_normalized) 
    #     self.set_patch(points_on_patch)
    #                #self.set_patch(point, self.patches[i])
    #                 # remake patch here


    def __autopatch(self):
        self.reset_patchwork()
        for i in range(self.dimensions):
            points_on_patch = []
            for point in self.points:
                norms = np.absolute(point)
                if norms[i] == max(norms):
                    point_normalized = self.normalize_point(point, i)
                    points_on_patch.append(point_normalized)
            print("point ")
            self.set_patch(points_on_patch, i)

    def __get_transition_function(self):
        return None

    def __get_holvolform(self):
        holvolform = []
        for i in range(len(self.patches)):
            holvolform.append(self.patches[i].holo_volume_form)
        return holvolform
    #Add class section
    #self. expr = sympy
    #def pt set
    #contains derivatives etc

