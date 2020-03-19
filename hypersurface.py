import numpy as np
import sympy as sp
from manifold import *
# In manifold and type
class Hypersurface(Manifold):

    def __init__(self, coordinates, function, dimensions, n_points):
        super().__init__(dimensions) # Add one more variable for dimension
        self.function = function
        self.coordinate = coordinates
        self.n_points = n_points
        self.__zpairs = self.__generate_random_pair()
        self.points = self.__solve_points()
        self.patches = []
        self.__autopatch()
    #def HolVolForm(F, Z, j)


    def reset_patchwork(self, n_patches):
        self.patches = [None]*n_patches

    def set_patch(self, point, patch):
        patch.append(point)
    # Private:

    def __generate_random_pair(self):
        z_random_pair = []
        for i in range(self.n_points):
            zv = []
            for j in range(2):
                zv.append([complex(c[0],c[1]) for c in np.random.normal(0.0,1.0,(self.dimensions,2))])
            z_random_pair.append(zv)
        return(z_random_pair)

    def __solve_points(self):
        points = []
        for zpair in self.__zpairs:
            a = sp.symbols('a')
            line = [zpair[0][i]+(a*zpair[1][i]) for i in range(self.dimensions)]
            function_valued = self.function.subs([(self.coordinate[i], line[i])
                                                  for i in range(self.dimensions)])
            a_solved = sp.solvers.solve(sp.Eq(function_valued),a)
            for pram_a in a_solved:
                points.append([sp.simplify(zpair[0][i]+(pram_a*zpair[1][i]))
                               for i in range(self.dimensions)])
        return(points)

    def __autopatch(self):
        self.reset_patchwork(self.dimensions)
        for i in range(self.dimensions):
            self.patches[i] = []
        for point in self.points:
            norms = np.absolute(point)
            for i in range(self.dimensions):
                if norms[i] == max(norms):
                    self.set_patch(point, self.patches[i])
    #Add class section
    #self. expr = sympy
    #def pt set
    #contains derivatives etc
