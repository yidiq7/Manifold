import numpy as np
from sympy import *

class Hypersurface(Manifold):

    def __init__(self, f):
    
        self.function = f
        self.points = solve_points(self.points)

    def solve_points(zpairs):
        points = []
        for zpair in zpairs:
            a = symbol('a')
            line = [zpair[0][i]+(a*zpair[1][i]) for i in range(self.d)]
            a_solved = solvers.solve(self.function(line),a)
            for pram_a in a_solved:
                points.append([zpair[0][i]+(pram_a*zpair[1][i]) for i in range(self.d)])
        return(points)
