import numpy as np
import sympy as sp
from manifold import *
#from patches import *

class Hypersurface(Manifold):

    def __init__(self, coordinates, function,
                 dimensions, n_pairs=0, points=None, norm_coordinate=None):
        super().__init__(dimensions) # Add one more variable for dimension
        self.function = function
        self.coordinates = np.array(coordinates)
        self.norm_coordinate = norm_coordinate
        # The symbolic coordiante is self.coordiante[self.norm_coordiante]
        self.patches = []
        if points is None:
            self.points = self.__solve_points(n_pairs)
            self.__autopatch()
        else:
            self.points = points
        self.n_points = len(self.points)
        self.n_patches = len(self.patches)
        self.initialize_basic_properties()
    
    def initialize_basic_properties(self):
        # This function is necessary because those variables need to be updated on
        # the projective patches after subpatches are created. Then this function will
        # be reinvoked.
        self.grad = self.get_grad()
        self.holo_volume_form = self.get_holvolform()
        self.omega_omegabar = self.get_omega_omegabar()
        self.sections, self.n_sections = self.get_sections()
        self.FS_Metric = self.get_FS()
        #self.transition_function = self.__get_transition_function()

    def reset_patchwork(self):
        self.patches = []

    def set_patch(self, points_on_patch, norm_coord=None):
        new_patch = Hypersurface(self.coordinates, self.function, self.dimensions,
                                 points=points_on_patch, norm_coordinate=norm_coord)
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

    def eval(self, expr, point):
        expr_array = np.array(expr)
        expr_array_evaluated = []
        for expr_i in np.nditer(expr_array, flags=['refs_ok']):
            # In case you want to integrate a constant
            try:
                expr_evaluated = expr_i.item(0).subs([(self.coordinates[i], point[i])
                                                      for i in range(self.dimensions)])
            except AttributeError:
                expr_evaluated = expr
            expr_array_evaluated.append(sp.simplify(expr_evaluated))
        return expr_array_evaluated


    def eval_all(self, expr_name):
        #expr_array = np.array(getattr(self, expr_name))
        expr_array = np.array(expr_name)
        expr_array_evaluated = []
        if self.patches == []:
            for point in self.points:
                expr_evaluated = []
                for expr in np.nditer(expr_array, flags=['refs_ok']):
                    expr = expr.item(0)
                    expr = expr.subs([(self.coordinates[i], point[i])
                                       for i in range(self.dimensions)])
                    expr_evaluated.append(sp.simplify(expr))
                expr_array_evaluated.append(expr_evaluated)
        else:
            for patch in self.patches:
                expr_array_evaluated.append(patch.eval_all(expr_name))
        return expr_array_evaluated

    def integrate(self, expr):
        summation = 0
        if self.patches == []:
            for point in self.points:
                expr_evaluated = self.eval(expr, point)
                # self.eval() will return a list.
                # Here we suppose there is only one element in that list
                # In other words, the expression being integrated is not a list itself
                summation += expr_evaluated[0]
        else:
            for patch in self.patches:
                summation += patch.integrate(expr) * patch.n_points
        # In case you want to try with few points and n_points might be zero on some
        # patch.
        try:
            integration = summation / self.n_points
        except ZeroDivisionError:
            integration = 0
        return integration


    # Private:

    def __generate_random_pair(self, n_pairs):
        z_random_pair = []
        for i in range(n_pairs):
            zv = []
            for j in range(2):
                zv.append([complex(c[0],c[1]) for c in np.random.normal(0.0, 1.0, (self.dimensions, 2))])
            z_random_pair.append(zv)
        return z_random_pair

    def __solve_points(self, n_pairs):
        points = []
        zpairs = self.__generate_random_pair(n_pairs)
        for zpair in zpairs:
            a = sp.symbols('a')
            line = [zpair[0][i]+(a*zpair[1][i]) for i in range(self.dimensions)]
            function_eval = self.function.subs([(self.coordinates[i], line[i])
                                                for i in range(self.dimensions)])
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

    def __autopatch(self):
        self.reset_patchwork()
        # projective patches
        points_on_patch = [[] for i in range(self.dimensions)]
        for point in self.points:
            norms = np.absolute(point)
            for i in range(self.dimensions):
                if norms[i] == max(norms):
                    point_normalized = self.normalize_point(point, i)
                    points_on_patch[i].append(point_normalized)
                
                    continue
        for i in range(self.dimensions):
            self.set_patch(points_on_patch[i], i)
        # Subpatches on each patch
        for patch in self.patches:
            points_on_patch = [[] for i in range(self.dimensions-1)]
            for point in patch.points:
                grad = patch.eval(patch.grad, point)
                grad_norm = np.absolute(grad)
                for i in range(self.dimensions-1):
                    if grad_norm[i] == max(grad_norm):
                        points_on_patch[i].append(point)
                        continue
            for i in range(self.dimensions-1):
                patch.set_patch(points_on_patch[i], patch.norm_coordinate)
            patch.initialize_basic_properties()

    def get_FS(self):
        FS_metric = self.kahler_metric(np.identity(self.n_sections))
        return FS_metric

    def get_grad(self):
        grad = []
        if self.patches == []:
            for i in range(len(self.coordinates)):
                if i == self.norm_coordinate:
                    continue
                grad_i = self.function.diff(self.coordinates[i])
                grad.append(grad_i)
        else:
            for patch in self.patches:
                grad.append(patch.grad)
        return grad

    def get_holvolform(self):
        holvolform = []
        if self.patches == []:
            for grad_i in self.grad:
                holvolform_i = 1 / grad_i
                holvolform.append(holvolform_i)
        else:
            for patch in self.patches:
                holvolform.append(patch.holo_volume_form)
        return holvolform

    def get_omega_omegabar(self):
        omega_omegabar = []
        if self.patches == []:
            for holvolform_i in self.holo_volume_form:
                omega_omegabar_i = holvolform_i * sp.conjugate(holvolform_i)
                omega_omegabar.append(omega_omegabar_i)
        else:
            for patch in self.patches:
                omega_omegabar.append(patch.omega_omegabar)
        return omega_omegabar

    def get_sections(self):
        sections = []
        t = sp.symbols('t')
        GenSec = sp.prod(1/(1-(t*zz)) for zz in self.coordinates)
        poly = sp.series(GenSec,t,n=self.dimensions+1).coeff(t**(self.dimensions))
        while poly!=0:
            sections.append(sp.LT(poly))
            poly = poly - sp.LT(poly)
        n_sections = len(sections)
        sections = np.array(sections)
        return sections, n_sections
    # just one potential
    def kahler_potential(self, h_matrix=None):
        #need to generalize this for when we start implementing networks
        # Kahler potential only exists locally
        if self.patches == []:
            ns = self.n_sections
            if h_matrix is None:
                h_matrix = sp.MatrixSymbol('H',ns,ns)
            zbar_H_z = np.matmul(sp.conjugate(self.sections),
                                 sp.simplify(np.matmul(h_matrix, self.sections)))
            zbar_H_z = zbar_H_z.subs(self.coordinates[self.norm_coordinate], 1)
            kahler_potential = sp.log(zbar_H_z)
        else:
            kahler_potential = []
            for patch in self.patches:
                kahler_potential.append(patch.kahler_potential(h_matrix))
        return kahler_potential

    def kahler_metric(self, h_matrix=None):
        if self.patches == []:
            pot = self.kahler_potential(h_matrix)
            metric = []
        #i holomorphc, j anti-hol
            for i in range(self.dimensions):
                if i == self.norm_coordinate:
                    continue
                else:
                    for j in range(self.dimensions):
                        if j == self.norm_coordinate:
                            continue
                        else:
                            a_holo_der = []
                            a_holo_der.append(diff_conjugate(pot,self.coordinates[j]))
                    metric.append([diff(ah, self.coordinates[i]) for ah in a_holo_der])
            metric = sp.Matrix(metric)
        else:
            metric = []
            for patch in self.patches:
                metric.append(patch.kahler_metric(h_matrix))
        # vol_element = metric.det() # Total vol_element
        return metric

    

#Can we just define conjugation in this way?
#Have a function inside the class self.conjugate?
def diff_conjugate(expr, coordinate):
    coord_bar = sp.symbols('coord_bar')
    expr_diff = expr.subs(sp.conjugate(coordinate), coord_bar).diff(coord_bar)
    expr_diff = expr_diff.subs(coord_bar, sp.conjugate(coordinate))
    return expr_diff

def diff(expr, coordinate):
    coord_bar = sp.symbols('coord_bar')
    expr_diff = expr.subs(sp.conjugate(coordinate), coord_bar).diff(coordinate)
    expr_diff = expr_diff.subs(coord_bar, sp.conjugate(coordinate))
    return expr_diff

    

#The integration of volume form should not depend on h
#So change h nd calculate the topology integration 
