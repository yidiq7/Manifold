import numpy as np
import sympy as sp
from manifold import *
from mpmath import *
from multiprocessing import Pool
import time
from loky import get_reusable_executor
#from patches import *

class Hypersurface(Manifold):

    def __init__(self, coordinates, function,
                 n_pairs=0, points=None, norm_coordinate=None,
                 max_grad_coordinate=None):
        #super().__init__(dimensions) # Add one more variable for dimension
        self.function = function
        self.coordinates = np.array(coordinates)
        self.dimensions = len(self.coordinates)
        self.norm_coordinate = norm_coordinate
        # The symbolic coordiante is self.coordiante[self.norm_coordiante]
        self.max_grad_coordinate = max_grad_coordinate
        # Range 0 to n-2, this works only on subpatches where max grad is calculated
        # Symbolically self.affin_coordinate[self.max_grad_coordinate]
        if norm_coordinate is not None:
            self.affine_coordinates = np.delete(self.coordinates, norm_coordinate)
        else:
            self.affine_coordinates = self.coordinates
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
        self.hol_n_form = self.get_hol_n_form()
        self.omega_omegabar = self.get_omega_omegabar()
        #self.sections, self.n_sections = self.get_sections(self.dimensions)
        self.FS_Metric = self.get_FS()
        #self.transition_function = self.__get_transition_function()

    def reset_patchwork(self):
        self.patches = []

    def set_patch(self, points_on_patch, norm_coord=None, max_grad_coord=None):
        new_patch = Hypersurface(self.coordinates, self.function, 
                                 points=points_on_patch, norm_coordinate=norm_coord,
                                 max_grad_coordinate=max_grad_coord)
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
            coordinate_normalized = coordinate / norm_coefficient
            point_normalized.append(coordinate_normalized)
        return point_normalized

    def print_all_points(self):
        print("All points on this hypersurface:")
        print(self.points)

    def eval(self, expr, point):
        f = sp.lambdify(self.coordinates, expr)
        expr_evaluated = f(*point)
        return expr_evaluated


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

    def summarize(self, lambda_expr):
        summation = 0
        points = np.array(self.points)
        if self.patches == []:
            time0 = time.time()
            f = sp.lambdify([self.coordinates], lambda_expr(self), "numpy")
            with get_reusable_executor() as executor:
                summation = sum(list(executor.map(f, self.points)))
                #print(value)
                #for value in executor.map(f, self.points):
                #    summation += value
            #for point in self.points:
            #     value = f(point)
            #     summation += value
                 #if np.absolute(value) < 100 and np.absolute(value) > -100:
                 #    summation += value
                 #else:
                 #    print("Possible division of a small number:", value)
        else:
            for patch in self.patches:
                summation += patch.summarize(lambda_expr)
        return summation

    def integrate(self, lambda_expr):
        summation = self.summarize(lambda_expr)
        integration = complex(summation / self.n_points)
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

    @staticmethod
    def solve_poly(zpair, coeff):
        # For each zpair there are d solutions, where d is the dimensions
        points_d = []
        c_solved = polyroots(coeff) 
        for pram_c in c_solved:
            points_d.append([pram_c * a + b for (a, b) in zip(zpair[0], zpair[1])]) 
        return points_d
    
    def __solve_points(self, n_pairs):
        points = []
        zpairs = self.__generate_random_pair(n_pairs)
        coeff_a = [sp.symbols('a'+str(i)) for i in range(self.dimensions)]
        coeff_b = [sp.symbols('b'+str(i)) for i in range(self.dimensions)]
        c = sp.symbols('c')
        coeff_zip = zip(coeff_a, coeff_b)
        line = [c*a+b for (a, b) in coeff_zip]
        function_eval = self.function.subs([(self.coordinates[i], line[i])
                                            for i in range(self.dimensions)])
        poly = sp.Poly(function_eval, c)
        coeff_poly = poly.coeffs()
        get_coeff = sp.lambdify([coeff_a, coeff_b], coeff_poly)
        # Multiprocessing. Then append the points to the same list in the main process
        with Pool() as pool:
            for points_d in pool.starmap(Hypersurface.solve_poly,
                                         zip(zpairs, [get_coeff(zpair[0], zpair[1])
                                                      for zpair in zpairs])):
                points.extend(points_d)
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
            grad_eval = sp.lambdify(self.coordinates, patch.grad)
            for point in patch.points:
                grad = grad_eval(*point)
                grad_norm = np.absolute(grad)
                for i in range(self.dimensions-1):
                    if grad_norm[i] == max(grad_norm):
                        points_on_patch[i].append(point)
                        continue
            for i in range(self.dimensions-1):
                patch.set_patch(points_on_patch[i], patch.norm_coordinate,
                                max_grad_coord=i)
            # Reinitialize the affine patches after generating subpatches
            patch.initialize_basic_properties()

    def get_FS(self):
        FS_metric = self.kahler_metric(np.identity(self.dimensions, dtype=int), k=1)
        return FS_metric

    def get_grad(self):
        grad = []
        if self.patches == []:
            for coord in self.affine_coordinates:
                grad_i = self.function.diff(coord)
                grad.append(grad_i)
        else:
            for patch in self.patches:
                grad.append(patch.grad)
        return grad

    def get_hol_n_form(self):
        hol_n_form = []
        if self.patches == [] and self.max_grad_coordinate is not None:
        # The later condition is neccessary due to the initialization
            hol_n_form = self.grad[self.max_grad_coordinate]
        else:
            for patch in self.patches:
                hol_n_form.append(patch.hol_n_form)
        return hol_n_form

    def get_omega_omegabar(self):
        omega_omegabar = []
        if self.patches == [] and self.max_grad_coordinate is not None:
            hol_n_form = self.hol_n_form
            omega_omegabar = hol_n_form * sp.conjugate(hol_n_form)
        else:
            for patch in self.patches:
                omega_omegabar.append(patch.omega_omegabar)
        return omega_omegabar

    def get_sections(self, k):
        sections = []
        t = sp.symbols('t')
        GenSec = sp.prod(1/(1-(t*zz)) for zz in self.coordinates)
        poly = sp.series(GenSec, t, n=k+1).coeff(t**k)
        while poly!=0:
            sections.append(sp.LT(poly))
            poly = poly - sp.LT(poly)
        n_sections = len(sections)
        sections = np.array(sections)
        return sections, n_sections
    # just one potential
    def kahler_potential(self, h_matrix=None, k=1):
        #need to generalize this for when we start implementing networks
        sections, n_sec = self.get_sections(k)
        if h_matrix is None:
            h_matrix = sp.MatrixSymbol('H', n_sec, n_sec)
        zbar_H_z = np.matmul(sp.conjugate(sections),
                             np.matmul(h_matrix, sections))
        if self.norm_coordinate is not None:
            zbar_H_z = zbar_H_z.subs(self.coordinates[self.norm_coordinate], 1)
        kahler_potential = sp.log(zbar_H_z)
        return kahler_potential

    def kahler_metric(self, h_matrix=None, k=1):
        pot = self.kahler_potential(h_matrix, k)
        metric = []
        #i holomorphc, j anti-hol
        for coord_i in self.affine_coordinates:
            a_holo_der = []
            for coord_j in self.affine_coordinates:
                a_holo_der.append(diff_conjugate(pot, coord_j))
            metric.append([diff(ah, coord_i) for ah in a_holo_der])
        metric = sp.Matrix(metric)
        return metric

    def get_restriction(self, ignored_coord=None):
        if ignored_coord is None:
            ignored_coord = self.max_grad_coordinate
        ignored_coordinate = self.affine_coordinates[ignored_coord]
        local_coordinates = sp.Matrix(self.affine_coordinates).subs(ignored_coordinate,                                                                   self.function)
        affine_coordinates = sp.Matrix(self.affine_coordinates)
        restriction = local_coordinates.jacobian(affine_coordinates).inv()
        restriction.col_del(ignored_coord)
        return restriction
        # Todo: Add try except in this function 

    def get_FS_volume_form(self, h_matrix=None, k=1):
        kahler_metric = self.kahler_metric(h_matrix, k)
        restriction = self.get_restriction()
        FS_volume_form = restriction.T.conjugate() * kahler_metric * restriction
        FS_volume_form = FS_volume_form.det()
        return FS_volume_form
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
