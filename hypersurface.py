import numpy as np
import sympy as sp
from manifold import *
from mpmath import *
from multiprocessing import Pool
import time
#from loky import get_reusable_executor

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
        self.indices = []
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
        #self.omega_omegabar = self.get_omega_omegabar()
        #self.sections, self.n_sections = self.get_sections(self.dimensions)
        #self.FS_Metric = self.get_FS()
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

    def sum_on_patch(self, f, numerical):
        summation = 0
        points = np.array(self.points)
        if self.patches == []:
            if numerical is True:
                # Currying if numerical is true
                def f_patch(patch):
                    def f_point(point):
                        return f(patch, point) 
                    return  f_point
                func = f_patch(self)
                for point in self.points:
                    value = func(point)
                    summation += value
            else:
                func = sp.lambdify([self.coordinates], f(self), "numpy") 
                for point in self.points:
                    value = func(point)
                    summation += value

                #if np.absolute(value) < 5 and np.absolute(value) > -5:
                #    summation += value
                #else:
                #    print("Possible division of a small number:", value)
        else:
            for patch in self.patches:
                summation += patch.sum_on_patch(f,numerical)

            # if numerical is True:
            #    for patch in self.patches:
            #        summation += patch.sum_on_patch(f, numerical)
            # else: 
            #     with get_reusable_executor() as executor:
            #     summation = sum(list(executor.map(lambda x: x.sum_on_patch(f, numerical), self.patches)))

        return summation

    def integrate(self, f, holomorphic=True, numerical=False):
        # f should be a lambda expression given by the user
        # holomorphic=True means integrating over Omega_Omegabar
        if numerical is True:

            # m is the mass formula
            def m(patch, point):
                mass = patch.omega_omegabar(point) / \
                       patch.num_FS_volume_form('identity', point, k=1)
                return mass

            def weighted_f(patch, point):
                weighted_f = f(patch, point) * m(patch, point) 
                return weighted_f
        else:
            m = lambda patch: patch.get_omega_omegabar() / \
                              patch.get_FS_volume_form(k=1)
            weighted_f = lambda patch: f(patch) * m(patch)

        if holomorphic is True:
            # Define a new f with an extra argument user_f and immediatly pass f as
            # the default value, so that f can be updated as f(x) * m(x)
            #f = lambda patch, user_f=f: user_f(patch) * m(patch)
            #m = lambda patch: patch.get_omega_omegabar() / \
            #    patch.get_FS_volume_form(k=1)
            summation = self.sum_on_patch(weighted_f, numerical)
            norm_factor = 1 / self.sum_on_patch(m, numerical)
        else:
            summation = self.sum_on_patch(f, numerical)
            norm_factor = 1 / self.n_points

        integration = summation * norm_factor
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
            points_d.append([complex(pram_c * a + b)
                             for (a, b) in zip(zpair[0], zpair[1])])
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
            grad_eval = sp.lambdify(self.coordinates, patch.grad, 'numpy')
            for point in patch.points:
                grad = grad_eval(*point)
                grad_norm = np.absolute(grad)
                for i in range(self.dimensions-1):
                    if grad_norm[i] == max(grad_norm):
                        points_on_patch[i].append(point)
                        patch.indices.append(i)
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
            hol_n_form = 1 / self.grad[self.max_grad_coordinate]
        else:
            for patch in self.patches:
                hol_n_form.append(patch.hol_n_form)
        return hol_n_form

    def get_omega_omegabar(self, lambdify=False):
        omega_omegabar = []
        if self.patches == [] and self.max_grad_coordinate is not None:
            hol_n_form = self.hol_n_form
            omega_omegabar = hol_n_form * sp.conjugate(hol_n_form)
        else:
            for patch in self.patches:
                omega_omegabar.append(patch.omega_omegabar)
        if lambdify is True:
            omega_omegabar = sp.lambdify([self.coordinates], omega_omegabar,'numpy')
        return omega_omegabar

    def get_sections(self, k, lambdify=False):
        sections = []
        t = sp.symbols('t')
        GenSec = sp.prod(1/(1-(t*zz)) for zz in self.coordinates)
        poly = sp.series(GenSec, t, n=k+1).coeff(t**k)
        while poly!=0:
            sections.append(sp.LT(poly))
            poly = poly - sp.LT(poly)
        n_sections = len(sections)
        sections = np.array(sections)
        if lambdify is True:
            sections = sp.lambdify([self.coordinates], sections, 'numpy')
        return sections, n_sections

    def kahler_potential(self, h_matrix=None, k=1):
        sections, n_sec = self.get_sections(k)
        if h_matrix is None:
            h_matrix = np.identity(n_sec)
        # Check if h_matrix is a string
        elif isinstance(h_matrix, str):
            if h_matrix == "identity":
                h_matrix = np.identity(n_sec)
            elif h_matrix == "symbolic":
                h_matrix = sp.MatrixSymbol('H', n_sec, n_sec)
        
        z_H_zbar = np.matmul(sections, np.matmul(h_matrix, sp.conjugate(sections)))
        if self.norm_coordinate is not None:
            z_H_zbar = z_H_zbar.subs(self.coordinates[self.norm_coordinate], 1)
        kahler_potential = sp.log(z_H_zbar)
        return kahler_potential

    def kahler_metric(self, h_matrix=None, k=1, point=None):
        if point is None:
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

    def get_restriction(self, ignored_coord=None, lambdify=False):
        if ignored_coord is None:
            ignored_coord = self.max_grad_coordinate
        ignored_coordinate = self.affine_coordinates[ignored_coord]
        local_coordinates = sp.Matrix(self.affine_coordinates).subs(ignored_coordinate, self.function)
        affine_coordinates = sp.Matrix(self.affine_coordinates)
        restriction = local_coordinates.jacobian(affine_coordinates).inv()
        restriction.col_del(ignored_coord)
        # Return a function is symbolic is flase
        if lambdify is True:
            restriction = sp.lambdify([self.coordinates], restriction, 'numpy')
        return restriction

    def get_FS_volume_form(self, h_matrix=None, k=1, lambdify=False):
        kahler_metric = self.kahler_metric(h_matrix, k)
        restriction = self.get_restriction()
        FS_volume_form = restriction.T.conjugate() * kahler_metric * restriction
        FS_volume_form = FS_volume_form.det()
        if lambdify is True:
            FS_volume_form = sp.lambdify([self.coordinates], FS_volume_form, 'numpy')
        return FS_volume_form

    # Numerical Methods:

    def set_k(self, k):
        sections, ns = self.get_sections(k, lambdify=False)
        sections_func, ns = self.get_sections(k, lambdify=True)
        self.n_sections = ns
        for patch in self.patches:
            # patch.k = k
            for subpatch in patch.patches:
                # subpatch.k = k
                subpatch.sections = sections_func
                jacobian = sp.Matrix(sections).jacobian(subpatch.affine_coordinates)
                subpatch.sections_jacobian = sp.lambdify([subpatch.coordinates],
                                                         jacobian,'numpy')
                subpatch.restriction = subpatch.get_restriction(lambdify=True)
                subpatch.omega_omegabar = subpatch.get_omega_omegabar(lambdify=True)

    def num_kahler_metric(self, h_matrix, point, k=-1):
        if k == 1:
            # k = 1 will be used in the mass formula during the integration
            s = point
            # Delete the correspoding row
            J = np.delete(np.identity(len(s)), self.norm_coordinate, 0)
        else:
            s = self.sections(point)
            J = self.sections_jacobian(point).T
        if isinstance(h_matrix, str) and h_matrix == 'identity':
            h_matrix = np.identity(len(s))
        H_Jdag = np.matmul(h_matrix, np.conj(J).T)
        A = np.matmul(J, H_Jdag)
        # Get the right half of B then reshape to transpose,
        # since b.T is still b if b is a 1d vector
        b = np.matmul(s, H_Jdag).reshape(-1, 1)
        B = np.matmul(np.conj(b), b.T)
        alpha = np.matmul(s, np.matmul(h_matrix, np.conj(s)))
        G = A / alpha - B / alpha**2
        return G

    def num_FS_volume_form(self, h_matrix, point, k=-1):
        kahler_metric = self.num_kahler_metric(h_matrix, point, k)
        r = self.restriction(point)
        FS_volume_form = np.matmul(np.conj(r).T, np.matmul(kahler_metric, r))
        FS_volume_form = np.matrix(FS_volume_form, dtype=complex)
        FS_volume_form = np.linalg.det(FS_volume_form).real
        return FS_volume_form

    def num_eta(self, h_matrix, point):
        FS_volume_form = self.num_FS_volume_form(h_matrix, point)
        Omega_Omegabar = self.omega_omegabar(point)
        eta = FS_volume_form / Omega_Omegabar
        return eta

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


