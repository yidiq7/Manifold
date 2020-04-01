import sympy as sp
class Patches():

    def __init__(self, coordinates, function, dimensions, points):
        self.coordinates = coordinates
        self.function = function
        self.dimensions = dimensions
        self.points = points
        self.n_points = len(self.points)
        self.holo_volume_form = self.__get_holvolform()

    def print_all_points(self):
        print("All points on this patch:")
        print(self.points)

    def __get_holvolform(self):
        holvolform = []
        for n in range(self.n_points):
            for i in range(len(self.coordinates)):
                holvolform_i = 1/self.function.diff(self.coordinates[i])
                holvolform.append(holvolform_i)
        return holvolform
'''
    def eval_holvolform(self):
        holvolform = []
        holvolform_npt = []
        for n in range(self.n_points):
            for i in range(len(self.coordinates)):
                holvolform_i = 1/self.function.diff(self.coordinates[i])
                holvolform_eval = holvolform_i.subs([(self.coordinates[j], self.points[n][j]) for j in range(len(self.coordinates))])
                holvolform_npt.append(sp.simplify(holvolform_eval))
            holvolform.append(holvolform_npt)
        return holvolform

   # def eval_omegaomegabar():
'''
