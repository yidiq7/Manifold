import sympy as sp
from sympy import I
a = sp.symbols("a")
func = -0.401515093225443*a**4 - 2.8986613515382*I*a**4 - 0.0603580239543025*a**3 + 6.17036348824273*I*a**3 + 11.5131559071023*a**2 - 21.4395093053674*I*a**2 + 13.8842177690785*a + 5.29634896476488*I*a + 10.2927012314434 + 1.6924278771672*I
#func = -a**4 - I*a**4 - a**3 + 6*I*a**3 + 11*a**2 - 21*I*a**2 + a + I*a + 10 + 1*I
print(sp.nsimplify(func, rational=True))
a_solved = sp.solvers.solve(sp.Eq(sp.nsimplify(func, rational=True)),a)
