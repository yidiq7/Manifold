# Changelog

## [Unreleased]
- Since the eval() function will slow down the code, we may change the get_ fuctions from returning a sympy expression to a lambdified function given some specific arguments. More details in the "changed" section in 0.1.0

## [0.1.1] - 2020-05-27
### Added
- Multiprocessing support for the integrate method. Since neither the standard library (using pickle) nor the pathos (using dill) can pickle lambda functions, a new library: loky is needed here. Loky uses cloudpickle, which is much slower when dealing with large lists. So the integration will actually be even slower when k is small, but it can save a lot of time with larger k (tested for k = 4)  

### Changed
- The "loop over patches" process is now wrapped inside the integrate method. User need to write the integrated expression in a lambda function.
- Updated the integration notebook with the method.

## [0.1.0] - 2020-05-25
### Added
- Multiprocessing support in solve_points() function
- A static method solve_poly() was separated to get multiprocessing working in a class

### Changed
- The solve_points() function was rewritten. The function now plugs c*a+b into the hypersurface and expresses it as a polynomial of c. Then the coeffients will be lambdified as a function of the zpairs a and b. After which, the numerical results will be calculated and sent to mpmath.polyroots to get the roots. In this way, the symbolic and numerical calculation can be seperated completely and the code will run faster.

- When creating the subpatches in autopatch(), eval() function was replaced since it wraps the lambdify and evalution together. Therefore, the lambdify function will be looped multiple times and slow down the code dramatically.
