# Changelog

## [Unreleased]
## [0.3.4] - 2020-07-29
### Changed
- Deleted @tf.functions for num\_s\_J\_t(), num\_Omega\_Omegabar\_tf(), etc, to speed up set\_k() function
## [0.3.3] - 2020-07-27
### Changed
- Changed the tensors from complex64 to complex128
- Wrap the \_tf functions with tf.functions. This will enbale the graph computation.

## [0.3.2] - 2020-07-25
### Added
- Method initial\_param\_from\_lowerk(HS, h\_sym, param\_low) in generate\_h.py. This basically take the optimized results from a lower k/2 case and and assign the new param so that the k polynomials is approximately the square of the k/2 poly

## [0.3.1] - 2020-07-21
### Added
- Attribute HS.k = k, set in method set\_k()

### Changed
- Separated the generation of h matrix in file generate\_h.py
- Simplified Optimization notebook and script
## [0.3.0] - 2020-07-20
### Added
- New file hypersurface\_tf.py and new optimization routine Optimization3-tensorflow.ipynb
- New \_tf functions which use tensorflow tensors to store the data, which speeds up the calculations.
- Option 'tensor' for method integrate(), tensor should be set as True if \_tf functions are being integrated

## [0.2.5] - 2020-07-12
### Fixed
- In method num\_FS\_volume\_form, it should be r.T \* kahler\_metric \* np.conj(r), instead of np.conj(r).T on the left, since it is z\_H\_zbar in the kahler potential. Same bug is fixed in method get\_FS\_volume\_form()

## [0.2.4] - 2020-07-11
### Added
- A new option 'FS' for kahler potential and metric, which corresponding to log(sum(Z)^k)

## [0.2.3] - 2020-06-23
### Changed
- Optimized the mass formula. Added a new argument k for num\_FS\_volume\_form. The default k=-1 means k will be decided by set\_k() method. k = 1 is used in the integrate() method. The user should not worry about the option.

- Updated the benchmark with the optimized version of integrate().

## [0.2.2] - 2020-06-22
### Added
- Attribute n\_sections after set\_k() is invoked.

- Comparison between "symbolic" and numeric integration in the aspect of speed and accuracy. See notebook Numerical\_benchmark.

### Removed
- Multiprocessing for "symbolic" integration. Since the set\_k() will change the definition of the instance, loky cannot pickle the patches after set\_k() is invoked. But you should only use it as a demonstration or cross check anyway, since now we have a faster numerical version for the integration. Pathos does not have this problem but you have to make the pool non-daemonic when used in a recursive function. See the following link:       
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic/8963618#8963618     
https://github.com/uqfoundation/pathos/issues/169       
This was not implemented because it is not necessary anymore.

## [0.2.1] - 2020-06-21
### Changed
- Method integrate() now supports the integration of numerical functions with argument numerical=True. The input will be a function with two arguments: patch and point. The format will be like lambda patch, point: patch.num\_eta('identity', point). 

- Holomorphic is True by defualt in method integrate().

- Lambdified numerical methods now take an array as input instead of mulitple arguments

## [0.2.0] - 2020-06-21
### Added
- A few pure numerical methods started with num\_

- A new method set\_k() to assgin the parameter k. The numerical methods need to be defined after k is given and before any potential loops to avoid unnecessary redefinition. So those methods will be defined when set\_k() is invoked.

- A new lambdify argument for the get\_symbolic methods. If lambdify=True, they will return a lamdified function from the original expressions.

### Changed
- Update the notebook eta\_on\_rational\_curve with new numerical methods.

## [0.1.4] - 2020-06-20
### Changed
- In kahler\_potential, zbar\_H\_z is changed to z\_H\_zbar

## [0.1.3] - 2020-06-18
### Fixed
- The expression of the holomophic n form Omega, which should be 1/grad.

### Added
- A notebook to plot eta on the rational curve.

## [0.1.2] - 2020-06-07
### Changed
- The multiprocessing to the patches level in sum\_on\_patch method. In this way, the lambdify function can also be multiprocessed. 

## [0.1.1] - 2020-05-27
### Added
- Mass formula in the integrate method. If the option holomorphic is True, the method will use omega\_omegabar as the measure and the mass formula will be applied automatically

- h\_matrix now takes options "identity" and "symbolic". If h\_matrix is omitted in the Kahler metric, etc, it will be set to identity.
- Multiprocessing support for the integrate method. Since neither the standard library (using pickle) nor the pathos (using dill) can pickle lambda functions, a new library: loky is needed here. Loky uses cloudpickle, which is much slower when dealing with large lists. So the integration will actually be even slower when k is small, but it can save a lot of time with larger k (tested for k = 4)  

### Changed
- The "loop over patches" process is now wrapped inside the integrate method. User need to write the integrated expression in a lambda function.
- Updated the integration notebook with the new method.

## [0.1.0] - 2020-05-25
### Added
- Multiprocessing support in solve\_points() function
- A static method solve\_poly() was separated to get multiprocessing working in a class

### Changed
- The solve\_points() function was rewritten. The function now plugs c*a+b into the hypersurface and expresses it as a polynomial of c. Then the coeffients will be lambdified as a function of the zpairs a and b. After which, the numerical results will be calculated and sent to mpmath.polyroots to get the roots. In this way, the symbolic and numerical calculation can be seperated completely and the code will run faster.

- When creating the subpatches in autopatch(), eval() function was replaced since it wraps the lambdify and evalution together. Therefore, the lambdify function will be looped multiple times and slow down the code dramatically.
