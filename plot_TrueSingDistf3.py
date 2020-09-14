import numpy as np 
import matplotlib.pyplot as plt 

data = np.genfromtxt('SingDist_f3.txt')

alpha = data[:,0]
singdist_patch3 = data[:,1]


#plt.plot(phi,singdist_patch1,'b-',label='Patch $z_1$')
plt.plot(alpha,singdist_patch3,'b-',label='Patch $z_3$')
plt.xlabel('$\\alpha$')
plt.ylabel('True Distance in $\mathbb{CP}^N$')
plt.title('Distance from Singularity $f_3$ $\psi=-0.5$')
plt.xlim(0,6.0)
plt.ylim(0,0.3)
plt.legend()
plt.show()