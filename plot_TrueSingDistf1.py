import numpy as np 
import matplotlib.pyplot as plt 

data = np.genfromtxt('TrueSingDist_f1.txt')

phi = data[:,1]
singdist_patch1 = data[:,2]
singdist_patch3 = data[:,3]


#plt.plot(phi,singdist_patch1,'b-',label='Patch $z_1$')
plt.plot(phi,singdist_patch3,'b-',label='Patch $z_3$')
plt.xlabel('$\phi$')
plt.ylabel('True Distance in $\mathbb{CP}^N$')
plt.title('Distance from Singularity $f_1$ $\psi=-0.5$')
plt.xlim(0,5.0)
plt.ylim(0,0.4)
plt.legend()
plt.show()

