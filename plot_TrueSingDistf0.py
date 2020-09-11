import numpy as np 
import matplotlib.pyplot as plt 

data = np.genfromtxt('TrueSingDist_f0.txt')

psi = data[:,0]
singdist_patch1 = data[:,1]



plt.plot(-psi,singdist_patch1,'b-',label='Patch $z_1$')
plt.xlabel('-$\psi$')
plt.ylabel('True Distance in $\mathbb{CP}^N$')
plt.title('Distance from Singularity $f_0$')
plt.xlim(0,10.0)
plt.ylim(0,0.4)
plt.legend()
plt.show()

