from hypersurface import *
from generate_h import *
import os
import configparser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if not os.path.exists('plot'):
    os.makedirs('plot')

z0, z1, z2, z3, z4= sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + 0.5*z0*z1*z2*z3*z4
HS = Hypersurface(Z, f, 1000)

config = configparser.RawConfigParser()
directory = './results'
k_old = 0
for filename in sorted(os.listdir(directory)):
    print('Plotting ', filename)
    full_path = os.path.join(directory, filename)
    config.read(full_path)

    k = int(config.get('results', 'k'))
    if k_old != k:
        HS.set_k(k)
        factor = HS.integrate(lambda patch, point: patch.num_eta('FS', point), numerical=True)
        h_sym = get_sym_info(HS)
    k_old = k     

    param_txt = config.get('results', 'param')
    param = np.fromstring(param_txt[1:-1], dtype=np.float, sep=',')

    h_minimal = param_to_matrix(param, h_sym)
    patch = HS.patches[0].patches[0]

    theta, phi = np.linspace(0.001,np.pi+0.001, 100), np.linspace(0.001, 2*np.pi+0.001, 100)
    R = []
    for j in phi:
        theta_list = []
        for i in theta:
            t = complex(math.sin(i)*math.sin(j), math.cos(i))/(sin(i)*cos(j)) 
            if np.absolute(t) <= 1:
                eta = patch.num_eta(h_minimal, [1, -1, complex(t), 0, -complex(t)])/factor 
            else:
                eta = patch.num_eta(h_minimal, [1, -1, complex(1/t), 0, -complex(1/t)])/factor
            theta_list.append(float(eta))
        R.append(theta_list)
    R = np.asarray(R)
    THETA, PHI = np.meshgrid(theta, phi)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    ZZ = R * np.cos(THETA)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    #ax.set_zlim3d(-0.8, 0.8)
    plot = ax.plot_surface(
        X, Y, ZZ, rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r,
        linewidth=0, antialiased=False)
    
    fig.savefig('./plot/'+filename+'.png', dpi=fig.dpi)
