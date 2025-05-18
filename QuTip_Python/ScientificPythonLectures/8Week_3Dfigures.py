#%% 3D figures
# we first need to create an instance of the Axes3D class. 3D axes can be added to a matplotlib figure canvas in exactly the same way as 2D axes; or, more conveniently, by passinga projection='3d' keyword argument to the add axes or add subplot methods

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.axes3d import *

fig = plt.figure(figsize =(14,6))
# ax is a 3D-aware axis instance because of the projection ='3d' keyword argument to add_subplot

alpha = 0.7
phi_ext = 2*np.pi*0.5

def flux_qubit_potential(phi_m,phi_p):
    return 2 + alpha -2*np.cos(phi_p)*np.cos(phi_m)-alpha*np.cos(phi_ext-2*phi_p)

phi_m = np.linspace(0,2*np.pi,100)
phi_p = np.linspace(0,2*np.pi,100)

X, Y =np.meshgrid(phi_p,phi_m)

Z =flux_qubit_potential(X,Y).T

ax = fig.add_subplot(1,2,1,projection = '3d')
p = ax.plot_surface(X,Y,Z, rstride=4, cstride=4, linewidth=0)

# surface grading with color grading and color bar

ax = fig.add_subplot(1,2,2,projection ="3d")
p = ax.plot_surface(X,Y,Z,rstride = 4, cstride =4, linewidth=0, cmap ="jet")
cb = fig.colorbar(p,shrink=0.5)

fig = plt.figure(figsize =(8,6))
ax =fig.add_subplot(1,1,1,projection ="3d")
p = ax.plot_wireframe(X,Y,Z,rstride=4, cstride=4)

# contour plot with projection

fig =plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1,projection="3d")

ax.plot_surface(X,Y,Z,rstride=4, cstride=4, alpha=0.25)

cset =ax.contour(X,Y,Z, zdir="z", offset=-np.pi,cmap =plt.cm.coolwarm)
cset = ax.contour(X,Y,Z,zdir="x", offset=-np.pi,cmap=plt.cm.coolwarm)
cset = ax.contour(X,Y,Z,zdir="y",offset=3*np.pi,cmap=plt.cm.jet)

ax.set_xlim3d(-np.pi,2*np.pi)
ax.set_ylim3d(0,3*np.pi)
ax.set_zlim3d(-np.pi,2*np.pi)

# ax.view_init(0,0)
# fig.tight_layout()

plt.savefig("test.jpg",dpi=100)

