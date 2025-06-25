#%%
# Plotting basics

import matplotlib.pyplot as plt

params = {'legend.fontsize': 'large',
          'axes.labelsize':'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'xtick.direction':"in",
          'ytick.direction':"in",
          }
plt.rcParams.update(params)

#%% Line Plots
import numpy as np
# 100 equally spaced points between 0 and 10
numpoints = 100
x = np.linspace(0,10,numpoints)

# calculate the value of sin(x)
y=np.sin(x)

# make plot and show the results
plt.plot(x,y)
plt.show()

# add x and y labels
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.plot(x,y)
plt.show()

# By default the data points are connected by lines
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.plot(x,y)
plt.plot(x,y,'o')
plt.show()


## multiple plots

numpoints = 100

x = np.linspace(0,10,numpoints)

ysin = np.sin(x)
ycos = np.cos(x)

plt.xlabel('x')
plt.plot(x,ysin, label="sin(x)", linestyle="-.")
plt.plot(x,ycos,label="cos(x)",linestyle="--")
plt.legend()
plt.show()

#%% Contour/density plots
import numpy as np
import matplotlib.pyplot as plt

# Grid of points
x = np.linspace(-5,5,200)
y = np.linspace(-5,5,200)


# Meshgrid xsize*xsize matrix yapıyor X ve Y iicnde ysize*ysize
X,Y = np.meshgrid(x,y)

# location of the poles
x1,y1 = -2.,0.
x2,y2 = 2.,0.

# electric potential of a dipole

Vdip = 1./np.sqrt((X-x1)**2+(Y-y1)**2)-1./np.sqrt((X-x2)**2+(Y-y2)**2)

# contour plot lines only
plt.title("Electric potential of a dipole")
plt.xlabel("x")
plt.ylabel("y")

CS =plt.contour(X,Y,Vdip, levels = np.linspace(-0.5,0.5,11))
plt.clabel(CS)
plt.show()

#Density plot (filled)
fig2,ax2 =plt.subplots(constrained_layout=True)
ax2.set_title("Electric potential of a dipole")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
CS2 = ax2.contourf(CS,levels = np.linspace(-0.5,0.5,11))
fig2.colorbar(CS2)

plt.show()


# density plot using imshow

plt.title("Electric potential of a dipole")
plt.xlabel("x")
plt.ylabel("y")
CS3 = plt.imshow(Vdip,vmax=1.5,vmin=-1.5, origin = "lower",extent = [-5,5,-5,5])
plt.colorbar(CS3)
plt.show()


