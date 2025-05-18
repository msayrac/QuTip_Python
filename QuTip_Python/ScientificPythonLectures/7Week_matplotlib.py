#%% matplotlib - 2D and 3D plotting in Python
# Introduction --> Matplotlib is an excellent 2D and 3D graphics library for generating scientific figures

# To get started using Matplotlib in a Python program, either include the symbols from the pylab module

from pylab import *

# or
import matplotlib.pyplot as plt

import numpy as np

#%%
import numpy as np
from pylab import *

x = np.linspace(0, 5,10)
y = x**2
figure()
plot(x,y,"r")
xlabel("x")
ylabel("y")
title("title")
show()

# Most of the plotting related functions in MATLAB are covered by the pylab module. For example, subplot and color/symbol selection:

subplot(1,2,1)
plot(x,y,"r--")
subplot(1,2,2)
plot(y,x,"g*-")

#%% The matplotlib object-oriented API --> is remarkably powerful. For advanced figures with subplots, insets and other components it is very nice to work with.

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5,10)
y = x**2

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8]) # left, bottom, width, height (range 0 to 1)

axes.plot(x,y,"r")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_title("title")

# a little bit more code is involved, the advantage is that we now have full control of where the plot axes are placed, and we can easily add more than one axis to the figure

fig = plt.figure()

axes1 = fig.add_axes([0.1,0.1,0.8,0.8]) # main axes
axes2 = fig.add_axes([0.2,0.5,0.35,0.3]) # inset axes

# main figure

axes1.plot(x,y,"r")
axes1.set_xlabel("x")
axes1.set_ylabel("y")
axes1.set_title("title")

#insert

axes2.plot(x,y,"g")
axes2.set_xlabel("y")
axes2.set_ylabel("x")
axes2.set_title("insert title")

#%% If we don't care about being explicit about where our plot axes are placed in the figure canvas, then we can use one of the many axis layout managers in matplotlib.
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplot()
axes.plot(x,y,"r")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_title("title")

fig, axes = plt.subplots(1,2)

for ax in axes:
    ax.plot(x,y,"r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("title")

fig.tight_layout() # automatically adjusts the positions of the axes on the figure canvas so that there is no overlapping content

#%% Figure size, aspect ratio and DPI (dots per inch)

# Matplotlib allows the aspect ratio, DPI and figure size to be specifed when the Figure object is created, using the figsize and dpi keyword arguments. figsize is a tuple of the width and height of the figure in inches, and dpi is the dots-per-inch (pixel per inch). To create an 800x400 pixel, 100 dots-per-inch figure

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize =(8,4), dpi = 100)

# The same arguments can also be passed to layout managers, such as the subplots function

fig, axes = plt.subplots(figsize=(4,3))
axes.plot(x,y,"r")
axes.set_xlabel("x") # Axis labels
axes.set_ylabel("y") # Axis labels
axes.set_title("title") # Figure titles

# Saving  figures --> savefig("filename.png") method 

fig.savefig("filename1.png", dpi = 100) # it save the path that you give upper right corner

# What formats are available and which ones should be used for best quality?

# Matplotlib can generate high-quality output in a number formats, including PNG, JPG, EPS, SVG, PGF and PDF.
fig.savefig("pdffilename.pdf", dpi = 100)

# Legends, labels and titles

#%% 
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5,10)
y = x**2

fig, axes = plt.subplots(figsize=(6,4), dpi = 100)

axes.plot(x,y,"r",label=r"$\alpha$")
axes.plot(y,x,"g*-",label="curve2")
axes.set_xlabel(r"x $y =\alpha$", fontsize=18)
axes.set_ylabel("y", fontsize=18)
axes.set_title("title", fontsize=18)
axes.legend(loc=0) # loc= 0-optimal location, 1-right corner,2-left corner,3-left corner,4-right corner
fig.savefig("text1.jpg", dpi=200)

#%% We can also change the global font size and font family, which applies to all text elements in a figure

import matplotlib.pyplot as plt
import numpy as np

# Update the matplotlib configuration parameters
plt.rcParams.update({"font.size":18,"font.family":"serif"})

x= np.linspace(0,5,10)
y = x **2

fig, ax = plt.subplots()
ax.plot(x,x**2, "b-", alpha=0.4, lw=5, ls=":", marker="+", label =r"$y = \alpha^2$") # b- blue line with dots alpha=0.5 transparancy
ax.plot(x,x**3, "b",linewidth=5, linestyle="-.", marker="*", markersize=16, markerfacecolor="red", markeredgewidth =2, markeredgecolor="yellow", label =r"$y = \alpha^3$") # green dashed line

ax.legend(loc = 2) # upper left corner
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$y$")
ax.set_title("title")

# Setting colors, linewidths, linetypes
# Control over axis appearance
# ax.axis("tight")
# ax.set_xlim([0,15])
# ax.set_ylim([0,150])

# logarithmic scale
ax.set_yscale("log")
ax.grid(color="b",alpha = 1, linestyle="dashed",linewidth=1)

ax.spines["bottom"].set_color("none")
ax.spines["left"].set_color("yellow")

ax2 =ax.twinx()
ax2.plot(x,x**4,lw=2,color="red")
ax2.set_ylabel(r"volume$(m^4)$",fontsize=18, color="red")

#%% 2D plot 

import matplotlib.pyplot as plt
import numpy as np

n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(1,4,figsize=(12,3))

axes[0].scatter(n,n+0.25*np.random.rand(len(n)))
axes[0].set_title("scatter")

axes[1].step(n,n**2,lw=2)
axes[0].set_title("step")

axes[2].bar(n,n**2,align="center",width=0.5,alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(n,n**2,n**3,color="green")
axes[3].set_title("fill_between")

#%% polar plot

import matplotlib.pyplot as plt
import numpy as np

fig =plt.figure()
ax =fig.add_axes([0,0,0.6,0.6],polar = True)
t = np.linspace(0, 2*np.pi,100)
ax.plot(t,t,color = "blue",lw=3)

ax.text(-4*np.pi,np.pi/2,"test",fontsize=20,color="green")

#%% subplots and insets

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2,3)
fig.tight_layout()

# subplot2grid

fig = plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan =3)
ax2 = plt.subplot2grid((3,3),(1,0),colspan=2)

ax3 = plt.subplot2grid((3,3),(1,2),rowspan = 2)
ax4 = plt.subplot2grid((3,3),(2,0))
ax5 = plt.subplot2grid((3,3),(2,1))

fig.tight_layout()

#%% add_axes
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,1,2,3,4,5])

fig, ax = plt.subplots()

ax.plot(x,x**2,x,x**3)
fig.tight_layout()

# inset

inset_ax =fig.add_axes([0.25,0.45,0.35,0.35]) # X, Y, width, height

inset_ax.plot(x,x**2,x,x**3)
inset_ax.set_title("zoom near origin")

inset_ax.set_xlim([0.2,2])
inset_ax.set_ylim([0.2,2])

#%% Colormap and contour figures

import matplotlib.pyplot as plt
import numpy as np

alpha = 0.7
phi_ext = 2*np.pi*0.5

def flux_qubit_potential(phi_m,phi_p):
    return 2 + alpha -2*np.cos(phi_p)*np.cos(phi_m)-alpha*np.cos(phi_ext-2*phi_p)

phi_m = np.linspace(0,2*np.pi,100)
phi_p = np.linspace(0,2*np.pi,100)

X, Y =np.meshgrid(phi_p,phi_m)

Z =flux_qubit_potential(X,Y).T

fig, ax = plt.subplots()
p= ax.pcolor(X/(2*np.pi),Y/(2*np.pi),Z,cmap=plt.cm.RdBu,vmin=abs(Z).min(),vmax=abs(Z).max())

cb = fig.colorbar(p,ax=ax)

# imshow

fig,ax = plt.subplots()

im = ax.imshow(Z,cmap =plt.cm.RdBu,vmin=abs(Z).min(),vmax = abs(Z).max())
im.set_interpolation("bilinear")

cb = fig.colorbar(im,ax=ax)

# contour

fig, ax = plt.subplots()
cnt = ax.contour(Z,cmap=plt.cm.RdBu,vmin=abs(Z).min(),vmax=abs(Z).max())




