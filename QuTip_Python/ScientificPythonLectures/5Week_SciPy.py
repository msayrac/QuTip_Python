#%% SciPy - Library of scientic algorithms for Python

# The SciPy framework builds on top of the low-level NumPy framework for multidimensional arrays, and provides a large number of higher-level scientic algorithms.

import matplotlib.pyplot as plt
from IPython.display import Image

from scipy.special import *
from numpy import *

n =0 # order
x =0

print(n,x,jn(n,x))

x =1

print(n,x,yn(n,x))

x = linspace(0,10,100)

fig,ax = plt.subplots()
for n in range(4):
    ax.plot(x,jn(n,x),label=f"J{n}(x)")
ax.legend()

#%% Integration
# Numerical integration : quadrature

# quad, dblquad and tplquad for single, double and triple integrals

from scipy.integrate import *
from numpy import *
# define a simple function for the integrand
def f(x):
    return x

x_lower = 1 # the lower limit of x
x_upper = 1 # the upper limit of x

val, abserr = quad(f,x_lower,x_upper)

print(f"integral value=",val,", abolute error=",abserr)


val, abserr = quad(lambda x: exp(-x**2),-Inf,Inf)

print(f"numerical=",val, abserr)

print(20*"*")
def integrand(x,y):
    return exp(-x**2-y**2)

x_lower =0
x_upper =10
y_lower = 0
y_upper = 10

val, abserr =dblquad(integrand,x_lower,x_upper, y_lower, y_upper)

print(val, abserr)

#%% Ordinary differential equations (ODEs)
# To use odeint, first import it from the scipy.integrate modul

from scipy.integrate import odeint,ode
from numpy import *

import matplotlib.pyplot as plt
from IPython.display import Image

# y_t = odeint(f, y_0, t)
# where t is and array with time-coordinates for which to solve the ODE problem. y t is an array with one row for each point in time in t, where each column corresponds to a solution y i(t) at that point in time.

#double pendulum
g=9.8
L=0.5
m=0.1

def dx(x,t):
    """
    The right hand side of the pendulum ODE
    """
    x1,x2,x3,x4 = x[0],x[1],x[2],x[3]
    
    dx1 = 6/(m*L**2)*(2*x3-3*cos(x1-x2)*x4)/(16-9*cos(x1-x2)**2)    
    dx2 = 6/(m*L**2)*(8*x4-3*cos(x1-x2)*x3)/(16-9*cos(x1-x2)**2)
    dx3 = -0.5*m*L**2*(dx1*dx2*sin(x1-x2)+3*(g/L)*sin(x1))
    dx4 = -0.5*m*L**2*(-dx1*dx2*sin(x1-x2)+(g/L)*sin(x2))
    
    return [dx1,dx2,dx3,dx4]

#choose an initial state
x0 = [pi/4,pi/2,0,0]

#time to solve the ODE for from 0 to 10 seconds
t = linspace(0,10,50)

# solve the ODE problem
x = odeint(dx,x0,t)
x.shape
# plot the angles as a function of time 
fig,axes = plt.subplots(1,2,figsize = (12,4))
axes[0].plot(t,x[:,0],"r",label="theta1")
axes[0].plot(t,x[:,1],"b",label="theta2")

x1 = +L*sin(x[:,0])
y1 = -L*cos(x[:,0])

x2 = x1 +L*sin(x[:,1])
y2 = y1-L*cos(x[:,1])

axes[1].plot(x1,y1,"r", label="pendulum1")
axes[1].plot(x2,y2,"b", label="pendulum2")
axes[1].set_ylim([-1,0])
axes[1].set_xlim([-1,1])

#%%
from IPython.display import display, clear_output,Image
import time
from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint,ode

g=9.8
L=0.5
m=0.1
#choose an initial state
x0 = [pi/4,pi/2,0,0]
#time to solve the ODE for from 0 to 10 seconds
t = linspace(0,10,50)

fig,ax = plt.subplots(figsize=(4,4))

for t_idx,tt in enumerate(t[:200]):
    x1 = +L*sin(x[t_idx,0])
    y1 = -L*cos(x[t_idx,0])
    
    x2 = x1 + L*sin(x[t_idx,1])
    y2 = y1 -L*cos(x[t_idx,1])

ax.cla()
ax.plot([0,x1],[0,y1],"r.-")
ax.plot([x1,x2],[y1,y2],"b.-")

ax.set_ylim([-1.5,0.5])
ax.set_xlim([1,-1])

clear_output()
display(fig)
time.sleep(0.1)

