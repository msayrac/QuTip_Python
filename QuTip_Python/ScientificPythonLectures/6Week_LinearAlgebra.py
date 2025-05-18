#%% Linear algebra
# The linear algebra module contains a lot of matrix related functions, including linear equation solving, eigenvalue solvers, matrix functions

from scipy.linalg import *
# import numpy as np
from numpy import *
A = array([[1,2,3],[4,5,6],[12,11,9]])
b = array([9,3,4])
# Ax = b soluton bulunur

x = solve(A,b)
print(x)

A =random.rand(3,3)
A
B =random.rand(3,3)
B
X = solve(A,B)

print(X)

#%% Eigenvalues and eigenvectors

# The eigenvalue problem for a matrix A: AVn = lambda*Vn where vn is the nth eigenvector and lambda is the nth eigenvalue. To calculate eigenvalues of a matrix, use the eigvals and for calculating both eigenvalues and eigenvectors, use the function eig:

from numpy import *
from scipy import *

A =random.rand(3,3)

evals =eigvals(A)

evals
evals.shape

evals,evecs = eig(A)

evals
evals.shape

evecs
evecs.shape

# The eigenvectors corresponding to the nth eigenvalue (stored in evals[n]) is the nth column in evecs, i.e., evecs[:,n]. To verify this, let's try mutiplying eigenvectors with the matrix and compare to the product of the eigenvector and the eigenvalue:

evecs[:,1]

n = 1

norm(dot(A,evecs[:,n])-evals[n]*evecs[:,n])

#%% Matrix operations
# the matrix inverse
from scipy import *
from numpy import *
#A*A^-1 = I
A =random.rand(3,3)
A
linalg.inv(A)

linalg.det(A)

#%% Optimization
# Optimization (finding minima or maxima of a function) is a large field in mathematics, and optimization of complicated functions or in many variables can be rather involved.

from scipy import *
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np

def f(x):
    return 4*x**3+(x-2)**2 + x**4

fig, ax = plt.subplots()

x = np.linspace(-5, 3,100)
ax.plot(x,f(x))

x_min = optimize.fmin_bfgs(f,-2) #2 initial guess
"""
bu fonksiyonun min degeri -2 dir degerinde sıfır olur. algoritma x = -2'den başlayarak bu noktayı bulur.
Bu örnekte, (x - 3)**2 + 2 fonksiyonunun minimumu x = 3'tedir, ama algoritma x = -2'den başlayarak bu noktayı bulur.
Başlangıç noktası, algoritmanın yakınsama hızını ve sonucunu etkileyebilir — özellikle çok tepe/çukur içeren fonksiyonlarda.

"""

#%%

# Finding a solution to a function

# To find the root for a function of the form f(x) = 0 we can use the fsolve function. It requires an initial guess

# fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None) --> Find the roots of a function.
# Return the roots of the (non-linear) equations defined by func(x) = 0 given a starting estimate.

"""
func : callable f(x, *args)
A function that takes at least one (possibly vector) argument, and returns a value of the same length.

x0 : ndarray
The starting estimate for the roots of func(x) = 0.

args : tuple, optional Any extra arguments to func.

fprime : callable f(x, *args), optional
A function to compute the Jacobian of func with derivatives across the rows. By default, the Jacobian will be estimated.

full_output: bool, optional If True, return optional outputs.

col_deriv : bool, optional Specify whether the Jacobian function computes derivatives down the columns (faster, because there is no transpose operation).

xtol: float, optional The calculation will terminate if the relative error between two consecutive iterates is at most xtol.

maxfev :int, optional The maximum number of calls to the function. If zero, then 100*(N+1) is the maximum where N is the number of elements in x0.

band : tuple, optional If set to a two-sequence containing the number of sub- and super-diagonals within the band of the Jacobi matrix, the Jacobi matrix is considered banded (only for fprime=None).

epsfcn : float, optional A suitable step length for the forward-difference approximation of the Jacobian (for fprime=None). If epsfcn is less than the machine precision, it is assumed that the relative errors in the functions are of the order of the machine precision.

factor : float, optional A parameter determining the initial step bound (factor * || diag * x||). Should be in the interval (0.1, 100).

diag : sequence, optional N positive entries that serve as a scale factors for the variables.
"""
# from scipy.optimize import fsolve
from scipy import *
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np

def f(x):
    return [x[0]*np.cos(x[1])-4, x[1]*x[0]-x[1]-5]

root =optimize.fsolve(f,[1,1])
root

from scipy import optimize
def f1(x):
    return x[0]**2 - 4   # DİKKAT: x[0] şeklinde aldık
root1 = optimize.fsolve(f1, [3])  # DİKKAT: [3] array olarak verildi
root2 = optimize.fsolve(f1, [-1])  # DİKKAT: [-3] array olarak verildi
print("Kök:", root1)
print("Kök:", root2)
# fsolve, başlangıç tahmininin çok yakınında bir çözüm ararıyor, ve 0'ı en yakın çözüm gibi kabul eder.


# solve, fonksiyonun kökünü (yani çıktısı 0 olan x değerini) bulur.
# x**2 - 4 = 0 denklemi çözülür → çözüm x = -2 veya x = 2'dir.
# Başlangıç tahminine (0) daha yakın olan kökü bulur, yani burada sonucu x = 2 verir.


import numpy as np
from scipy import optimize

def system(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 4
    eq2 = x - y
    return [eq1, eq2]

# Başlangıç tahmini (x=1, y=1)
initial_guess = [1, 1]

solution = optimize.fsolve(system, initial_guess)
print("Çözüm (x, y):", solution)

# Başlangıç tahmini, çözümün yakınında olduğunu düşündüğün bir değerdir. doğrudan çözüm olmasına gerek yok — ama yakın olması, çözümün daha hızlı ve doğru bulunmasını sağlar.

#%% Interpolation
# Interpolation is simple and convenient in scipy: The interp1d function, when given arrays describing X and Y data, returns and object that behaves like a function that can be called for an arbitrary value of x (in the range covered by X), and it returns the corresponding interpolated y value

# from scipy.interpolate import *
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
from IPython.display import Image

def f(x):
    return sin(x)

n = arange(0,10)
x =linspace(0,9,100)

y_meas = f(n) +0.1*random.rand(len(n))
y_real = f(x)

linear_interpolation = interpolate.interp1d(n,y_meas)
y_interp1 = linear_interpolation(x)

cubic_interpolation = interpolate.interp1d(n,y_meas,kind="cubic")
y_interp2 = cubic_interpolation(x)

fig, ax = plt.subplots()

ax.plot(n,y_meas, "bs", label = "noisy data")
ax.plot(x,y_real, "k", label = "true funciton")
ax.plot(x,y_interp1, "r", label = "lineer interp")
ax.plot(x,y_interp2, "g", label = "cubic interp")
ax.legend(loc=3)

#%%  Statistics

# from scipy import stats

from scipy.stats import *
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
from IPython.display import Image

# create a discreet random variable with poissionian distribution

X =stats.poisson(3.5) # photon distribution for a coherent state with n=3.5 photons

n = arange(0,15)

fig, axes = plt.subplots(3, 1, sharex = True) # sharex x axis ortak paylasır
# plot the probability mass function (PMF)
axes[0].step(n,X.pmf(n))

# plot the commulative distribution function (CDF)
axes[1].step(n,X.cdf(n))

# plot histogram of 1000 random realizations of the stochastic variable X
axes[2].hist(X.rvs(size=1000))


# create a (continous) random variable with normal distribution
Y = stats.norm()
x=linspace(-5, 5,100)

fig,axes = plt.subplots(3,1,sharex = True)
# plot the probability distribution function (PDF)
axes[0].plot(x,Y.pdf(x))

# plot the commulative distributin function (CDF)
axes[1].plot(x,Y.cdf(x))

# plot histogram of 1000 random realizations of the stochastic variable Y
axes[2].hist(Y.rvs(size=1000),bins=50)

X.mean(), X.std(), X.var() # poission distribution

Y.mean(), Y.std(), Y.var()

# Statistical tests
# Test if two sets of (independent) random data comes from the same distribution

t_statistic, p_value =stats.ttest_ind(X.rvs(size=1000),X.rvs(size=1000))

print(f"t-statistic =",t_statistic)
print(f"p-value =",p_value)


