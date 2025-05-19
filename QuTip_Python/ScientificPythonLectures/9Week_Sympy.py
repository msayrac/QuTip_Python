#%% Sympy - Symbolic algebra in Python

import matplotlib.pyplot as plt
import numpy as np

from sympy import *


# In SymPy we need to create symbols for the variables we want to work with. We can create a new symbol using the Symbol class

x = Symbol("x")

(pi+x)**2
# alternative way of defining symbols
a,b,c = symbols("a,b,c")

type(a)

# We can add assumptions to symbols when we create them:

x = Symbol("x",real=True)
x.is_imaginary

x = Symbol("x",positive=True)
x>0

# Complex numbers --> The imaginary unit is denoted I in Sympy

1+1*I

I**2

(x*I+1)**2


# Rational numbers --> There are three different numerical types in SymPy: Real, Rational, Integer

r1 = Rational(4,5)
r2 = Rational(5,4)

r1, r2

r1+r2

r1/r2

# Numerical evaluation --> To evaluate an expression numerically we can use the evalf function (or N). It takes an argument n which specifies the number of significant digits

pi.evalf(n=50)

y = (x+pi)**2
N(y,5) # same as evalf

# When we numerically evaluate algebraic expressions we often want to substitute a symbol with a numerical value. In SymPy we do that using the subs function

y.subs(x,1.5)

N(y.subs(x,1.5))

# The subs function can of course also be used to substitute Symbols and expressions

y.subs(x,a+pi)

# We can also combine numerical evolution of expressions with NumPy arrays

import numpy as np
x_vec = np.arange(0,10,0.1)
y_vec = np.array([N(((x+pi)**2).subs(x,xx)) for xx in x_vec])

fig, ax = plt.subplots()
ax.plot(x_vec,y_vec)

# Algebraic manipulations
# One of the main uses of an CAS (Computer Algebra Systems) is to perform algebraic manipulations of expressions. For example, we might want to expand a product, factor an expression, or simply an expression. The functions for doing these basic operations in SymPy are demonstrated in this section

# Expand and factor --> The first steps in an algebraic manipulation

test = (x+1)*(x+2)*(x+3)

expand(test)

# The expand function takes a number of keywords arguments which we can tell the functions what kind of expansions we want to have performed. For example, to expand trigonometric expressions, use the trig=True keyword argument:

expand(sin(a+b),trig=True)

help(expand) #detailed explanation of the various types of expansions the expand functions can perform.

# The opposite a product expansion is of course factoring. The factor an expression in SymPy use the factor function

factor(x**3+6*x**2+11*x+6)

factor(x**2+2*x+1)

# Simplify --> The simplify tries to simplify an expression into a nice looking expression, using various techniques. More specific alternatives to the simplify functions also exists: trigsimp, powsimp, logcombine, etc. The basic usages of these functions are as follows
# simplify expands a product

simplify((x+1)*(x+2)*(x+3)+4-4)

# simplify uses trigonometric identities
simplify(sin(a)**2+cos(a)**2)

simplify(cos(a)/sin(a))

# apart and together --> To manipulate symbolic expressions of fractions, we can use the apart and together functions

f1 = 1/((a+1)*(a+2))

f1

apart(f1)

f2 = 1/(a+2) + 1/(a+3)
f2

together(f2)

# Differentiation --> Differentiation is usually simple. Use the diff function. The first argument is the expression to take the derivative of, and the second argument is the symbol by which to take the derivative

y 

diff(y**2,x)

# For higher order derivatives we can do
diff(y**2,x,x)

diff(y**2,x,2) # same as above

diff(y**2,x,4) # take derivative 4 times

# To calculate the derivative of a multivariate expression, we can do
import matplotlib.pyplot as plt
import numpy as np
from sympy import *

x,y,z =symbols("x,y,z")

f = sin(x*y)+cos(y*z)

diff(f,x,1,y,2) # x 1 , y 2 derivative

# Integration --> Integration is done in a similar fashion

f=x

integrate(f,x)

# By providing limits for the integration variable we can evaluate definite integrals

integrate(f,(x,-1,1))

# and also improper integrals

integrate(exp(-x**2),(x,-np.inf,np.inf))

# Sums and products --> We can evaluate sums and products using the functions

n = Symbol("n")
Sum(1/n**2, (n,1,10))
Sum(1/n**2, (n,1,10)).evalf(n=15)
Sum(1/n**2, (n,1,np.inf)).evalf()

# products work much the same way

fact = Product(n,(n,1,5)) # 10!

fact.evalf(n=3) # first n=3 digits

# limits

limit(sin(x)/x,x,0)

# We can change the direction from which we approach the limiting point using the dir keywork argument

limit(1/x,x,0,dir="+")

limit(1/x,x,0,dir="-")

# Series --> Series expansion is also one of the most useful features of a CAS. In SymPy we can perform a series expansion of an expression using the series function

series(exp(x),x)

# By default it expands the expression around x = 0, but we can expand around any value of x by explicitly include a value in the function call
series(exp(x),x,1)

# And we can explicitly define to which order the series expansion should be carried out

series(exp(x),x,1,10)

# Linear algebra
# Matrices --> Matrices are defined using the Matrix class

m11,m12,m21,m22 = symbols("m11,m12,m21,m22")
b1, b2 = symbols("b1,b2")

A = Matrix([[m11,m12],[m21,m22]])
A

b = Matrix([[b1],[b2]])
b

A**2

A*b

A.det()

# Solving equations --> For solving equations and systems of equations we can use the solve function

solveTest = solve(x**2 - 1,x)

solveTest[0]
solveTest[1]

solve(x**4 -x**2 -1,x)

# System of equations
solve([x+y-1,x-y-1],[x,y])

# In terms of other symbolic expressions
solve([x + y - a, x - y - c], [x,y])



