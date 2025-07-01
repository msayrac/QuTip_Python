#%%
# Here we explore how computers represent numbers and the implications for scientific computing. We’ll examine:

# Integer representation using binary digits

# Floating-point representation (IEEE 754 standard)

# Precision limitations and their effects on calculations

# Common numerical issues like overflow, underflow, and round-off errors

# Strategies for mitigating precision problems in scientific computing

x = 1.1 + 2.2

print("x :",x)
print(3.3)

if(x==3.3):
    print("x ==3.3 is True")
else:
    print("x==3.3 is False")

# A safer way to compare two floats is to check the equality only within a certain precision epsilon

# The desired precision
eps =1.e-15

# The comparison
if(x-3.3 < eps):
    print("x ==3.3 to a preciison of ",eps, " is True")
else:
    print("x==3.3 to a precision of ",eps," is False")

#%%
import numpy as np

print(np.sqrt(25))

if(np.sqrt(36.+1.e-13)==np.round(np.sqrt(36))):
    print("x==3.3 is True")
else:
    print("x == 3.3 is False")


#%%
# Subtracting two large numbers with a small difference

from math import sqrt

delta = 1.e-14
x =1.
y = 1. + delta*sqrt(2)

print("x = ",x)
print("y = ",y)

res = (1./delta)*(y-x)

print("y-x = ",y-x)
print("(1/delta) * (y-x) = ", res)

print("The accurate value is sqrt(2) = ",sqrt(2))
print("The difference is ",res-sqrt(2))

