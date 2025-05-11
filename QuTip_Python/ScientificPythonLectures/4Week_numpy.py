#%% Numpy - multidimensional data arrays
# The numpy package (module) is used in almost all numerical computation using Python. It is a package that provide high-performance vector, matrix and higher-dimensional data structures for Python. It is implemented in C and Fortran so when calculations are vectorized (formulated with vectors and matrices), performance is very good.

# To use numpy you need to import the module, using for example:
from numpy import *
# In the numpy package the terminology used for vectors, matrices and higher-dimensional data sets is array.

# Creating numpy arrays
# There are a number of ways to initialize new numpy arrays, for example from 
# •a Python list or tuples
# •using functions that are dedicated to generating numpy arrays, such as arange, linspace, etc
# •reading data from  files

# From lists
# For example, to create new vector and matrix arrays from Python lists we can use the numpy.array function.

v = array([1,2,3,4])

print(lambda x: x , v)

# a matrix: the argument to the array function is a nested Python list
M = array([[1,2],[3,4]])

print(type(v), type(M))

# The difference between the v and M arrays is only their shapes. We can get information about the shape of an array by using the ndarray.shape property.
print(M.shape)
print(v.shape)





