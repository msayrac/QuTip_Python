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

#%% array-generating functions
import numpy as np

x= np.arange(0,10,1) # start, stop, step
print(x)

x = np.arange(-1,1,0.1)
print(x)


# using linspace, both end points ARE included

print(np.linspace(0,10,25)) #↓ 0 dan 10 a kadar 25 adet eşit aralıklı linspace list olusturur

print(np.logspace(0,10,10))

#%% mgrid similar to mesgrid in MATLAB
from numpy import *
x, y = mgrid[0:3,0:5] # rows and columns olarak 0:3 x 0:5 y matrix olusturur
print(x)
print(y)

#%% random data
# uniform random numbers in [0,1]
from numpy import random
x= random.rand(5,5) # 5 by 5 matrix 0-1 range
print(shape(x))
print(x[0])

# standard normal distributed random numbers
print(random.randn(5,5))

# a diagonal matrix
x = diag([1,2,3])
print(x)

# diagonal with offset from the main diagonal
x = diag([1,2,3],k=1)
print(x)

# zeros and ones

print(zeros((3))) # 3 zeros in vector print 0

print(zeros((2,3))) # 2 rows and 3 columns print 0

print(ones((2,3))) # 2 rows 3 columns print 1 

#%% File I/O
from numpy import *

M = random.rand(3,3)
savetxt("random_matrix.csv",M)

save("random_matrix.npy",M)

load("random_matrix.npy")

# M is a matrix, or a 2 dimensional array, taking two indices

M[1]
M[1,1]

M[1,:] # row 1 : --> column all

M[:,1] # all rows column 1

M
M[1,:] = 0
M[:,2] = -1

M

#%% index slicing
# M[lower:upper:step]
A = array([1,2,3,4,5])
A

A[1:3]

# Array slices are mutable: if they are assigned a new value the original array from which the slice was extracted is modified

A[1:3] = 0
A

A[::] # lower, upper, step all take the default values

A[::2] # step is 2, lower and upper defaults to the beginning and end of the array

A[:3]

A[3:] # elements from index 3

# Negative indices counts from the end of the array (positive index from the begining)

A= array([1,2,3,4,5])
A[-1] # gives the last element in the array

A[-3:] # last 3 elements

#%%
from numpy import *

A = array([[n+m*10 for n in range(5)] for m in range(5)])

A
# Index slicing works exactly the same way for multidimensional arrays
A[1:4,1:4] # a block from the original array

A[::2,::2] # bastan basla 2 ser 2 ser git

# Fancy indexing --> Fancy indexing is the name for when an array or list is used in-place of an index

row_indices = [1,2,3]
A[row_indices]

col_indices = [1,2,-1]
A[row_indices,col_indices]


#%%  Functions for extracting data from arrays and creating arrays



x = arange(0,10,0.5)

mask = (5<x)*(x<7.5) # x > 5 ve x<7.5 oldugu duurmları true dondurur

print(mask)

print(x[mask])

# where --> The index mask can be converted to position index using the where function

indices = where(mask) # mask True indices leri verir
print(indices)

# With the diag function we can also extract the diagonal and subdiagonals of an array:
A = array([[n+m*10 for n in range(5)] for m in range(5)])
A
diag(A) # array diag elementlerini alır

diag(A,k=-1) # 1 satir sonra diag alır

# take --> The take function is similar to fancy indexing described above

v2 =arange(-3,3)
v2
row_indices = [1,3,5]
print(v2[row_indices]) # fancy indexing

v2.take(row_indices) # ilgili index elemanlarını ulasabiliriz
liste = [-3,-2,-1,0,1,2]
take(liste,row_indices)

# choose --> Constructs an array by picking elements from several arrays

which = [1,0,1,0]
choices = [[-2,-2,-2,-2],[5,5,5,5]]

choose(which,choices)

#%% Linear algebra
# Scalar-array operations
from numpy import *

v1 = arange(0,5)
v1*2

v1+ 2

A = array([[n+m*10 for n in range(5)] for m in range(5)])
A
A*2 , A+2

# Element-wise array-array operations
# When we add, subtract, multiply and divide arrays with each other, the default behaviour is element-wise operations

A*A # element wise multiplication

v1*v1

# If we multiply arrays with compatible shapes, we get an element-wise multiplication of each row

A.shape, v1.shape # m*n ve n*k  == m*k matris cıkar

A*v1


# Matrix algebra
# There are two ways. We can either use the dot function, which applies a matrix-matrix, matrix-vector, or inner vector multiplication to its two arguments
A
dot(A,A)

v1
dot(A,v1)

dot(v1,v1)


# Alternatively, we can cast the array objects to the type matrix. This changes the behavior of the standard arithmetic operators +, -, * to use matrix algebra

M = matrix(A)
M
v = matrix(v1).T # this makes v a column vector
v

M*M

M*v
# inner product
v.T*v

# with matrix objects, standard matrix algebra applies

v + M*v

# If we try to add, subtract or multiply objects with incomplatible shapes we get an error

v = matrix([1,2,3,4,5,6]).T # column vector
v.T
shape(M), shape(v)
# M*v # value error --> shape not aligned

#%% Array/Matrix transformations

# Above we have used the .T to transpose the matrix object v. We could also have used the transpose function to accomplish the same thing.

v = matrix([1,2,3])
v
v.T
transpose(v) # same things

C = matrix([[1j,2j],[3j,4j]])
C
conjugate(C)

# Hermitian conjugate: transpose + conjugate
C.H

# We can extract the real and imaginary parts of complex-valued arrays using real and imag

real(C)

imag(C)

angle(3+4j, deg=True) # tan-1(y/x) Return the angle of the complex argument
angle(3+4j, deg=False) # in radyan (deg * pi/180)

C
abs(C)

#%% Matrix computations
# Inverse
C
linalg.inv(C) # Compute the inverse of a matrix. 

# Determinant
import numpy as np
linalg.det(C)
C
C.I # inversini hesaplar

# Data processing

# sum, prod, and trace
d = arange(0, 10)
d
sum(d) # sum up all elements

prod(d+1) # product of all elements

# cummulative sum
d
cumsum(d) # toplayarak gider

d
cumprod(d+1) # cummulative product

# same as diag(A).sum()

A = array([[n+m*10 for n in range(5)] for m in range(5)])
A
trace(A) # diagon sum alıyor

diag(A).sum()

#%% Calculations with higher-dimensional data

# When functions such as min, max, etc. are applied to a multidimensional arrays, it is sometimes useful to apply the calculation to the entire array, and sometimes only on a row or column basis. Using the axis argument we can specify how these functions should behave

from numpy import *

m = random.rand(3,3)
m
m.max() # global max

m.max(axis = 1) # max in each row

m
m.max(axis = 0) # max in each column

# Reshaping, resizing and stacking arrays

A = array([[n+m*10 for n in range(5)] for m in range(5)])
A

n, m = A.shape
n, m
A
B=A.reshape((1,m*n))
B
B.shape

B[0,0:5] = 5 # modify the array
B

A # and the original variable is also changed. B is only a different view of the same data

# We can also use the function flatten to make a higher dimensional array into a vector. But this function create a copy of the data.
B = A.flatten()
B[0:5]=10
B
A # now A has not changed, because B's data is a copy of A's, not refering to the same data

# Adding a new dimension: newaxis --> With newaxis, we can insert new dimensions in an array, for example converting a vector to a column or row matrix

v = array([1,2,3])
shape(v)

v[:, newaxis] # make a column matrix of the vector v

v[:,newaxis].shape # column matrix

# row matrix
v[newaxis,:].shape

# Stacking and repeating arrays

a = array([[1,2],[3,4]])
a
a.shape
b= repeat(a,3) # repeat each element 3 times
b
b.shape

c =tile(a,3) # tile the matrix 3 times
c
c.shape

# concatenate
a
b=array([[5,6]])

concatenate((a,b), axis=0) # row a b matrixi ekler

concatenate((a,b.T), axis = 1) # column B matrisi ekler

# Copy and 'deep' copy"
A = array([[1, 2], [3, 4]])
A
B=A
B[0,0] = 10
B
A # element of A is changet to 10

# If we want to avoid this behavior, so that when we get a new completely independent object B copied from A, then we need to do a so-called '\'deep copy' using the function copy

A = array([[1, 2], [3, 4]])
A
B = copy(A) # now, if we modify B, A is not affected
B[0,0] = 10
B
A

#%% Iterating over array elements
from numpy import *
v= array([1,2,3,4])
v.shape
for element in v:
    print(element)

M = array([[1,2],[3,4]])

for row in M:
    print("row : ", row)
    for col in row:
        print(col)

M
for row_idx, row in enumerate(M):
    print("row_idx", row_idx, "row",row)
    
    for col_idx, element in enumerate(row):
        print("col_idx",col_idx, "element",element)

#%% Using arrays in conditions
# When using arrays in conditions,for example if statements and other boolean expressions, one needs to use any or all, which requires that any or all elements in the array evalutes to True

from numpy import *

M = array([[1,2],[9,16]])
M

if (M>5).any():
    print("5 ten buyuk herhangi bir eleman vardır")
else:
    print("5 ten buyuk herhangi bir eleman yoktur")
print(20*"*")            
if(M>5).all():
    print("Butun elemanlar 5 ten buyuktur")
else:
    print("Tum elemanlar 5 ten buyuk degildir")
    
M.dtype
M2 = M.astype(float)
M2.dtype
M2
