#%% Introduction
# Science has traditionally been divided into experimental and theoretical disiplines. During the last several decades computing has emerged as a very important part of science. Ä±t is viewed a theory, also it has many charactesitics in common with experimental work. It is therefore seen a a new third branch of science. Compotational work is an important complement to both experiment and theory. Majority of research papers involve numerical calculations, simulations or computer modeling.

#  What is Python? --> Python is a modern, general-purpose, object-oriented, high-level programming language.

# Modules -->  Most of the functionality in Python is provided by modules. To use a module in a Python program it first has to be imported. A module can be imported using the import statement. For example, to import the module math, which contains many standard mathematical functions.

import math
x=math.cos(2*math.pi)
print(x)

# Alternatively, we can chose to import all symbols (functions and variables) in a module to the current
# namespace (so that we don't need to use the pre
# x \math." every time we use something from the math
# module:

from math import *
x=cos(2*pi)
print(x)

# As a third alternative, we can chose to import only a few selected symbols from a module by explicitly
# listing which ones we want to import instead of using the wildcard character *:

from math import cos,pi
print(cos(2*pi))

# Looking at what a module contains, and its documentation. Once a module is imported, we can list the symbols it provides using the dir function:

import math
print(dir(math))

import math
print(math.ceil(5.4))

# And using the function help we can get a description of each function

help(math.log)

print(log(100,10)) # base 10
print(log(e)) # base e 


# We can also use the help function directly on modules
help(math)

print(type(5) is float) # False doner

print(type("5") is not float) # True doner

# We can also use the isinstance method for testing types of variables
print(isinstance("x", float))

print(isinstance("x", str))

# Type casting
x=1
print(x, type(x))

z=complex(1,0)
print(z)

print(z.real) # real part
print(z.imag) # imag part

print(bool(z.real))
print(bool(z.imag))

# Operators and comparisons --> Arithmetic operators +, -, *, /, // (integer division), '**' power

print(3.0//2.0) # int part of division

print(2**3) # power operator in python


# and not or operator

print(True and False)

print([1,2] == [1,2])

l1 = l2 = [1,2]
print(l1 is l2)


# Compound types: Strings, List and dictionaries
# Strings --> Strings are the variable type that is used for storing text messages.
s = "Hello world"
type(s)

# length of the string: the number of characters
len(s)

# replace a substring in a string with something else. [start:end:step]
s = "Hello world"
s2 = s.replace("world","space")
print(s2)

print(s[0:5:1])
print(s[4:5])

print(s[:5])
print(s[6:])
print(s[0:5:2])


x=1.0
print(f"value ={x}")

# alternative, more intuitive way of formatting a string
s3 = "value1 = {0}, value2 = {1}".format(3.1415, 1.5)
print(s3)


#%% List are very similar to strings, except that each element can be of any type.
l = [1,2,3,4]
print(type(l))
print(l)
print(l[1:3])
print(l[::2]) # cifter gider


# Python lists can be inhomogeneous and arbitrarily nested:
nested_list = [1, [2, [3, [4, [5]]]]]
print(nested_list)

# range
start = 10
stop = 30
step = 2
print(list(range(start, stop, step)))

print(list(range(-10,10)))

s = "Hello world"
s2=list(s)
print(s2)

# create an empty list

l=[]
print(l)
l.append("c")
l.append(1)
print(l)

# assigning new values to elements in the list
l[0]=0
l[1]="python"
print(l)

l.insert(0, "Welcome")
l.append("course")
print(l)

l.remove(0) # remove specific element
print(l)

#%% tuples are like lists, except that they cannot be modified once created. --> immutable

point=(10,20)
print(point,type(point))

# We can unpack a tuple by assigning it to a comma-separated list of variables

x,y =point
print("x=",x)
print("y=",y)

# If we try to assign a new value to an element in a tuple we get an error immutable

# point[0]=20

#%% Dictionaries
# Dictionaries are also like lists, except that each element is a key-value pair

params ={"key1":1,"key2":3, 3:"Ali"}
print(params["key1"])
print(params[3])

# add a new entry
params[5]="New Parameters"

print(params)
print(params[5])

