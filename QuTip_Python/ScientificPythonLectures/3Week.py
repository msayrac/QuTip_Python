#%% unnamed funciton lambda funciton

f1 = lambda x : x**2

print(f1(2))

# this is similar to 

def f2(x):
    return x**2

f2(2)

# This technique is useful for example when we want to pass a simple function as an argument to another function

# map funciton 
mapValues = list(map(lambda x: x**2, range(-3,4)))

print(mapValues)

#%% Classes
# A class is a structure for representing an object and the operations that can be performed on the object. In Python a class can contain attributes (variables) and methods (functions).

# Each class method should have an argument self as its first argument. This object is a self-reference
# Some class method names have special meaning, for example

 # __init__ : The name of the method that is invoked when the object is first created.

# __str__ : : A method that is invoked when a simple string representation of the class is needed, as for example when printed.

class Point:
    
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def translate(self, dx,dy):
        """
        Translate the point by dx and dy in the x and y direction
        """
        self.x += dx
        self.y += dy
    
    def __str__(self):
        return (f"Point at [{self.x},{self.y}]")
        
    
p1 = Point(0,0) # this will invoke the __init__ method in the Point class. it calls constructor

print(p1) # this will invoke the __str__ method

p2 = Point(1,1)
p1.translate(0.25, 1.5)
print(p1)
print(p2)


#%% Modules

# One of the most important concepts in good programming is to reuse code and avoid repetitions.
# The idea is to write functions and classes with a well-de ned purpose and scope, and reuse these instead of repeating similar code in different part of a program (modular programming). The result is usually that readability and maintainability of a program is greatly improved. What this means in practice is that our programs have fewer bugs, are easier to extend and debug/troubleshoot.

# A python module is defined in a python file (with file-ending .py), and it can be made accessible to other Python modules and programs using the import statement.

import mymodule

help(mymodule) # get a summary of what the module provides

print(mymodule.my_variable)

print(mymodule.my_funciton())

my_class = mymodule.MyClass()
my_class.set_variable(10)
print(my_class.get_variable())

#%% Exception
# In Python errors are managed with a special language construct called Exceptions. When errors occur exceptions can be raised, which interrupts the normal program flow and fallback to somewhere else in the code where the closest try-except statement is defined.

# To generate an exception we can use the raise statement, which takes an argument that must be an instance of the class BaseException or a class derived from it.

raise Exception("description of the error")
#%%
try:
    print("test")
except:
    print("Caught an exception")

#%% To get information about the error, we can access the Exception class instance that describes the exception by using for example
try:
    print("test")
    # print(a)
except Exception as e:
    print("Caught an exception",e)

























