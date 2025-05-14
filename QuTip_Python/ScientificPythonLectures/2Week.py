#%% Control Flow if, elif, else

a=5
b=9
if a>b:
    print("a > b")
elif  a<b:
    print("a < b")
else:
    print("a = b")

#%% Loops can be programmed in a number of different ways. The most common is the for loop

name = "Ali"
for x in name:
    print(x)
    
for i in range(0,10,2):
    print(i)
    
liste =["scientific", "computing", "with", "python"]
for word in liste:
    print(word)
    
# access to the indices of the values when iterating over a list.
ranges =range(-3,3)
for idx,x in enumerate(ranges):
    print(idx,x)
    
# Creating lists using for loops

l1 = [x**2 for x in range(0,5)]    
print(l1)   

print("****")
l2=[]
for i in range(5):
    l2.append(i**2)
print(l2)    

# while loops
i=0
while i<5:
    print(i)
    i+=1
print("done")

#%% Functions is defined using the keyword def, followed by a function name, a signature within parentheses (), and a colon:

def func():
    print("test")
func()
    
# Optionally, but highly recommended, we can define a so called "docstring", which is a description of the functions purpose and behaivor. The docstring should follow directly after the function definition, before the code in the function body

def func1(s):
    """
    print a string 's' and tell how many character is has
    """
    print("'" +s + "'" + " has "+str(len(s))+ " characters")

func1("Hello World")

help(func1) # it gives docstring

# Functions that returns a value use the return keyword:
    
def square(x):
    """
    Return a few power of the x
    """
    return x**2, x**3, x**4

x2,x3,x4 =square(4)

print(x2, x3, x4)
    
# Default argument and keyword arguments: In a definition of a function, we can give default values to the arguments the function takes

def myfunc(x, p=2, debug=False):
    if debug:
        print("Evaluating myfunc for x = "+ str(x) + " using exponent p = "+ str(p))
    return x**p
    
print(myfunc(5))

print(myfunc(5, debug=True))

print(myfunc(p=0, x=5, debug=True))

# sometime we want to access to the indices of the values when iterating over a list.

liste=range(-3,3)
print(*liste)
for idx, value in enumerate(liste):
    print(idx,value)

