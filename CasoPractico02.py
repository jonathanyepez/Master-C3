# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:34:31 2020

@author: Jonathan A. Yepez M.
"""

"""
Crea un conjunto de datos utilizando el siguiente código:
trX = np.linspace(-1, 1, 101)
trY = np.linspace(-1, 1, 101)
for i in range(len(trY)):
    trY[i] = math.log(1 + 0.5 * abs(trX[i])) + trX[i] / 3 + np.random.randn() * 0.033

Ahora, utiliza Theano para obtener los parámetros w_0 y w_1 del siguiente modelo:
    y= log(1+ w0|x|) + w1x 
"""

#Import Libraries for this exercise
import numpy as np
import math
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

#Initialize data
trX = np.linspace(-1,1,101) #create vector X
trY = np.linspace(-1,1,101) #create vector Y

for i in range(len(trY)): #for every element in Y, 
    trY[i] = math.log(1 + 0.5*abs(trX[i])) + trX[i]/3 + np.random.randn()*0.033 #update based on a formula, adding random noise

#Theano Implementation
X = T.scalar() #initialize the scalar representation for X in Theano 
Y = T.scalar() #initialize the scalar representation for Y in Theano

def model(X,w0,w1): #specify the function here
    return T.log(1+w0*T.abs_(X)) + w1*X

w0 = theano.shared(np.asarray(0., dtype = theano.config.floatX)) #shared variable for parameter w0, initialized at 0
w1 = theano.shared(np.asarray(0., dtype = theano.config.floatX)) #shared variable for parameter w1, initialized at 0
y = model(X,w0, w1) #run the model

print("The initial value for w0:")
print(w0.get_value()) #should print 0
print("and the initial value for w1:" )
print(w1.get_value()) #should print 0

cost = T.mean(T.sqr(y - Y)) #define the cost function (MSE)
gradient_w0 = T.grad(cost = cost, wrt = w0) #calculate the gradient of the cost w.r.t. w0
gradient_w1 = T.grad(cost = cost, wrt = w1) #calculate the gradient of the cost w.r.t. w1
updates = [[w0,w0-gradient_w0*0.001],[w1, w1-gradient_w1*0.001]] #establish the updates that need to occur in this case
#learning rate = 0.001
train = theano.function(inputs = [X, Y], outputs = cost, updates = updates)

costs = [] #initialize an empty list to store costs

for i in range(101): #run 100 iterations
    for x, y in zip(trX, trY): #tuples of trX, trY values to be passed in train()
        cost_m = train(x, y) #calculate the cost based on the previous tuple
    print('En el paso', i, 'el valor de w\'s son', w0.get_value(), 'y', w1.get_value(), 'con un coste', cost_m)
    costs.append(cost_m)

plt.plot(range(101), costs)
plt.title("Error vs Number of Iterations")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.grid()
plt.show()

print("The final parameter values are approximately:\n")
print("w0 = ", round(float(w0.get_value()),3))
print("w1 = ", round(float(w1.get_value()),3))
