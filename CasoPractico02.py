# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:34:31 2020

@author: Jonathan A. Yepez M.
"""

"""
Crear un conjunto de datos

Utilizar Theano para obtener parametros w0 y w1 de:
    
    y= log(1+ w0|x|) + w1x 

"""

#Import Libraries for this case

import numpy as np
import math
import theano
import theano.tensor as T

#Initialize data

trX = np.linspace(-1,1,101)
trY = np.linspace(-1,1,101)

for i in range(len(trY)):
    trY[i] = math.log(1 + 0.5*abs(trX[i])) + trX[i]/3 + np.random.randn()*0.033

#######################ESTIMACION LINEAL#####################

X = T.scalar() #initialize the scalar representation for X in Theano 
Y = T.scalar() #initialize the scalar representation for Y in Theano
"""
def model(X, w, c):
    return T.dot(X,w) + c
w = theano.shared(np.asarray([0,0], dtype = theano.config.floatX))
c = theano.shared(np.asarray(0., dtype = theano.config.floatX))
y = model(X, w, c)
"""

def model(X,w0,w1): #specify the function here
    return T.log(1+w0*T.abs_(X)) + w1*X

w0 = theano.shared(np.asarray(0., dtype = theano.config.floatX))
w1 = theano.shared(np.asarray(0., dtype = theano.config.floatX))
y = model(X,w0, w1)

print("The initial value for w0:")
print(w0.get_value())
print("and the initial value for w1:" )
print(w1.get_value())
#print(c.get_value())

cost = T.mean(T.sqr(y - Y))
gradient_w0 = T.grad(cost = cost, wrt = w0)
gradient_w1 = T.grad(cost = cost, wrt = w1)
#gradient_c = T.grad(cost = cost, wrt = c)
#updates  = [[w, w - gradient_w * 0.001], [c, c - gradient_c * 0.001]] #learning rate (alpha) = 0.001
updates = [[w0,w0-gradient_w0*0.001],[w1, w1-gradient_w1*0.001]]
train = theano.function(inputs = [X, Y], outputs = cost, updates = updates)

for i in range(101):
    for x, y in zip(trX, trY):
        cost_i = train(x, y)
    #print('En el paso', i, 'el valor de w es', w.get_value(),
    #       'y c es', c.get_value(), 'con un coste', cost_i)
    print('En el paso', i, 'el valor de w\'s son', w0.get_value(), 'y', w1.get_value(), 'con un coste', cost_i)
##################################################################
    
#a = T.switch(T.lt(z,0),0,1) #switch statement, if z lt(less than) 0, then 0 else 1

#neuron = theano.function([x],f)