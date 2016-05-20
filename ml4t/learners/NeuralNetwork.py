import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

class NeuralNetwork(object):
    def __init__(self, inputLayer=2, outputLayer=1, hiddenLayer=3):
        #Define HyperParameters
        self.inputLayerSize = inputLayer
        self.outputLayerSize = outputLayer
        self.hiddenLayerSize = hiddenLayer

        #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, \
                                  self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, \
                                  self.outputLayerSize)

    def forward(self, X):
        #Propogate
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def predict(self, X):
        yHat = self.forward(X)
        ypred = np.argmax(yHat, axis=1)
        if yHat.shape[1] > 1:
            return ypred
        return yHat
    
    def sigmoid(self, z):
        #Apply the sigmoid activation function
        return 1./(1+np.exp(-z))

    def sigmoidPrime(self,z):
        prime = self.sigmoid(z)
        return prime*(1-prime)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class
        self.yHat = self.forward(X)
        m = y.shape[0]
        J = 0.5*sum((y-self.yHat)**2)
##        J = np.multiply( y, np.log(self.yHat) ) + \
##            np.multiply( (1-y), np.log( 1-self.yHat ) )
##        J *= -(1./m)
        return J.sum()
        
    def costFunctionPrime(self, X, y):
        '''Need to work on this implementation. I believe the cost functioni is working properly.
Refer to Octave construction for proper implementation.'''
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)        

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2
    
    def learning_rate_adjust(self):
        '''Adjust learning rate alpha to speed up learning.'''
        if (self.costs[len(self.costs)-1]<self.costs[len(self.costs)-2]):
            self.alpha *= 1.05
        else:
            self.alpha *= 0.6

    def display_error(self):
        '''#Inform the user of the current error as iterations increase.'''
        #self.draw(self.costs[len(self.costs)-1])
        sys.stdout.write(
            "ERROR LEVEL: {0:.5g}\r".format(
                self.costs[len(self.costs)-1]
                                             )
            )
        sys.stdout.flush()
        self.hidden_history.append(self.a2)
        
    def gradDescent(self, iterations, X, y):
        self.costs = []
        self.costs.append(self.costFunction(X,y))
        self.hidden_history = []
        self.hidden_history.append(self.a2)
        self.alpha = 0.001
        iteration = 0
        self.display_error()
        for i in range(iterations):
            #Calculate gradients
            gradients = self.costFunctionPrime(X,y)
            #Add gradients to theta
            self.W1 -= self.alpha*gradients[0]
            self.W2 -= self.alpha*gradients[1]

            #Add cost to vector to ensure decrease with each iteration
            self.costs.append(self.costFunction(X,y))

            #Adjust learning rate to improve optimization speed.
            self.learning_rate_adjust()

            #Inform the user of the current error as it decreases.
            if iteration%500==0:
                self.display_error()
            
            if iteration > 1000  and \
            (self.costs[len(self.costs)-1]>self.costs[len(self.costs)-1000]):
                break

##            if len(self.costs) > 1000:
##                del self.costs[0]
            
            iteration += 1
        self.display_error()
        print
        
##        print iteration
        
    
    def clear(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

    def draw(self, cost):
        self.clear()
        print "The current error is {}".format(cost[0])

