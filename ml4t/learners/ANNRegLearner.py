"""
A wrapper for neural network regression. 
"""
print "Not currently in production. Sorry"
exit()

import numpy as np
from FFNeuralNet import *

class ANNRegLearner(object):

    def __init__(self, sizes, cost=QuadraticCost, verbose = False):
        self.name = "Neural net Regression Learner"
        self.network = Network(sizes,
                               cost=QuadraticCost)
        # pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        training_data=zip(dataX,dataY)
        self.network.SGD(training_data,
                           epochs=10,
                           mini_batch_size=3,
                           eta=0.01,
                           lmbda = 0.0)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.network.feedforward(points)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
