##"""
##A wrapper for neural network regression. 
##"""
##print "Not currently in production. Sorry"
##exit()

import numpy as np
from NeuralNet import *

class ANNRegLearner(object):

    def __init__(self, sizes=[], verbose = False):
        self.name = "Neural net Regression Learner"
        # pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        dataX = dataX[:,5:]
        feature_size = dataX.shape[1]
        dataY =  dataY.reshape(dataY.shape[0], 1)
        output_size = dataY.shape[1]
        self.min_y = dataY.min()
        self.network = Network(sizes = [feature_size, 20, 10, output_size], activations=[ReLU, ReLU, ReLU])
        self.network.sgd(dataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        points = points.values
        points = points[:,5:]
        return self.network.forward(points)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
