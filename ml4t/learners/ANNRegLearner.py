"""
A wrapper for neural network regression. 
"""


import numpy as np
from NeuralNet import *

class ANNRegLearner(object):

    def __init__(self, sizes=[], lmbda=0.0, verbose = False):
        self.name = "Neural net Regression Learner: lambda={}".format(lmbda)
        self.lmbda = lmbda
        # pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY, use_trained=False):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        feature_size = dataX.shape[1]
        dataY =  dataY.reshape(dataY.shape[0], 1)
        output_size = dataY.shape[1]
        # Create potential network
        sizes = [feature_size, 50, 50, output_size]
        if dataY.mean()<5.0:
        	activations = [Tanh for i in sizes[1:]]
        else:
        	activations = [ReLU for i in sizes[1:]]
        net = Network(
        		sizes = sizes, 
            	activations = activations,
            	lmbda = self.lmbda
            	)
        # Load saved network for continued training?
        filename = "network_models/stocknet.txt"
        if use_trained:
            try:
               self.network = load(filename)
            except IOError:
               self.network = net
        else:
            self.network = net
        self.network.sgd(dataX, dataY)
        self.network.save(filename)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        points = points.values
        return self.network.forward(points)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
