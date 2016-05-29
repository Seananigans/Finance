"""
A wrapper for neural network regression. 
"""


import numpy as np
from NeuralNet import *

class ANNRegLearner(object):

    def __init__(self, sizes=[], lmbda=0.0, use_trained=False, verbose = False):
        self.name = "Neural net Regression Learner: lambda={}".format(lmbda)
        self.lmbda = lmbda
        self.use_trained = use_trained
        # pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        feature_size = dataX.shape[1]
        dataY =  dataY.reshape(dataY.shape[0], 1)
        output_size = dataY.shape[1]
        # Create potential network
        sizes = [feature_size, 50, output_size]
        if dataY.mean()<5.0:
            cost = CrossEntropyCost
            if dataY.min()==0.0 and dataY.max()==1.0:
                activations = [Sigmoid for i in sizes[1:]]
            elif dataY.min()==-1.0 and dataY.max()==1.0:
                activations = [Tanh for i in sizes[1:]]
            else:
                activations = [ReLU for i in sizes[1:]]
        else:
            activations = [Linear for i in sizes[1:]]
            cost = QuadraticCost
        net = Network(
                sizes = sizes, 
                cost = cost,
                activations = activations,
                lmbda = self.lmbda,
                dropout=1.0
                )
        print net.activations
        # Load saved network for continued training?
        filename = "network_models/stocknet.txt"
        if self.use_trained:
            try:
               self.network = load(filename)
            except IOError:
               self.network = net
        else:
            self.network = net
        self.network.sgd(dataX, dataY)
        
        #check if worth saving
        preds = self.network.forward(dataX)
        net_cost = self.network.cost.fn(preds, dataY)
        try:
            net2 = load(filename)
            try:
                preds2 = net2.forward(dataX)
                net2_cost = net2.cost.fn(preds2, dataY)
            except ValueError:
                net2_cost = np.inf
        except IOError:
            net2_cost = np.inf
        if net2_cost>net_cost:
            self.network.save(filename)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        if type(points) != np.ndarray:
            points = points.as_matrix()
        return self.network.forward(points)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
