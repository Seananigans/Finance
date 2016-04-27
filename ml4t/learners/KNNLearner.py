"""
A simple wrapper for k-nearest neighbors regression.
"""

import numpy as np

class KNNLearner(object):

    def __init__(self, k=3, verbose = False):
        self.k = k
        self.verbose = verbose
        self.name = "{}-Nearest Neighbors Learner".format(k)
    
    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term

        # build and save the model
        self.Xtrain = dataX
        self.Ytrain = dataY
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        
        estimates = np.zeros(points.shape[0])
        
        for i, point in enumerate(points):
            diff = (point - self.Xtrain)
            dist = np.sum(diff**2, axis=1)**0.5
            nearest_neighbors = np.argsort(dist)[0:self.k]
            estimates[i] = np.mean(self.Ytrain[nearest_neighbors])
        return estimates

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
