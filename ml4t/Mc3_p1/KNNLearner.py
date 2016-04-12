"""
A simple wrapper for k-nearest neighbors regression.  (c) 2015 Tucker Balch
"""

import numpy as np

class LinRegLearner(object):

    def __init__(self, k=3, verbose = False):
        self.k = k
    
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
        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
