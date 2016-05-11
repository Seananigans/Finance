"""
A simple wrapper for SVM regression.
"""

import numpy as np
from sklearn.svm import SVR

class SVMLearner(object):

    def __init__(self, verbose = False):
		self.name = "Linear Regression Learner"
        # pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # build and save the model
        self.svr = SVR()
        self.svr.fit(dataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.svr.predict(points)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
