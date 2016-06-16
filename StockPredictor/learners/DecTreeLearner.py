"""
A simple wrapper for SVM regression.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class DecTreeLearner(object):

    def __init__(self, depth=3, verbose = False):
		self.name = "Linear Support Vector Machine Learner"
		self.depth = depth

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # build and save the model
        self.dt = DecisionTreeRegressor(max_depth=self.depth)
        self.dt.fit(dataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.dt.predict(points)
