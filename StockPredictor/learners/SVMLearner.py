"""
A simple wrapper for SVM regression.
"""

import numpy as np
from sklearn.svm import SVR

class SVMLearner(object):

    def __init__(self, kernel="linear", C=1e3, gamma=0.1, degree=2, verbose = False):
		self.name = "{} Support Vector Machine Learner".format(kernel.capitalize())
		self.kernel=kernel
		if kernel=="linear":
			self.svr = SVR(kernel=kernel, C=C)
		elif kernel=="rbf":
			self.svr = SVR(kernel=kernel, C=C, gamma=gamma)
		elif kernel=="poly":
			self.svr = SVR(kernel=kernel, C=C, degree=degree)

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # build and save the model
        self.svr.fit(dataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.svr.predict(points)