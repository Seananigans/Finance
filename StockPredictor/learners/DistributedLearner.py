"""
A simple wrapper for k-nearest neighbors regression.
"""

import numpy as np
from KNNLearner import KNNLearner
from LinRegLearner import LinRegLearner

class DistributedLearner(object):
    
    def __init__(self, learners=[KNNLearner, LinRegLearner], kwargs = [{"k":20}, {}], 
    					distribution = [0.4,0.6], verbose = False):
        self.learners = [learner(**kwarg) for learner,kwarg in zip(learners, kwargs)]
        self.distribution = distribution
        self.verbose = verbose
        self.name = "Distributed Learner: {}".format(distribution)
        	
    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        for learner in self.learners:
            learner.addEvidence(dataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        
        estimates = [trust*learner.query(points) for learner,trust in zip(self.learners, self.distribution)]
        return np.sum(estimates, axis=0)

