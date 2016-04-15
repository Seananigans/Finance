"""
A simple wrapper for k-nearest neighbors regression.
"""

import numpy as np

class BagLearner(object):
    
    def __init__(self, learner, kwargs = {"k":3}, bags = 20, boost = False, verbose = False):
        self.learners = [learner(**kwargs) for i in range(0, bags)]
        self.boost = boost
        self.verbose = verbose
        try:
        	self.name = "Bag Learner: {}".format(self.learners[0].name)
        except AttributeError:
        	self.name = "Bag Learner"
        	
    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        n = dataX.shape[0]
        # build and save the model
        idx = np.random.choice(n, size=n, replace=True)

        for learner in self.learners:
            learner.addEvidence(dataX[idx,:], dataY[idx])
            if self.boost:
                errors = np.abs(learner.query(dataX)-dataY)
                weights = errors/sum(errors)
                idx = np.random.choice(n, size=n, replace=True, p=weights)
            else:
                idx = np.random.choice(n, size=n, replace=True)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        
        estimates = [learner.query(points) for learner in self.learners]
        return np.mean(estimates, axis=0)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
