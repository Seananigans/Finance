"""
A simple wrapper for k-nearest neighbors regression.
"""

import numpy as np

class BagLearner(object):
	
    def __init__(self, learner, kwargs = {"k":3}, bags = 20, boost = False, verbose = False):
        self.learners = [learner(**kwargs) for i in range(0, bags)]
        self.boost = boost
        self.verbose = verbose
        self.name = "Bag Learner"
    
    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        n = dataX.shape[0]
        n_prime = int(0.6 * n)
        # build and save the model
        for learner in self.learners:
			idx = np.random.choice(n, size=n_prime, replace=True)
			learner.addEvidence(dataX[idx,:], dataY[idx])
        
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