"""
In Production
"""
print "Not currently in production. Sorry"
exit()


import numpy as np
from sknn.mlp import Regressor, Layer

class NeuralRegLearner(object):

    def __init__(self, verbose = False):
        self.name = "Neural net Regression Learner"
        self.network =  Regressor( layers=[
										Layer("Rectifier", units=100),
										Layer("Linear")],
									learning_rate=0.02,
									n_iter=10)

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        self.network.fit(dataX, dataY) 
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.network.predict(points)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
