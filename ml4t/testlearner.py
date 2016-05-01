"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from learners import LinRegLearner as lrl
from learners import KNNLearner as knn
from learners import BagLearner as bag

def mean_normalization(trainX, testX):
    trnX = ( trainX - trainX.mean(axis=0) )/trainX.std(axis=0)
    tstX = ( testX - trainX.mean(axis=0) )/trainX.std(axis=0)
    return trnX, tstX
    
def max_normalization(trainX, testX):
    trnX = trainX / trainX.max(axis=0)
    tstX = testX / trainX.max(axis=0)
    return trnX, tstX
	
	
if __name__=="__main__": 
#     inf = open('simData/ripple.csv')
#     inf = open('simData/simple.csv')
#     inf = open('simData/3_groups.csv')
    inf = open('simData/example.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6* data.shape[0]))
    test_rows = int(data.shape[0] - train_rows)

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape
    
    trainX, testX = mean_normalization(trainX, testX)
    
    # create a learner and train it
    learners = [lrl.LinRegLearner(verbose = True), # create a LinRegLearner
    			knn.KNNLearner(k=5, verbose = True), # create a KNNLearner
    			bag.BagLearner(learner = knn.KNNLearner, # create a BagLearner
    							kwargs = {"k":5}, 
    							bags = 50, 
    							boost = True, 
    							verbose = False),
    			bag.BagLearner(learner = lrl.LinRegLearner, # create a BagLearner
    							kwargs = {}, 
    							bags = 10, 
    							boost = True, 
    							verbose = False),
    			bag.BagLearner(learner = lrl.LinRegLearner, # create a BagLearner
    							kwargs = {}, 
    							bags = 50, 
    							boost = False, 
    							verbose = False)]
    for learner in learners:
        print learner.name
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        print
        
        

    #learners = []
    #for i in range(0,10):
        #kwargs = {"k":i}
        #learners.append(lrl.LinRegLearner(**kwargs))
