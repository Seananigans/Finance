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
    
def plot_histogram(trainY):
    mn = trainY.mean()
    sd = trainY.std()
    plt.hist(trainY)
    plt.xlabel("Daily Returns")
    plt.ylabel("Counts")
    mean_line = plt.axvline(mn, color="k", lw=3)
    std_line = plt.axvline(mn + sd, color="r", lw=2)
    plt.axvline(mn - sd, color="r", lw=2)
    plt.legend([mean_line, std_line],
               ["Avg. {} Day\nReturn:   {}%".format(5,round(mn*100,2)),
                "Std. Dev.\nof Returns: {}%".format(round(sd*100,2))]
               )
    plt.show()
    
if __name__=="__main__":

    # get actual data
    df = pd.read_csv("simData/example.csv", index_col='Date',
                    parse_dates=True, na_values=['nan'])
    data = df.values
    
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
    print "Average Return: {}".format(round( trainY.mean(),3 ))
    plot_histogram(trainY)
    exit()
    trainX, testX = mean_normalization(trainX, testX)
    
    # create a learner and train it
##    learners = [lrl.LinRegLearner(verbose = True), # create a LinRegLearner
##                knn.KNNLearner(k=6, verbose = True)] # create a KNNLearner
##    
##    learners = [lrl.LinRegLearner(verbose = True), # create a LinRegLearner
##                knn.KNNLearner(k=6, verbose = True), # create a KNNLearner
##                bag.BagLearner(learner = knn.KNNLearner, # create a BagLearner
##                                kwargs = {"k":6}, 
##                                bags = 10, 
##                                boost = True, 
##                                verbose = False),
##                bag.BagLearner(learner = lrl.LinRegLearner, # create a BagLearner
##                                kwargs = {}, 
##                                bags = 10, 
##                                boost = True, 
##                                verbose = False)]
    
    learners = [knn.KNNLearner(k=i) for i in range(1,25)]
    
    cors, rmsestrain, rmsestest = [], [], []
    for i, learner in enumerate(learners):
        print learner.name
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predYtrain = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predYtrain, y=trainY)
        print "corr: ", c[0,1]
        rmsestrain.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        print
        cors.append(c[0,1])
        rmsestest.append(rmse)
        predicted = pd.DataFrame(predY,
                           columns=["Predicted"],
                           index=df.ix[train_rows:,-1].index)
        predicted = predicted.join(pd.DataFrame(testY,
                           columns=["Actual"],
                           index=df.ix[train_rows:,-1].index))
        if i%5==0:
            plt.figure(1)
            plt.subplot(211)
            pre, = plt.plot(predicted[['Predicted']])
            act, = plt.plot(predicted[['Actual']])
            plt.legend([pre, act], ["Predicted", "Actual"])
            plt.xlabel("Date")
            plt.ylabel("Returns")
            plt.subplot(212)
            plt.scatter(predY, testY)
            plt.xlabel("Predicted Returns")
            plt.ylabel("Actual Returns")
            plt.show()
    
    plt.plot(range(len(cors)), cors)
    plt.ylabel("Correlation")
    plt.xlabel("Model Complexity")
    plt.show()
    
    # Plot testing & training error on the same plot to 
    # show how error behaves with different models.
    testerr, = plt.plot(range(len(cors)), rmsestest, label="Test Error")
    trainerr, = plt.plot(range(len(cors)), rmsestrain, label="Training Error")
    plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
    plt.xlabel("Model Complexity")
    plt.ylabel("RMSE")
    plt.show()
