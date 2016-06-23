"""
Test a learner.  (c) 2015 Tucker Balch

Acknowldgements to the original code provided by Tucker Balch for the 
Machine Learning For Trading Course offered at Georgia Tech.

This code has been changed significantly and is used for testing the
value of individual and lists of machine learning algorithms.
"""

import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from learners import LinRegLearner as lrl
from learners import KNNLearner as knn
from learners import BagLearner as bag
from learners import SVMLearner as svm
from learners import DecTreeLearner as dt
from learners import DistributedLearner as dst
from helpers.util import calculate_returns
from helpers.error_metrics import rmse, mape
from helpers.normalization import mean_normalization, max_normalization
from helpers.plotting import plot_histogram
from dataset_construction import create_input, create_output, get_and_store_web_data

try:
    from learners import SVMLearner as svm
except:
    pass
    
if __name__=="__main__":

    # get stock data
    filename= "webdata/AAPL.csv"
    
    
    df = pd.read_csv(filename, index_col='Date',
                    parse_dates=True, na_values=['nan'])
    output = create_output("IBM")
    features = get_and_store_web_data("IBM", online=False)
    df = features.join(output).dropna()
    
    data = df.values
    cols = [col for col in df.columns if not col.startswith("y")]
    
    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6* data.shape[0]))
    test_rows = int(data.shape[0] - train_rows)

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    # Analyze datasets and returns
    print testX.shape
    print testY.shape
    
    # Formatting for printing
    if trainY.max()<=3.0:
        output_type="Return"
    else:
        output_type="Price"
    
    # Calculate average value of returns or prices for output
    print "Average Training {}: {}".format(output_type, round( trainY.mean(),3 ))
    print "Average Training {}: {}".format(output_type, round( testY.mean(),3 ))
    
    # Calculate the error that would be achieved by predicting the average price of 
    # the training output onto the testing output
    err = np.zeros(testY.shape)
    for i in range(err.shape[0]):
            err[i] = trainY.mean()
    print "RMSE of Train Average on Test Set: {}".format(
            math.sqrt(((testY - err) ** 2).sum()/testY.shape[0])
            )
    
    # Calculate how each feature correlates with the output
    for i in range(trainX.shape[1]):
            print cols[i], np.corrcoef(trainX[:,i], trainY)[0][1], trainX[:,i].mean()
    
    # Normalize training and test features
    trainX, testX = max_normalization(trainX, testX)
    
    # Create a learners
    ######################################################################################
    """Linear Regression: Uncomment the lines below to test."""
    learners = [lrl.LinRegLearner(verbose = True),lrl.LinRegLearner(verbose = True)]
    opt_var = range(len(learners))
    plot_name = "LinearRegressionTuning"
    """Bagged Linear Regression: Uncomment the lines below to test."""
#     opt_var = range(3,60,1)
#     learners = [bag.BagLearner(learner = lrl.LinRegLearner,
#                                    kwargs = {},
#                                    bags = i,
#                                    boost = True,
#                                    verbose = False) for i in opt_var]
#     plot_name = "BaggedLinearRegressionTuning"
    ######################################################################################
    """k-Nearest Neighbors: Uncomment the lines below to test."""
#     opt_var = range(5,100,5)
#     learners = [knn.KNNLearner(k=i, verbose = True) for i in opt_var]
#     plot_name = "KNNTuning"
    """Bagged k-Nearest Neighbors: Uncomment the lines below to test."""
#     opt_var = range(3,20,1)
#     learners = [bag.BagLearner(learner = knn.KNNLearner,
#                                    kwargs = {"k":15},
#                                    bags = i,
#                                    boost = False,
#                                    verbose = False) for i in opt_var]
#     plot_name = "BaggedKNNTuning"
    ######################################################################################
    """Decision Trees: Uncomment the lines below to test."""
#     opt_var = range(1,100,5)
#     learners = [dt.DecTreeLearner(depth=i, verbose = True) for i in opt_var]
#     plot_name = "DecisionTreeTuning"
    """Bagged Decision Trees: Uncomment the lines below to test."""
#     opt_var = range(3,60,1)
#     learners = [bag.BagLearner(learner = dt.DecTreeLearner,
#                                    kwargs = {"depth":3},
#                                    bags = i,
#                                    boost = False,
#                                    verbose = False) for i in opt_var]
#     plot_name = "BaggedDecisionTreeTuning"
    ######################################################################################
    """Support Vector Machines: Uncomment the lines below to test."""
#     opt_var = [10**i for i in range(-5,5)]
#     opt_var = range(6)
#     learners = [svm.SVMLearner(kernel='poly', C=1e2, degree=i) for i in opt_var]
#     svr_rbf = svm.SVMLearner(kernel='rbf', C=1e2, gamma=0.1)
#     svr_lin = svm.SVMLearner(kernel='linear', C=1e2)
#     svr_poly = svm.SVMLearner(kernel='poly', C=1e2, degree=2)
#     learners = [svr_rbf, svr_lin, svr_poly]
#     opt_var = [i for i in range(len(learners))]
#     plot_name = "SVMTuning"
    ######################################################################################
    """Distributed Learner Regression: Uncomment the lines below to test."""
#     opt_var = [i for i in range(1,20)]
#     learners = [dst.DistributedLearner(distribution=[i/20.,(1-i/20.)]) for i in opt_var]
#     plot_name = "DistributedRegressionTuning"

    # Collect scoring metrics for each learner for later comparison
    cors, rmsestrain, rmsestest = [], [], []
    
    for i, learner in enumerate(learners):
        print learner.name
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predYtrain = learner.query(trainX) # get the predictions
        print predYtrain.shape
        print
        print "In sample results"
        # Calculate TRAINING Root Mean Squared Error
        RMSE = rmse(trainY, predYtrain)
        print "RMSE: ", RMSE
        # Calculate correlation between predicted and TRAINING results
        c = np.corrcoef(predYtrain, y=trainY)
        print "corr: ", c[0,1]
        rmsestrain.append(RMSE)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        print
        print "Out of sample results"
        # Calculate TEST Root Mean Squared Error
        RMSE = rmse(testY,predY)
        print "RMSE: ", RMSE
        # Calculate correlation between predicted and TEST results
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        print
        cors.append(c[0,1])
        rmsestest.append(RMSE)
        
        # Join predicted values and actual values into a dataframe.
        predicted = pd.DataFrame(predY,
                           columns=["Predicted"],
                           index=df.ix[train_rows:,:].index)
        predicted = predicted.join(pd.DataFrame(testY,
                           columns=["Actual"],
                           index=df.ix[train_rows:,:].index))
        
        predicted = calculate_returns(predicted, 5)
        
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
        
        df1 = pd.DataFrame(testX,
                           columns=cols,
                           index= df.index[train_rows:])
        df1 = df1.join(predicted)
        df1.to_csv("test.csv", index_label="Date")
    
    if len(learners)>4:
        plt.plot(opt_var, cors)
        plt.ylabel("Correlation")
        plt.xlabel("Model Complexity")
        plt.savefig("tuning_figures/{}.png".format("{}Correlation".format(plot_name)))
        plt.show()
    
    try:
        print predicted.ix[dt.date.today()]
    except:
        print predicted.iloc[-2,:]
    
    # Plot testing & training error on the same plot to 
    # show how error behaves with different models.
    testerr, = plt.plot(opt_var, rmsestest, label="Test Error")
    trainerr, = plt.plot(opt_var, rmsestrain, label="Training Error")
    plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
    plt.xlabel("Model Complexity")
    plt.ylabel("RMSE")
#     plt.xticks(range(len(cors)),opt_var)
    plt.savefig("tuning_figures/{}.png".format(plot_name))
    plt.show()
