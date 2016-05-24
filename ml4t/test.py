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
##from learners import NeuralRegLearner as net
from learners import ANNRegLearner as net

def mean_normalization(trainX, testX):
    trnX = ( trainX - trainX.mean(axis=0) )/trainX.std(axis=0)
    tstX = ( testX - trainX.mean(axis=0) )/trainX.std(axis=0)
    return trnX, tstX
    
def max_normalization(trainX, testX):
    trnX = trainX / trainX.max(axis=0)
    tstX = testX / trainX.max(axis=0)
    return trnX, tstX
    
def plot_histogram(trainY):
    mns = trainY.mean(axis=0)
    sds = trainY.std(axis=0)
    plt.hist(trainY)
    plt.xlabel("Daily Returns")
    plt.ylabel("Counts")
    mean_lines = [plt.axvline(mn, color="k", lw=3) for mn in mns]
    upper_std_lines = [plt.axvline(mn + sd, color="r", lw=2) for sd in sds]
    lower_std_lines = [plt.axvline(mn - sd, color="r", lw=2) for sd in sds]
    std_lines = upper_std_lines+lower_std_lines
    lines = mean_lines+std_lines
    #Create Labels
    labels = [None for i in lines]
    labels[:mns.shape[0]] = [
        "Avg. {} Day\nReturn:   {}%".format(5,
                                            round(mn*100,2)) for mn in mns]
    labels[mns.shape[0]:] = [
        "Std. Dev.\nof Returns: {}%".format(round(sd*100,2)) for sd in sds]
    plt.legend(lines,#[mean_line, std_line],
               labels)
    plt.show()
    
if __name__=="__main__":

	# get actual data
	filename= "simData/example.csv"
	
	df = pd.read_csv(filename, index_col='Date',
					parse_dates=True, na_values=['nan'])
					
	data = df.values
	cols = [col for col in df.columns if not col.startswith("Returns")]
	outcome = [col for col in df.columns if col.startswith("Returns")]
	
	# compute how much of the data is training and testing
	train_rows = int(math.floor(0.6* df.shape[0]))
	test_rows = int(df.shape[0] - train_rows)
	
	# separate out training and testing data
	trainX = df.ix[:train_rows,0:-1]
	trainY = df.ix[:train_rows,-1]
	testX = df.ix[train_rows:,0:-1]
	testY = df.ix[train_rows:,-1]
	
	trainX, testX = mean_normalization(trainX, testX)
	
	# create a learner and train it
	learners = [ net.ANNRegLearner() ]
    
	cors, rmsestrain, rmsestest = [], [], []
	for i, learner in enumerate(learners):
		print learner.name
		learner.addEvidence(trainX.values, trainY, use_trained=True) # train it

		# evaluate in sample
		predYtrain = learner.query(trainX) # get the predictions
		rmse = math.sqrt(((trainY.values - predYtrain) ** 2).sum()/trainY.shape[0])
		print
		print "In sample results"
		print "RMSE: ", rmse
		predYtrain = predYtrain.reshape(trainY.values.shape)
		c = np.corrcoef(predYtrain, y=trainY.values)
		print "corr: ", c[0,1]
		rmsestrain.append(rmse)

		# evaluate out of sample
		predY = learner.query(testX) # get the predictions
		rmse = math.sqrt(((testY.values - predY) ** 2).sum()/testY.shape[0])
		print
		print "Out of sample results"
		print "RMSE: ", rmse
		predY = predY.reshape(testY.values.shape)
		c = np.corrcoef(predY, y=testY.values)
		print "corr: ", c[0,1]
		print
		cors.append(c[0,1])
		rmsestest.append(rmse)
		predicted = pd.DataFrame(predY,
                                         columns=["Predicted"],
                                         index=df.ix[train_rows:,:].index)
		predicted = predicted.join(testY)
		
		if i%5==0:
			plt.figure(1)
			plt.subplot(211)
			pre, = plt.plot(predicted[['Predicted']])
			act, = plt.plot(predicted[outcome])
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
						   index=df.ix[train_rows:,:].index)
		df1 = df1.join(predicted)
		df1.to_csv("test.csv", index_label="Date")
	
	if len(learners)>4:
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
