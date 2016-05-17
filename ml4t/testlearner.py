"""
Test a learner.	 (c) 2015 Tucker Balch
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from learners import LinRegLearner as lrl
from learners import KNNLearner as knn
from learners import BagLearner as bag
from learners import SVMLearner as svm

def mean_normalization(trainX, testX):
	"""Returns the features normalized by the mean and standard deviation of
	feature values in the training set."""
	trnX = ( trainX - trainX.mean(axis=0) )/trainX.std(axis=0)
	tstX = ( testX - trainX.mean(axis=0) )/trainX.std(axis=0)
	return trnX, tstX
	
def max_normalization(trainX, testX):
	"""Returns the features normalized by the maximum 
	feature values in the training set."""
	trnX = trainX / trainX.max(axis=0)
	tstX = testX / trainX.max(axis=0)
	return trnX, tstX
	
def plot_histogram(trainY):
	"""Plots a histogram of the input with vertical lines 
	indicating the mean and +/- 1 standard deviation."""
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
		"Avg. {} Day\nReturn:	{}%".format(5,
											round(mn*100,2)) for mn in mns]
	labels[mns.shape[0]:] = [
		"Std. Dev.\nof Returns: {}%".format(round(sd*100,2)) for sd in sds]
	plt.legend(lines,#[mean_line, std_line],
			   labels)
	plt.show()
	
if __name__=="__main__":

	# get stock data
	filename= "simData/example.csv"
	
	df = pd.read_csv(filename, index_col='Date',
					parse_dates=True, na_values=['nan'])
					
	data = df.values
	cols = [col for col in df.columns if not col.startswith("Returns")]
	
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
	print "Average Training Return: {}".format(round( trainY.mean(),3 ))
	print "Average Training Return: {}".format(round( testY.mean(),3 ))
	err = np.zeros(testY.shape)
	for i in range(err.shape[0]):
			err[i] = trainY.mean()
	print "RMSE of Train Average on Test Set: {}".format(
			math.sqrt(((testY - err) ** 2).sum()/testY.shape[0])
			)
	err[0] += 0.00000001
	print "Correlation of Train Average on Test Set: {}".format(
			np.corrcoef(err, y=testY)[0,1]
			)
	for i in range(trainX.shape[1]):
			print np.corrcoef(trainX[:,i], trainY)[0][1]
#	plot_histogram(trainY)
	
	# Normalize training and test features
	trainX, testX = mean_normalization(trainX, testX)
	
	# create a learner and train it
#	learners = [lrl.LinRegLearner(verbose = True), # create a LinRegLearner
#				knn.KNNLearner(k=6, verbose = True)] # create a KNNLearner
   
#	learners = [lrl.LinRegLearner(verbose = True), # create a LinRegLearner
#			   knn.KNNLearner(k=6, verbose = True), # create a KNNLearner
#			   bag.BagLearner(learner = knn.KNNLearner, # create a BagLearner
#							   kwargs = {"k":6}, 
#							   bags = 10, 
#							   boost = True, 
#							   verbose = False),
#			   bag.BagLearner(learner = lrl.LinRegLearner, # create a BagLearner
#							   kwargs = {}, 
#							   bags = 10, 
#							   boost = True, 
#							   verbose = False)]
	learners = [bag.BagLearner(learner = lrl.LinRegLearner,# knn.KNNLearner, #create a BagLearner
								   kwargs = {},#{"k":3}, #
								   bags = i,
								   boost = True,
								   verbose = False) for i in range(5,100,5)]
	
	# Collect scoring metrics for each learner for later comparison
	cors, rmsestrain, rmsestest = [], [], []
	
	for i, learner in enumerate(learners):
		print learner.name
		learner.addEvidence(trainX, trainY) # train it

		# evaluate in sample
		predYtrain = learner.query(trainX) # get the predictions
		print predYtrain.shape
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
		
		# Join predicted values and actual values into a dataframe.
		predicted = pd.DataFrame(predY,
						   columns=["Predicted"],
						   index=df.ix[train_rows:,:].index)
		predicted = predicted.join(pd.DataFrame(testY,
						   columns=["Actual"],
						   index=df.ix[train_rows:,:].index))
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
