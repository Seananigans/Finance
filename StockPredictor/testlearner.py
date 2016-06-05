"""
Test a learner.	 (c) 2015 Tucker Balch
"""

import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from learners import LinRegLearner as lrl
from learners import KNNLearner as knn
from learners import BagLearner as bag
from util import calculate_returns
from error_metrics import rmse, mape
from normalization import mean_normalization, max_normalization
from plotting import plot_histogram
try:
	from learners import SVMLearner as svm
except:
	pass
	
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
	
	# Formatting for printing
	if trainY.max()<=1.0:
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
	opt_var = range(5,50,7)
	learners = [bag.BagLearner(learner = lrl.LinRegLearner,# knn.KNNLearner, #create a BagLearner
								   kwargs = {},#{"k":3}, #
								   bags = i,
								   boost = True,
								   verbose = False) for i in opt_var]
	
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
		RMSE = rmse(trainY, predYtrain)#math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0])
		print "RMSE: ", RMSE
		# Calculate TRAINING Mean Absolute Percent Error
		MAPE = mape(trainY, predYtrain)
		print "MAPE: ", MAPE
		# Calculate correlation between predicted and TRAINING results
		c = np.corrcoef(predYtrain, y=trainY)
		print "corr: ", c[0,1]
		rmsestrain.append(RMSE)

		# evaluate out of sample
		predY = learner.query(testX) # get the predictions
		print
		print "Out of sample results"
		# Calculate TEST Root Mean Squared Error
		RMSE = rmse(testY,predY)#math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
		print "RMSE: ", RMSE
		# Calculate TEST Mean Absolute Percent Error
		MAPE = mape(testY, predY)#(np.abs(testY - predY)/testY).mean()
		print "MAPE: ", MAPE
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
						   index=df.ix[train_rows:,:].index)
		df1 = df1.join(predicted)
		df1.to_csv("test.csv", index_label="Date")
	
	if len(learners)>4:
		plt.plot(range(len(cors)), cors)
		plt.xticks(range(len(cors)),opt_var)
		plt.ylabel("Correlation")
		plt.xlabel("Model Complexity")
		plt.show()
	
	try:
		print predicted.ix[dt.date.today()]
	except:
		print predicted.iloc[-2,:]
	
	# Plot testing & training error on the same plot to 
	# show how error behaves with different models.
	testerr, = plt.plot(range(len(cors)), rmsestest, label="Test Error")
	trainerr, = plt.plot(range(len(cors)), rmsestrain, label="Training Error")
	plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
	plt.xlabel("Model Complexity")
	plt.ylabel("RMSE")
	plt.xticks(range(len(cors)),opt_var)
	plt.show()
