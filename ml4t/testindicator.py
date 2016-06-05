"""
Test a learner.	 (c) 2015 Tucker Balch
"""

import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
# Import learners
from learners import LinRegLearner as lrl
from learners import KNNLearner as knn
from learners import BagLearner as bag
# Import indicator libraries
from indicators.Bollinger import Bollinger
from indicators.Momentum import Momentum
from indicators.SimpleMA import SimpleMA
from indicators.ExponentialMA import ExponentialMA
from indicators.Volatility import Volatility
from indicators.Lag import Lag
from indicators.RSI import RSI
from indicators.Weekdays import Weekdays
# Import dataset constructor library
from dataset_construction import create_input, create_output
# Import data processing libraries
from util import calculate_returns
from error_metrics import rmse, mape
from normalization import mean_normalization, max_normalization

# from plotting import plot_histogram
##from learners import SVMLearner as svm
	
if __name__=="__main__":
	symbol="MMM"
	# get stock data
	filename= "webdata/{}.csv".format(symbol)
##	upper_length = 1
##	opt_var = range(1,20)
##	indicators = [[RSI(i)] for i in opt_var]
	indicators = [RSI(9), Weekdays(), Lag(5), Volatility(15), SimpleMA(10), ExponentialMA(10), Bollinger(18), Momentum(10)] + [Lag(i) for i in range(1,5)]
	empty_list = []
	upper_length = 4
	for j in range(1,upper_length):
            for i in combinations(indicators,j):
                empty_list.append(list(i))
	indicators = empty_list
##      Best 1 set: EMA_10
##      Best 2 set: Bollinger_18, SMA_10
##      Best 3 set: Weekdays, SMA_10, Bollinger_18
##      Best 4 set: Weekdays, Lag_5, Bollinger_18, SMA_10

        opt_var = range(len(indicators))
##        opt_var = [[ind.name for ind in indicator] for indicator in indicators]
	# create a learner and train it
##        learner = bag.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=15, boost=True)
	learner = lrl.LinRegLearner()
	# Collect scoring metrics for each learner for later comparison
	cors, rmsestrain, rmsestest, mapestrain, mapestest = [], [], [], [], []
	for i, indicator in enumerate(indicators):
                dataX = create_input(symbol,indicators=indicator)
                dataY = create_output(symbol, use_prices=True)
                data = dataX.join(dataY)
                df = data.dropna()
                data = data.dropna().values
                # compute how much of the data is training and testing
                train_rows = int(math.floor(0.6* data.shape[0]))
                test_rows = int(data.shape[0] - train_rows)

                # separate out training and testing data
                trainX = data[:train_rows,0:-1]
                trainY = data[:train_rows,-1]
                testX = data[train_rows:,0:-1]
                testY = data[train_rows:,-1]
                trainX, testX = mean_normalization(trainX, testX)
		print [ind.name for ind in indicator]
		learner.addEvidence(trainX, trainY) # train it

		# evaluate in sample
		predYtrain = learner.query(trainX) # get the predictions
		print predYtrain.shape
		print
		print "In sample results"
		# Calculate TRAINING Root Mean Squared Error
		RMSE = rmse(trainY, predYtrain)
		rmsestrain.append(RMSE)
		print "RMSE: ", RMSE
		# Calculate TRAINING Mean Absolute Percent Error
		MAPE = mape(trainY, predYtrain)
		mapestrain.append(MAPE)
		print "MAPE: ", MAPE
		# Calculate correlation between predicted and TRAINING results
		c = np.corrcoef(predYtrain, y=trainY)
		print "corr: ", c[0,1]
		

		# evaluate out of sample
		predY = learner.query(testX) # get the predictions
		print
		print "Out of sample results"
		# Calculate TEST Root Mean Squared Error
		RMSE = rmse(testY,predY)
		rmsestest.append(RMSE)
		print "RMSE: ", RMSE
		# Calculate TEST Mean Absolute Percent Error
		MAPE = mape(testY, predY)
		mapestest.append(MAPE)
		print "MAPE: ", MAPE
		# Calculate correlation between predicted and TEST results
		c = np.corrcoef(predY, y=testY)
		print "corr: ", c[0,1]
		print
		cors.append(c[0,1])
		
		# Join predicted values and actual values into a dataframe.
		predicted = pd.DataFrame(predY,
						   columns=["Predicted"],
						   index=df.ix[train_rows:,:].index)
		predicted = predicted.join(pd.DataFrame(testY,
						   columns=["Actual"],
						   index=df.ix[train_rows:,:].index))
		
##		predicted = calculate_returns(predicted, 5)
		
		if i%5==0 and False:
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
##		df1 = pd.DataFrame(testX,
##                                   columns=cols,
##                                   index=df.ix[train_rows:,:].index)
##		df1 = df1.join(predicted)
##		df1.to_csv("test.csv", index_label="Date")
	
	if len(indicators)>4:
		plt.plot(range(len(cors)), cors)
		plt.xticks(range(len(cors)),opt_var)
		plt.ylabel("Correlation")
		plt.xlabel("Model Complexity")
		plt.show()
	
	try:
		print predicted.ix[dt.date.today()]
	except:
		print predicted.iloc[-2,:]
	
	# Plot testing & training RMSE on the same plot to 
	# show how error behaves with different indicators.
	testerr, = plt.plot(opt_var, rmsestest, label="Test Error")
	trainerr, = plt.plot(opt_var, rmsestrain, label="Training Error")
	plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
	plt.xlabel("Model Complexity")
	plt.ylabel("RMSE")
	plt.show()
	# Plot testing & training MAPE on the same plot to 
	# show how error behaves with different indicators.
	testerr, = plt.plot(opt_var, mapestest, label="Test Error")
	trainerr, = plt.plot(opt_var, mapestrain, label="Training Error")
	plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
	plt.xlabel("Model Complexity")
	plt.ylabel("MAPE")
	plt.show()
        csv_name = "test_{}_indicators.csv".format(upper_length)
	pd.DataFrame(np.array([[len(x)]+[y]+[ind.name for ind in x] for y,x in sorted(zip(rmsestest, indicators))])).to_csv(csv_name)
