import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math
import os
from util import get_data, plot_data, create_training_data, calculate_returns
# from learners import KNNLearner as knn
from learners import LinRegLearner as lrl
from learners import BagLearner as bag

symbol='IBM'
# Where are we saving the data to and retrieving it from?
filename = "{}.csv".format(symbol)
horizon = 5
# Retrieve data from web and create a csv @ filename
create_training_data(
						symbol, 
						
						use_web = True, #Use the web to gether Adjusted Close data?
						horizon = horizon, # Num. days ahead to predict
						filename = filename, #Save location
						use_vol = False, #Use the volume of stocks traded that day?
						use_prices = True, #Use future Adj. Close as opposed to future returns
						direction = False, #Use the direction of the market returns as output
						indicators = ['Bollinger',
									   'Momentum',
									   'Volatility',
									   'SimpleMA',
									   'ExponentialMA',
									   'Lagging',
									   'Weekdays'],
						num_lag = 5
					)

# get stock data
df = pd.read_csv(filename, index_col='Date', parse_dates=True, na_values=['nan'])


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

# create learner
learners = [bag.BagLearner(learner = lrl.LinRegLearner,# knn.KNNLearner, #create a BagLearner
								   kwargs = {},#{"k":3}, #
								   bags = 5,
								   boost = True,
								   verbose = False)]
learner=learners[0]


print learner.name
learner.addEvidence(trainX, trainY) # train it

# evaluate in sample
predYtrain = learner.query(trainX) # get the predictions
rmse = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0])
print
print "In sample results"
# Calculate TRAINING Root Mean Squared Error
rmse = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0])
print "RMSE: ", rmse
# Calculate TRAINING Mean Absolute Percent Error
mape = (np.abs(trainY - predYtrain)/trainY).mean()
print "MAPE: ", mape
# Calculate correlation between predicted and TRAINING results
c = np.corrcoef(predYtrain, y=trainY)
print "corr: ", c[0,1]

# evaluate out of sample
predY = learner.query(testX) # get the predictions
print
print "Out of sample results"
# Calculate TEST Root Mean Squared Error
rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
print "RMSE: ", rmse
# Calculate TEST Mean Absolute Percent Error
mape = (np.abs(testY - predY)/testY).mean()
print "MAPE: ", mape
# Calculate correlation between predicted and TEST results
c = np.corrcoef(predY, y=testY)
print "corr: ", c[0,1]

train_predictions = predicted = pd.DataFrame(predYtrain,
						   columns=["Predicted_Train"],
						   index=df.ix[:train_rows,-1].index)
test_predictions = predicted = pd.DataFrame(predY,
						   columns=["Predicted_Test"],
						   index=df.ix[train_rows:,:].index)
train_predictions = calculate_returns(train_predictions,horizon)
test_predictions = calculate_returns(test_predictions,horizon)
train_predictions.to_csv('train_preds.csv', index_label="Date")
test_predictions.to_csv('test_preds.csv', index_label="Date")


buy_threshold = 0.03
print np.sum(train_predictions > buy_threshold)
print np.sum(test_predictions > buy_threshold)

def learner_strategy(data, threshold=0.03, sym="IBM"):
	df = pd.DataFrame()
	df = df.append({'Date': np.nan, 'Symbol': np.nan, 'Order': np.nan, 'Shares': np.nan}, ignore_index=True)
	cols = [col for col in data.columns if col.startswith("Pred")]
	position = None
	print "Date,Symbol,Order,Shares"
	for i in data.index:
		if data.ix[i,:].values[0]>threshold:
			df = df.append( {
						'Date': i.strftime("%Y-%m-%d"), 
						'Symbol': sym, 
						'Order': "BUY", 
						'Shares': 100
						}, 
						ignore_index=True)
			print "{0},{1},BUY,100".format(i.strftime("%Y-%m-%d"),sym)
		if data.ix[i,:].values[0]<-threshold:
			df = df.append( {
						'Date': i.strftime("%Y-%m-%d"), 
						'Symbol': sym, 
						'Order': "SELL", 
						'Shares': 100
						}, 
						ignore_index=True)
			print "{0},{1},SELL,100".format(i.strftime("%Y-%m-%d"),sym)
	df = df.dropna()
	df.index = df.Date
	df = df[[col for col in df.columns if col!="Date"]]
	colms = ['Symbol', 'Order', 'Shares']
	df =  df[colms]
	df.to_csv("orders/learner_orders.csv", index_label="Date")

learner_strategy(test_predictions)

