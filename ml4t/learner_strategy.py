import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math
import os
import sys
from util import get_data, plot_data, create_training_data
# from learners import KNNLearner as knn
from learners import LinRegLearner as lrl
from learners import BagLearner as bag


def create_model_predict(symbol, horizon = 5):
	# Where are we saving the data to and retrieving it from?
	filename = "training_data/{}.csv".format(symbol)
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
									   bags = 15,
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

	train_predictions.ix[:,:] = train_predictions.values/df.ix[:train_rows,[symbol]].values - 1.0
	test_predictions.ix[:,:] = test_predictions.values/df.ix[train_rows:,[symbol]].values - 1.0
	train_predictions.to_csv('train_preds.csv', index_label="Date")
	test_predictions.to_csv('test_preds.csv', index_label="Date")
	return train_predictions, test_predictions

def issue_stock_order(df, date, sym, order, shares):
	"""Adds a stock order to the dataframe"""
	df = df.append( {
					'Date': date, 
					'Symbol': sym, 
					'Order': order, 
					'Shares': shares
					}, 
					ignore_index=True)
	return df

def learner_strategy(data, threshold=0.05, sym="IBM", horizon=5, num_shares=100, shorting=False):
	df = pd.DataFrame()
	df = df.append({'Date': np.nan, 'Symbol': np.nan, 'Order': np.nan, 'Shares': np.nan}, ignore_index=True)
	cols = [col for col in data.columns if col.startswith("Pred")]
	position = None
	days = 0
# 	print "Date,Symbol,Order,Shares"
	for i in data.index:
		if position==None:
			if data.ix[i,:].values[0]>threshold:
				if data.ix[i,:].values[0]>2*threshold: multiplier = 2
				else: multiplier=1
				position="LONG"
				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "BUY", int(multiplier*num_shares))
# 				print "{0},{1},BUY,{2}".format(i.strftime("%Y-%m-%d"),sym,multiplier*num_shares)
				days += 1
			if data.ix[i,:].values[0]<-threshold and shorting:
				if data.ix[i,:].values[0]>2*threshold: multiplier = 2
				else: multiplier=1
				position="SHORT"
				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "SELL", int(multiplier*num_shares))
# 				print "{0},{1},SELL,{2}".format(i.strftime("%Y-%m-%d"),sym,multiplier*num_shares)
				days += 1
		else:
			if position:
				days += 1
			if (data.ix[i,:].values[0]>threshold and position=="SHORT" or days>=horizon) and shorting:
				last_shares = int(df.Shares.iloc[-1])
				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "BUY", last_shares)
# 				print "{0},{1},BUY,{2}".format(i.strftime("%Y-%m-%d"), sym, last_shares)
				position, days = None, 0
			if (data.ix[i,:].values[0]<-threshold and position=="LONG") or days>=horizon:
				last_shares = int(df.Shares.iloc[-1])
				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "SELL", last_shares)
# 				print "{0},{1},SELL,{2}".format(i.strftime("%Y-%m-%d"), sym, last_shares)
				position, days = None, 0
				
	df = df.dropna()
	df.index = df.Date
	df = df[[col for col in df.columns if col!="Date"]]
	colms = ['Symbol', 'Order', 'Shares']
	df =  df[colms]
	df.to_csv("orders/learner_orders.csv", index_label="Date")


if __name__=="__main__":
	try:
		symbol = sys.argv[1]
	except IndexError:
		symbol='AAPL'
	try:
		horizon = float(sys.argv[2])
	except IndexError:
		horizon = 5
	try:
		buy_threshold = float(sys.argv[3])
	except IndexError:
		buy_threshold = 0.05

	dummy, test_predictions = create_model_predict(symbol, horizon)
	learner_strategy(test_predictions, threshold=buy_threshold, sym=symbol, num_shares=5, horizon = horizon)
	# learner_strategy(train_predictions, num_shares=5)

