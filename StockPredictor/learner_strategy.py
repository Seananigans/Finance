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
# Import indicators
from indicators.Weekdays import Weekdays
from indicators.Lag import Lag
from indicators.Bollinger import Bollinger
from indicators.SimpleMA import SimpleMA as SMA
# Import dataset retrieval
from dataset_construction import create_input, create_output
# Import error metrics
from error_metrics import rmse, mape
# Import normalization
from normalization import mean_normalization

def create_model_predict(symbol, horizon = 5):
	"""Creates predictions for future value of a stock over a specified horizon."""
	# Where are we saving the data to and retrieving it from?
	filename = "training_data/{}.csv".format(symbol)
	# Retrieve data from web and create a csv @ filename
	create_input(symbol, indicators = [], store=False)
	create_output(symbol, horizon=5, use_prices=False)
	# get stock data
	df = create_input(symbol, indicators = [Weekdays(), Bollinger(18), SMA(10), Lag(3)], store=False)
	output = create_output(symbol, horizon=5, use_prices=False)
	df = df.join(output).dropna()

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

	trainX, testX = mean_normalization(trainX, testX)
	
	# create learner
	learner = bag.BagLearner(learner = lrl.LinRegLearner,# knn.KNNLearner, #create a BagLearner
								 kwargs = {},#{"k":3}, #
								 bags = 15,
								 boost = True,
								 verbose = False)
	learner = lrl.LinRegLearner()
	learner.addEvidence(trainX, trainY) # train it

	# evaluate in sample
	predYtrain = learner.query(trainX) # get the predictions
	print
	print "In sample results"
	# Calculate TRAINING Root Mean Squared Error
	RMSE = rmse(trainY, predYtrain)
	print "RMSE: ", RMSE
	# Calculate TRAINING Mean Absolute Percent Error
	MAPE = mape(trainY, predYtrain)
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
	print "RMSE: ", RMSE
	# Calculate TEST Mean Absolute Percent Error
	MAPE = mape(testY,predY)
	print "MAPE: ", MAPE
	# Calculate correlation between predicted and TEST results
	c = np.corrcoef(predY, y=testY)
	print "corr: ", c[0,1]

	train_predictions = predicted = pd.DataFrame(predYtrain,
							   columns=["Predicted_Train"],
							   index=df.ix[:train_rows,-1].index)
	test_predictions = predicted = pd.DataFrame(predY,
							   columns=["Predicted_Test"],
							   index=df.ix[train_rows:,:].index)
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
#	print "Date,Symbol,Order,Shares"
	t=0
	df = issue_stock_order(df, data.index[t].strftime("%Y-%m-%d"), sym, "BUY", 0)
	for i in data.index:
		multiplier=1
		pred = data.ix[i,:].values
		if np.abs(pred)>5*threshold: multiplier=2
		if pred>threshold:
			try:
				df = issue_stock_order(df, data.index[t+horizon].strftime("%Y-%m-%d"), sym, "SELL", int(multiplier*num_shares))
				df = issue_stock_order(df, data.index[t].strftime("%Y-%m-%d"), sym, "BUY", int(multiplier*num_shares))
			except IndexError:
				continue
		elif pred<-threshold and shorting:
			try:
				df = issue_stock_order(df, data.index[t+horizon].strftime("%Y-%m-%d"), sym, "BUY", int(multiplier*num_shares))
				df = issue_stock_order(df, data.index[t].strftime("%Y-%m-%d"), sym, "SELL", int(multiplier*num_shares))
			except IndexError:
				continue
		t+=1
				
##		if position==None:
##			if data.ix[i,:].values[0]>threshold:
##				if data.ix[i,:].values[0]>2*threshold: multiplier = 2
##				else: multiplier=1
##				position="LONG"
##				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "BUY", int(multiplier*num_shares))
###					print "{0},{1},BUY,{2}".format(i.strftime("%Y-%m-%d"),sym,multiplier*num_shares)
##				days += 1
##			if data.ix[i,:].values[0]<-threshold and shorting:
##				if data.ix[i,:].values[0]>2*threshold: multiplier = 2
##				else: multiplier=1
##				position="SHORT"
##				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "SELL", int(multiplier*num_shares))
###					print "{0},{1},SELL,{2}".format(i.strftime("%Y-%m-%d"),sym,multiplier*num_shares)
##				days += 1
##		else:
##			if position:
##				days += 1
##			if (data.ix[i,:].values[0]>threshold and position=="SHORT" or days>=horizon) and shorting:
##				last_shares = int(df.Shares.iloc[-1])
##				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "BUY", last_shares)
###					print "{0},{1},BUY,{2}".format(i.strftime("%Y-%m-%d"), sym, last_shares)
##				position, days = None, 0
##			if (data.ix[i,:].values[0]<-threshold and position=="LONG") or days>=horizon:
##				last_shares = int(df.Shares.iloc[-1])
##				df = issue_stock_order(df, i.strftime("%Y-%m-%d"), sym, "SELL", last_shares)
###					print "{0},{1},SELL,{2}".format(i.strftime("%Y-%m-%d"), sym, last_shares)
##				position, days = None, 0
				
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
		horizon = int(sys.argv[2])
	except IndexError:
		horizon = 5
	try:
		buy_threshold = float(sys.argv[3])
	except IndexError:
		buy_threshold = 0.05
	try:
		num_shares = int(sys.argv[4])
	except IndexError:
		num_shares = 5
	try:
		if sys.argv[5].lower()=="t":
			short=True
		else:
			short=False
	except IndexError:
		short=False

	print symbol, horizon, buy_threshold, num_shares, short

	train_predictions, test_predictions = create_model_predict(symbol, horizon)
	learner_strategy(test_predictions,
						 threshold = buy_threshold,
						 sym = symbol,
						 num_shares = num_shares,
						 horizon = horizon,
						 shorting=short)

