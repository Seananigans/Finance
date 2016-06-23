import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math, os, sys

# Import Learners
from learners import LinRegLearner as lrl
from learners import BagLearner as bag
from learners import KNNLearner as knn
# from learners import SVMLearner as svm
# from learners import DecTreeLearner as dt

# Import indicators
from indicators.Weekdays import Weekdays
from indicators.Lag import Lag
from indicators.Bollinger import Bollinger
from indicators.SimpleMA import SimpleMA as SMA

# Import dataset retrieval
from dataset_construction import create_input, create_output, get_and_store_web_data

# Import error metrics
from helpers.error_metrics import rmse, mape

# Import normalization
from helpers.normalization import mean_normalization



def predict_spy_future(symbol= None, horizon=5, learner=None, use_prices=False, verbose=False):
	"""Predict future prices or returns over a user defined horizon and machine learner."""
	if not symbol:
                fhand = pd.read_csv("spy_list.csv")
                spy_list = list(fhand.Symbols)
        else:
                spy_list = [symbol]
	results = pd.DataFrame()
	if not use_prices:
		results = results.append({'Date': np.nan,
					  "ReturnDate": np.nan,
					  'Symbol': np.nan,
					  'Return': np.nan,
					  'TestError(RMSE)': np.nan,
					  'Bench_0(RMSE)': np.nan,
					  'TestCorr': np.nan}, ignore_index=True)
	else:
		results = results.append({'Date': np.nan,
					  "FutureDate": np.nan,
					  'Symbol': np.nan,
					  'Future_Price': np.nan,
					  'TestError(MAPE)': np.nan,
					  'Bench_Last_Price(MAPE)': np.nan,
					  'TestCorr': np.nan}, ignore_index=True)

	for sym in spy_list:
		if sym in set(["ADT","NEE","WLTW","ARG","BXLT","SNDK","SNI","UA.C"]): continue #Something about ADT and NEE screws up the results
##		features = create_input(sym, [Weekdays(), Bollinger(18), SMA(10), Lag(3)], store=False)
##		features = create_input(sym, [Weekdays(), Lag(1), SMA(2), SMA(4)], store=False)
		features = get_and_store_web_data(sym, online=False)
		features["HmL_{}".format(sym)] = features["High_{}".format(sym)]-features["Low_{}".format(sym)]
		features["OmC_{}".format(sym)] = features["Open_{}".format(sym)]-features["Close_{}".format(sym)]
		features[["AdjClose_{}".format(sym),"Volume_{}".format(sym)]] = features[["AdjClose_{}".format(sym),"Volume_{}".format(sym)]].pct_change()
		output = create_output(sym, horizon=horizon, use_prices=use_prices)
		df = features.join(output).dropna()

		data = df.values
		cols = [col for col in df.columns if not col.startswith("Returns")]

		# compute how much of the data is training and testing
		train_rows = int(math.floor(0.9* data.shape[0]))
		test_rows = int(data.shape[0] - train_rows)

		# separate out training and testing data
		trainX = data[:train_rows,0:-1]
		trainY = data[:train_rows,-1]
		testX = data[train_rows:,0:-1]
		testY = data[train_rows:,-1]
		trainX, testX = mean_normalization(trainX, testX)
		
		if not learner:
			learners = [
			lrl.LinRegLearner(),
                        lrl.LinRegLearner(),
                        lrl.LinRegLearner(),
                        lrl.LinRegLearner(),
			knn.KNNLearner(k=55)
			]
		else:
			learners = [learner()]
		
		learners[0].addEvidence(trainX, trainY)
		predY = learners[0].query(testX)
		for learn in learners[1:]:
			learn.addEvidence(trainX, trainY)
			# evaluate out of sample
			predY += learn.query(testX)
		predY = predY/len(learners)
		
		# Calculate TEST Root Mean Squared Error
		RMSE = rmse(testY,predY)
		# Calculate TEST Mean Absolute Percent Error
		MAPE = mape(testY,predY)
		# Calculate correlation between predicted and TEST results
		c = np.corrcoef(predY, y=testY)
		if verbose:
			print "corr: ", c[0,1]
			print
			print "Out of sample results"
			print "RMSE: ", RMSE
			print "MAPE: ", MAPE
		# Calculate Benchmark
		if not use_prices:
			bench = rmse(testY,0.0)
		else:
			bench = mape(testY, df[[col for col in df.columns if col.startswith("Adj")]].values[train_rows:,-1])

		six_months = 5*4*6 # 5 trading days/week * 4 weeks/month * 6 months
		_, six_month_data = mean_normalization(data[:train_rows,0:-1], features.iloc[-six_months:].values)
		six_month_output = data[-six_months:,-1]
		_, todays_values = mean_normalization(data[:train_rows,0:-1], features.iloc[-1].values)
		if not learner:
			learners = [
			lrl.LinRegLearner(),
                        lrl.LinRegLearner(),
                        lrl.LinRegLearner(),
                        lrl.LinRegLearner(),
			knn.KNNLearner(k=15)
			]
		else:
			learners = [learner()]
		learners[0].addEvidence(six_month_data, six_month_output)
		future_pred = learners[0].query(todays_values.reshape(1,-1))
		for learn in learners[1:]:
                        learn.addEvidence(six_month_data, six_month_output)
			future_pred += learn.query(todays_values.reshape(1,-1))
		future_pred = future_pred/len(learners)
		
		if not use_prices:
			results = results.append( {
							'Date': features.index[-1], 
							'ReturnDate': features.index[-1] + BDay(horizon),
							'Symbol': sym, 
							'Return': float(future_pred), 
							'TestError(RMSE)': RMSE,
							'Bench_0(RMSE)': bench,
							'TestCorr': c[0,1]
							}, 
							ignore_index=True)
		else:
			results = results.append( {
							'Date': features.index[-1], 
							'FutureDate': features.index[-1] + BDay(horizon),
							'Symbol': sym, 
							'Future_Price': float(future_pred), 
							'TestError(MAPE)': MAPE,
							'Bench_Last_Price(MAPE)': bench,
							'TestCorr': c[0,1]
							}, 
							ignore_index=True)

			
	results = results.dropna()
	if not use_prices:
		results	= results.sort_values(by=["TestError(RMSE)"],ascending=True)
		results = results[["Date","ReturnDate","Symbol", "Return","TestError(RMSE)","Bench_0(RMSE)","TestCorr"]]
		results.columns = ["Date","Return_Date","Symbol", "Return(%)","Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]
		results.to_csv('return_results.csv', index="Date")
	else:
		results	= results.sort_values(by=["TestError(MAPE)"],ascending=True)
		results = results[["Date","FutureDate","Symbol", "Future_Price","TestError(MAPE)","Bench_Last_Price(MAPE)","TestCorr"]]
		results.columns = ["Date","Future_Date","Symbol", "Future_Price($)","Test_Error(MAPE)","Bench_Last_Price(MAPE)","Test_Corr"]
		results.to_csv('price_results.csv', index="Date")
	results = results.set_index("Date")
	return results.iloc[:10]

if __name__=="__main__":
	try:
		horizon = int(sys.argv[1])
	except (IndexError, ValueError):
		horizon = 5
	
	print predict_spy_future(horizon=horizon)
