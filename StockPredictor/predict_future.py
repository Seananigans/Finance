import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math, os, sys

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

def predict_spy_future(horizon=5, verbose=False):
	fhand = pd.read_csv("spy_list.csv")
	spy_list = list(fhand.Symbols)
	results = pd.DataFrame()
	results = results.append({'Date': np.nan,
                                  "ReturnDate": np.nan,
                                  'Symbol': np.nan,
                                  'Return': np.nan,
                                  'TestError(RMSE)': np.nan,
                                  'Benchmark_0(RMSE)': np.nan,
                                  'TestCorrelation': np.nan}, ignore_index=True)

	for sym in spy_list:
		if sym == "ADT": continue #Something about ADT screws up the results
		features = create_input(sym, [Weekdays(), Bollinger(18), SMA(10), Lag(3)], store=False)
		output = create_output(sym, horizon=horizon, use_prices=False)
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
	
		learner = lrl.LinRegLearner()
		learner.addEvidence(trainX, trainY)

		# evaluate out of sample
		predY = learner.query(testX) # get the predictions
		# Calculate TEST Root Mean Squared Error
		RMSE = rmse(testY,predY)
		bench = rmse(testY,0.0)
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
	
		_, todays_values = mean_normalization(data[:train_rows,0:-1], features.iloc[-1].values)
		future_pred = learner.query(todays_values.reshape(1,-1))
	
		results = results.append( {
						'Date': features.index[-1], 
						'ReturnDate': features.index[-1] + BDay(horizon),
						'Symbol': sym, 
						'Return': float(future_pred), 
						'TestError(RMSE)': RMSE,
                                                'Benchmark_0(RMSE)': bench,
                                                'TestCorrelation': c[0,1]
						}, 
						ignore_index=True)
	results = results.dropna()
	results = results[["Date","ReturnDate","Symbol", "Return","TestError(RMSE)","Benchmark_0(RMSE)","TestCorrelation"]]
	results = results.set_index("Date")
	results	= results.sort_values(by=["Return"],ascending=False)
	results.to_csv('results.csv', index="Date")
	return results.iloc[:10]

if __name__=="__main__":
	try:
		horizon = int(sys.argv[1])
	except (IndexError, ValueError):
		horizon = 5
	
	print predict_spy_future(horizon)
