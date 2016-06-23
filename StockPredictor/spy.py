import os
import pandas as pd
import numpy as np
from random import sample

from learner_strategy import learner_strategy
from marketsim import compute_portvals, test_code
from learners.LinRegLearner import LinRegLearner as lrl
from learners.KNNLearner import KNNLearner as knn
from helpers.normalization import mean_normalization
from dataset_construction import get_and_store_web_data, create_input, create_output
from predict_future import predict_spy_future

fhand = pd.read_csv("spy_list.csv")
spy_list = list(fhand.Symbols)
# spy_list = sample(spy_list,15)

# Learner Strategy args = [1]symbol [2]horizon [3]threshold [4]num_shares [5]shorting?
horizon=5
shares=10
shorting=True
threshold=0.01

# 	os.system('python learner_strategy.py {0:} 5 0.01 10 T'.format(i))
# 	# Market Simulator args = [1]create a plot [2]portfolio start value
# 	os.system('python marketsim.py T 1000')

for _, i in enumerate(spy_list):
	ibm_future_returns = create_output(i, use_prices=False)
	ibm_future_returns.columns = ["Returns_{}".format(i)]
	# Create full dataset
	dataset = get_and_store_web_data(i, online=False).join(ibm_future_returns).dropna()
	# Calculate Open minus Close and High minus Low
	dataset["HmL_{}".format(i)] = dataset["High_{}".format(i)]-dataset["Low_{}".format(i)]
	# Choose features that will be used for training model
	dataset = create_input(i, []).join(
		dataset[["Volume_{}".format(i),"HmL_{}".format(i)]]).join(
		create_output(i, use_prices=False)).dropna()
	# Change actual Adjusted Close and Volume to percent change to deal with trends in prices and 
	# improve stationarity (decrease time dependence)
	dataset[["AdjClose_{}".format(i),"Volume_{}".format(i)]] = dataset[["AdjClose_{}".format(i),"Volume_{}".format(i)]].pct_change()
	# Remove any NaN values
	dataset = dataset.dropna()
	# Create testing set
	num_rows = dataset.shape[0]
	test_rows = range(num_rows-int(0.2*num_rows), num_rows)
	testingX, testingY = (dataset.ix[test_rows,:-1], dataset.ix[test_rows,-1])
	# Remove test rows
	features = dataset.iloc[range(0, num_rows-int(0.2*num_rows))]
	# Create datasets to use for cross-validation
	dividend = 3
	n_rows = features.shape[0]
	section = int(float(n_rows)/dividend + 1)

	# Retreive data comparable in rows to those previously trained on.
	trainingX, trainingY = (dataset.ix[-section:,:-1], dataset.ix[-section:,-1])
	validX, validY = trainingX.values, trainingY.values
	# Mean Normalize by data used to train the model
	validX, testX = mean_normalization( validX, testingX )
	# Create, train, and query Linear Regression Learner
	learner = lrl()
	learner.addEvidence(validX, validY)
	resultslrl = learner.query(testX.values)
	# Create, train, and query Linear Regression Learner
	learner = knn()
	learner.addEvidence(validX, validY)
	resultsknn = learner.query(testX.values)
	# Calculate Avg(kNN and Linear) Regression Predictions
	resultsavg = np.add(resultslrl , resultsknn)/2

	pred = pd.DataFrame(resultsavg, columns = ["Predicted"], index=testingY.index)
	validY = pd.DataFrame(validY, columns = ["Past"], index=trainingY.index)
	dframe = pd.DataFrame(pred.join(trainingY.to_frame(name="Past").join(testingY, how="outer"), how="outer"))
	returns = dframe.Predicted.dropna()
	returns = returns.to_frame()
	returns.columns = ["Returns"]
	predict_spy_future(symbol=i, horizon=horizon, use_prices=False)

	symbol = i
	horizon = 5
	num_shares = 10
	orders_file = "./orders/learner_orders.csv"
	start_val = 1000
	learner_strategy(data = returns, 
					 threshold = 0.01, 
					 sym = symbol, 
					 horizon = horizon, 
					 num_shares = num_shares, 
					 shorting = True)
	test_code(of = orders_file, sv = start_val, plotting="T", testing=True)