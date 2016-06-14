"""
Test sets of indicators.
"""

import datetime as dt
from itertools import combinations
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
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

def run_test(symbol, indicator_list, learner, plotting=False, verbose=False):
        dataX = create_input(symbol,indicators=indicator_list)
        dataY = create_output(symbol, use_prices=False)
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
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predYtrain = learner.query(trainX) # get the predictions
        # Calculate TRAINING Root Mean Squared Error
        train_rmse = rmse(trainY, predYtrain)
        # Calculate TRAINING Mean Absolute Percent Error
        train_mape = mape(trainY, predYtrain)
        # Calculate correlation between predicted and TRAINING results
        train_cor = np.corrcoef(predYtrain, y=trainY)

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        # Calculate TEST Root Mean Squared Error
        test_rmse = rmse(testY,predY)
        # Calculate TEST Mean Absolute Percent Error
        test_mape = mape(testY, predY)
        # Calculate correlation between predicted and TEST results
        c = np.corrcoef(predY, y=testY)
        
        if verbose:
                print [ind.name for ind in indicator_list]
                print predYtrain.shape
                print
                print "In sample results"
                print "RMSE: ", train_rmse
                if train_mape!=np.inf:
                        print "MAPE: ", train_mape
                print "corr: ", train_cor[0,1]
                print
                print "Out of sample results"
                print "RMSE: ", test_rmse
                print "MAPE: ", test_mape
                print "corr: ", c[0,1]
                print
        
        # Join predicted values and actual values into a dataframe.
        predicted = pd.DataFrame(predY,
                                 columns=["Predicted"],
                                 index=df.ix[train_rows:,:].index)
        predicted = predicted.join(pd.DataFrame(testY,
                                                columns=["Actual"],
                                                index=df.ix[train_rows:,:].index))
        
        if plotting:
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

        return predicted, c[0,1], train_cor[0,1], train_mape, test_mape, train_rmse, test_rmse

def plot_error_curves(opt_var, train_error, test_error, error_type="RMSE"):
        testerr, = plt.plot(opt_var, test_error, label="Test Error")
	trainerr, = plt.plot(opt_var, train_error, label="Training Error")
	plt.legend([testerr,trainerr], ["Test Error", "Train Error"])
	plt.xlabel("Model Complexity")
	plt.ylabel(error_type)
	plt.title("{} Training and Test Curves".format(error_type))
	plt.show()

def test_indicator(horizon=5,test_one_indicator=True, verbose=False, plotting=False):
        fhand = pd.read_csv("spy_list.csv")
	spy_list = list(fhand.Symbols)
	use_prices=False
	
        # Get Indicators
        if test_one_indicator:
                upper_length = 1
                opt_var = range(2,21)
                indicators = [[SimpleMA(i)] for i in opt_var]
        else:
                indicators = [
                        Bollinger(4), Bollinger(5), Bollinger(19),
                        ExponentialMA(2), ExponentialMA(4), ExponentialMA(8), ExponentialMA(20),
                        Lag(1), Lag(3), Lag(8),
                        Momentum(2), Momentum(3), Momentum(10), Momentum(19),
                        SimpleMA(2), SimpleMA(4), SimpleMA(10), SimpleMA(20),
                        RSI(2), RSI(5),
                        Volatility(2), Volatility(3),
                        Weekdays()
                        ]
                for i in indicators: print i.name
                empty_list = []
                upper_length = 4
                for j in range(1,upper_length+1):
                    for i in combinations(indicators,j):
                        empty_list.append(list(i))
                indicators = empty_list
                opt_var = range(len(indicators))
##                opt_var = [[ind.name for ind in indicator] for indicator in indicators]
        
        # create a learner and train it
##        learner = bag.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags= 50, boost=False)
        learner = lrl.LinRegLearner()

        best_ind_dict = {}
        all_ind_dict = {}
        total = 1
	for symbol in spy_list:
                # Get stock data
                filename= "webdata/{}.csv".format(symbol)
                if total%20==0:
                        print round(float(total)/len(spy_list),2)
                total+=1
                
                # Collect scoring metrics for each learner for later comparison
                cors, rmsestrain, rmsestest = [], [], []
                if use_prices: mapestrain, mapestest = [], []
                
                best_rmse = np.inf
                for i, indicator in enumerate(indicators):
                        predicted, c, train_cor, train_mape, test_mape, train_rmse, test_rmse = run_test(symbol, indicator, learner)
                        if test_rmse < best_rmse:
                                best_rmse = test_rmse
                                best_indicator_set = ", ".join([ind.name for ind in indicator])
                        indicator_set = ", ".join([ind.name for ind in indicator])
                        if test_rmse<4.0: all_ind_dict[indicator_set] = all_ind_dict.get(indicator_set, 0.0) + test_rmse
                        rmsestrain.append(train_rmse)
                        rmsestest.append(test_rmse)
                        if use_prices:
                                mapestrain.append(train_mape)
                                mapestest.append(test_mape)
                        cors.append(c)
                
                if len(indicators)>4 and plotting:
                        plt.plot(range(len(cors)), cors)
                        plt.xticks(range(len(cors)),opt_var)
                        plt.ylabel("Correlation")
                        plt.xlabel("Model Complexity")
                        plt.show()
                
                # Plot testing & training RMSE on the same plot to 
                # show how error behaves with different indicators.
                if plotting:
                        plot_error_curves(opt_var, rmsestrain, rmsestest, error_type="RMSE")
                        if use_prices:
                                # Plot testing & training MAPE on the same plot to 
                                # show how error behaves with different indicators.
                                plot_error_curves(opt_var, mapestrain, mapestest, error_type="MAPE")

                best_ind_dict[best_indicator_set] = [best_ind_dict.get(best_indicator_set, [0,0.0])[0] + 1,
                                                     best_ind_dict.get(best_indicator_set, [0,0.0])[1] + best_rmse]

	if test_one_indicator:
##                sorted_indicators = pd.DataFrame(
##                        {"Indicator":[re.search("([A-Za-z]+)", string).groups(0)[0] for string in best_ind_dict.keys()],
##                         "Window":[int(re.search("(\d+)", string).groups(0)[0]) for string in best_ind_dict.keys()],
##                         "Value":best_ind_dict.values()})
##                sorted_indicators.sort_values("Value", ascending=False, inplace=True)
##                sorted_indicators = sorted_indicators[["Indicator","Window","Value"]]
##                sorted_indicators.to_csv("best_indicators/{}_best.csv".format(sorted_indicators.Indicator.iloc[0]))
                sorted_indicators = pd.DataFrame(
                        {"Indicator":[re.search("([A-Za-z]+)", string).groups(0)[0] for string in all_ind_dict.keys()],
                         "Window":[int(re.search("(\d+)", string).groups(0)[0]) for string in all_ind_dict.keys()],
                         "Value":[val/len(spy_list) for val in all_ind_dict.values()]})
                sorted_indicators.sort_values("Value", ascending=True, inplace=True)
                sorted_indicators = sorted_indicators[["Indicator","Window","Value"]]
                sorted_indicators.to_csv("best_indicators/{}_best.csv".format(sorted_indicators.Indicator.iloc[0]))
        else:
##                sorted_indicators = pd.DataFrame(
##                        {"Indicator":best_ind_dict.keys(),
##                         "Error":[val[1]/val[0] for val in best_ind_dict.values()],
##                         "Value":[val[0] for val in best_ind_dict.values()]})
##                sorted_indicators.sort_values("Value", ascending=False, inplace=True)
##                sorted_indicators = sorted_indicators[["Indicator", "Error", "Value"]]
##                sorted_indicators.to_csv("best_indicators/{}_best.csv".format(upper_length))
                sorted_indicators = pd.DataFrame(
                        {"Indicator":all_ind_dict.keys(),
                         "Error":all_ind_dict.values()})
                sorted_indicators.sort_values("Error", ascending=True, inplace=True)
                sorted_indicators = sorted_indicators[["Indicator", "Error"]]
                sorted_indicators.to_csv("best_indicators/average_{}_best.csv".format(upper_length))
        csv_name = "test_{}_indicators.csv".format(upper_length)
	pd.DataFrame(np.array([[len(x)]+[y]+[ind.name for ind in x] for y,x in sorted(zip(rmsestest, indicators))])).to_csv(csv_name)

if __name__=="__main__":
	test_one_indicator = False
	horizon=10
	test_indicator(horizon=horizon, test_one_indicator=True)
	
