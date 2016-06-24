"""PART 1"""
# Import Python Libraries necessary for the report
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
# Import personal libraries developed for project available on github for exploration:
# https://github.com/Seananigans/Finance/tree/master/StockPredictor
from dataset_construction import create_input, create_output, get_and_store_web_data
from error_metrics import rmse
from learners.LinRegLearner import LinRegLearner as lrl
from learners.KNNLearner import KNNLearner as knn
from normalization import mean_normalization
from predict_future import predict_spy_future

"""PART 2"""
# Create features dataset and output dataset
# fhand = pd.read_csv("spy_list.csv")
# spy_list = list(fhand.Symbols)
symbol = "AAPL"
# avg_above_zero = 0.
# avg_below_zero = 0.
# avg_return = 0.0
# avg_std = 0.0
# length = len(spy_list)
# for symbol in spy_list:
ibm = create_input(symbol)
ibm_future_prices = create_output(symbol, use_prices=True)
ibm_future_returns = create_output(symbol, use_prices=False)
ibm_future_prices.columns = ["Future_{}".format(symbol)]
ibm_future_returns.columns = ["Returns_{}".format(symbol)]
# Display the joined Dataset
ibm_price_returns_df = ibm.join(ibm_future_prices).join(ibm_future_returns)
num_days = ibm_price_returns_df.dropna().shape[0]
# if num_days<1: 
# 	length-=1
# 	continue
print ibm_price_returns_df.iloc[:4]
print
# Display Statistics about output variable
print "There are {} days/observations in the raw data.".format(num_days+5)
print "{} of those days observe a 5 day future return that is not a number (eg. NaN).".format(
	ibm_price_returns_df[np.isnan(ibm_price_returns_df["Returns_{}".format(symbol)])].shape[0])
above_zero = ibm_price_returns_df[ibm_price_returns_df["Returns_{}".format(symbol)]>0.0].shape[0]
percent_above = ibm_price_returns_df[ibm_price_returns_df["Returns_{}".format(symbol)]>0.0].shape[0]*100./num_days
print "{} or {:.2f}% of those days observe a 5 day future return ABOVE zero.".format( above_zero, percent_above)
below_zero = ibm_price_returns_df[ibm_price_returns_df["Returns_{}".format(symbol)]<=0.0].shape[0]
percent_below = ibm_price_returns_df[ibm_price_returns_df["Returns_{}".format(symbol)]<=0.0].shape[0]*100./num_days
print "{} or {:.2f}% of those days observe a 5 day future return EQUAL TO or BELOW zero.".format(
	below_zero, percent_below)
avg_return_stock = ibm_price_returns_df["Returns_{}".format(symbol)].mean()
avg_std_stock = ibm_price_returns_df["Returns_{}".format(symbol)].std()
print "The average 5 day future return for all days in this \
dataset is {:.4f} with a standard deviation of {:.4f}.".format(avg_return_stock,avg_std_stock)
#     avg_above_zero += percent_above
#     avg_below_zero += percent_below
#     avg_return += avg_return_stock
#     avg_std += avg_std_stock
#     
# print "Average percent above zero:", avg_above_zero/length
# print "Average percent below zero:", avg_below_zero/length
# print "Average of average returns:", avg_return/length
# print "Average of standard deviation of returns:", avg_std/length 

"""PART 3"""
# Display Prices and returns over whole timeline for example dataset IBM
plots = ibm.join(ibm_future_returns).plot(subplots=True,figsize=(15,4), title=symbol, legend=False)
plots[0].set_ylabel("{} Prices".format(symbol))
plots[1].set_ylabel("{} Returns".format(symbol))
plt.savefig("report_figures/{}_prices_returns.png".format(symbol))
# Display how current price relates to 5 day future returns and prices
prices_corr = ibm.join(ibm_future_prices).corr()["AdjClose_{}".format(symbol)]["Future_{}".format(symbol)]
returns_corr = ibm.join(ibm_future_returns).corr()["AdjClose_{}".format(symbol)]["Returns_{}".format(symbol)]
f, axarr = plt.subplots(1,2, figsize=(15,3))
axarr[0].scatter(ibm, ibm_future_prices, label="Correlation: {}".format(round(prices_corr,2)))
axarr[0].set_xlabel('Current Prices')
axarr[0].set_ylabel('Price 5 Days Later')
axarr[0].legend(loc="lower right", frameon=False)
axarr[1].scatter(ibm, ibm_future_returns, label="Correlation: {}".format(round(returns_corr,2)))
axarr[1].set_xlabel('Current Prices')
axarr[1].set_ylabel('5 Day Return')
axarr[1].legend(loc="lower left", frameon=False)
plt.show()
exit()
"""PART 4"""
# Creates input data with a 4 day window bollinger value indicator
from indicators.Bollinger import Bollinger
df = create_input("IBM", indicators=[Bollinger(3)])
# Display how feature data appears before removing NA values.
print df.iloc[:5]

"""PART 5"""
# Display how feature data appears after removing NA values.
print df.iloc[:6].dropna()
# Display mean normalized feature data
print "====================After Normalizing===================="
df = (df-df.mean())/ df.std()
print df.iloc[:6].dropna()

"""PART 6"""
# Creates output data of 3 day future returns
df_output = create_output("IBM",horizon=3)
# Display output data as it appears before removing NA values.
print df_output.iloc[-6:]
# Display output data as it appears before removing NA values.
print "=========After Processing==========="
print df_output.iloc[-6:].dropna()

"""PART 7"""
from indicators import Bollinger, SimpleMA, ExponentialMA, RSI, Volatility, Weekdays, Lag
ibm_indicators = create_input("IBM",[
        Bollinger.Bollinger(i) for i in range(3,20)] + [ 
        SimpleMA.SimpleMA(i) for i in range(3,20)] + [
        RSI.RSI(i) for i in range(3,20)] + [
        ExponentialMA.ExponentialMA(i) for i in range(3,20)] + [ 
        Weekdays.Weekdays() ] + [
        Lag.Lag(i) for i in range(5)])
overall_data = get_and_store_web_data("IBM", online=False)
ibm_indicators = ibm_indicators.join( overall_data[
        [col for col in overall_data.columns if not col.startswith("Adj")]] )
print (ibm_indicators.join(ibm_future_returns).corr().ix[:,-1])[np.abs(
        ibm_indicators.join(ibm_future_returns).corr().ix[:,-1])>0.08]

"""PART 8"""
# Reports 5 symbols with the lowest test prediction error (RMSE)
predictions = predict_spy_future(horizon=5, learner=lrl)
print predictions[[col for col in predictions.columns if not col.startswith("Return_Date")]].iloc[:5]

"""PART 9"""
# Display performance statistics as they relate to the benchmark for all S&P 500 companies in spy_list.csv
predicted_returns_results = pd.read_csv("return_results.csv", index_col="Date")
# Calculate meaningful statistics
averages = predicted_returns_results[["Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]].mean(
    axis=0).to_frame(name="Average")
medians = predicted_returns_results[["Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]].median(
    axis=0).to_frame(name="Median")
stdevs = predicted_returns_results[["Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]].std(
    axis=0).to_frame(name="Standard Deviation")
df = averages.join(medians).join(stdevs)
print df
# Plot histogram to show performance against benchmark.
predicted_returns_results[["Test_Error(RMSE)","Bench_0(RMSE)"]].plot.hist(color=["r","b"], bins=30,
                                                                          alpha=0.3, figsize=(10,3))
plt.axvline(predicted_returns_results.median(axis=0)[["Test_Error(RMSE)"]].values, color="r")
plt.axvline(predicted_returns_results.median(axis=0)[["Bench_0(RMSE)"]].values, color="b")
plt.xlabel("RMSE")
plt.xlim((0,.25))
plt.show()

"""PART 10"""
# Create full dataset
dataset = get_and_store_web_data("IBM", online=False).join(ibm_future_returns).dropna()
# Calculate Open minus Close and High minus Low
dataset["HmL_IBM"] = dataset["High_IBM"]-dataset["Low_IBM"]
dataset["OmC_IBM"] = dataset["Open_IBM"]-dataset["Close_IBM"]
# Choose features that will be used for training model
dataset = create_input("IBM", []).join(
    dataset[["Volume_IBM","HmL_IBM"]]).join(
    create_output("IBM", use_prices=False)).dropna()
# Change actual Adjusted Close and Volume to percent change to deal with trends in prices and 
# improve stationarity (decrease time dependence)
dataset[["AdjClose_IBM","Volume_IBM"]] = dataset[["AdjClose_IBM","Volume_IBM"]].pct_change()
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
print "There are {} datasets with {} rows each, from the total {} rows in the dataset.".format(
    dividend, section, n_rows)
d_rows = [(section*(i-1), section*i) for i in range(1,dividend+1)]
print "Feature dataset indices: {}".format(d_rows)
features = [features.iloc[i[0]:i[1]] for i in d_rows]
print "Test dataset indices: {} to {}".format(test_rows[0],test_rows[-1])

"""PART 11"""
for dataset in features:
    # Create a training and validation set
    train_rows = range(0, int(0.8*dataset.shape[0]))
    valid_rows = range(int(0.8*dataset.shape[0]), dataset.shape[0])
    print "Trained from {} to {}.".format(dataset.iloc[train_rows].index[0].strftime("%B %d, %Y"),
                                         dataset.iloc[train_rows].index[-1].strftime("%B %d, %Y"))
    print "Tested from {} to {}.".format(dataset.iloc[valid_rows].index[0].strftime("%B %d, %Y"),
                                         dataset.iloc[valid_rows].index[-1].strftime("%B %d, %Y"))
    # Split into training and validation sets.
    trainX, trainY = (dataset.ix[train_rows,:-1].values, dataset.ix[train_rows,-1].values)
    validX, validY = (dataset.ix[valid_rows,:-1].values, dataset.ix[valid_rows,-1].values)
    # Normalize training and validation set by the values of the training data
    trainX, validX = mean_normalization(trainX, validX)
    # Create two learners (Linear Regression and k-Nearest Neighbors Regression)
    learner = lrl()
    learner2 = knn(k=15)
    # Train the models on the training data
    learner.addEvidence(trainX, trainY)
    learner2.addEvidence(trainX, trainY)
    # Predict the future data in the validation set
    resultslrl = learner.query(validX)
    resultsknn = learner2.query(validX)
    # Average prediction for an ensemble prediction
    resultsavg = np.add(resultslrl , resultsknn)/2
    # Draw plots to compare predictions vs actual values
    f, (plot1, plot2, plot3) = plt.subplots(1,3, figsize=(12,3), sharex=True)
    # Linear Regression Prediction plot
    plot1.scatter(x = resultslrl, y = validY, 
                  label="Correlation: {}\nTest RMSE: {}\nBench RMSE: {}".format(
            round(np.corrcoef(resultslrl, validY)[0,1],4),
            round(rmse(resultslrl, validY),4),
            round(rmse(np.zeros(validY.shape), validY),4)))
    plot1.legend(loc="upper left")
    plot1.set_xlabel("Linear Regression")
    plot1.set_ylabel("Actual Returns")
    # k-Nearest Neighbors Regression Prediction plot
    plot2.scatter(x = resultsknn, y = validY, 
                  label="Correlation: {}\nTest RMSE: {}\nBench RMSE: {}".format(
            round(np.corrcoef(resultsknn, validY)[0,1],4),
            round(rmse(resultsknn, validY),4),
            round(rmse(np.zeros(validY.shape), validY),4)))
    plot2.legend(loc="upper left")
    plot2.set_xlabel( "kNN Regression" )
    # Avg(kNN and Linear) Regression Prediction plot
    plot3.scatter(x = resultsavg, y = validY, 
                  label="Correlation: {}\nTest RMSE: {}\nBench RMSE: {}".format(
            round(np.corrcoef(resultsavg, validY)[0,1],4),
            round(rmse(resultsavg, validY),4),
            round(rmse(np.zeros(validY.shape), validY),4)))
    plot3.legend(loc="upper left")
    plot3.set_xlabel( "Avg(kNN and Linear) Regression" )
    plt.show()
# Retreive data comparable in rows to those previously trained on.
trainingX, trainingY = (dataset.ix[-section:,:-1], dataset.ix[-section:,-1])
validX, validY = trainingX.values, trainingY.values
# Mean Normalize by data used to train the model
validX, testX = mean_normalization( validX, testingX )
# Print dates used for training and testing
print "Trained from {} to {}".format(dataset.iloc[-section:].index[0].strftime("%B %d, %Y"),
                                     dataset.iloc[-section:].index[-1].strftime("%B %d, %Y"))
print "Tested from {} to {}".format(testX.index[0].strftime("%B %d, %Y"),
                                    testX.index[-1].strftime("%B %d, %Y"))
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
# Plot correlation between predicted and actual returns
plt.figure(figsize=(12,4))
plt.scatter(resultsavg, 
            testingY, 
            color="r",
            label="Correlation: {}\nTest RMSE: {}\nBench RMSE: {}".format(
        round(np.corrcoef(resultsavg, testingY)[0,1],4),
        round(rmse(resultsavg, testingY),4),
        round(rmse(np.zeros(testingY.shape), testingY),4)))
plt.legend(loc="upper left")
plt.xlabel( "Avg(kNN and Linear) Regression Test Results" )
plt.ylabel("Actual Returns")
plt.axvline(0.0)
plt.axhline(0.0)
plt.show()
# Join returns data into one dataframe for plotting.
pred = pd.DataFrame(resultsavg, columns = ["Predicted"], index=testingY.index)
validY = pd.DataFrame(validY, columns = ["Past"], index=trainingY.index)
dframe = pd.DataFrame(pred.join(trainingY.to_frame(name="Past").join(testingY, how="outer"), how="outer"))
# Calculate sensitivity.
print "Only {:.2f}% of the predicted returns were greater than 0.0, \
while {:.2f}% of the actual test data returns were greater than 0.0.".format( 
    (np.sum(pred>0)*100./testingY.shape[0]).values[0], np.sum(testingY>0)*100./testingY.shape[0] )
print "{:.2f}% of the time the return was positive, the model predicted it would be positive.".format( 
    (dframe[dframe.Predicted>0])[dframe.y_IBM>0].shape[0]*100./dframe[dframe.y_IBM>0].shape[0] )
print "{:.2f}% of the time the return was negative, the model predicted it would be negative.".format( 
    (dframe[dframe.Predicted<=0])[dframe.y_IBM<=0].shape[0]*100./dframe[dframe.y_IBM<=0].shape[0] )
# Plot returns dataframe.
dframe.plot(figsize=(12,4))
plt.ylabel("Returns")
plt.axhline(0.0, color='k')
plt.show()

"""PART 12"""
# Predict 5 day returns using Ensemble of Regression Models
predict_spy_future(horizon=5, use_prices=False)
# Display performance statistics as they relate to the benchmark for all S&P 500 companies in spy_list.csv
predicted_returns_results = pd.read_csv("return_results.csv", index_col="Date")
# Calculate meaningful statistics
averages = predicted_returns_results[
    ["Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]].mean(axis=0).to_frame(name="Average")
medians = predicted_returns_results[
    ["Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]].median(axis=0).to_frame(name="Median")
stdevs = predicted_returns_results[
    ["Test_Error(RMSE)","Bench_0(RMSE)","Test_Corr"]].std(axis=0).to_frame(name="Standard Deviation")
df = averages.join(medians).join(stdevs)
print df
# Plot histogram to show performance against benchmark.
predicted_returns_results[
    ["Test_Error(RMSE)","Bench_0(RMSE)"]].plot.hist( color=["r","b"], bins=30, alpha=0.3, figsize=(10,3) )
plt.axvline(predicted_returns_results.median(axis=0)[["Test_Error(RMSE)"]].values, color="r")
plt.axvline(predicted_returns_results.median(axis=0)[["Bench_0(RMSE)"]].values, color="b")
plt.xlabel("RMSE")
plt.xlim((0,.25))
plt.show()

"""PART 13"""
bench_list, error_list = [], []
bench_std_list, error_std_list = [], []
weeks = range(1,5)
for i in range(len(weeks)):
    try:
        predict_spy_future(horizon=5*weeks[i])
        test = pd.read_csv("return_results.csv", index_col="Date")
        error_list.append(test.mean(axis=0)[["Test_Error(RMSE)"]])
        bench_list.append(test.mean(axis=0)[["Bench_0(RMSE)"]])
        error_std_list.append(test.std(axis=0)[["Test_Error(RMSE)"]])
        bench_std_list.append(test.std(axis=0)[["Bench_0(RMSE)"]])
    except:
        weeks[i] = None
        continue
weeks = [w for w in weeks if w]

"""PART 14"""
error_upper_band = [mn + 3*std for mn, std in zip(error_list, error_std_list)]
bench_upper_band = [mn + 3*std for mn, std in zip(bench_list, bench_std_list)]

"""PART 15"""
fig, ax = plt.subplots(figsize=(12,3))

test_error_plot, = plt.plot(weeks, error_list, 'r', lw=3)
test_error_upper_plot, = plt.plot(weeks, error_upper_band, 'r--')
bench_error_plot, = plt.plot(weeks, bench_list, 'b', lw=3)
bench_error_upper_plot, = plt.plot(weeks, bench_upper_band, 'b--')
plt.legend([test_error_plot, test_error_upper_plot, bench_error_plot, bench_error_upper_plot], 
           ["Average Test Error", 
            "Avg. Test Error + 3 Sigma", 
            "Average Benchmark Error", 
            "Avg. Bench Error + 3 Sigma"], 
           loc="upper left")
plt.xlabel("Weeks")
plt.ylabel("Estimate Error (RMSE)")

plt.show()

"""PART 16"""
from learner_strategy import learner_strategy
from marketsim import compute_portvals, test_code

"""PART 17"""
returns = dframe.Predicted.dropna()
returns = returns.to_frame()
returns.columns = ["Returns"]

symbol = "IBM"
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
test_code(of = orders_file, sv = start_val)
# test_code(of = orders_file, sv = start_val) uses compute_portvals(orders_file, start_val, allowed_leverage=2.0)
# to compute the portfolio value.
