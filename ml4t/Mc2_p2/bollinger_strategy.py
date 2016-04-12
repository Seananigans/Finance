import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
from util import get_data, plot_data

def bollinger_bands(data, window=20):
	sma = pd.DataFrame( pd.rolling_mean(data, window=window))
	upper = pd.DataFrame( sma + 2*pd.rolling_std(data, window=window) )
	lower = pd.DataFrame( sma - 2*pd.rolling_std(data, window=window) )
	
	sma.columns = ["SMA"]
	upper.columns = ["UPPER"]
	lower.columns = ["LOWER"]
	df = data.join(sma.join(upper).join(lower))
	return df

def bollinger_plot(data):
	b_bands = bollinger_bands(data)
	sym = data.columns.values[0]
	symbol=data
	b_bands[[sym,"SMA","UPPER","LOWER"]].plot(color=['b','y','c','c'])
	handles, labels = plt.gca().get_legend_handles_labels()
	handles = handles[:-1]
	labels = labels[:-1]
	labels[-1] = "Bollinger Bands"
	plt.legend(handles, labels, loc="upper left")
	plt.title("{} Adjusted Close".format(sym))
	plt.ylabel("{} Prices".format(sym))
	
	position = None
	print "Date,Symbol,Order,Shares"
	for i in range(1,b_bands.shape[0]):
		#Long Entry Signal
		below_lower_before = (data.iloc[i-1]<b_bands["LOWER"].iloc[i-1]).values
		above_lower_now = (data.iloc[i]>=b_bands["LOWER"].iloc[i]).values
		long_entry = below_lower_before and above_lower_now
		#Long Exit Signal
		below_sma_before = (data.iloc[i-1]<b_bands["SMA"].iloc[i-1]).values
		above_sma_now = (data.iloc[i]>=b_bands["SMA"].iloc[i]).values
		long_exit = below_sma_before and above_sma_now and position=="Long"
		#Short Entry Signal
		above_upper_before = (data.iloc[i-1]>b_bands["UPPER"].iloc[i-1]).values
		below_upper_now = (data.iloc[i]<=b_bands["UPPER"].iloc[i]).values
		short_entry = above_upper_before and below_upper_now
		#Short Exit Signal
		above_sma_before = (data.iloc[i-1]>b_bands["SMA"].iloc[i-1]).values
		below_sma_now = (data.iloc[i]<=b_bands["SMA"].iloc[i]).values
		short_exit = above_sma_before and below_sma_now and position=="Short"
		if position == None:
			#Short Entry
			if short_entry:
				plt.axvline(x=data.index[i], color="r", linewidth=1)
				position = "Short"
				#issue sell order
				print "{0},{1},SELL,100".format(data.index[i].strftime("%Y-%m-%d"),
												sym)
			#Long Entry
			if long_entry:
				plt.axvline(x=data.index[i], color="g", linewidth=1)
				position = "Long"
				#issue buy order
				print "{0},{1},BUY,100".format(data.index[i].strftime("%Y-%m-%d"),
												sym)
		else:
			#Short Exit
			if short_exit:
				plt.axvline(x=data.index[i], color="k", linewidth=1)
				position = None
				#issue buy order
				print "{0},{1},BUY,100".format(data.index[i].strftime("%Y-%m-%d"),
												sym)
			#Long Exit
			if long_exit:
				plt.axvline(x=data.index[i], color="k", linewidth=1)
				position = None
				#issue sell order
				print "{0},{1},SELL,100".format(data.index[i].strftime("%Y-%m-%d"),
												sym)
	plt.show()
	
def test_code():
	# Read in adjusted closing prices for given symbols, date range
	sd = dt.datetime(2007,12,31)
	ed = dt.datetime(2009,12,31)
	dates = pd.date_range(sd, ed)
	syms=['$SPX','IBM','GOOG','AAPL','GLD','XOM']
	prices_all = get_data(syms, dates)
	symbol = pd.DataFrame(prices_all[syms[1]])
	bollinger_plot(symbol)
	
if __name__ == "__main__":
	test_code()
