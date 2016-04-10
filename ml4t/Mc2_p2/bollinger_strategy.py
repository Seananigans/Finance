import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
from util import get_data, plot_data

class Bollinger(object):
	def __init__(self, window=20):
		self.window=window
		self.data = None
		
bg = Bollinger(10)

def bollinger_bands(data, window=20):
	sma = pd.DataFrame( pd.rolling_mean(data, window=window))
	upper = pd.DataFrame( sma + 2*pd.rolling_std(data, window=window) )
	lower = pd.DataFrame( sma - 2*pd.rolling_std(data, window=window) )
	
	sma.columns = ["_SMA"]
	upper.columns = ["_UPPER"]
	lower.columns = ["_LOWER"]
	df = sma.join(upper).join(lower)
	return df

def test_code():
	# Read in adjusted closing prices for given symbols, date range
	sd = dt.datetime(2007,12,31)
	ed = dt.datetime(2009,12,31)
	dates = pd.date_range(sd, ed)
	syms=['IBM','GOOG','AAPL','GLD','XOM']
	prices_all = get_data(syms, dates)
	ibm = pd.DataFrame(prices_all[syms[0]])
	b_bands = bollinger_bands(ibm)
	plt.plot(prices_all[syms[0]])
	plt.plot(b_bands.ix[:,0], color='y')
	plt.plot(b_bands.ix[:,1], color='c')
	plt.plot(b_bands.ix[:,2], color='c')
	
	position = None
	for i in range(1,b_bands.shape[0]):
		#Long Entry Signal
		below_lower_before = (ibm.iloc[i-1]<b_bands["_LOWER"].iloc[i-1]).values
		above_lower_now = (ibm.iloc[i]>=b_bands["_LOWER"].iloc[i]).values
		long_entry = below_lower_before and above_lower_now
		#Long Exit Signal
		below_sma_before = (ibm.iloc[i-1]<b_bands["_SMA"].iloc[i-1]).values
		above_sma_now = (ibm.iloc[i]>=b_bands["_SMA"].iloc[i]).values
		long_exit = below_sma_before and above_sma_now and position=="Long"
		#Short Entry Signal
		above_upper_before = (ibm.iloc[i-1]>b_bands["_UPPER"].iloc[i-1]).values
		below_upper_now = (ibm.iloc[i]<=b_bands["_UPPER"].iloc[i]).values
		short_entry = above_upper_before and below_upper_now
		#Short Exit Signal
		above_sma_before = (ibm.iloc[i-1]>b_bands["_SMA"].iloc[i-1]).values
		below_sma_now = (ibm.iloc[i]<=b_bands["_SMA"].iloc[i]).values
		short_exit = above_sma_before and below_sma_now and position=="Short"
		if position == None:
			#Short Entry
			if short_entry:
				plt.axvline(x=ibm.index[i], color="r", linewidth=1)
				position = "Short"
				#issue sell order
				print "{},SYMBOL,SELL,100".format(ibm.index[i].strftime("%Y-%m-%d"))
			#Long Entry
			if long_entry:
				plt.axvline(x=ibm.index[i], color="g", linewidth=1)
				position = "Long"
				#issue buy order
				print "{},SYMBOL,BUY,100".format(ibm.index[i].strftime("%Y-%m-%d"))
		else:
			#Short Exit
			if short_exit:
				plt.axvline(x=ibm.index[i], color="k", linewidth=1)
				position = None
				#issue buy order
				print "{},SYMBOL,BUY,100".format(ibm.index[i].strftime("%Y-%m-%d"))
			#Long Exit
			if long_exit:
				plt.axvline(x=ibm.index[i], color="k", linewidth=1)
				position = None
				#issue sell order
				print "{},SYMBOL,SELL,100".format(ibm.index[i].strftime("%Y-%m-%d"))
	plt.show()

if __name__ == "__main__":
	test_code()