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
		
	def add_series(self, series):
		if self.data:
			self.data.join(series, how="inner")
		else: self.data = series
	
	def moving_average(self):
		mv_avg = pd.rolling_mean(self.data, window=window)
		return mv_avg
		
	def moving_std(self):
		mv_std = pd.rolling_std(self.data, window=window)
		return mv_std
		
	def get_bands(self):
		ma = self.moving_average()
		mstd = self.moving_std()
		self.upper_band = ma+mtd
		self.lower_band = ma-mtd
		
bg = Bollinger(10)

def bollinger_bands(data, window=20):
	mv_avg = pd.rolling_mean(data, window)
	mv_std = pd.rolling_std(data, window)
	data.join(pd.DataFrame(mv_avg + mv_std, columns=["Upper"], index=index))
	data.join(pd.DataFrame(mv_avg - mv_std, columns=["Lower"], index=index))

def test_code():
	# Read in adjusted closing prices for given symbols, date range
	sd = dt.datetime(2008,2,28)
	ed = dt.datetime(2009,12,29)
	dates = pd.date_range(sd, ed)
	syms=['GOOG','AAPL','GLD','XOM']
	prices_all = get_data(syms, dates)
	aapl = prices_all["AAPL"]
	bollinger_bands(aapl)
	print aapl

if __name__ == "__main__":
	test_code()