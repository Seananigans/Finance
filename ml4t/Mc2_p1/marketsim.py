"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
	# this is the function the autograder will call to test your code
	# TODO: Your code here
	# 1 Read CSV into trades array
	trades = pd.read_csv(orders_file, index_col="Date", 
						parse_dates=True, na_values=['nan'])
	print trades

	# 2 Scan trades for symbols
	symbols = list(trades.Symbol.unique())
	# 3 Scan trades for dates
	start_date = pd.to_datetime(trades.index.min())
	end_date = pd.to_datetime(trades.index.max())
	
	# 4 Read in data
	portvals = get_data(symbols, pd.date_range(start_date, end_date))
	portvals = portvals[symbols]  # remove SPY
	
	# 5 Scan trades to update cash
	on_hand = portvals.copy()
	on_hand[:] = 0.0
	# 6 Scan trades to create ownership array
	portfolio = portvals.copy()
	portfolio[:] = 0.
	# 5 + 6
	for i in range(trades.shape[0]):
		sym = trades.iloc[i].Symbol
		date = pd.to_datetime(trades.iloc[i].name)
		price = portvals.ix[date, sym]
		stock_order = trades.iloc[i].Order
		n_shares = trades.iloc[i].Shares
		if stock_order == "BUY":
			on_hand.ix[date, sym] -= n_shares*price
			portfolio.ix[date, sym] += n_shares
		else:
			on_hand.ix[date, sym] += n_shares*price
			portfolio.ix[date, sym] -= n_shares

	on_hand = np.sum(on_hand, axis=1)
	
	for i in range(on_hand.shape[0]):
		if i==0:
			on_hand.iloc[i] += start_val
		else:
			on_hand.iloc[i] += on_hand[i-1]
			portfolio.iloc[i] += portfolio.iloc[i-1]
	portfolio = portfolio*portvals
	portfolio = np.sum(portfolio, axis=1)
	
	# 7 Scan cash and value to create total fund value
	portvals = portfolio + on_hand
	return portvals

def get_portfolio_value(prices, allocs, start_val):
	"""Given a starting value and prices of
stocks in portfolio with allocations
return the portfolio value over time."""
	normed = prices/prices.iloc[0]
	alloced = np.multiply(allocs, normed)
	pos_vals = alloced * start_val
	port_val = pos_vals.sum(axis=1)
	return port_val

def get_portfolio_stats(port_val, daily_rf, samples_per_year):
	"""Given portfolio values return:
Cummulative return (cr)
Average Daily Return (adr)
Standard Deviation of Daily Return (sddr)
Sharpe Ratio (sr)."""
	cr = port_val.iloc[-1]/port_val.iloc[0] - 1.0
	daily_returns = (port_val / port_val.shift(1)) - 1
	daily_returns = daily_returns.iloc[1:]
	adr = daily_returns.mean()
	sddr = daily_returns.std()
	sr = (daily_returns-daily_rf).mean()/sddr *np.sqrt(samples_per_year)
	return cr, adr, sddr, sr

def get_sharpe_ratio(allocs, prices):
	"""Calculate sharpe ratio for minimizer."""
	port_val = get_portfolio_value(prices, allocs, start_val=1.0)
	sharpe_ratio = get_portfolio_stats(port_val, daily_rf=0.0, samples_per_year=252)[3]
	return -sharpe_ratio
	
def test_code():
	# this is a helper function you can use to test your code
	# note that during autograding his function will not be called.
	# Define input parameters

	of = "./orders/orders2.csv"
	of = "./orders/orders3.csv"
#	  of = "./orders/orders-test.csv"
# 	of = "./orders/orders-short.csv"
# 	of = "./orders/orders.csv"
	sv = 1000000

	# Process orders
	portvals = compute_portvals(orders_file = of, start_val = sv)
	if isinstance(portvals, pd.DataFrame):
		portvals = portvals[portvals.columns[0]] # just get the first column
	else:
		"warning, code did not return a DataFrame"
	
	# Get portfolio stats
	# Here we just fake the data. you should use your code from previous assignments.
	portvals = portvals.sort_index()
	start_date = pd.to_datetime(portvals.index.min())
	end_date = pd.to_datetime(portvals.index.max())
	dates = pd.date_range(start_date, end_date)
	
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals,0.0,252)
	dfSPY = get_data(["$SPX"], dates)
	dfSPY = dfSPY[["$SPX"]]
	cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(
	dfSPY, 0.0,252)

	# Compare portfolio against $SPX
	print "Date Range: {} to {}".format(start_date, end_date)
	print
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
	print
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
	print
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
	print
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
	print
	print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
	test_code()
