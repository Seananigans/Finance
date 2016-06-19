"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os, sys
from util import get_data, plot_data
from dataset_construction import create_input

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, allowed_leverage=2.0, testing=False):
	print "Starting Cash Value = ${}".format( start_val )
	# 1 Read CSV into trades array
	trades = pd.read_csv(orders_file, index_col="Date", 
						parse_dates=True, na_values=['nan'])
	
	# 2 Scan trades for symbols
	symbols = list(trades.Symbol.unique())
	
	# 3 Scan trades for dates
	start_date = pd.to_datetime(trades.index.min())
	
	# 4 Read in data
	adj_close = create_input(symbols[0], indicators=[], store=False).ix[start_date:,:]
	adj_close = pd.DataFrame(adj_close[[col for col in adj_close.columns if col.startswith("Adj")]])
	adj_close.columns = [symbols[0]]
	portvals= adj_close
	portvals = portvals[symbols]  # remove SPY
	
	# 5 Scan trades to update cash
	on_hand = portvals.copy()
	on_hand[:] = 0.0
	# 6 Scan trades to create ownership array
	portfolio = portvals.copy()
	portfolio[:] = 0.

	# 5 + 6
	shorts = longs = 0
	time_own = portfolio.iloc[0].copy()
	cash= start_val
	for i in range(trades.shape[0]):
		sym = trades.iloc[i].Symbol
		date = pd.to_datetime(trades.iloc[i].name)
		price = portvals.ix[date, sym]
		stock_order = trades.iloc[i].Order
		n_shares = trades.iloc[i].Shares

		#Store old portfolio in case order is rejected
		longs0 = longs
		shorts0 = shorts
		cash0 = cash
		time_own0 = time_own.copy()
		
		if stock_order == "BUY":
			time_own[sym] += n_shares
			cash -= n_shares*price
		else:
			time_own[sym] -= n_shares
			cash += n_shares*price

		longs = sum((time_own[time_own>0]*portvals.ix[date,:]).fillna(0.0))
		shorts = sum((time_own[time_own<0]*portvals.ix[date,:]).fillna(0.0))
		leverage = (longs + abs(shorts)) / (longs - abs(shorts) + cash)
		if testing:
			print "Longs:\t\t{}".format(longs)
			print "Shorts:\t\t{}".format(shorts)
			print "Cash:\t\t{}".format(cash)
			print "Leverage:\t{}".format(leverage)
		
		if leverage >= allowed_leverage:
			shorts, longs, cash = shorts0, longs0, cash0
			time_own = time_own0.copy()
			print "LEVERAGE EXCEEDED"
			print "Potential Leverage:\t{}".format(leverage)
			print "REJECTING ORDER"
			print "Reset Leverage:\t{}".format((longs + abs(shorts)) / (longs - abs(shorts) + cash))
			continue
		
		if cash<0 and testing:
			bad_cash = cash
			shorts, longs, cash = shorts0, longs0, cash0
			time_own = time_own0.copy()
			print "CASH EXCEEDED"
			print "Potential Cash:\t{}".format(bad_cash)
			print "REJECTING ORDER"
			print "Reset Cash:\t{}".format(cash)
			continue
		
		if stock_order == "BUY":
			on_hand.ix[date, sym] -= n_shares*price
			portfolio.ix[date, sym] += n_shares
		else:
			on_hand.ix[date, sym] += n_shares*price
			portfolio.ix[date, sym] -= n_shares

	
	cash = np.sum(on_hand, axis=1)

	for i in range(on_hand.shape[0]):
		if i==0:
			cash.iloc[i] += start_val
		else:
			cash.iloc[i] += cash.iloc[i-1]
			portfolio.iloc[i] += portfolio.iloc[i-1]

	portfolio = portfolio*portvals
	portfolio = np.sum(portfolio, axis=1)
	
	# 7 Scan cash and value to create total fund value
	portvals = pd.DataFrame(portfolio + cash)
	portvals.columns = ["Portfolio"]
	print "Ending Cash Value = ${}".format( cash.iloc[-1] )
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

def plot_normalized(data, symbol=None):
	"""Plot portfolio values for @data normalized by the first value."""
	if symbol:
		title = "Daily portfolio value and {}".format(symbol)
	else:
		title = "Daily portfolio value and $SPX"
	ax = (data/data.iloc[0]).plot()
	plt.title(title)
	plt.ylabel("Normalized Price")
	if symbol:
		plt.savefig("figures/{}.png".format(symbol))
	else:
		plt.savefig("figures/$SPX.png")
        
def test_code(of="./orders/learner_orders.csv",sv = 1000, plotting="f"):
	try:
		symbol = pd.read_csv(of, index_col="Date", parse_dates=True, 
								na_values=['nan']).Symbol[0]
	except IndexError:
		print "There are no orders in this dataframe."
		exit()

	# Process orders
	portvals = compute_portvals(orders_file = of, start_val = sv, allowed_leverage=2.0)
	if isinstance(portvals, pd.DataFrame):
		portvals = portvals[portvals.columns[0]] # just get the first column
	else:
		"warning, code did not return a DataFrame"
	
	# Get portfolio stats
	portvals = portvals.sort_index()
	start_date = pd.to_datetime(portvals.index.min())
	end_date = pd.to_datetime(portvals.index.max())
	dates = pd.date_range(start_date, end_date)

	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals,0.0,252)
        dfSPY = create_input(symbol, indicators = [], store=False).ix[start_date:end_date,:]

	dfSPY = pd.DataFrame(dfSPY[[col for col in dfSPY.columns if col.startswith("Adj")]])
	dfSPY.columns = [symbol]
	
	dfSPY = dfSPY[[symbol]]
	cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(
	dfSPY, 0.0,252)
	
	# Compare portfolio against $SPX
	print "Date Range: {} to {}".format(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
	print
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Sharpe Ratio of {}: {}".format(symbol, float(sharpe_ratio_SPY))
	print
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Cumulative Return of {}: {}".format(symbol, float(cum_ret_SPY))
	print
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Standard Deviation of {}: {}".format(symbol, float(std_daily_ret_SPY))
	print
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Average Daily Return of {}: {}".format(symbol, float(avg_daily_ret_SPY))
	print
	print "Final Portfolio Value: {}".format(portvals[-1])

	temp = dfSPY.join(portvals)
	if plotting=='t':
		plot_normalized(temp, symbol)
	elif plotting=='tt':
		plot_normalized(temp, symbol)
		plt.show()

	
if __name__ == "__main__":
	try:
		plotting=sys.argv[1].lower()
	except IndexError:
		plotting="t"
	
	try:
		sv = float(sys.argv[2])
	except IndexError, ValueError:
		sv = 1000
	test_code(sv=sv, plotting=plotting)
