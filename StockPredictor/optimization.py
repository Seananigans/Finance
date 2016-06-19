import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo


def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    """Optimizes the distribution of allocations for a set of stock symbols."""

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

	# find the allocations for the optimal portfolio
    #1 provide an initial guess for x
    allocs = np.ones(len(syms))/len(syms)
    #2 Provide constraints to the optimizer
    bounds = [(0,1) for i in syms]
    constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })
    #3 call the optimizer
    res = spo.minimize(get_sharpe_ratio, allocs, 
    					args=prices, 
    					bounds = bounds,
    					constraints=constraints)
    allocs = res.x
    
    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs, 1.0)
    
    # Get portfolio statistics
    cr, adr, sddr, sr = get_portfolio_stats(port_val, 
    										daily_rf=0.0, 
    										samples_per_year=252)
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp)

    return allocs, cr, adr, sddr, sr

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

def plot_normalized_data(df, title="Daily portfolio value and SPY", 
						 xlabel="Date", ylabel="Normalized price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    plot_data(df/df.iloc[0], title, xlabel, ylabel)
    
def test_code():
	"""Tests the optimization code: optimize_portfolio."""
    # Define input parameters
    
    # Example 
    start_date = dt.datetime(2004,1,1)
    end_date = dt.datetime(2006,1,1)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
