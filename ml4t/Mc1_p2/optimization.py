"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = np.asarray([0.2, 0.2, 0.3, 0.3, 0.0]) # add code here to find the allocations
    
    spo.minimize(get_portfolio_value, allocs, args=(prices, 1.0))
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    return allocs, cr, adr, sddr, sr

def get_portfolio_value(allocs, prices, start_val=1.0):
    """Given a starting value and prices of
stocks in portfolio with allocations
return the portfolio value over time."""
    normed = prices/prices.iloc[0]
    alloced = np.multiply(allocs, normed)
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val

def get_portfolio_stats(port_val, daily_rf=0.0, samples_per_year=252):
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
    return sr #cr, adr, sddr, sr

def plot_normalized_data(df, title="Daily portfolio value and SPY", 
						 xlabel="Date", ylabel="Normalized price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    df_temp = pd.concat([df, prices_SPY/sv], keys=['Portfolio', 'SPY'], axis=1)
    plot_data(df_temp, title, xlabel, ylabel)
    
def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

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
