"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY = (prices_SPY/prices_SPY.iloc[0])*sv

    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs, sv)
    
    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = get_portfolio_stats(port_val, rfr, sf)
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val/sv, prices_SPY/sv], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp,
                  title="Daily portfolio value and SPY",
                  ylabel="Normalized price")
        
    # Add code here to properly compute end value
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev

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

def plot_normalized_data(df, title="Daily portfolio value and SPY", xlabel="Date", ylabel="Normalized price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    df_temp = pd.concat([df, prices_SPY/sv], keys=['Portfolio', 'SPY'], axis=1)
    plot_data(df_temp, title, xlabel, ylabel)

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
##    start_date = dt.datetime(2009,1,1)
##    end_date = dt.datetime(2010,1,1)
##    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
##    allocations = [0.2, 0.3, 0.4, 0.1]
    #Example 1
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    #Example 2
##    start_date = dt.datetime(2010,1,1)
##    end_date = dt.datetime(2010,12,31)
##    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
##    allocations = [0.0, 0.0, 0.0, 1.0]
    #Example 3
#     start_date = dt.datetime(2010,6,1)
#     end_date = dt.datetime(2010,12,31)
#     symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
#     allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

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
    test_code()
