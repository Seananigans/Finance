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
    print on_hand
    print portfolio
    on_hand = np.sum(on_hand, axis=1)
    
    for i in range(on_hand.shape[0]):
        if i==0:
            on_hand.iloc[i] += start_val
        else:
            on_hand.iloc[i] += on_hand[i-1]
            portfolio.iloc[i] += portfolio.iloc[i-1]
    print portfolio
    print portvals
    portfolio = portfolio*portvals
    print portfolio
    portfolio = np.sum(portfolio, axis=1)
    
    # 7 Scan cash and value to create total fund value
    portvals = portfolio + on_hand
    print portvals
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
#     of = "./orders/orders-test.csv"
#     of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = pd.to_datetime(portvals.idxmin())
    end_date = pd.to_datetime(portvals.idxmax())
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

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
