"""MLT: Utility code."""

import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

def symbol_to_path(symbol, base_dir=os.path.join("", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, vol=False):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
    	if vol:
			df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
					parse_dates=True, usecols=['Date', 'Volume'], na_values=['nan'])
			df_temp = df_temp.rename(columns={'Volume': 'Volume_'+symbol})
        else:
			df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
					parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
			df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", filename=None):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if filename is not None:
        plt.savefig(filename)
        
    plt.show()

def create_training_data(symbol, start_date, end_date,
                         horizon = 5, # Num. days ahead to predict
                         filename = "simData/example.csv", #Save location
                         use_web = False, #Use the web to gether Adjusted Close data?
                         use_vol = False, #Use the volume of stocks traded that day?
                         use_prices = False, #Use future Adj. Close as opposed to future returns
                         indicators = ['Bollinger',
                                       'Momentum',
                                       'Volatility',
                                       'SimpleMA',
                                       'ExponentialMA']
                         ):
    
        """Retrieve historical data based off start and end dates for selected symbol.
Create and store a training dataframe:
        Features - adj close
                   indicators from indicator list
        Prediction - Future return or
                     Future Adj. Close
"""
        
        if use_web:
                import pandas_datareader.data as web
                adj_close = web.DataReader(name=symbols[0], data_source='yahoo', start=start_date, end=end_date)
                adj_close = pd.DataFrame(adj_close["Adj Close"])
                adj_close.columns = [symbol]
        else:
                dates = pd.date_range(start_date, end_date)
                adj_close = get_data([symbol], dates, False, vol=False)

        # Fill any missing data
        adj_close = adj_close.ffill()
        adj_close = adj_close.bfill()
        df1 = adj_close

        # Add trade volume data to training data
        if use_vol:
                vol = get_data([symbol], dates, False, vol=use_vol)
                vol = vol.fillna(0.0)
                df1 = adj_close.join(vol)

        # Add Indicators from List provided by user
        indicator_list = []
        if "Bollinger" in indicators:
                from indicators.Bollinger import Bollinger
                indicator_list.append(Bollinger())
        if "Momentum" in indicators:
                from indicators.Momentum import Momentum
                indicator_list.append(Momentum())
        if "Volatility" in indicators:
                from indicators.Volatility import Volatility
                indicator_list.append(Volatility())
        if "SimpleMA" in indicators:
                from indicators.SimpleMA import SimpleMA
                indicator_list.append(SimpleMA())
        if "ExponentialMA" in indicators:
                from indicators.ExponentialMA import ExponentialMA
                indicator_list.append(ExponentialMA())

        for indicator in indicator_list:
                indicator.addEvidence(adj_close)
                ind_values = indicator.getIndicator()
                df1 = df1.join(ind_values)
                
        # Add output column ***(output should be returns, not prices)***
        if not use_prices:
                returns = adj_close[[symbol]].shift(-horizon)/adj_close[[symbol]] - 1
        else:
                returns = adj_close[[symbol]].shift(-horizon)
        returns.columns = ["Returns_"+symbol]
        df1 = df1.join(returns)

        # Drop rows without information (ie. NaN for Lagging Indicators)
        df1 = df1.dropna()

        # Write csv to simData folder so learners can be tested on the financial data
        df1.to_csv(filename, index_label="Date")

def test_create_training_data():
    start_date = dt.datetime(2004,1,1)
    end_date = dt.datetime(2006,1,1)
    symbol = 'AAPL'
    
    create_training_data(symbol, start_date, end_date)

if __name__ == "__main__":
	test_create_training_data()
