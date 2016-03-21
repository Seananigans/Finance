import matplotlib.pyplot as plt
import os
import pandas as pd

def symbol_to_path(symbol, base_dir="data"):
	"""Return CSV file path given ticker symbol."""
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
	"""Read stock data (adjusted close) for given symbols from CSV files."""
	df = pd.DataFrame(index=dates)
	if 'SPY' not in symbols:  # add SPY for reference if absent
		symbols.insert(0, "SPY")
	
	for symbol in symbols:
		df_symbol = pd.read_csv(symbol_to_path(symbol), 
								index_col="Date", 
								parse_dates=True, 
								usecols=['Date','Adj Close'], 
								na_values=['nan'])
		df_symbol = df_symbol.rename(columns={"Adj Close":symbol})
		df = df.join(df_symbol, how="inner")
		if symbol=="SPY":
			df = df.dropna(subset=['SPY'])
	return df

def get_max_close(symbol):
	"""Return the maximum closing value for stock indicated by symbol
	
	"""
	df = pd.read_csv('data/{}.csv'.format(symbol))
	return df['Close'].max()

def get_mean_volume(symbol):
	
	df = pd.read_csv('data/{}.csv'.format(symbol))
	return df['Volume'].mean()

def plot_multiple(data, *args):
	data[list(args)].plot()
	plt.show()

def start_end(start, end):
	return pd.date_range(start, end)

def read_and_join(df1, *args):
	for symbol in list(args):
		df_symbol = pd.read_csv("data/{}.csv".format(symbol), 
								index_col="Date", 
								parse_dates=True, 
								usecols=['Date','Adj Close'], 
								na_values=['nan'])
		df_symbol = df_symbol.rename(columns={"Adj Close":symbol})
		df1 = df1.join(df_symbol, how="inner")
	return df1
		
def test_run():
	df = pd.read_csv('data/AAPL.csv')
	print df.head()
	
	for symbol in ['AAPL','HCP']:
		print "Max Close"
		print symbol, get_max_close(symbol)
		print "Mean Volume"
		print symbol, get_mean_volume(symbol)
		
# 	plot_multiple(df, "Close","Adj Close")
	start_date = '2010-01-22'
	end_date = '2010-01-26'
	dates =  start_end(start_date, end_date)
	df1 = pd.DataFrame(index=dates)
# 	dfSPY = pd.read_csv("data/SPY.csv", index_col="Date",
# 						parse_dates=True, usecols=['Date','Adj Close'],
# 						na_values=['nan'] )
	df1 = df1.dropna()
# 	df1 = df1.join(dfSPY, how="inner")
	print read_and_join(df1, "SPY", "HCP", "AAPL","IBM")
	
if __name__ == "__main__":
	test_run()