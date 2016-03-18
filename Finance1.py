import matplotlib.pyplot as plt
import pandas as pd


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

def test_run():
	df = pd.read_csv('data/AAPL.csv')
	print df.head()
	
	for symbol in ['AAPL','HCP']:
		print "Max Close"
		print symbol, get_max_close(symbol)
		print "Mean Volume"
		print symbol, get_mean_volume(symbol)
		
	print df['Adj Close']
	plot_multiple(df, "Close","Adj Close")
# 	df['Adj Close'].plot()
# 	plt.show()
# 	

if __name__ == "__main__":
	test_run()