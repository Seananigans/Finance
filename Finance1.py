import pandas as pd

def get_max_close(symbol):
	"""Return the maximum closing value for stock indicated by symbol
	
	"""
	df = pd.read_csv('data/{}.csv'.format(symbol))
	return df['Close'].max()

def test_run():
	df = pd.read_csv('data/AAPL.csv')
	print df.head()
	
	for symbol in ['AAPL','HCP']:
		print "Max Close"
		print symbol, get_max_close(symbol)

if __name__ == "__main__":
	test_run()