import datetime as dt
import os
import pandas as pd

def get_and_store_web_data(symbol, online=False):
		"""Retrieve historical data from yahoo finance based off start and end dates for 
		selected symbol."""
		filename = "webdata/{}.csv".format(symbol) #Save location
		
		if online:
			end_date = dt.date.today()
			year = dt.timedelta(days=365*3)
			start_date = end_date - year
		
			import pandas_datareader.data as web
			dframe = web.DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date)
			dframe.columns = [(col+"_"+symbol).replace(" ","") for col in dframe.columns]

			# Fill any missing data
			dframe = dframe.ffill()
			dframe = dframe.bfill()

			# Write csv to webdata folder
			dframe.to_csv(filename, index_label="Date")
		else:
			# Read from webdata folder
			dframe = pd.read_csv(
								filename, 
								index_col='Date', 
								parse_dates=True, 
								na_values=['nan'])
		return dframe

def populate_webdata(replace=True):
	"""Populate webdata folder with All tickers in S&P 500"""
	fhand = pd.read_csv("spy_list.csv")
	spy_list = list(fhand.Symbols)
	for ticker in spy_list:
		if replace:
			try:
				get_and_store_web_data(ticker, True)
			except: continue
		else:
			in_webdata_folder = sum([True for file in os.listdir("webdata") if file.startswith(ticker+".csv")])
			if in_webdata_folder:
				continue
			else:
				try: 
					get_and_store_web_data(ticker, True)
				except: continue

def create_output(symbol, horizon=5, use_prices=False):
	"""Retrieve """
	dframe = get_and_store_web_data(symbol, online=False)
	output = dframe[[col for col in dframe if col.startswith("Adj")]]
	output.columns = ["y_"+symbol for col in output.columns]
	if use_prices:
		return output.shift(-horizon)
	else:
		return output.shift(-horizon)/output - 1

def create_input(symbol, indicators = [], store=False):
	"""Retrieve historical data based off start and end dates for selected symbol."""
	filename = symbol+"_training.csv"
	dframe = get_and_store_web_data(symbol, online=False)
	dframe = dframe[[col for col in dframe.columns if col.startswith("Adj")]]
	adj_close = dframe
    
	for indicator in indicators:
		indicator.addEvidence(adj_close)
		ind_values = indicator.getIndicator()
		dframe = dframe.join(ind_values)
        
	# Write training data to csv in training data folder
	if store:
		dframe.to_csv(filename, index_label="Date")
	return dframe

if __name__ == "__main__":
	symbol = "IBM"
	out = create_output(symbol, horizon=5, use_prices=True)
	print create_input(symbol).join(out)
