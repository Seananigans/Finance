def create_training_data(symbol, 
                         start_date = None, 
                         end_date = None,
                         horizon = 5, # Num. days ahead to predict
                         filename = "simData/example.csv", #Save location
                         use_web = False, #Use the web to gether Adjusted Close data?
                         use_vol = False, #Use the volume of stocks traded that day?
                         use_prices = False, #Use future Adj. Close as opposed to future returns
                         direction = False, #Use the direction of the market returns as output
                         indicators = ['Bollinger',
                                       'Momentum',
                                       'Volatility',
                                       'SimpleMA',
                                       'ExponentialMA',
                                       'Lagging',
                                       'Weekdays'],
                        num_lag = 20,
                        check_correlations=False
                         ):
    
        """Retrieve historical data based off start and end dates for selected symbol.
Create and store a training dataframe:
        Features - adj close
                   indicators from indicator list
        Prediction - Future return or
                     Future Adj. Close
"""
        if not end_date or not start_date:
			end_date = dt.date.today()
			year = dt.timedelta(days=365)
			start_date = end_date - year
			
        if use_web:
                import pandas_datareader.data as web
                adj_close = web.DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date)
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
            # Add Bollinger value as indicator
            from indicators.Bollinger import Bollinger
            indicator_list.append(Bollinger())
        if "Momentum" in indicators:
            # Add Momentum value as indicator
            from indicators.Momentum import Momentum
            indicator_list.append(Momentum())
        if "Volatility" in indicators:
            # Add Volatility value as indicator
            from indicators.Volatility import Volatility
            indicator_list.append(Volatility())
        if "SimpleMA" in indicators:
            # Add Simple moving average value as indicator
            from indicators.SimpleMA import SimpleMA
            indicator_list.append(SimpleMA())
        if "ExponentialMA" in indicators:
            # Add exponential moving average value as indicator
            from indicators.ExponentialMA import ExponentialMA
            indicator_list.append(ExponentialMA())
        if "Lagging" in indicators:
            # Add Lagging values as indicators
            from indicators.Lagging import Lag
            for i in range(1,num_lag+1):
                lag = Lag(i)
                lag.addEvidence(adj_close)
                lag_values = lag.getIndicator()
                df1 = df1.join(lag_values)
        if "Weekdays" in indicators:
        	# Add weekdays as indicators
            from indicators.Weekdays import Weekdays
            indicator_list.append(Weekdays())
        
        for indicator in indicator_list:
            indicator.addEvidence(adj_close)
            ind_values = indicator.getIndicator()
            df1 = df1.join(ind_values)
                
        # Add output column ***(output should be returns, not prices)***
        if not use_prices:
            returns = calculate_returns(adj_close[[symbol]],horizon)
            if direction:
                returns[returns.values>0.0] = 1.0
                returns[returns.values<=0.0] = 0.0
        else:
            returns = adj_close[[symbol]].shift(-horizon)
        returns.columns = ["Returns_"+symbol]
        df1 = df1.join(returns)

        # Drop rows without information (ie. NaN for Lagging Indicators)
        df1 = df1.dropna()
        ind_names = [col for col in df1.columns
                       if not col.startswith("Lag") and not col.startswith("Returns")]
        
        # Check correlations between Input features and output
        if check_correlations:
			for name in ind_names:
				print "{}\t".format(name), np.corrcoef(df1[name],
								  df1["Returns_"+symbol])[0][1]
			if "Lagging" in indicators:
				for i in range(1,num_lag+1):
					print "Lag {}\t".format(i), np.corrcoef(df1["Lag{}_".format(i) + symbol],
									  df1["Returns_"+symbol])[0][1]
        # Write csv to simData folder so learners can be tested on the financial data
        df1.to_csv(filename, index_label="Date")

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

import datetime as dt
import os
import pandas as pd

def get_and_store_web_data(symbol, online=False):
		"""Retrieve historical data from yahoo finance based off start and end dates for 
		selected symbol."""
		filename = "webdata/{}.csv".format(symbol) #Save location
		
		if online:
			end_date = dt.date.today()
			year = dt.timedelta(days=365)
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