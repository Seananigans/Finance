import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from util import *

start_date = dt.datetime(2004,1,1)
end_date = dt.datetime(2006,1,1)
dates = pd.date_range(start_date, end_date)
symbols = ['AAPL',"IBM"]

chosen_symbol = symbols[0]
prices_preferred = False

use_vol=False
# df = get_data(symbols, dates, False, vol=False)

df = web.DataReader(name=symbols[0], data_source='yahoo', start=start_date, end=end_date)
df = pd.DataFrame(df["Adj Close"])
df.columns = [chosen_symbol]

df = df.ffill()
df = df.bfill()
df1 = df

if use_vol:
	vol = get_data(symbols, dates, False, vol=use_vol)
	vol = vol.fillna(0.0)
	df1 = df.join(vol)

# Add Bollinger Indicator
from indicators.Bollinger import Bollinger
from indicators.Momentum import Momentum
from indicators.Volatility import Volatility
from indicators.SimpleMA import SimpleMA
from indicators.ExponentialMA import ExponentialMA


indicators = [Bollinger(),
              Momentum(),
              Volatility(),
              SimpleMA(),
              ExponentialMA()]

for indicator in indicators:
	indicator.addEvidence(df)
	ind_values = indicator.getIndicator()
	df1 = df1.join(ind_values)
	
# Add output column ***(output should be returns, not prices)***
if not prices_preferred:
        returns = df[[chosen_symbol]]/df[[chosen_symbol]].shift(5) - 1
else:
        returns = df[[chosen_symbol]].shift(-5)
returns.columns= ["Returns_"+chosen_symbol]
df1 = df1.join(returns)

# Drop rows without information (ie. NaN for Lagging Indicators)
df1 = df1.dropna()

# Write csv to simData folder so learners can be tested on the financial data
df1.to_csv("simData/example.csv", index_label="Date")
