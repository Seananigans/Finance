import datetime as dt
import pandas as pd
from util import *

start_date = dt.datetime(2004,1,1)
end_date = dt.datetime(2006,1,1)
dates = pd.date_range(start_date, end_date)
symbols = ['AAPL']

use_vol=False
df = get_data(symbols, dates, False, vol=use_vol)

df = df.ffill()
df = df.bfill()
df1 = df

if use_vol:
	vol = df['Volume']
	vol = vol.fillna(0.0)
	df1 = df.join(vol)

# Add Bollinger Indicator
from indicators.Bollinger import Bollinger
from indicators.Momentum import Momentum
from indicators.Volatility import Volatility
from indicators.SimpleMA import SimpleMA
from indicators.ExponentialMA import ExponentialMA

indicators = [Bollinger(), Momentum(), Volatility(), SimpleMA(), ExponentialMA()]
for indicator in indicators:
	indicator.addEvidence(df)
	ind_values = indicator.getIndicator()
	df1 = df1.join(ind_values)

# Add output column ***(output should be returns, not prices)***
returns = df.shift(-5)/df - 1
returns.columns=["Returns"]
df1 = df1.join(returns)

# Drop rows without information (ie. NaN for Lagging Indicators)
df1 = df1.dropna()
print df1
# Write csv to simData folder so learners can be tested on the financial data
df1.to_csv("simData/example.csv", header=False, index=False)
