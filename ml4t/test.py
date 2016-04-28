import datetime as dt
import pandas as pd
from util import *

start_date = dt.datetime(2004,1,1)
end_date = dt.datetime(2006,1,1)
dates = pd.date_range(start_date, end_date)
symbols = ['AAPL']

df = get_data(symbols, dates, False)
df = df.ffill()
df = df.bfill()

# Add Moving Average Indicator
mva = pd.rolling_mean(df, 20)
mva.columns = ["Moving Average"]
df1 = df.join(mva)

# Add Bollinger Indicator
from indicators.Bollinger import Bollinger
boll = Bollinger()
boll.addEvidence(df)
boll = boll.getIndicator()
df1 = df1.join(boll)

# Add output column
shifted = df.shift(-1)
shifted.columns = ["Output"]
df1 = df1.join(shifted)

# Drop rows without information (ie. NaN for Lagging Indicators)
df1 = df1.dropna()

# Write csv to simData folder so learners can be tested on the financial data
df1.to_csv("simData/axp_example.csv", header=False, index=False)
